# file: app.py
from __future__ import annotations

import json, math, re, hashlib, random, os
from dataclasses import dataclass
from io import StringIO, BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# PDF (ReportLab)
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# ====================== Canonical dimensions ======================

MOTIVATIONS: List[str] = [
    "Sattva", "Rajas", "Tamas",
    "Prajna", "Personal_Unconscious", "Collective_Unconscious",
    "Cheng", "Wu_Wei", "Anatta",
    "Relational_Balance", "Thymos", "Eros",
]
STRATEGIES: List[str] = ["Conform", "Control", "Flow", "Risk"]

# Final orientations
ORIENTATIONS: List[str] = ["Cognitive", "Energy", "Relational", "Surrender"]

SELF_SCALES: List[str] = ["Self_Insight", "Self_Serving_Bias"]
ALL_DIMS: List[str] = MOTIVATIONS + STRATEGIES + ORIENTATIONS
ALL_REQ_FOR_Z: List[str] = ALL_DIMS + SELF_SCALES

DEFAULT_DOMAIN_MAP = {
    "Energy": ["Sattva","Rajas","Tamas"],
    "Cognitive": ["Prajna","Personal_Unconscious","Collective_Unconscious"],
    "Integrative": ["Cheng","Wu_Wei","Anatta"],
    "Relational": ["Relational_Balance","Thymos","Eros"],
}

# ====================== Header normalization ======================

HEADER_TO_CANON = {
    # motivations
    "sattva":"Sattva","rajas":"Rajas","tamas":"Tamas",
    "prajna":"Prajna","prajna logos":"Prajna","prajna-logos":"Prajna",
    "personal unconscious":"Personal_Unconscious","personal_unconscious":"Personal_Unconscious","pers.u":"Personal_Unconscious","pers u":"Personal_Unconscious",
    "collective unconscious":"Collective_Unconscious","collective_unconscious":"Collective_Unconscious","coll.u":"Collective_Unconscious","coll u":"Collective_Unconscious",
    "cheng":"Cheng","wu wei":"Wu_Wei","wu_wei":"Wu_Wei","anatta":"Anatta",
    "relational balance":"Relational_Balance","relational_balance":"Relational_Balance","rel.bal":"Relational_Balance","rel bal":"Relational_Balance",
    "thymos":"Thymos","eros":"Eros",
    # strategies
    "conform":"Conform","control":"Control","flow":"Flow","risk":"Risk",
    # orientations (new canon + aliases)
    "cognitive":"Cognitive","energy":"Energy","relational":"Relational","relationship":"Relational","relationship value":"Relational","surrender":"Surrender",
    "inward":"Cognitive","outward":"Energy",
    # self scales
    "self insight":"Self_Insight","self_insight":"Self_Insight","self-insight":"Self_Insight","self–insight":"Self_Insight",
    "self serving bias":"Self_Serving_Bias","self-serving bias":"Self_Serving_Bias","self_serving_bias":"Self_Serving_Bias",
}
def canon(name: str) -> Optional[str]:
    if not name: return None
    s = name.strip().lstrip("\ufeff")
    s = s.replace("—","-").replace("–","-").replace("：",":")
    s = re.sub(r"\(.*?\)", " ", s).lower()
    s = re.sub(r"[^a-z0-9_ :\-]+", " ", s).replace(":", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if s in HEADER_TO_CANON: return HEADER_TO_CANON[s]
    best = None
    for k, v in HEADER_TO_CANON.items():
        if k in s and (best is None or len(k) > len(best[0])): best = (k, v)
    return best[1] if best else None

# ====================== Centroids (embedded, adjusted peaks) ======================

COLUMN_NORMALIZATION = {
    "Sattva":"Sattva","Rajas":"Rajas","Tamas":"Tamas","Prajna":"Prajna",
    "Pers.U":"Personal_Unconscious","Personal Unconscious":"Personal_Unconscious","Personal_Unconscious":"Personal_Unconscious",
    "Coll.U":"Collective_Unconscious","Collective Unconscious":"Collective_Unconscious","Collective_Unconscious":"Collective_Unconscious",
    "Cheng":"Cheng","Wu Wei":"Wu_Wei","Wu_Wei":"Wu_Wei",
    "Anatta":"Anatta","Rel.Bal":"Relational_Balance","Relational Balance":"Relational_Balance","Relational_Balance":"Relational_Balance",
    "Thymos":"Thymos","Eros":"Eros",
    "Conform":"Conform","Control":"Control","Flow":"Flow","Risk":"Risk",
    "Outward":"Energy","Inward":"Cognitive","Relational":"Relational","Relationship":"Relational","Surrender":"Surrender",
}
def normalize_centroid_headers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={k:v for k,v in COLUMN_NORMALIZATION.items() if k in df.columns})
    missing = set(ALL_DIMS) - set(out.columns)
    if missing: raise KeyError(f"Archetype centroids missing columns: {sorted(missing)}")
    return out[ALL_DIMS].copy()

_ARC_ROWS = [
    "Lucerna (Lantern)","Arbor (Tree)","Sharin (Wheel)","Keras (Rhino)",
    "Hayabusa (Falcon)","Arachna (Spider)","Tempus (Hourglass)","Simia (Monkey)",
    "Polvo (Octopus)","Tigre (Tiger)","Enguia (Eel)","Dacia (Nomad)",
]

# Columns: motivations..., Conform, Control, Flow, Risk, Outward, Inward, Surrender, Relational
_ARC_RAW = pd.DataFrame(
    [
        # Lucerna → Energy peak
        [4.8,6.8,3.8,6.0,4.7,4.5,5.0,4.9,4.3,4.7,6.2,5.0, 4.5,4.8,6.0,4.2, 6.8,4.6,4.6,5.2],
        # Arbor → Relational peak (top-3 group)
        [6.4,4.5,4.7,4.4,4.5,4.8,6.0,4.8,4.3,6.7,5.0,5.3, 6.0,4.8,4.2,3.8, 4.9,4.9,5.1,6.5],
        # Sharin → Relational peak (highest)
        [5.0,6.2,4.2,4.8,4.5,4.7,5.4,5.8,4.5,5.3,6.5,5.0, 5.5,4.8,6.0,4.5, 5.2,4.7,5.2,6.7],
        # Keras → Surrender peak (highest)
        [4.8,6.2,5.7,4.8,4.4,4.5,6.7,4.7,4.0,4.5,5.2,4.4, 4.3,6.0,4.5,5.8, 5.0,4.6,6.7,4.6],
        # Hayabusa → Cognitive peak (highest)
        [4.6,5.3,4.8,4.7,6.7,4.5,6.3,4.4,4.2,4.7,5.7,4.4, 5.8,6.0,4.3,4.2, 5.2,6.7,4.8,4.9],
        # Arachna → Energy peak (third)
        [4.2,4.8,6.5,4.5,4.7,6.3,4.3,4.8,6.4,4.2,4.8,4.5, 3.8,4.3,4.8,6.0, 6.6,4.8,4.9,4.3],
        # Tempus → Cognitive peak (second)
        [6.2,5.0,4.3,6.6,5.3,5.0,6.1,4.8,4.5,4.9,4.8,4.7, 4.5,6.0,5.8,4.3, 5.1,6.6,5.1,4.9],
        # Simia → Surrender peak (second)
        [4.8,5.8,4.2,4.6,4.7,4.8,4.6,6.5,4.3,5.0,5.0,6.2, 4.0,4.3,6.0,4.8, 5.0,4.7,6.6,5.6],
        # Polvo → Surrender peak (third)
        [4.6,4.7,5.5,6.3,5.4,5.1,4.8,4.5,6.4,4.4,4.6,4.5, 4.8,4.8,4.2,4.0, 4.6,4.8,6.5,4.4],
        # Tigre → Energy peak (second)
        [6.5,6.2,4.4,4.9,4.7,5.8,4.9,5.6,4.3,4.8,4.8,4.7, 3.9,4.7,4.8,6.0, 6.7,4.7,4.9,4.8],
        # Enguia → Relational peak (second)
        [6.3,4.7,4.3,4.5,4.8,4.9,4.8,5.0,4.4,6.5,5.2,6.7, 4.6,4.4,5.8,4.3, 5.3,4.9,5.3,6.6],
        # Dacia → Cognitive peak (third)
        [4.8,6.5,4.5,5.6,4.8,6.7,4.7,5.1,5.2,4.6,6.1,5.5, 4.2,4.8,5.8,5.7, 5.0,6.5,5.0,5.1],
    ],
    index=_ARC_ROWS,
    columns=[
        "Sattva","Rajas","Tamas","Prajna","Pers.U","Coll.U","Cheng","Wu Wei","Anatta","Rel.Bal","Thymos","Eros",
        "Conform","Control","Flow","Risk","Outward","Inward","Surrender","Relational",
    ],
)
ARCHETYPE_CENTROIDS = normalize_centroid_headers(_ARC_RAW)

# --- Sanity check: orientation top-3 peaks as specified ---
def _check_orientation_peaks(centroids: pd.DataFrame) -> None:
    expected_order = {
        "Energy":     ["Lucerna (Lantern)", "Tigre (Tiger)", "Arachna (Spider)"],
        "Cognitive":  ["Hayabusa (Falcon)", "Tempus (Hourglass)", "Dacia (Nomad)"],
        "Surrender":  ["Keras (Rhino)", "Simia (Monkey)", "Polvo (Octopus)"],
        "Relational": ["Sharin (Wheel)", "Enguia (Eel)", "Arbor (Tree)"],
    }
    problems = []
    for col, want in expected_order.items():
        if col not in centroids.columns:
            problems.append(f"Missing orientation column: {col}")
            continue
        top3 = centroids[col].nlargest(3).index.tolist()
        want_set, top_set = set(want), set(top3)
        missing = list(want_set - top_set)
        extra   = list(top_set - want_set)
        order_ok = (not missing and not extra and top3 == want)
        if missing or extra or not order_ok:
            msg = f"**{col}** expected: {want} | actual top-3: {top3}"
            if missing: msg += f" | missing: {missing}"
            if extra:   msg += f" | unexpected: {extra}"
            if not order_ok and not (missing or extra):
                msg += " | note: membership matches but order differs"
            problems.append(msg)
    if problems:
        st.warning("Centroid orientation sanity check failed:\n\n- " + "\n- ".join(problems))
    else:
        st.success("Centroid orientation sanity check passed for Energy, Cognitive, Surrender, Relational.")

_check_orientation_peaks(ARCHETYPE_CENTROIDS)

# ====================== Scoring engine ======================

EPS = 1e-8
W_MOT_ABS, W_STRAT_MATCH, W_ORIENT_MATCH = 0.60, 0.20, 0.20

@dataclass
class ZParams:
    mean: Dict[str,float]; std: Dict[str,float]
    @classmethod
    def fit(cls, df: pd.DataFrame, cols: List[str]) -> "ZParams":
        mu = df[cols].mean().to_dict()
        sd = (df[cols].std(ddof=0) + EPS).to_dict()
        return cls(mu, sd)
    def zrow(self, row: pd.Series, cols: List[str]) -> np.ndarray:
        return np.array([(row[c] - self.mean[c]) / self.std[c] for c in cols], float)

def intra_person_z(vals: np.ndarray) -> np.ndarray:
    mu, sd = float(vals.mean()), float(vals.std()) + EPS
    return (vals - mu) / sd

def rowwise_motivation_z(arch: pd.DataFrame) -> pd.DataFrame:
    df = arch.copy()
    mot = df[MOTIVATIONS].to_numpy(float)
    mu = mot.mean(1, keepdims=True); sd = mot.std(1, keepdims=True) + EPS
    df[MOTIVATIONS] = (mot - mu) / sd
    return df

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = float(a.std()), float(b.std())
    if sa < EPS or sb < EPS: return 0.0
    return float(np.corrcoef(a, b)[0,1])

def euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a-b)**2)))

def sigmoid(x: float) -> float:
    x = max(min(x, 35.0), -35.0)
    return 1/(1+math.exp(-x))

def normalize_probs(v: np.ndarray) -> np.ndarray:
    s = float(v.sum())
    return v/s if s>0 else np.full_like(v, 1.0/len(v))

def quadrant(z_control: float, z_flow: float, z_conform: float, z_risk: float) -> Tuple[str, Dict[str,float]]:
    x, y = z_control - z_flow, z_conform - z_risk
    if x>=0 and y>=0: lab="Controlled–Conformist"
    elif x>=0 and y<0: lab="Controlled–Risk"
    elif x<0 and y>=0: lab="Flow–Conformist"
    else: lab="Flow–Risk"
    return lab, {"axis_CF": x, "axis_CR": y}

def domain_means(row: pd.Series) -> Dict[str,float]:
    return {d: float(np.nanmean([row[k] for k in ks])) for d, ks in DEFAULT_DOMAIN_MAP.items()}

def score_single(person: pd.Series, z: ZParams, arch: pd.DataFrame, arch_mz: pd.DataFrame, arch_std: pd.DataFrame) -> Dict:
    z_mot = z.zrow(person, MOTIVATIONS)
    z_str = z.zrow(person, STRATEGIES)
    z_ori = z.zrow(person, ORIENTATIONS)
    z_pat = intra_person_z(np.array([person[m] for m in MOTIVATIONS], float))
    z_SI  = (person["Self_Insight"]-z.mean["Self_Insight"])/z.std["Self_Insight"]
    z_SSB = (person["Self_Serving_Bias"]-z.mean["Self_Serving_Bias"])/z.std["Self_Serving_Bias"]
    C = 0.7*sigmoid(float(z_SI)) + 0.3*(1.0 - 2.0*abs(sigmoid(float(z_SSB))-0.5))

    names = list(arch.index); vals=[]
    for name in names:
        ap = arch_mz.loc[name, MOTIVATIONS].to_numpy(float)
        R = pearson(z_pat, ap); Rp = (R+1)/2
        am = arch_std.loc[name, MOTIVATIONS].to_numpy(float)
        as_ = arch_std.loc[name, STRATEGIES].to_numpy(float)
        ao = arch_std.loc[name, ORIENTATIONS].to_numpy(float)
        Dall = euclid(np.concatenate([z_mot,z_str,z_ori]), np.concatenate([am,as_,ao]))
        S_abs = 1/(1+Dall); S_A = 0.65*Rp + 0.35*S_abs
        DS = euclid(z_str, as_); SS = 1/(1+DS)
        DO = euclid(z_ori, ao); SO = 1/(1+DO)
        S_total = W_MOT_ABS*S_A + W_STRAT_MATCH*SS + W_ORIENT_MATCH*SO
        vals.append(S_total*(0.75+0.25*C))

    probs = normalize_probs(np.array(vals,float))
    order = np.argsort(-probs); top3=[(names[i], float(probs[i])) for i in order[:3]]
    qc, axes = quadrant(
        z_str[STRATEGIES.index("Control")],
        z_str[STRATEGIES.index("Flow")],
        z_str[STRATEGIES.index("Conform")],
        z_str[STRATEGIES.index("Risk")]
    )
    return {"probs": {names[i]: float(probs[i]) for i in range(len(names))}, "top3": top3,
            "quadrant": qc, "quadrant_axes": axes, "confidence": float(C),
            "domain_centroids": domain_means(person)}

# ====================== TXT parser ======================

ITEM_KV_RE = re.compile(r"\[(\w+)\s*=\s*(.*?)\]")  # [KEY=VALUE]
def _kv_blocks(s: str) -> Dict[str,str]:
    return {k.upper(): v.strip() for k, v in ITEM_KV_RE.findall(s)}

def parse_txt_questions(raw: str) -> Dict:
    raw = raw.lstrip("\ufeff")
    lines = [l.rstrip() for l in raw.splitlines()]
    spec = {"scale": {"min":1, "max":7, "step":1}, "questions":[], "parse_report": {"unknown_headers":[], "items_without_dim":[]}}
    cur_dim: Optional[str] = None
    counts: Dict[str,int] = {}

    for idx, line in enumerate(lines, start=1):
        s = line.strip()
        if not s: continue
        if s.endswith(":") or s.endswith("："):
            header_txt = s[:-1]
            maybe = canon(header_txt)
            if maybe is None:
                spec["parse_report"]["unknown_headers"].append({"line": idx, "text": line})
                cur_dim = None
            else:
                cur_dim = maybe
            continue

        text = s
        kvs = _kv_blocks(s)
        if kvs: text = ITEM_KV_RE.sub("", s).strip()

        dim = canon(kvs.get("DIM", cur_dim or ""))
        if dim is None:
            spec["parse_report"]["items_without_dim"].append({"line": idx, "text": line})
            continue

        vmin = int(kvs.get("MIN", "1"))
        vmax = int(kvs.get("MAX", "7"))
        vstep = int(kvs.get("STEP", "1"))
        if vmax <= vmin or vstep <= 0:
            raise ValueError(f"Bad range for '{text}' on line {idx}: MIN={vmin} MAX={vmax} STEP={vstep}")
        npoints = (vmax - vmin) // vstep + 1

        labels: Optional[List[str]] = None
        if "LABELS" in kvs:
            labels = [p.strip() for p in kvs["LABELS"].split("|")]
            if len(labels) != npoints:
                raise ValueError(f"LABELS count ({len(labels)}) must equal number of points ({npoints}) for '{text}' (line {idx})")
        else:
            L = kvs.get("L"); R = kvs.get("R")
            if L and R and npoints == 7:
                labels = [L, "Slightly "+L, "Somewhat "+L, "Neutral", "Somewhat "+R, "Slightly "+R, R]
            elif L and R and npoints == 5:
                labels = [L, "Somewhat "+L, "Neutral", "Somewhat "+R, R]

        counts[dim] = counts.get(dim, 0) + 1
        qid = f"q_{dim}_{counts[dim]}"
        spec["questions"].append({
            "id": qid, "dimension": dim, "text": text,
            "min": vmin, "max": vmax, "step": vstep,
            "labels": labels, "L": kvs.get("L"), "R": kvs.get("R"),
        })
    return spec

# ====================== Aggregation & z-params ======================

def aggregate_to_scales(responses: Dict[str,int], spec: Dict) -> Dict[str,float]:
    buckets: Dict[str, List[float]] = {d: [] for d in (ALL_DIMS + SELF_SCALES)}
    for q in spec["questions"]:
        dim = q["dimension"]; qid = q["id"]
        if qid in responses and responses[qid] is not None:
            buckets[dim].append(float(responses[qid]))
    means: Dict[str, float] = {}
    for d in ALL_DIMS + SELF_SCALES:
        vals = buckets.get(d, [])
        means[d] = float(np.mean(vals)) if len(vals) > 0 else np.nan
    return means

def zparams_from_norms_or_single(person_scales: Dict[str,float], norms_df: Optional[pd.DataFrame]) -> ZParams:
    if norms_df is not None:
        missing = [c for c in ALL_REQ_FOR_Z if c not in norms_df.columns]
        if missing: raise ValueError(f"Norms CSV missing columns: {missing}")
        return ZParams.fit(norms_df, list(ALL_REQ_FOR_Z))
    df1 = pd.DataFrame([person_scales])[ALL_REQ_FOR_Z]
    return ZParams.fit(df1, list(ALL_REQ_FOR_Z))

def prepare_archetype_pieces(z: ZParams) -> Tuple[pd.DataFrame,pd.DataFrame]:
    arch_mz = rowwise_motivation_z(ARCHETYPE_CENTROIDS)
    arch_std = ARCHETYPE_CENTROIDS.copy()
    for c in ALL_DIMS:
        arch_std[c] = (arch_std[c] - z.mean[c]) / z.std[c]
    return arch_mz, arch_std

def to_result_df(res: Dict, pid: str) -> pd.DataFrame:
    (p1,p1v),(p2,p2v),(p3,p3v) = res["top3"]
    row = {"participant_id": pid,"confidence": res["confidence"],"quadrant": res["quadrant"],
           "axis_CF": res["quadrant_axes"]["axis_CF"],"axis_CR": res["quadrant_axes"]["axis_CR"],
           "primary": p1,"primary_prob": p1v,"secondary": p2,"secondary_prob": p2v,"tertiary": p3,"tertiary_prob": p3v,
           **res["domain_centroids"], **res["probs"]}
    return pd.DataFrame([row])

def top3_percentages(top3: List[Tuple[str,float]]) -> List[Tuple[str,int]]:
    vals = [p for _, p in top3]; s = sum(vals) or 1.0
    raw = [p / s * 100.0 for p in vals]
    a = int(round(raw[0])); b = int(round(raw[1])); c = 100 - a - b
    return [(top3[0][0], a), (top3[1][0], b), (top3[2][0], c)]

# ====================== Load questions from repo or override ======================

def load_questions_from_repo() -> Dict:
    q_path = Path(os.getenv("QUESTIONS_PATH", "questions.txt"))
    if not q_path.exists():
        raise FileNotFoundError(f"questions file not found at: {q_path.resolve()}")
    text = q_path.read_text(encoding="utf-8").lstrip("\ufeff")
    return parse_txt_questions(text)

# ====================== UI ======================

st.set_page_config(page_title="Motivational Archetypes – Test", page_icon="🧭", layout="wide")
st.title("🧭 Motivational Archetypes – Test")

with st.sidebar:
    st.header("Inputs")
    st.caption("Reads **questions.txt** from the repo. You can override it below.")
    q_up = st.file_uploader("Override questions.txt (optional)", type=["txt"])
    norms_up = st.file_uploader("Optional norms.csv", type=["csv"])
    participant_id = st.text_input("Participant ID", value="P001")
    ranking_mode = st.selectbox("Motivation ranking metric", ["Raw means (1–7)", "Z-scores (vs norms)"])
    st.caption("Two-column layout. Labels update live.")

# Load spec (repo first; uploader overrides if provided)
try:
    spec = parse_txt_questions(q_up.read().decode("utf-8")) if q_up is not None else load_questions_from_repo()
except Exception as e:
    st.error(f"Failed to load/parse questions: {e}")
    st.stop()

rep = spec.get("parse_report", {})
if rep.get("unknown_headers") or rep.get("items_without_dim"):
    with st.expander("Parsing report (click to review)"):
        if rep.get("unknown_headers"):
            st.warning("Unknown headers (not matched to a dimension):")
            st.json(rep["unknown_headers"])
        if rep.get("items_without_dim"):
            st.warning("Items skipped (no active dimension):")
            st.json(rep["items_without_dim"])

items: List[Dict] = list(spec.get("questions", []))
if not items:
    st.error("No items parsed. Check your headers and items.")
    st.stop()

# Stable per-user shuffle
def stable_shuffle(items: List[Dict], pid: str, spec_obj: Dict) -> List[Dict]:
    spec_bytes = json.dumps({k:v for k,v in spec_obj.items() if k != "parse_report"}, sort_keys=True).encode("utf-8")
    seed_hex = hashlib.sha256((pid + "|").encode("utf-8") + spec_bytes).hexdigest()[:16]
    seed_int = int(seed_hex, 16)
    rng = random.Random(seed_int)
    out = items.copy(); rng.shuffle(out); return out

spec_fingerprint = hashlib.sha256(json.dumps({k:v for k,v in spec.items() if k != "parse_report"}, sort_keys=True).encode("utf-8")).hexdigest()
if "shuffle_meta" not in st.session_state or st.session_state.shuffle_meta != (participant_id, spec_fingerprint):
    st.session_state.shuffle_meta = (participant_id, spec_fingerprint)
    st.session_state.shuffled_items = stable_shuffle(items, participant_id, spec)
shuffled = st.session_state.shuffled_items

# ============ Live label mapping ============

def value_to_label(item: Dict, val: int) -> str:
    vmin = int(item.get("min", 1)); vmax = int(item.get("max", 7)); step = int(item.get("step", 1))
    labels = item.get("labels"); idx = (val - vmin) // step
    if labels and 0 <= idx < len(labels): return labels[idx]
    L, R = item.get("L"), item.get("R")
    npoints = (vmax - vmin)//step + 1
    if L and R and npoints == 7:
        return [L, f"Slightly {L}", f"Somewhat {L}", "Neutral", f"Somewhat {R}", f"Slightly {R}", R][idx]
    if L and R and npoints == 5:
        return [L, f"Somewhat {L}", "Neutral", f"Somewhat {R}", R][idx]
    return ""

# ============ Render (live updates) ============

responses: Dict[str,int] = {}
scale_defaults = spec.get("scale", {"min":1,"max":7,"step":1})

st.subheader("📝 Questionnaire")
for it in shuffled:
    vmin = int(it.get("min", scale_defaults.get("min", 1)))
    vmax = int(it.get("max", scale_defaults.get("max", 7)))
    step = int(it.get("step", scale_defaults.get("step", 1)))
    default_val = vmin + ((vmax - vmin) // (2 * step)) * step

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown(f"**{it['text']}**")
    with c2:
        current_val = st.session_state.get(it["id"], default_val)
        curr_label = value_to_label(it, current_val)
        if curr_label:
            st.markdown(f"<div style='font-size:0.9rem; opacity:0.8; margin-bottom:-0.5rem'><b>{curr_label}</b></div>", unsafe_allow_html=True)
        elif it.get("L") or it.get("R"):
            st.markdown(f"<div style='font-size:0.9rem; opacity:0.7; margin-bottom:-0.5rem'><b>{it.get('L','')}</b></div>", unsafe_allow_html=True)

        val = st.slider(label="", min_value=vmin, max_value=vmax, step=step,
                        value=current_val, key=it["id"],
                        help=None if not (it.get("L") or it.get("R")) else f"{it.get('L','')}  ↔  {it.get('R','')}",)
    st.divider()
    responses[it["id"]] = val

compute = st.button("Compute Results")
if not compute: st.stop()

# Aggregate & validate
person_scales = aggregate_to_scales(responses, spec)
missing = [d for d in ALL_DIMS + SELF_SCALES if np.isnan(person_scales.get(d, np.nan))]
if missing:
    st.error(f"Missing responses for: {missing}")
    st.stop()

# Norms (optional)
norms_df = None
if norms_up is not None:
    try:
        norms_df = pd.read_csv(norms_up)
    except Exception as e:
        st.error(f"Failed to read norms.csv: {e}"); st.stop()

# Score
try:
    z = zparams_from_norms_or_single(person_scales, norms_df)
    arch_mz, arch_std = prepare_archetype_pieces(z)
    person_row = pd.Series({**person_scales, "participant_id": participant_id})
    res = score_single(person_row, z, ARCHETYPE_CENTROIDS, arch_mz, arch_std)
except Exception as e:
    st.error(str(e)); st.stop()

# ====================== Report: Top-3 + ranking + downloads ======================

left, right = st.columns([1,1])

probs = pd.Series(res["probs"]).sort_values(ascending=False).rename("probability")
(p1,p1v),(p2,p2v),(p3,p3v) = res["top3"]
mix = top3_percentages(res["top3"])
mix_text = " · ".join([f"{pct}% {name}" for name, pct in mix])

# Motivation ranking (raw or z)
if ranking_mode.startswith("Z-scores"):
    mot_series = pd.Series({m: (person_scales[m]-z.mean[m])/z.std[m] for m in MOTIVATIONS}).sort_values(ascending=False).rename("z")
    mot_df = mot_series.to_frame()
else:
    mot_series = pd.Series({m: person_scales[m] for m in MOTIVATIONS}).sort_values(ascending=False).rename("mean")
    mot_df = mot_series.to_frame()
mot_df["rank"] = np.arange(1, len(mot_df)+1)

with left:
    st.subheader("🏆 Top Archetypes")
    st.metric("Primary", p1, f"{p1v:.3f}")
    st.metric("Secondary", p2, f"{p2v:.3f}")
    st.metric("Tertiary", p3, f"{p3v:.3f}")
    st.markdown(f"**Top-3 mix:** {mix_text}")

    st.subheader("🧭 Strategic Quadrant")
    st.write(f"**{res['quadrant']}**")
    st.caption(f"axis_CF = {res['quadrant_axes']['axis_CF']:.2f} | axis_CR = {res['quadrant_axes']['axis_CR']:.2f}")

    st.subheader("🔒 Confidence")
    st.metric("Confidence Index", f"{res['confidence']:.3f}")

    st.subheader("📥 Download Full Scores (CSV)")
    out_df = to_result_df(res, participant_id)
    buf = StringIO(); out_df.to_csv(buf, index=False)
    st.download_button("Download scores.csv", data=buf.getvalue(), file_name=f"{participant_id}_scores.csv", mime="text/csv")

with right:
    st.subheader("📊 Archetype Probabilities")
    st.dataframe(probs.to_frame())

    st.subheader(f"🧩 Motivation Ranking — { 'Z' if ranking_mode.startswith('Z') else 'Raw' }")
    st.dataframe(mot_df[["rank"] + ([ "z" ] if "z" in mot_df.columns else [ "mean" ])])

    mot_buf = StringIO(); mot_df.reset_index(names="motivation").to_csv(mot_buf, index=False)
    st.download_button("Download motivation_ranking.csv", data=mot_buf.getvalue(), file_name=f"{participant_id}_motivation_ranking.csv", mime="text/csv")

# ====================== PDF Export ======================

def build_pdf_report(
    participant_id: str,
    top3: List[Tuple[str,float]],
    mix: List[Tuple[str,int]],
    quadrant_label: str,
    axes: Dict[str,float],
    confidence: float,
    probs_series: pd.Series,
    mot_df: pd.DataFrame,
    ranking_mode_label: str
) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story: List = []

    story.append(Paragraph(f"Motivational Archetypes Report – {participant_id}", styles["Title"]))
    story.append(Spacer(1, 8))

    (tp1,v1),(tp2,v2),(tp3,v3) = top3
    story.append(Paragraph("<b>Top Archetypes</b>", styles["Heading2"]))
    story.append(Paragraph(f"Primary: <b>{tp1}</b> ({v1:.3f})", styles["Normal"]))
    story.append(Paragraph(f"Secondary: <b>{tp2}</b> ({v2:.3f})", styles["Normal"]))
    story.append(Paragraph(f"Tertiary: <b>{tp3}</b> ({v3:.3f})", styles["Normal"]))
    mix_line = " · ".join([f"{pct}% {name}" for name, pct in mix])
    story.append(Paragraph(f"Top-3 mix: <b>{mix_line}</b>", styles["Normal"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Strategic Quadrant</b>", styles["Heading2"]))
    story.append(Paragraph(f"{quadrant_label}", styles["Normal"]))
    story.append(Paragraph(f"axis_CF = {axes['axis_CF']:.2f} | axis_CR = {axes['axis_CR']:.2f}", styles["Normal"]))
    story.append(Paragraph(f"Confidence Index: <b>{confidence:.3f}</b>", styles["Normal"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Archetype Probabilities</b>", styles["Heading2"]))
    probs_tbl_data = [["Archetype", "Probability"]]
    for name, val in probs_series.items():
        probs_tbl_data.append([name, f"{val:.3f}"])
    probs_tbl = Table(probs_tbl_data, hAlign="LEFT")
    probs_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('ALIGN', (1,1), (1,-1), 'RIGHT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ]))
    story.append(probs_tbl)
    story.append(Spacer(1, 8))

    story.append(Paragraph(f"<b>Motivation Ranking — {ranking_mode_label}</b>", styles["Heading2"]))
    col_label = "Z" if "z" in mot_df.columns else "Mean"
    mot_tbl_data = [["Rank", "Motivation", col_label]]
    mot_iter = mot_df.reset_index().rename(columns={"index":"Motivation"})
    for _, r in mot_iter.iterrows():
        mot_tbl_data.append([int(r["rank"]), r["Motivation"], f"{float(r[col_label.lower()]):.3f}"])
    mot_tbl = Table(mot_tbl_data, hAlign="LEFT")
    mot_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('ALIGN', (0,1), (0,-1), 'CENTER'),
        ('ALIGN', (2,1), (2,-1), 'RIGHT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ]))
    story.append(mot_tbl)

    doc.build(story)
    return buf.getvalue()

pdf_bytes = build_pdf_report(
    participant_id=participant_id,
    top3=res["top3"],
    mix=mix,
    quadrant_label=res["quadrant"],
    axes=res["quadrant_axes"],
    confidence=res["confidence"],
    probs_series=probs,
    mot_df=mot_df[["rank"] + ([ "z" ] if "z" in mot_df.columns else [ "mean" ])],
    ranking_mode_label=("Z-scores" if ranking_mode.startswith("Z") else "Raw means (1–7)")
)
st.download_button("📄 Download PDF report", data=pdf_bytes,
                   file_name=f"{participant_id}_report.pdf", mime="application/pdf")
