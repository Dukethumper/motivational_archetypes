# file: app.py
from __future__ import annotations

import json, math, re, hashlib, random, os
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ====================== Canonical dimensions ======================

MOTIVATIONS: List[str] = [
    "Sattva", "Rajas", "Tamas",
    "Prajna", "Personal_Unconscious", "Collective_Unconscious",
    "Cheng", "Wu_Wei", "Anatta",
    "Relational_Balance", "Thymos", "Eros",
]
STRATEGIES: List[str] = ["Conform", "Control", "Flow", "Risk"]
ORIENTATIONS: List[str] = ["Inward", "Outward", "Surrender", "Relationship"]
SELF_SCALES: List[str] = ["Self_Insight", "Self_Serving_Bias"]
ALL_DIMS: List[str] = MOTIVATIONS + STRATEGIES + ORIENTATIONS
ALL_REQ_FOR_Z: List[str] = ALL_DIMS + SELF_SCALES

DEFAULT_DOMAIN_MAP = {
    "Energetic": ["Sattva","Rajas","Tamas"],
    "Cognitive": ["Prajna","Personal_Unconscious","Collective_Unconscious"],
    "Integrative": ["Cheng","Wu_Wei","Anatta"],
    "Relational": ["Relational_Balance","Thymos","Eros"],
}

# ====================== Header normalization ======================

HEADER_TO_CANON = {
    "sattva":"Sattva","rajas":"Rajas","tamas":"Tamas",
    "prajna":"Prajna","prajna-logos":"Prajna",
    "personal unconscious":"Personal_Unconscious","personal_unconscious":"Personal_Unconscious","pers.u":"Personal_Unconscious",
    "collective unconscious":"Collective_Unconscious","collective_unconscious":"Collective_Unconscious","coll.u":"Collective_Unconscious",
    "cheng":"Cheng","wu wei":"Wu_Wei","wu_wei":"Wu_Wei","anatta":"Anatta",
    "relational balance":"Relational_Balance","rel.bal":"Relational_Balance",
    "thymos":"Thymos","eros":"Eros",
    "conform":"Conform","control":"Control","flow":"Flow","risk":"Risk",
    "inward":"Inward","outward":"Outward","surrender":"Surrender",
    "relationship":"Relationship","relational":"Relationship","relationship value":"Relationship",
    "self insight":"Self_Insight","self_insight":"Self_Insight",
    "self serving bias":"Self_Serving_Bias","self-serving bias":"Self_Serving_Bias","self_serving_bias":"Self_Serving_Bias",
}
def canon(name: str) -> Optional[str]:
    return HEADER_TO_CANON.get(name.strip().lower())

# ====================== Centroids (embedded) ======================

COLUMN_NORMALIZATION = {
    "Sattva":"Sattva","Rajas":"Rajas","Tamas":"Tamas","Prajna":"Prajna",
    "Pers.U":"Personal_Unconscious","Personal Unconscious":"Personal_Unconscious","Personal_Unconscious":"Personal_Unconscious",
    "Coll.U":"Collective_Unconscious","Collective Unconscious":"Collective_Unconscious","Collective_Unconscious":"Collective_Unconscious",
    "Cheng":"Cheng","Wu Wei":"Wu_Wei","Wu_Wei":"Wu_Wei",
    "Anatta":"Anatta","Rel.Bal":"Relational_Balance","Relational Balance":"Relational_Balance","Relational_Balance":"Relational_Balance",
    "Thymos":"Thymos","Eros":"Eros",
    "Conform":"Conform","Control":"Control","Flow":"Flow","Risk":"Risk",
    "Outward":"Outward","Inward":"Inward","Surrender":"Surrender","Relational":"Relationship","Relationship":"Relationship",
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
_ARC_RAW = pd.DataFrame(
    [
        [4.8,6.8,3.8,6.0,4.7,4.5,5.0,4.9,4.3,4.7,6.2,5.0, 4.5,4.8,6.0,4.2, 6.3,4.8,4.7,5.2],
        [6.4,4.5,4.7,4.4,4.5,4.8,6.0,4.8,4.3,6.7,5.0,5.3, 6.0,4.8,4.2,3.8, 4.5,4.8,5.1,6.3],
        [5.0,6.2,4.2,4.8,4.5,4.7,5.4,5.8,4.5,5.3,6.5,5.0, 5.5,4.8,6.0,4.5, 6.4,4.6,5.2,5.9],
        [4.8,6.2,5.7,4.8,4.4,4.5,6.7,4.7,4.0,4.5,5.2,4.4, 4.3,6.0,4.5,5.8, 5.9,4.4,4.5,4.6],
        [4.6,5.3,4.8,4.7,6.7,4.5,6.3,4.4,4.2,4.7,5.7,4.4, 5.8,6.0,4.3,4.2, 5.5,5.9,4.8,4.9],
        [4.2,4.8,6.5,4.5,4.7,6.3,4.3,4.8,6.4,4.2,4.8,4.5, 3.8,4.3,4.8,6.0, 5.8,6.0,6.2,4.3],
        [6.2,5.0,4.3,6.6,5.3,5.0,6.1,4.8,4.5,4.9,4.8,4.7, 4.5,6.0,5.8,4.3, 5.3,6.3,5.1,4.9],
        [4.8,5.8,4.2,4.6,4.7,4.8,4.6,6.5,4.3,5.0,5.0,6.2, 4.0,4.3,6.0,4.8, 6.2,4.9,6.4,5.6],
        [4.6,4.7,5.5,6.3,5.4,5.1,4.8,4.5,6.4,4.4,4.6,4.5, 4.8,4.8,4.2,4.0, 4.3,6.5,6.2,4.4],
        [6.5,6.2,4.4,4.9,4.7,5.8,4.9,5.6,4.3,4.8,4.8,4.7, 3.9,4.7,4.8,6.0, 6.6,4.8,4.9,4.8],
        [6.3,4.7,4.3,4.5,4.8,4.9,4.8,5.0,4.4,6.5,5.2,6.7, 4.6,4.4,5.8,4.3, 5.8,4.9,5.3,6.5],
        [4.8,6.5,4.5,5.6,4.8,6.7,4.7,5.1,5.2,4.6,6.1,5.5, 4.2,4.8,5.8,5.7, 6.3,5.5,5.0,5.1],
    ],
    index=_ARC_ROWS,
    columns=[
        "Sattva","Rajas","Tamas","Prajna","Pers.U","Coll.U","Cheng","Wu Wei","Anatta","Rel.Bal","Thymos","Eros",
        "Conform","Control","Flow","Risk","Outward","Inward","Surrender","Relational",
    ],
)
def _norm_headers(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_centroid_headers(df)
ARCHETYPE_CENTROIDS = _norm_headers(_ARC_RAW)

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
    z_mot, z_str, z_ori = z.zrow(person, MOTIVATIONS), z.zrow(person, STRATEGIES), z.zrow(person, ORIENTATIONS)
    z_pat = intra_person_z(np.array([person[m] for m in MOTIVATIONS], float))
    z_SI = (person["Self_Insight"]-z.mean["Self_Insight"])/z.std["Self_Insight"]
    z_SSB= (person["Self_Serving_Bias"]-z.mean["Self_Serving_Bias"])/z.std["Self_Serving_Bias"]
    C = 0.7*sigmoid(float(z_SI)) + 0.3*(1.0 - 2.0*abs(sigmoid(float(z_SSB))-0.5))
    names = list(arch.index); vals=[]
    for name in names:
        ap = arch_mz.loc[name, MOTIVATIONS].to_numpy(float)
        R = pearson(z_pat, ap); Rp = (R+1)/2
        am, as_, ao = arch_std.loc[name, MOTIVATIONS].to_numpy(float), arch_std.loc[name, STRATEGIES].to_numpy(float), arch_std.loc[name, ORIENTATIONS].to_numpy(float)
        Dall = euclid(np.concatenate([z_mot,z_str,z_ori]), np.concatenate([am,as_,ao]))
        S_abs = 1/(1+Dall); S_A = 0.65*Rp + 0.35*S_abs
        DS = euclid(z_str, as_); SS = 1/(1+DS)
        DO = euclid(z_ori, ao); SO = 1/(1+DO)
        S_total = W_MOT_ABS*S_A + W_STRAT_MATCH*SS + W_ORIENT_MATCH*SO
        vals.append(S_total*(0.75+0.25*C))
    probs = normalize_probs(np.array(vals,float))
    order = np.argsort(-probs); top3=[(names[i], float(probs[i])) for i in order[:3]]
    qc, axes = quadrant(z_str[STRATEGIES.index("Control")], z_str[STRATEGIES.index("Flow")], z_str[STRATEGIES.index("Conform")], z_str[STRATEGIES.index("Risk")])
    return {"probs": {names[i]: float(probs[i]) for i in range(len(names))}, "top3": top3, "quadrant": qc, "quadrant_axes": axes, "confidence": float(C), "domain_centroids": domain_means(person)}

# ====================== TXT parser with per-item Likert ======================

ITEM_KV_RE = re.compile(r"\[(\w+)\s*=\s*(.*?)\]")  # [KEY=VALUE]
def _kv_blocks(s: str) -> Dict[str,str]:
    return {k.upper(): v.strip() for k, v in ITEM_KV_RE.findall(s)}

def parse_txt_questions(raw: str) -> Dict:
    """
    Format:
      Dimension headers (hidden):   Sattva:
      Item lines: <question> [LABELS=a|b|...|z] [MIN=1][MAX=7][STEP=1]
      Short-form endpoints: [L=Never][R=Always]
      Optional per-item override: [DIM=Rajas]
    """
    lines = [l.rstrip() for l in raw.splitlines()]
    spec = {"scale": {"min":1, "max":7, "step":1}, "questions":[]}
    cur_dim: Optional[str] = None
    counts: Dict[str,int] = {}

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.endswith(":"):
            cur_dim = canon(s[:-1])
            continue

        text = s
        kvs = _kv_blocks(s)
        if kvs:
            text = ITEM_KV_RE.sub("", s).strip()

        dim = canon(kvs.get("DIM", cur_dim or ""))
        if dim is None:
            continue

        vmin = int(kvs.get("MIN", "1"))
        vmax = int(kvs.get("MAX", "7"))
        vstep = int(kvs.get("STEP", "1"))
        if vmax <= vmin or vstep <= 0:
            raise ValueError(f"Bad range for '{text}': MIN={vmin} MAX={vmax} STEP={vstep}")
        npoints = (vmax - vmin) // vstep + 1

        labels: Optional[List[str]] = None
        if "LABELS" in kvs:
            labels = [p.strip() for p in kvs["LABELS"].split("|")]
            if len(labels) != npoints:
                raise ValueError(f"LABELS count ({len(labels)}) must equal number of points ({npoints}) for '{text}'")
        else:
            L = kvs.get("L"); R = kvs.get("R")
            if L and R and npoints == 7:
                labels = [L, "Slightly "+L, "Somewhat "+L, "Neutral", "Somewhat "+R, "Slightly "+R, R]
            elif L and R and npoints == 5:
                labels = [L, "Somewhat "+L, "Neutral", "Somewhat "+R, R]

        counts[dim] = counts.get(dim, 0) + 1
        qid = f"q_{dim}_{counts[dim]}"
        spec["questions"].append({
            "id": qid,
            "dimension": dim,
            "text": text,
            "min": vmin, "max": vmax, "step": vstep,
            "labels": labels,
            "L": kvs.get("L"), "R": kvs.get("R"),
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

# ====================== Load questions from repo or override ======================

def load_questions_from_repo() -> Dict:
    """Reads questions from local repo file. Path from env QUESTIONS_PATH or './questions.txt'."""
    q_path = Path(os.getenv("QUESTIONS_PATH", "questions.txt"))
    if not q_path.exists():
        raise FileNotFoundError(f"questions file not found at: {q_path.resolve()}")
    text = q_path.read_text(encoding="utf-8")
    return parse_txt_questions(text)

# ====================== UI ======================

st.set_page_config(page_title="Motivational Archetypes – Test", page_icon="🧭", layout="wide")
st.title("🧭 Motivational Archetypes – Test")

with st.sidebar:
    st.header("Inputs")
    st.caption("By default the app reads **questions.txt** from the repo. You can override it below.")
    q_up = st.file_uploader("Override questions.txt (optional)", type=["txt"])
    norms_up = st.file_uploader("Optional norms.csv", type=["csv"])
    participant_id = st.text_input("Participant ID", value="P001")
    st.caption("Two-column layout. Labels above sliders update live as you move them.")

# Load spec (repo first; uploader overrides if provided)
try:
    spec = parse_txt_questions(q_up.read().decode("utf-8")) if q_up is not None else load_questions_from_repo()
except Exception as e:
    st.error(f"Failed to load/parse questions: {e}")
    st.stop()

items: List[Dict] = list(spec.get("questions", []))
if not items:
    st.error("No items parsed. Check your headers (e.g., 'Sattva:') and item lines.")
    st.stop()

# Stable per-user shuffle
def stable_shuffle(items: List[Dict], pid: str, spec_obj: Dict) -> List[Dict]:
    spec_bytes = json.dumps(spec_obj, sort_keys=True).encode("utf-8")
    seed_hex = hashlib.sha256((pid + "|").encode("utf-8") + spec_bytes).hexdigest()[:16]
    seed_int = int(seed_hex, 16)
    rng = random.Random(seed_int)
    out = items.copy()
    rng.shuffle(out)
    return out

spec_fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()
if "shuffle_meta" not in st.session_state or st.session_state.shuffle_meta != (participant_id, spec_fingerprint):
    st.session_state.shuffle_meta = (participant_id, spec_fingerprint)
    st.session_state.shuffled_items = stable_shuffle(items, participant_id, spec)

shuffled = st.session_state.shuffled_items

# ============ Live label mapping ============

def value_to_label(item: Dict, val: int) -> str:
    vmin, vmax, step = int(item.get("min", 1)), int(item.get("max", 7)), int(item.get("step", 1))
    labels = item.get("labels")
    idx = (val - vmin) // step
    if labels and 0 <= idx < len(labels):
        return labels[idx]
    L, R = item.get("L"), item.get("R")
    npoints = (vmax - vmin)//step + 1
    if L and R and npoints == 7:
        fallback = [L, f"Slightly {L}", f"Somewhat {L}", "Neutral", f"Somewhat {R}", f"Slightly {R}", R]
        return fallback[idx]
    if L and R and npoints == 5:
        fallback = [L, f"Somewhat {L}", "Neutral", f"Somewhat {R}", R]
        return fallback[idx]
    return ""

# ============ Compact two-column render (no form → live updates) ============

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
            st.markdown(
                f"<div style='font-size:0.9rem; opacity:0.8; margin-bottom:-0.5rem'><b>{curr_label}</b></div>",
                unsafe_allow_html=True,
            )
        elif it.get("L") or it.get("R"):
            st.markdown(
                f"<div style='font-size:0.9rem; opacity:0.7; margin-bottom:-0.5rem'><b>{it.get('L','')}</b></div>",
                unsafe_allow_html=True,
            )

        val = st.slider(
            label="", min_value=vmin, max_value=vmax, step=step,
            value=current_val, key=it["id"],
            help=None if not (it.get("L") or it.get("R")) else f"{it.get('L','')}  ↔  {it.get('R','')}",
        )

    st.divider()
    responses[it["id"]] = val

compute = st.button("Compute Results")
if not compute:
    st.stop()

# Aggregate & validate
person_scales = aggregate_to_scales(responses, spec)
missing = [d for d in ALL_DIMS + SELF_SCALES if np.isnan(person_scales.get(d, np.nan))]
if missing:
    st.error(f"Missing responses for: {missing}\n\nAdd at least one item for each missing dimension.")
    st.stop()

# Norms (optional)
norms_df = None
if norms_up is not None:
    try:
        norms_df = pd.read_csv(norms_up)
    except Exception as e:
        st.error(f"Failed to read norms.csv: {e}")
        st.stop()

# Score
try:
    z = zparams_from_norms_or_single(person_scales, norms_df)
    arch_mz, arch_std = prepare_archetype_pieces(z)
    person_row = pd.Series({**person_scales, "participant_id": participant_id})
    res = score_single(person_row, z, ARCHETYPE_CENTROIDS, arch_mz, arch_std)
except Exception as e:
    st.error(str(e)); st.stop()

# Output
left, right = st.columns([1,1])
with left:
    st.subheader("🏆 Top Archetypes")
    (p1,p1v),(p2,p2v),(p3,p3v) = res["top3"]
    st.metric("Primary", p1, f"{p1v:.3f}")
    st.metric("Secondary", p2, f"{p2v:.3f}")
    st.metric("Tertiary", p3, f"{p3v:.3f}")

    st.subheader("🧭 Strategic Quadrant")
    st.write(f"**{res['quadrant']}**")
    st.caption(f"axis_CF = {res['quadrant_axes']['axis_CF']:.2f} | axis_CR = {res['quadrant_axes']['axis_CR']:.2f}")

    st.subheader("🔒 Confidence")
    st.metric("Confidence Index", f"{res['confidence']:.3f}")

    st.subheader("📥 Download CSV")
    out_df = to_result_df(res, participant_id)
    buf = StringIO(); out_df.to_csv(buf, index=False)
    st.download_button("Download scores.csv", data=buf.getvalue(), file_name=f"{participant_id}_scores.csv", mime="text/csv")

with right:
    st.subheader("📊 Archetype Probabilities")
    probs = pd.Series(res["probs"]).sort_values(ascending=False).rename("probability")
    st.dataframe(probs.to_frame())

    st.subheader("🧩 Domain Centroids (means)")
    st.json(res["domain_centroids"])

