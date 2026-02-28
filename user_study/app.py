"""
SAE Bug Detection – User Study
Run with:  streamlit run app.py
"""

import html as html_lib
import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SAE Bug Detection · User Study",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CURATED = Path(__file__).parent / "data" / "curated_study_data.jsonl"
_FULL    = Path(__file__).parent / "data" / "study_data.jsonl"
DATA_PATH = _CURATED if _CURATED.exists() else _FULL

EXT_TO_LANG = {
    "java": "java",
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "c": "c",
    "cpp": "cpp",
    "cc": "cpp",
    "php": "php",
    "go": "go",
    "rb": "ruby",
    "rs": "rust",
    "cs": "csharp",
}

CONFIDENCE_COLOR = {
    "high": "#2ecc71",
    "medium": "#f39c12",
    "low": "#e74c3c",
    "": "#95a5a6",
}


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading study data …")
def load_data() -> list[dict]:
    if not DATA_PATH.exists():
        return []
    records = []
    with open(DATA_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    return records


# ── Helpers ────────────────────────────────────────────────────────────────────
def lang(ext: str) -> str:
    return EXT_TO_LANG.get(ext.lower().lstrip("."), "text")


def render_code_box(code: str, height: int = 600) -> None:
    """Render a scrollable code block using a styled HTML div."""
    escaped = html_lib.escape(code)
    st.markdown(
        f"""<div style="
            height:{height}px;
            overflow-y:auto;
            overflow-x:auto;
            background:#0e1117;
            padding:14px 16px;
            border-radius:6px;
            border:1px solid #2d2d2d;
            font-family:'Source Code Pro','Courier New',monospace;
            font-size:13px;
            line-height:1.6;
            color:#e8e8e8;
            white-space:pre-wrap;
            word-break:break-word;
            tab-size:4;
        ">{escaped}</div>""",
        unsafe_allow_html=True,
    )


def activation_chart(features: list[dict]) -> go.Figure:
    labels = [f"F{f['feature_idx']}" for f in features]
    secure_vals = [f["secure_activation"] for f in features]
    vuln_vals = [f["vulnerable_activation"] for f in features]
    diff_vals = [f["diff"] for f in features]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Secure",
            x=labels,
            y=secure_vals,
            marker_color="#3498db",
            opacity=0.85,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Vulnerable",
            x=labels,
            y=vuln_vals,
            marker_color="#e74c3c",
            opacity=0.85,
        )
    )
    fig.update_layout(
        barmode="group",
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="SAE Feature",
        yaxis_title="Activation",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    return fig


def diff_chart(features: list[dict]) -> go.Figure:
    """Horizontal bar showing vuln - secure difference."""
    labels = [f"F{f['feature_idx']}" for f in reversed(features)]
    diffs = [f["diff"] for f in reversed(features)]
    colors = ["#e74c3c" if d > 0 else "#3498db" for d in diffs]

    fig = go.Figure(
        go.Bar(
            x=diffs,
            y=labels,
            orientation="h",
            marker_color=colors,
            opacity=0.85,
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="gray")
    fig.update_layout(
        height=max(300, 20 * len(features)),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Δ activation  (vulnerable − secure)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    return fig


# ── App ────────────────────────────────────────────────────────────────────────
records = load_data()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("SAE Bug Detection")
    st.caption("User Study · Layer 11")
    st.divider()

    if not records:
        st.error(
            "No study data found.\n\n"
            "Run `python prepare_data.py` first to generate `data/study_data.jsonl`."
        )
        st.stop()

    # CWE filter
    all_cwes = sorted({r["cwe"] for r in records})
    selected_cwes = st.multiselect(
        "Filter by CWE", all_cwes, placeholder="All CWEs"
    )

    # Language filter
    all_langs = sorted({r["file_extension"] for r in records if r["file_extension"]})
    selected_langs = st.multiselect(
        "Filter by language", all_langs, placeholder="All languages"
    )

    filtered = records
    if selected_cwes:
        filtered = [r for r in filtered if r["cwe"] in selected_cwes]
    if selected_langs:
        filtered = [r for r in filtered if r["file_extension"] in selected_langs]

    if not filtered:
        st.warning("No records match the current filters.")
        st.stop()

    st.caption(f"{len(filtered)} samples available")
    st.divider()

    # Sample navigation
    idx = st.number_input(
        "Sample index",
        min_value=1,
        max_value=len(filtered),
        value=1,
        step=1,
    )
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("◀ Prev", use_container_width=True):
            idx = max(1, idx - 1)
    with col_next:
        if st.button("Next ▶", use_container_width=True):
            idx = min(len(filtered), idx + 1)

    st.divider()

    # Top-K slider
    top_k = st.slider("Top features shown", min_value=5, max_value=20, value=10)

    # Chart style
    chart_type = st.radio(
        "Chart style",
        ["Grouped bars (secure vs vulnerable)", "Difference only"],
        index=0,
    )

sample = filtered[idx - 1]
features = sample["top_features"][:top_k]

# ─── Main content ──────────────────────────────────────────────────────────────
header_col, meta_col = st.columns([3, 1])
with header_col:
    title = sample.get("title") or sample["vuln_id"]
    st.subheader(title)
    st.caption(sample["vuln_id"])
with meta_col:
    st.markdown(
        f"**CWE:** `{sample['cwe']}`  \n"
        f"**Language:** `{sample['file_extension'] or 'unknown'}`"
    )
    if sample.get("placeholder_activations"):
        st.warning("Placeholder activations", icon="⚠️")

st.divider()

# Code panels
code_left, code_right = st.columns(2)
language = lang(sample["file_extension"])

with code_left:
    st.markdown("### ✅ Secure version")
    render_code_box(sample["secure_code"])

with code_right:
    st.markdown("### ⚠️ Vulnerable version")
    render_code_box(sample["vulnerable_code"])

st.divider()

# Activation chart
st.markdown(f"### SAE Feature Activations — top {top_k} by |Δ|")

if chart_type.startswith("Grouped"):
    st.plotly_chart(activation_chart(features), use_container_width=True)
else:
    st.plotly_chart(diff_chart(features), use_container_width=True)

# Feature details
st.divider()
st.markdown("### Feature Details")
st.caption(
    "Each row shows what the SAE feature has learned to detect.  \n"
    "**Δ > 0** → feature fires more on the *vulnerable* version.  "
    "**Δ < 0** → feature fires more on the *secure* version."
)

for feat in features:
    conf = feat["confidence"]
    conf_color = CONFIDENCE_COLOR.get(conf, "#95a5a6")
    delta = feat["diff"]
    direction = "🔴 higher on vulnerable" if delta > 0 else "🔵 higher on secure"

    with st.expander(
        f"**Feature {feat['feature_idx']}**  —  "
        f"Δ = {delta:+.4f}  ({direction})",
        expanded=False,
    ):
        col1, col2, col3 = st.columns(3)
        col1.metric("Secure activation", f"{feat['secure_activation']:.4f}")
        col2.metric(
            "Vulnerable activation",
            f"{feat['vulnerable_activation']:.4f}",
            delta=f"{delta:+.4f}",
        )
        col3.markdown(
            f"**Confidence:** "
            f"<span style='color:{conf_color};font-weight:bold'>{conf.upper() or 'N/A'}</span>",
            unsafe_allow_html=True,
        )

        st.markdown("**Hypothesis**")
        st.info(feat["hypothesis"] or "No hypothesis available.")

        if feat.get("notes"):
            st.markdown("**Notes**")
            st.markdown(feat["notes"])
