"""
SAE Bug Detection – User Study
Run with:  streamlit run app.py
"""

import base64
import datetime
import html as html_lib
import json
import os
import random
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SAE Bug Detection · User Study",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CURATED      = Path(__file__).parent / "data" / "curated_study_data.jsonl"
_FULL         = Path(__file__).parent / "data" / "study_data.jsonl"
DATA_PATH     = _CURATED if _CURATED.exists() else _FULL
FEEDBACK_PATH = Path(__file__).parent / "data" / "feedback.jsonl"
PDF_PATH      = Path(__file__).parent / "SAE.pdf"

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


# ── Hypothesis lookup (for sandbox) ───────────────────────────────────────────
@st.cache_data(show_spinner="Loading feature hypotheses …")
def load_hypotheses() -> dict:
    """Load feature hypotheses. Prefers data/hypotheses.json; falls back to study records."""
    hyp_path = Path(__file__).parent / "data" / "hypotheses.json"
    if hyp_path.exists():
        with open(hyp_path) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    # Fall back: extract from study records already in memory
    hyps: dict = {}
    for rec in load_data():
        for feat in rec.get("top_features", []):
            fid = feat["feature_idx"]
            if fid not in hyps:
                hyps[fid] = {
                    "hypothesis": feat.get("hypothesis", ""),
                    "confidence": feat.get("confidence", ""),
                    "notes": feat.get("notes", ""),
                    "max_activation": feat.get("max_activation", 0.5),
                }
    return hyps


def sandbox_activations(hypotheses: dict, top_k: int, seed: int) -> list[dict]:
    """Randomly sample features and assign placeholder activations."""
    rng = random.Random(seed)
    pool = list(hypotheses.items())
    rng.shuffle(pool)
    results = []
    for fid, h in pool[:top_k]:
        act = round(rng.uniform(0.02, max(0.05, h.get("max_activation", 0.5) * 0.9)), 6)
        results.append(
            {
                "feature_idx": int(fid),
                "activation": act,
                "hypothesis": h.get("hypothesis") or "No hypothesis available.",
                "confidence": h.get("confidence", ""),
                "notes": h.get("notes", ""),
            }
        )
    results.sort(key=lambda x: x["activation"], reverse=True)
    return results


def sandbox_chart(features: list[dict]) -> go.Figure:
    labels = [f"F{f['feature_idx']}" for f in reversed(features)]
    acts = [f["activation"] for f in reversed(features)]
    fig = go.Figure(
        go.Bar(x=acts, y=labels, orientation="h", marker_color="#9b59b6", opacity=0.85)
    )
    fig.update_layout(
        height=max(300, 22 * len(features)),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Activation",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    return fig


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


def _push_feedback_to_hub() -> None:
    """Push feedback.jsonl to a HF dataset repo. Silently skips if not configured."""
    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("FEEDBACK_REPO")  # e.g. "your-username/sae-study-feedback"
    if not token or not repo_id or not FEEDBACK_PATH.exists():
        return
    try:
        from huggingface_hub import HfApi
        HfApi().upload_file(
            path_or_fileobj=str(FEEDBACK_PATH),
            path_in_repo="feedback.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="update feedback",
        )
    except Exception:
        pass  # never crash the study over a failed upload


def save_feedback(sample: dict, features: list[dict], comment: str) -> None:
    """Append one feedback record to feedback.jsonl and push to HF Hub."""
    ratings = {}
    for feat in features:
        key = f"rating_{sample['vuln_id']}_{feat['feature_idx']}"
        val = st.session_state.get(key)
        if val is not None:
            ratings[feat["feature_idx"]] = val

    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "participant": st.session_state.get("participant_name", "anonymous"),
        "vuln_id": sample["vuln_id"],
        "cwe": sample["cwe"],
        "feature_ratings": ratings,
        "comment": comment.strip(),
    }
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    _push_feedback_to_hub()


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


# ── Intro page ────────────────────────────────────────────────────────────────
if "intro_done" not in st.session_state:
    st.session_state.intro_done = False

if not st.session_state.intro_done:
    st.title("SAE-Based Bug Detection — User Study")
    st.markdown(
        "Welcome! Before you start, please read this short introduction so you "
        "know what you are evaluating and why."
    )
    st.divider()

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("## What is a Sparse Autoencoder (SAE)?")
        st.markdown(
            """
Large Language Models (LLMs) encode concepts in a **superimposed** way — many different
ideas are mixed together inside the same neurons, making it hard to understand what
the model has learned.

A **Sparse Autoencoder** is a small network trained on top of an LLM to *decompose*
those tangled representations into thousands of individual **features**, each one
ideally corresponding to a single, human-interpretable concept.

> Think of it as a prism that splits white light into distinct colours — the SAE
> splits a dense model activation into individual semantic signals.
"""
        )

        st.markdown("## What is a Feature?")
        st.markdown(
            """
Each SAE feature is a **direction** in the model's internal space.
For a given piece of code, the feature produces an **activation value** — a non-negative
number that measures how strongly the feature "fires" on that input.

- **High activation** → the feature pattern is strongly present in the code.
- **Zero activation** → the feature is inactive (most features are zero for any given input).

In this study the SAE was trained on a dataset of *secure* and *vulnerable* code pairs.
Features that fire differently on the two versions may capture security-relevant concepts
such as:
- missing input validation
- SQL query construction patterns
- error message information leakage
- unescaped HTML output
"""
        )

        st.markdown("## What are you being asked to do?")
        st.markdown(
            """
For each example you will see:

1. **Two code panels** — the *secure* version on the left, the *vulnerable* version on the right.
2. **An activation chart** — showing the SAE features that differ most between the two versions.
3. **Feature cards** — each card shows an automatically generated *hypothesis* describing
   what concept the feature may encode.

Your task is to **rate each hypothesis** (👍 Agree / 👎 Disagree) based on whether it
makes sense given the code you see, and optionally leave a comment.

There are no right or wrong answers — your judgement is the ground truth here.
"""
        )

    with col_right:
        st.markdown("## Reference slides")
        if PDF_PATH.exists():
            b64 = base64.b64encode(PDF_PATH.read_bytes()).decode()
            st.markdown(
                f'<iframe src="data:application/pdf;base64,{b64}" '
                f'width="100%" height="620px" style="border:none;border-radius:6px;"></iframe>',
                unsafe_allow_html=True,
            )
        else:
            st.info("SAE.pdf not found — place it next to app.py to display it here.")

    st.divider()
    name = st.text_input(
        "Your name",
        placeholder="Enter your name to begin …",
        key="participant_name_input",
    )
    if st.button(
        "Start the study →",
        type="primary",
        disabled=not name.strip(),
    ):
        st.session_state.intro_done = True
        st.session_state.participant_name = name.strip()
        st.rerun()
    st.stop()


# ── App ────────────────────────────────────────────────────────────────────────
records = load_data()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("SAE Bug Detection")
    st.caption("User Study · Layer 11")
    participant = st.session_state.get("participant_name", "")
    if participant:
        st.caption(f"Participant: **{participant}**")
    if st.button("← Back to intro", use_container_width=True):
        st.session_state.intro_done = False
        st.rerun()
    st.divider()

    page = st.radio(
        "page",
        ["📋 Study", "🔬 Sandbox"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()

    if page == "📋 Study":
        if not records:
            st.error(
                "No study data found.\n\n"
                "Run `python prepare_data.py` first to generate `data/study_data.jsonl`."
            )
            st.stop()

        # CWE filter
        all_cwes = sorted({r["cwe"] for r in records})
        selected_cwes = st.multiselect("Filter by CWE", all_cwes, placeholder="All CWEs")

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
            "Sample index", min_value=1, max_value=len(filtered), value=1, step=1
        )
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("◀ Prev", use_container_width=True):
                idx = max(1, idx - 1)
        with col_next:
            if st.button("Next ▶", use_container_width=True):
                idx = min(len(filtered), idx + 1)

        st.divider()
        top_k = st.slider("Top features shown", min_value=5, max_value=20, value=10)
        chart_type = st.radio(
            "Chart style",
            ["Grouped bars (secure vs vulnerable)", "Difference only"],
            index=0,
        )

    else:  # Sandbox
        top_k = st.slider("Features to show", min_value=5, max_value=20, value=10)

# ══════════════════════════════════════════════════════════════════════════════
# ── STUDY PAGE ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if page == "📋 Study":
    sample = filtered[idx - 1]
    features = sample["top_features"][:top_k]

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

    code_left, code_right = st.columns(2)
    language = lang(sample["file_extension"])
    with code_left:
        st.markdown("### ✅ Secure version")
        render_code_box(sample["secure_code"])
    with code_right:
        st.markdown("### ⚠️ Vulnerable version")
        render_code_box(sample["vulnerable_code"])

    st.divider()
    st.markdown(f"### SAE Feature Activations — top {top_k} by |Δ|")
    if chart_type.startswith("Grouped"):
        st.plotly_chart(activation_chart(features), use_container_width=True)
    else:
        st.plotly_chart(diff_chart(features), use_container_width=True)

    st.divider()
    st.markdown("### Feature Details")
    st.caption(
        "**Δ > 0** → feature fires more on the *vulnerable* version.  "
        "**Δ < 0** → feature fires more on the *secure* version."
    )

    for feat in features:
        conf = feat["confidence"]
        conf_color = CONFIDENCE_COLOR.get(conf, "#95a5a6")
        delta = feat["diff"]
        direction = "🔴 higher on vulnerable" if delta > 0 else "🔵 higher on secure"

        with st.expander(
            f"**Feature {feat['feature_idx']}**  —  Δ = {delta:+.4f}  ({direction})",
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
            st.markdown("**Does this hypothesis match what you see in the code?**")
            st.radio(
                "rating",
                options=["👍  Agree", "👎  Disagree"],
                horizontal=True,
                index=None,
                key=f"rating_{sample['vuln_id']}_{feat['feature_idx']}",
                label_visibility="collapsed",
            )

    st.divider()
    st.markdown("### Feedback")
    rated = sum(
        1 for feat in features
        if st.session_state.get(f"rating_{sample['vuln_id']}_{feat['feature_idx']}")
    )
    st.caption(f"{rated} / {len(features)} features rated for this sample.")
    with st.form(key=f"feedback_form_{sample['vuln_id']}"):
        comment = st.text_area(
            "Additional comments (optional)",
            placeholder="Any observations about the code, hypotheses, or activations …",
            height=120,
        )
        submitted = st.form_submit_button("Submit feedback", type="primary")
    if submitted:
        save_feedback(sample, features, comment)
        st.success("Feedback saved — thank you!", icon="✅")


# ══════════════════════════════════════════════════════════════════════════════
# ── SANDBOX PAGE ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
else:
    hypotheses = load_hypotheses()

    st.title("🔬 Sandbox")
    st.caption(
        "Paste or write any code snippet below and click **Analyze** to see which "
        "SAE features fire and what they may mean.  "
        "Activations are **placeholder** values until real inference is wired up."
    )
    st.divider()

    sb_lang_options = list(EXT_TO_LANG.keys())
    sb_col_lang, sb_col_btn = st.columns([2, 1])
    with sb_col_lang:
        sb_lang = st.selectbox("Language", sb_lang_options, index=sb_lang_options.index("java"))
    with sb_col_btn:
        st.markdown("<div style='margin-top:28px'/>", unsafe_allow_html=True)
        analyze = st.button("Analyze ▶", type="primary", use_container_width=True)

    user_code = st.text_area(
        "Your code",
        height=380,
        placeholder="// Paste your Java (or other) code here …",
        key="sandbox_code",
        label_visibility="collapsed",
    )

    if analyze:
        if not user_code.strip():
            st.warning("Please enter some code before clicking Analyze.")
        else:
            st.session_state.sandbox_seed = random.randint(0, 10_000)
            st.session_state.sandbox_lang = sb_lang

    if st.session_state.get("sandbox_seed") and st.session_state.get("sandbox_code", "").strip():
        if not hypotheses:
            st.error(
                "No feature hypotheses found.  \n"
                "Run `python prepare_data.py` or `python generate_curated_data.py` first."
            )
        else:
            sb_features = sandbox_activations(
                hypotheses, top_k, st.session_state.sandbox_seed
            )

            st.divider()
            st.markdown(f"### SAE Feature Activations — top {top_k} features")
            st.plotly_chart(sandbox_chart(sb_features), use_container_width=True)

            st.divider()
            st.markdown("### Feature Details")
            st.caption(
                "These are the features with the highest placeholder activations for your code.  \n"
                "Replace with real SAE inference to get meaningful results."
            )
            for feat in sb_features:
                conf = feat["confidence"]
                conf_color = CONFIDENCE_COLOR.get(conf, "#95a5a6")
                with st.expander(
                    f"**Feature {feat['feature_idx']}**  —  activation = {feat['activation']:.4f}",
                    expanded=False,
                ):
                    col1, col2 = st.columns([1, 2])
                    col1.metric("Activation", f"{feat['activation']:.4f}")
                    col2.markdown(
                        f"**Confidence:** "
                        f"<span style='color:{conf_color};font-weight:bold'>"
                        f"{conf.upper() or 'N/A'}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Hypothesis**")
                    st.info(feat["hypothesis"])
                    if feat.get("notes"):
                        st.markdown("**Notes**")
                        st.markdown(feat["notes"])
                    st.markdown("**Does this hypothesis seem relevant to your code?**")
                    st.radio(
                        "sb_rating",
                        options=["👍  Relevant", "👎  Not relevant"],
                        horizontal=True,
                        index=None,
                        key=f"sb_rating_{st.session_state.sandbox_seed}_{feat['feature_idx']}",
                        label_visibility="collapsed",
                    )
