"""
token_feature_viz.py
====================
Per-token SAE feature activation visualisation.

For each selected feature, finds the top-N paired (vulnerable, secure) examples
by mean activation on the *vulnerable* code, runs a per-token forward pass through
Qwen2.5-7B-Instruct + the SAE, and renders a publication-quality PDF figure where
each token is coloured by its activation value for that feature.

This directly addresses the reviewer request for mechanistic token-level evidence.

Usage
-----
python token_feature_viz.py \\
    --features 1797 9193 \\
    --cwe CWE-119 \\
    --n_examples 2 \\
    --out_dir /path/to/paper/figures \\
    --device cuda          # or cpu / mps

Dependencies (install in your inference env)
--------------------------------------------
    pip install torch>=2.4 transformers safetensors huggingface_hub
    matplotlib seaborn numpy

The sae conda env may need torch upgraded:
    conda activate sae && pip install torch>=2.4 --upgrade
"""

import argparse
import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────

HERE = Path(__file__).parent
ARTIFACTS = HERE.parents[1] / "artifacts" / "activations"
# Original 2493-record run — used only for record selection (pre-computed activations + code text)
SAE_RUN = ARTIFACTS / "run_20260218_134529_vulnerable_code_qwen_coder_standard_16384_10M"
PAPER_FIGS = (
    HERE.parents[3]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)

# ── SAE model identifiers ─────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SAE_REPO  = "rufimelo/vulnerable_code_qwen_coder_standard_16384"
SAE_LAYER = 11

# ── Style (NeurIPS-compatible) ────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":     "monospace",
    "font.size":       8,
    "axes.titlesize":  9,
    "figure.dpi":      150,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load pre-computed meta + mean-pooled activations from JSONL
# ─────────────────────────────────────────────────────────────────────────────

def load_records(sae_run: Path) -> list[dict]:
    """
    Load all records from the SAE JSONL file.
    Each record has: vuln_id, cwe, file_extension,
                     secure_code (b64), vulnerable_code (b64),
                     secure (list[float] len=16384),
                     vulnerable (list[float] len=16384).
    """
    jsonl = next(sae_run.glob("activations_layer_*.jsonl"))
    records = []
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # Decode source code from base64
            r["secure_code_text"]    = base64.b64decode(r["secure_code"]).decode("utf-8", errors="replace")
            r["vulnerable_code_text"] = base64.b64decode(r["vulnerable_code"]).decode("utf-8", errors="replace")
            r["secure_acts"]    = np.array(r["secure"],    dtype=np.float32)
            r["vulnerable_acts"] = np.array(r["vulnerable"], dtype=np.float32)
            records.append(r)
    print(f"Loaded {len(records)} records from {jsonl.name}")
    return records


def top_examples(records: list[dict], feature_idx: int, cwe: str | None,
                 n: int) -> list[dict]:
    """
    Select top-N records by vulnerable_acts[feature_idx].
    Optionally filter to a specific CWE.
    """
    pool = [r for r in records if cwe is None or r["cwe"] == cwe]
    if not pool:
        print(f"  [WARN] No records match CWE={cwe}; using all CWEs.")
        pool = records
    pool.sort(key=lambda r: r["vulnerable_acts"][feature_idx], reverse=True)
    return pool[:n]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load model and SAE weights
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_id: str, device: str):
    """Load Qwen2.5-7B-Instruct in half precision."""
    print(f"Loading model {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print("  Model loaded.")
    return tokenizer, model


def load_sae_weights(repo_id: str, device: str) -> dict[str, torch.Tensor]:
    """
    Download and load SAE encoder weights from HuggingFace.

    SAELens stores weights as safetensors with keys:
      W_enc  [d_sae, d_model]   encoder weight matrix
      b_enc  [d_sae]            encoder bias
      W_dec  [d_model, d_sae]   decoder weight matrix
      b_dec  [d_model]          decoder bias (pre-encoder centering)

    Tries sae_weights.safetensors first, then falls back to any .safetensors
    or .pt file in the repo.
    """
    print(f"Loading SAE from {repo_id} ...")

    # list_repo_files returns strings in huggingface_hub >= 0.20, RepoFile objects in older versions
    _repo_iter     = list_repo_files(repo_id)
    all_repo_files = [f if isinstance(f, str) else f.rfilename for f in _repo_iter]

    # Prefer the layer-specific SAELens path, then generic fallbacks
    candidate_files = [
        f"blocks.{SAE_LAYER}.hook_resid_post/sae_weights.safetensors",  # layer-specific (SAELens)
        "sae_weights.safetensors",
        "model.safetensors",
    ]
    safetensor_files = [f for f in all_repo_files if f.endswith(".safetensors")]
    pt_files         = [f for f in all_repo_files if f.endswith(".pt")]

    weights = None
    for fname in candidate_files + safetensor_files:
        if fname in all_repo_files:
            local = hf_hub_download(repo_id=repo_id, filename=fname)
            weights = load_safetensors(local, device="cpu")
            print(f"  Loaded safetensors: {fname}  keys={list(weights.keys())}")
            break

    if weights is None and pt_files:
        local = hf_hub_download(repo_id=repo_id, filename=pt_files[0])
        weights = torch.load(local, map_location="cpu", weights_only=True)
        print(f"  Loaded .pt: {pt_files[0]}  keys={list(weights.keys())}")

    if weights is None:
        raise FileNotFoundError(f"Could not find SAE weights in {repo_id}. Files: {all_repo_files}")

    # Normalise key names (handle nested dicts from some SAELens versions)
    if "encoder.weight" in weights:
        weights = {
            "W_enc": weights["encoder.weight"],
            "b_enc": weights["encoder.bias"],
            "W_dec": weights["decoder.weight"],
            "b_dec": weights.get("b_dec", torch.zeros(weights["encoder.weight"].shape[1])),
        }

    return {k: v.float().to(device) for k, v in weights.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-token forward pass
# ─────────────────────────────────────────────────────────────────────────────

def get_token_activations(
    model,
    tokenizer,
    sae: dict[str, torch.Tensor],
    code_text: str,
    layer_idx: int,
    device: str,
    max_tokens: int = 200,
) -> tuple[list[str], np.ndarray]:
    """
    Run code_text through Qwen, capture layer `layer_idx` residual stream,
    apply SAE encoder, return (tokens, feature_acts).

    Returns
    -------
    tokens : list[str]  — one string per token (readable display form)
    feature_acts : np.ndarray  shape [seq_len, d_sae]
    """
    inputs = tokenizer(
        code_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
        add_special_tokens=True,
    ).to(device)

    hidden = {}

    def _hook(module, inp, out):
        # out is a tuple; out[0] is the hidden state [batch, seq, d_model]
        hidden["resid"] = out[0].detach().float().cpu()

    handle = model.model.layers[layer_idx].register_forward_hook(_hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    resid = hidden["resid"][0]  # [seq_len, d_model]

    # SAE encode: relu((x - b_dec) @ W_enc + b_enc)
    # W_enc is stored as [d_model, d_sae] in this SAELens release (no transpose needed)
    W_enc = sae["W_enc"].cpu()   # [d_model, d_sae]
    b_enc = sae["b_enc"].cpu()   # [d_sae]
    b_dec = sae.get("b_dec", torch.zeros(resid.shape[-1])).cpu()  # [d_model]

    x_cent     = resid - b_dec.unsqueeze(0)       # [seq, d_model]
    pre_acts   = x_cent @ W_enc + b_enc           # [seq, d_sae]
    feature_acts = F.relu(pre_acts).numpy()        # [seq, d_sae]

    # Readable token strings
    token_ids = inputs["input_ids"][0].tolist()
    raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = [
        t.replace("▁", " ").replace("Ġ", " ").replace("Ċ", "\n").replace("Ä", " ")
        if t is not None else "[UNK]"
        for t in raw_tokens
    ]

    return tokens, feature_acts


# ─────────────────────────────────────────────────────────────────────────────
# 4. Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _count_display_rows(tokens: list[str], max_chars_per_line: int = 60) -> int:
    """Count how many wrapped display rows a token list will occupy."""
    col, rows = 0, 1
    for tok in tokens:
        if tok and "\n" in tok:
            rows += 1
            col = 0
            continue
        tok_len = max(len(tok) if tok else 1, 1)
        if col > 0 and col + tok_len > max_chars_per_line:
            rows += 1
            col = 0
        col += tok_len
    return rows


def render_token_heatmap(
    ax,
    tokens: list[str],
    activations: np.ndarray,  # 1-D, one value per token
    title: str,
    vmax: float,
    cmap,
    max_chars_per_line: int = 60,
):
    """
    Draw tokens as coloured text boxes on `ax` using data coordinates.

    Each token is rendered with ax.text(..., bbox=dict(facecolor=color, ...)).
    x = character column, y = -row (rows go downward from 0).
    """
    vmax_safe = max(float(vmax), 1e-6)
    norm = plt.Normalize(vmin=0, vmax=vmax_safe)
    col = 0   # x: character column
    row = 0   # display row (y = -row so rows go down)

    for tok, act in zip(tokens, activations):
        # Hard newline token → move to next row
        if tok and "\n" in tok:
            row += 1
            col = 0
            continue

        display = tok if tok else " "
        tok_len = max(len(display), 1)

        # Soft line-wrap
        if col > 0 and col + tok_len > max_chars_per_line:
            row += 1
            col = 0

        color = cmap(norm(act))
        # Luminance-based text colour for readability
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = "white" if lum < 0.45 else "#111111"

        ax.text(
            col + tok_len * 0.5,  # x center
            -row,                 # y: top row = 0, increases downward
            display,
            fontsize=6,
            fontfamily="monospace",
            ha="center", va="center",
            color=text_color,
            bbox=dict(facecolor=color, edgecolor="none",
                      pad=0.25, boxstyle="square,pad=0.25"),
            zorder=2,
            clip_on=True,
        )
        col += tok_len

    ax.set_xlim(-0.5, max_chars_per_line + 0.5)
    ax.set_ylim(-row - 1.5, 1.0)
    ax.axis("off")
    ax.set_title(title, fontsize=7, fontweight="bold", pad=4)


def make_feature_figure(
    records_and_acts: list[tuple[dict, tuple, tuple]],
    feature_idx: int,
    vmax: float,
    out_path: Path,
    max_chars_per_line: int = 60,
):
    """
    records_and_acts : list of (record, (v_tokens, v_acts[seq,d_sae]),
                                        (s_tokens, s_acts[seq,d_sae]))
    Each row in the figure = one example pair (vulnerable | secure).
    """
    try:
        cmap = plt.colormaps["YlOrRd"]
    except AttributeError:
        import matplotlib.cm as _cm
        cmap = _cm.get_cmap("YlOrRd")  # matplotlib < 3.7 fallback

    n_rows = len(records_and_acts)

    # Compute per-subplot height from number of wrapped display rows
    row_heights = []
    for _rec, (v_tokens, _), (s_tokens, _) in records_and_acts:
        n_disp = max(
            _count_display_rows(v_tokens, max_chars_per_line),
            _count_display_rows(s_tokens, max_chars_per_line),
        )
        row_heights.append(max(n_disp * 0.20 + 0.5, 1.5))  # inches

    total_h = sum(row_heights) + 1.0  # suptitle space
    fig = plt.figure(figsize=(13.0, total_h))
    gs = fig.add_gridspec(
        n_rows, 2,
        height_ratios=row_heights,
        hspace=0.55, wspace=0.06,
        left=0.01, right=0.91, top=0.93, bottom=0.02,
    )

    for row_i, (rec, (v_tokens, v_acts_seq), (s_tokens, s_acts_seq)) in enumerate(records_and_acts):
        ax_v = fig.add_subplot(gs[row_i, 0])
        ax_s = fig.add_subplot(gs[row_i, 1])

        v_feature = v_acts_seq[:, feature_idx]
        s_feature = s_acts_seq[:, feature_idx]

        print(f"  row {row_i}: vuln max={v_feature.max():.4f}  secure max={s_feature.max():.4f}")

        vuln_title = (
            f"Vulnerable | {rec['cwe']} | {rec.get('file_extension', '?')}  "
            f"(mean={v_feature.mean():.3f}, max={v_feature.max():.3f})"
        )
        sec_title = (
            f"Secure (patched)  "
            f"(mean={s_feature.mean():.3f}, max={s_feature.max():.3f})"
        )

        render_token_heatmap(ax_v, v_tokens, v_feature, vuln_title, vmax, cmap, max_chars_per_line)
        render_token_heatmap(ax_s, s_tokens, s_feature, sec_title,  vmax, cmap, max_chars_per_line)

    # Shared colorbar on the right
    vmax_safe = max(float(vmax), 1e-6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax_safe))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.08, 0.015, 0.80])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"Feature {feature_idx} activation", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Per-token SAE Feature {feature_idx} activation  (Layer {SAE_LAYER})",
        fontsize=9, y=0.98,
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features",    nargs="+", type=int, default=[1797, 9193],
                   help="SAE feature indices to visualise")
    p.add_argument("--cwe",         type=str,  default="CWE-119",
                   help="Filter examples to this CWE (e.g. CWE-119). Use 'all' for no filter.")
    p.add_argument("--n_examples",  type=int,  default=2,
                   help="Number of paired examples per feature")
    p.add_argument("--max_tokens",  type=int,  default=180,
                   help="Truncate code to this many tokens")
    p.add_argument("--device",      type=str,  default="cuda",
                   help="Inference device: cuda / mps / cpu")
    p.add_argument("--out_dir",     type=Path, default=PAPER_FIGS,
                   help="Output directory for PDF figures")
    p.add_argument("--model_id",    type=str,  default=MODEL_ID)
    p.add_argument("--sae_repo",    type=str,  default=SAE_REPO)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cwe_filter = None if args.cwe.lower() == "all" else args.cwe

    # ── Load meta + mean-pooled activations ───────────────────────────────────
    records = load_records(SAE_RUN)

    # ── Load model + SAE (heavy — only once) ──────────────────────────────────
    tokenizer, model = load_model(args.model_id, args.device)
    sae = load_sae_weights(args.sae_repo, args.device)

    # ── Per-feature loop ──────────────────────────────────────────────────────
    for feat_idx in args.features:
        print(f"\n── Feature {feat_idx} ──")

        examples = top_examples(records, feat_idx, cwe_filter, args.n_examples)
        if not examples:
            print(f"  No examples found, skipping.")
            continue

        # Compute vmax from mean-pooled activations for a stable colour scale
        vmax_seq = max(r["vulnerable_acts"][feat_idx] for r in examples)
        # Per-token max may be higher; we'll rescale after the forward passes
        per_token_data = []

        for i, rec in enumerate(examples):
            print(f"  Example {i+1}/{len(examples)}: {rec['vuln_id']}  {rec['cwe']}")

            v_tokens, v_acts = get_token_activations(
                model, tokenizer, sae,
                rec["vulnerable_code_text"], SAE_LAYER, args.device, args.max_tokens,
            )
            s_tokens, s_acts = get_token_activations(
                model, tokenizer, sae,
                rec["secure_code_text"], SAE_LAYER, args.device, args.max_tokens,
            )

            # Update vmax from per-token observations
            vmax_seq = max(vmax_seq,
                           v_acts[:, feat_idx].max(),
                           s_acts[:, feat_idx].max())

            per_token_data.append((rec, (v_tokens, v_acts), (s_tokens, s_acts)))

        vmax = float(vmax_seq) * 1.05  # 5% headroom

        out_path = args.out_dir / f"fig_token_feature_{feat_idx}.pdf"
        make_feature_figure(per_token_data, feat_idx, vmax, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
