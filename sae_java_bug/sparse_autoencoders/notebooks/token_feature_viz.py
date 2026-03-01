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

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────

HERE = Path(__file__).parent
ARTIFACTS = HERE.parents[1] / "artifacts" / "activations"
SAE_RUN = ARTIFACTS / "run_20260218_134529_vulnerable_code_qwen_coder_standard_16384_10M"
PAPER_FIGS = (
    HERE.parents[3]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)

# ── SAE model identifiers ─────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SAE_REPO  = "rufimelo/vulnerable_code_qwen_coder_standard_16384_10M"
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

    # Try the canonical SAELens filename first
    candidate_files = ["sae_weights.safetensors", "model.safetensors"]
    all_repo_files  = [f.rfilename for f in list_repo_files(repo_id)]

    # Prefer safetensors, fall back to .pt
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

    # SAE encode: relu((x - b_dec) @ W_enc.T + b_enc)
    W_enc = sae["W_enc"].cpu()   # [d_sae, d_model]
    b_enc = sae["b_enc"].cpu()   # [d_sae]
    b_dec = sae.get("b_dec", torch.zeros(resid.shape[-1])).cpu()  # [d_model]

    x_cent     = resid - b_dec.unsqueeze(0)       # [seq, d_model]
    pre_acts   = x_cent @ W_enc.T + b_enc         # [seq, d_sae]
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

def render_token_heatmap(
    ax,
    tokens: list[str],
    activations: np.ndarray,  # 1-D, one value per token
    title: str,
    vmax: float,
    cmap,
    max_chars_per_line: int = 55,
):
    """
    Draw tokens as coloured boxes on `ax`.
    Line-wraps at actual newlines in the code or at max_chars_per_line.
    """
    norm   = plt.Normalize(vmin=0, vmax=vmax)
    x0, y  = 0.01, 0.96
    char_w = 1.0 / max_chars_per_line      # normalised width per character
    row_h  = 0.10
    gap    = char_w * 0.3

    col = 0  # character column tracker

    for tok, act in zip(tokens, activations):
        # Hard line-break tokens
        if tok.strip() == "" and "\n" in tok:
            y  -= row_h
            col = 0
            x0  = 0.01
            continue

        display = tok if tok else " "
        w = max(len(display), 1) * char_w + gap

        # Soft wrap
        if col + len(display) > max_chars_per_line:
            y  -= row_h
            col = 0
            x0  = 0.01

        if y < 0.02:
            break

        color      = cmap(norm(act))
        text_color = "white" if act > vmax * 0.60 else "#111111"

        # Coloured background
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y - row_h * 0.88), w - gap * 0.5, row_h * 0.85,
            boxstyle="round,pad=0.002",
            facecolor=color, edgecolor="none",
            transform=ax.transAxes, clip_on=True, zorder=1,
        ))

        # Token text
        ax.text(
            x0 + w * 0.5, y - row_h * 0.44,
            display,
            fontsize=6.5, fontfamily="monospace",
            ha="center", va="center",
            color=text_color,
            transform=ax.transAxes, zorder=2,
        )

        x0  += w
        col += len(display)

    ax.set_xlim(0, 1)
    ax.set_ylim(max(0, y - row_h * 2), 1.0)
    ax.axis("off")
    ax.set_title(title, fontsize=8, fontweight="bold", pad=3)


def make_feature_figure(
    records_and_acts: list[tuple[dict, np.ndarray, np.ndarray]],
    feature_idx: int,
    vmax: float,
    out_path: Path,
):
    """
    records_and_acts : list of (record, vuln_token_acts[seq, d_sae], sec_token_acts[seq, d_sae])
    Each row in the figure = one example pair (vulnerable | secure).
    """
    n_rows  = len(records_and_acts)
    fig_h   = 2.8 * n_rows + 0.4
    fig, axes = plt.subplots(n_rows, 2, figsize=(6.75, fig_h))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    cmap = cm.get_cmap("YlOrRd")

    for row_i, (rec, vuln_acts, sec_acts) in enumerate(records_and_acts):
        v_tokens, v_acts_seq = vuln_acts
        s_tokens, s_acts_seq = sec_acts

        v_feature = v_acts_seq[:, feature_idx]
        s_feature = s_acts_seq[:, feature_idx]

        vuln_title = (
            f"Vulnerable  |  {rec['cwe']}  |  {rec['file_extension']}\n"
            f"mean act = {v_feature.mean():.3f}  max = {v_feature.max():.3f}"
        )
        sec_title = (
            f"Secure (patched)\n"
            f"mean act = {s_feature.mean():.3f}  max = {s_feature.max():.3f}"
        )

        render_token_heatmap(axes[row_i, 0], v_tokens, v_feature, vuln_title, vmax, cmap)
        render_token_heatmap(axes[row_i, 1], s_tokens, s_feature, sec_title,  vmax, cmap)

    # Shared colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    cbar.set_label(f"Feature {feature_idx} activation", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Per-token activation of SAE Feature {feature_idx} (Layer {SAE_LAYER})\n"
        f"Colour intensity = feature activation; white = zero",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


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
