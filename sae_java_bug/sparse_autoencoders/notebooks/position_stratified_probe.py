"""
position_stratified_probe.py
============================
Position-stratified SAE feature activation analysis.

For the top discriminative SAE features at Layer 11 (those with the largest
secure-vs-vulnerable activation frequency shifts), computes the mean activation
as a function of NORMALISED token position (0 = sequence start, 1 = end) for
secure and vulnerable code separately.

Hypothesis (negative-space encoding): discriminative features — those that fire
more on secure code globally — should show a position-dependent gap between
secure and vulnerable.  If the gap is non-uniform, it reveals WHERE in the
sequence defensive coding patterns are absent in vulnerable code.

Outputs
-------
  <out_dir>/
    positional_profiles_raw.jsonl     — per-sample binned activations (checkpoint)
    fig_positional_profiles.pdf       — 5×2 panel, one subplot per feature
    fig_position_delta_heatmap.pdf    — features × position bins (secure - vuln diff)
    probe_results.json                — aggregated numbers for paper

Usage
-----
# Full run (2493 samples, needs GPU; slow on CPU/MPS):
python position_stratified_probe.py --n_sample -1 --device mps

# Quick local test (200 samples):
python position_stratified_probe.py --n_sample 200 --device mps

# Regenerate figures from a previous checkpoint (no inference):
python position_stratified_probe.py --load_from artifacts/token_viz/positional_profiles_raw.jsonl
"""

import argparse
import base64
import json
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths ──────────────────────────────────────────────────────────────────────

HERE       = Path(__file__).parent
ARTIFACTS  = HERE.parents[1] / "artifacts" / "activations"
OUT_ROOT   = HERE.parents[1] / "artifacts" / "token_viz"
PAPER_FIGS = (
    HERE.parents[3]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)

DEFAULT_JSONL = (
    ARTIFACTS
    / "run_20260218_134529_vulnerable_code_qwen_coder_standard_16384_10M"
    / "activations_layer_11_sae_blocks.11.hook_resid_post_component_hook_resid_post.hook_sae_acts_post.jsonl"
)

MODEL_ID  = "Qwen/Qwen2.5-7B-Instruct"
SAE_REPO  = "rufimelo/vulnerable_code_qwen_coder_standard_16384"
SAE_LAYER = 11

# Top 10 discriminative features from paper Table 1
# (all Δf < 0: fire more on secure code)
TOP_FEATURES_META = [
    (80,    2.50, -0.030, "Cond. branching & resource dispatch"),
    (4135,  2.42, -0.045, "Function entry & parameter setup"),
    (1185,  2.36, -0.034, "Security config & input sanitisation"),
    (6401,  2.32, -0.036, "Short snippets, incomplete security"),
    (13920, 2.29, -0.037, "Control flow past validation"),
    (2860,  2.28, -0.037, "Input parsing, inadequate validation"),
    (14439, 2.20, -0.039, "End-of-file & file termination"),
    (15720, 2.20, -0.036, "Config/protocol & HTTP headers"),
    (15765, 2.20, -0.037, "Raw memory ops & pointer manip."),
    (9193,  2.17, -0.031, "Resource cleanup & error-handling"),
]
TOP_FEATURE_IDS = [f[0] for f in TOP_FEATURES_META]

N_BINS    = 20
SEED      = 42

# ── Plot style ─────────────────────────────────────────────────────────────────

mpl.rcParams.update({
    "font.family":     "serif",
    "font.size":       8,
    "axes.titlesize":  8,
    "axes.labelsize":  8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi":      150,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

# ── Data loading ───────────────────────────────────────────────────────────────

def load_records(jsonl_path: Path, n_sample: int, seed: int = SEED) -> list[dict]:
    """Load records from JSONL; subsample if n_sample > 0."""
    records = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["secure_code_text"]    = base64.b64decode(r["secure_code"]).decode("utf-8", errors="replace")
            r["vulnerable_code_text"] = base64.b64decode(r["vulnerable_code"]).decode("utf-8", errors="replace")
            # Keep pre-computed last-token SAE activations for reference
            r["secure_acts"]   = np.array(r["secure"],   dtype=np.float32)
            r["vulnerable_acts"] = np.array(r["vulnerable"], dtype=np.float32)
            records.append(r)

    if 0 < n_sample < len(records):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(records), size=n_sample, replace=False)
        records = [records[i] for i in sorted(idx)]

    print(f"Loaded {len(records)} records")
    return records


# ── SAE loading (same pattern as token_feature_viz.py) ───────────────────────

def load_sae_weights(repo_id: str, device: str) -> dict[str, torch.Tensor]:
    _repo_iter     = list_repo_files(repo_id)
    all_repo_files = [f if isinstance(f, str) else f.rfilename for f in _repo_iter]

    candidates = [
        f"blocks.{SAE_LAYER}.hook_resid_post/sae_weights.safetensors",
        "sae_weights.safetensors",
        "model.safetensors",
    ]
    safetensor_files = [f for f in all_repo_files if f.endswith(".safetensors")]
    pt_files         = [f for f in all_repo_files if f.endswith(".pt")]

    weights = None
    for fname in candidates + safetensor_files:
        if fname in all_repo_files:
            local = hf_hub_download(repo_id=repo_id, filename=fname)
            weights = load_safetensors(local, device="cpu")
            print(f"  SAE loaded: {fname}  keys={list(weights.keys())}")
            break

    if weights is None and pt_files:
        local = hf_hub_download(repo_id=repo_id, filename=pt_files[0])
        weights = torch.load(local, map_location="cpu", weights_only=True)

    if weights is None:
        raise FileNotFoundError(f"Could not find SAE weights in {repo_id}")

    if "encoder.weight" in weights:
        weights = {
            "W_enc": weights["encoder.weight"],
            "b_enc": weights["encoder.bias"],
            "W_dec": weights["decoder.weight"],
            "b_dec": weights.get("b_dec", torch.zeros(weights["encoder.weight"].shape[1])),
        }

    return {k: v.float().to(device) for k, v in weights.items()}


# ── Per-token forward pass ─────────────────────────────────────────────────────

def get_token_feature_acts(
    model,
    tokenizer,
    sae: dict[str, torch.Tensor],
    code_text: str,
    feature_indices: list[int],
    device: str,
    max_tokens: int = 2048,
) -> tuple[int, np.ndarray]:
    """
    Run code_text through Qwen + SAE encoder, return per-token activations for
    the requested feature indices.

    Returns
    -------
    seq_len   : int
    feat_acts : np.ndarray  shape [seq_len, len(feature_indices)]
    """
    inputs = tokenizer(
        code_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
        add_special_tokens=False,
    ).to(device)

    hidden = {}

    def _hook(module, inp, out):
        hidden["resid"] = inp[0].detach().float().cpu()

    handle = model.model.layers[SAE_LAYER].register_forward_hook(_hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    resid = hidden["resid"][0]  # [seq_len, d_model]
    seq_len = resid.shape[0]

    W_enc = sae["W_enc"].cpu()   # [d_model, d_sae]
    b_enc = sae["b_enc"].cpu()   # [d_sae]
    b_dec = sae.get("b_dec", torch.zeros(resid.shape[-1])).cpu()

    x_cent    = resid - b_dec.unsqueeze(0)
    pre_acts  = x_cent @ W_enc + b_enc
    all_acts  = F.relu(pre_acts).numpy()  # [seq_len, d_sae]

    return seq_len, all_acts[:, feature_indices]  # [seq_len, n_features]


# ── Position binning ───────────────────────────────────────────────────────────

def bin_acts_by_position(feat_acts: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """
    Average feature activations within each normalised-position bin.

    Parameters
    ----------
    feat_acts : [seq_len, n_features]

    Returns
    -------
    binned : [n_bins, n_features]  mean activation per bin
    """
    seq_len, n_features = feat_acts.shape
    binned = np.zeros((n_bins, n_features), dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)

    for t in range(seq_len):
        b = min(int(t / seq_len * n_bins), n_bins - 1)
        binned[b] += feat_acts[t]
        counts[b] += 1

    safe_counts = np.maximum(counts, 1)[:, np.newaxis]
    return (binned / safe_counts).astype(np.float32)


# ── Inference loop ─────────────────────────────────────────────────────────────

def run_inference(
    records: list[dict],
    model,
    tokenizer,
    sae: dict[str, torch.Tensor],
    feature_indices: list[int],
    device: str,
    max_tokens: int,
    checkpoint_path: Path,
) -> list[dict]:
    """
    For each record, extract per-token SAE feature activations and bin by position.
    Writes a checkpoint JSONL after each sample.

    Returns list of result dicts (one per record).
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    t0 = time.time()

    with checkpoint_path.open("w") as ckpt:
        for i, rec in enumerate(records):
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - t0
                eta     = elapsed / (i + 1) * (len(records) - i - 1)
                print(f"  [{i+1:>4}/{len(records)}]  {elapsed:.0f}s elapsed  ETA {eta:.0f}s")

            try:
                v_len, v_acts = get_token_feature_acts(
                    model, tokenizer, sae,
                    rec["vulnerable_code_text"], feature_indices, device, max_tokens,
                )
                s_len, s_acts = get_token_feature_acts(
                    model, tokenizer, sae,
                    rec["secure_code_text"], feature_indices, device, max_tokens,
                )
            except Exception as e:
                print(f"  [WARN] {rec['vuln_id']}: {e}")
                continue

            v_binned = bin_acts_by_position(v_acts, N_BINS)  # [n_bins, n_feat]
            s_binned = bin_acts_by_position(s_acts, N_BINS)

            row = {
                "vuln_id":        rec["vuln_id"],
                "cwe":            rec["cwe"],
                "file_extension": rec.get("file_extension", ""),
                "v_len":          v_len,
                "s_len":          s_len,
                # shape [n_bins, n_features] stored as list-of-lists
                "v_binned": v_binned.tolist(),
                "s_binned": s_binned.tolist(),
            }
            results.append(row)
            ckpt.write(json.dumps(row) + "\n")
            ckpt.flush()

    print(f"Inference complete: {len(results)} samples in {time.time()-t0:.0f}s")
    return results


def load_checkpoint(ckpt_path: Path) -> list[dict]:
    results = []
    with ckpt_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    print(f"Loaded {len(results)} samples from checkpoint: {ckpt_path}")
    return results


# ── Aggregate positional profiles ─────────────────────────────────────────────

def aggregate_profiles(results: list[dict], n_features: int) -> dict:
    """
    Compute mean positional profiles (secure and vulnerable) across all samples,
    with standard error.

    Returns dict with keys:
        v_mean, s_mean  — shape [n_bins, n_features]
        v_se,   s_se    — shape [n_bins, n_features]
        diff_mean       — s_mean - v_mean  (positive = secure fires more)
        n               — number of samples
    """
    n = len(results)
    v_stack = np.zeros((n, N_BINS, n_features), dtype=np.float64)
    s_stack = np.zeros((n, N_BINS, n_features), dtype=np.float64)

    for i, r in enumerate(results):
        v_stack[i] = np.array(r["v_binned"], dtype=np.float64)
        s_stack[i] = np.array(r["s_binned"], dtype=np.float64)

    v_mean = v_stack.mean(axis=0).astype(np.float32)   # [n_bins, n_features]
    s_mean = s_stack.mean(axis=0).astype(np.float32)
    v_se   = (v_stack.std(axis=0) / np.sqrt(n)).astype(np.float32)
    s_se   = (s_stack.std(axis=0) / np.sqrt(n)).astype(np.float32)

    return {
        "v_mean":    v_mean,
        "s_mean":    s_mean,
        "v_se":      v_se,
        "s_se":      s_se,
        "diff_mean": s_mean - v_mean,  # positive = secure fires more
        "n":         n,
    }


# ── Figures ────────────────────────────────────────────────────────────────────

BIN_CENTERS = np.linspace(0, 1, N_BINS, endpoint=False) + 0.5 / N_BINS

SECURE_COLOR = "#2166ac"   # blue
VULN_COLOR   = "#d6604d"   # red


def fig_positional_profiles(agg: dict, feature_meta: list, out_path: Path):
    """
    5×2 panel: one subplot per feature, showing mean activation (± 1 SE) vs.
    normalised position for secure (blue) and vulnerable (red) code.
    """
    n_feat = len(feature_meta)
    n_rows, n_cols = 5, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.5, 10.0), sharey=False)
    axes = axes.flatten()

    for fi, (feat_id, score, delta_f, label) in enumerate(feature_meta):
        ax = axes[fi]
        vm = agg["v_mean"][:, fi]
        sm = agg["s_mean"][:, fi]
        ve = agg["v_se"][:, fi]
        se = agg["s_se"][:, fi]

        ax.plot(BIN_CENTERS, sm, color=SECURE_COLOR, lw=1.4, label="Secure")
        ax.fill_between(BIN_CENTERS, sm - se, sm + se, color=SECURE_COLOR, alpha=0.18)

        ax.plot(BIN_CENTERS, vm, color=VULN_COLOR, lw=1.4, label="Vulnerable")
        ax.fill_between(BIN_CENTERS, vm - ve, vm + ve, color=VULN_COLOR, alpha=0.18)

        ax.axhline(0, color="#888888", lw=0.5, ls=":")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
        ax.set_title(f"F{feat_id} | {label}\n(Δf={delta_f:+.3f})", fontsize=7)
        ax.set_xlabel("Normalised position", fontsize=7)
        ax.set_ylabel("Mean activation", fontsize=7)
        if fi == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Hide unused axes (if any)
    for ax in axes[n_feat:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Per-feature mean activation vs. normalised token position (Layer {SAE_LAYER}, n={agg['n']})\n"
        "Shaded = ±1 SE across samples. Blue = secure, red = vulnerable.",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def fig_delta_heatmap(agg: dict, feature_meta: list, out_path: Path):
    """
    Heatmap: features × position bins, colour = (secure_mean - vuln_mean).
    Green = secure fires more (supports negative-space hypothesis).
    """
    diff = agg["diff_mean"]  # [n_bins, n_features]
    data = diff.T             # [n_features, n_bins]

    labels = [f"F{fid}\n{lbl[:22]}" for fid, _, _, lbl in feature_meta]
    vabs   = np.abs(data).max()
    if vabs < 1e-6:
        vabs = 1e-6

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    im = ax.imshow(
        data,
        aspect="auto",
        cmap="RdBu",          # red = vuln fires more, blue = secure fires more
        vmin=-vabs, vmax=vabs,
        origin="upper",
        interpolation="nearest",
    )

    ax.set_xticks(range(N_BINS))
    ax.set_xticklabels(
        [f"{c:.2f}" for c in BIN_CENTERS],
        rotation=45, ha="right", fontsize=6,
    )
    ax.set_yticks(range(len(feature_meta)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Normalised token position")
    ax.set_title(
        f"Secure − Vulnerable mean activation by position (Layer {SAE_LAYER}, n={agg['n']})\n"
        "Blue = secure fires more  |  Red = vulnerable fires more",
        fontsize=8,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label("Δ activation (secure − vuln)", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    # Add zero-contour for clarity
    ax.contour(
        np.arange(N_BINS), np.arange(len(feature_meta)),
        data, levels=[0], colors=["black"], linewidths=0.5, linestyles=["--"],
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def fig_aggregate_gap(agg: dict, feature_meta: list, out_path: Path):
    """
    Bar chart: for each feature, the mean (secure - vuln) activation averaged
    over all positions, with ±1 SE error bar.
    Shows which features have the largest global positional gap.
    """
    n_feat = len(feature_meta)
    diff   = agg["diff_mean"]  # [n_bins, n_features]
    # Mean over bins
    gap_mean = diff.mean(axis=0)            # [n_features]
    # SE of the mean gap across bins (treat bins as independent)
    gap_se   = diff.std(axis=0) / np.sqrt(N_BINS)

    feat_ids = [f[0] for f in feature_meta]
    colors   = [SECURE_COLOR if g > 0 else VULN_COLOR for g in gap_mean]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    xs = np.arange(n_feat)
    ax.bar(xs, gap_mean, color=colors, alpha=0.75, yerr=gap_se,
           capsize=3, error_kw={"lw": 0.8})
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"F{fid}" for fid in feat_ids], rotation=45, ha="right")
    ax.set_ylabel("Mean Δ activation (secure − vuln)")
    ax.set_title(
        f"Position-averaged secure−vulnerable gap per feature (Layer {SAE_LAYER}, n={agg['n']})\n"
        "Blue bars = secure fires more (consistent with negative-space hypothesis)",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl",       type=Path, default=DEFAULT_JSONL,
                   help="JSONL with code text + last-token SAE activations")
    p.add_argument("--n_sample",    type=int,  default=200,
                   help="Number of pairs to analyse (-1 = all)")
    p.add_argument("--max_tokens",  type=int,  default=2048)
    p.add_argument("--device",      type=str,  default=None,
                   help="cuda / mps / cpu  (auto-detected if omitted)")
    p.add_argument("--out_dir",     type=Path, default=None,
                   help="Output directory (default: artifacts/token_viz/figures)")
    p.add_argument("--paper_figs",  action="store_true", default=False,
                   help="Also copy figures to the paper figures directory")
    p.add_argument("--load_from",   type=Path, default=None,
                   help="Skip inference; reload from this checkpoint JSONL")
    p.add_argument("--features",    nargs="+", type=int, default=None,
                   help="Override feature indices (default: top-10 discriminative)")
    p.add_argument("--model_id",    type=str,  default=MODEL_ID)
    p.add_argument("--sae_repo",    type=str,  default=SAE_REPO)
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # ── Features ──────────────────────────────────────────────────────────────
    if args.features is not None:
        feature_meta = [(fid, 0, 0, f"Feature {fid}") for fid in args.features]
    else:
        feature_meta = TOP_FEATURES_META
    feature_indices = [f[0] for f in feature_meta]
    n_features      = len(feature_indices)
    print(f"Features: {feature_indices}")

    # ── Output dirs ───────────────────────────────────────────────────────────
    out_dir = args.out_dir or (OUT_ROOT / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "positional_profiles_raw.jsonl"

    # ── Inference or load ──────────────────────────────────────────────────────
    if args.load_from is not None:
        results = load_checkpoint(args.load_from)
    else:
        records = load_records(
            args.jsonl,
            n_sample=args.n_sample if args.n_sample > 0 else 10**9,
        )
        print(f"\nLoading model {args.model_id} ...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        model     = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()

        sae = load_sae_weights(args.sae_repo, device)

        print(f"\nRunning inference on {len(records)} pairs ...")
        results = run_inference(
            records, model, tokenizer, sae,
            feature_indices, device, args.max_tokens, ckpt_path,
        )

    if not results:
        print("[ERROR] No results to analyse.")
        return

    # ── Aggregate ──────────────────────────────────────────────────────────────
    agg = aggregate_profiles(results, n_features)
    print(f"\nAggregated over {agg['n']} samples, {N_BINS} position bins, {n_features} features.")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Feature':>8}  {'Label':<36}  {'Gap (s-v)':>10}  {'Max|diff|':>10}")
    print("-" * 70)
    diff = agg["diff_mean"]  # [n_bins, n_features]
    for fi, (fid, score, df, label) in enumerate(feature_meta):
        gap   = diff[:, fi].mean()
        maxd  = np.abs(diff[:, fi]).max()
        print(f"  F{fid:>5}  {label:<36}  {gap:>+10.4f}  {maxd:>10.4f}")
    print("=" * 70)

    # ── Save aggregated results ────────────────────────────────────────────────
    results_path = out_dir / "probe_results.json"
    save_payload = {
        "n_samples":      agg["n"],
        "n_bins":         N_BINS,
        "bin_centers":    BIN_CENTERS.tolist(),
        "features":       [
            {
                "feature_idx": fid,
                "score":       score,
                "delta_f":     df,
                "label":       label,
                "gap_mean":    float(diff[:, fi].mean()),
                "gap_max":     float(np.abs(diff[:, fi]).max()),
                "s_profile":   agg["s_mean"][:, fi].tolist(),
                "v_profile":   agg["v_mean"][:, fi].tolist(),
                "diff_profile": diff[:, fi].tolist(),
            }
            for fi, (fid, score, df, label) in enumerate(feature_meta)
        ],
    }
    with results_path.open("w") as f:
        json.dump(save_payload, f, indent=2)
    print(f"Results saved: {results_path}")

    # ── Figures ───────────────────────────────────────────────────────────────
    fig_profiles_path = out_dir / "fig_positional_profiles.pdf"
    fig_heatmap_path  = out_dir / "fig_position_delta_heatmap.pdf"
    fig_gap_path      = out_dir / "fig_position_gap_bar.pdf"

    fig_positional_profiles(agg, feature_meta, fig_profiles_path)
    fig_delta_heatmap(agg, feature_meta, fig_heatmap_path)
    fig_aggregate_gap(agg, feature_meta, fig_gap_path)

    if args.paper_figs:
        import shutil
        PAPER_FIGS.mkdir(parents=True, exist_ok=True)
        for src in [fig_profiles_path, fig_heatmap_path, fig_gap_path]:
            dst = PAPER_FIGS / src.name
            shutil.copy2(src, dst)
            print(f"  Copied to paper: {dst}")

    print("\nDone.")


if __name__ == "__main__":
    main()