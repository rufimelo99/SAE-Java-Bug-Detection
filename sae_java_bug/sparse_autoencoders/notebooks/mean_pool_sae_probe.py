"""
Mean-token SAE probe — reviewer Q3 response.

Compares four pooling × representation combinations at Layer 11:
  (1) last-token  × SAE features    [from pre-saved .pt, AUROC ~0.47]
  (2) last-token  × raw residual    [from pre-saved .pt, AUROC ~0.50]
  (3) mean-token  × raw residual    [from pre-saved mean_pool .pt, AUROC ~0.63]
  (4) mean-token  × SAE features    [NEW — encode each token → mean-pool features]
  (5) encode-of-mean × SAE features [encode the mean-pooled residual once, for comparison]

The key new experiment (4) is what the reviewer asked in Q3:
  "Have you tried mean-token pooling in SAE feature space (encoding per token
   then averaging features)?"

Usage (on GPU server):
    conda run -n sae python mean_pool_sae_probe.py [--checkpoint_dir PATH]

Saves:
    artifacts/activations/mean_pool_sae/<run_ts>/
        safe_mean_sae_layer_11.pt        shape [N, 16384]  — mean-token SAE
        vulnerable_mean_sae_layer_11.pt  shape [N, 16384]
        safe_enc_of_mean_layer_11.pt     shape [N, 16384]  — encode-of-mean
        vulnerable_enc_of_mean_layer_11.pt
        probe_results.json
    Paper figures dir / fig_mean_pool_sae_comparison.pdf

Checkpointing (to survive interruption):
    A JSONL checkpoint is written incrementally to <out_dir>/checkpoint.jsonl.
    If the script is re-run with --checkpoint_dir pointing to an existing out_dir,
    it skips already-processed samples and resumes from where it left off.
    Pass --checkpoint_dir auto to automatically find the most recent incomplete run.
"""

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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

import base64
import ctypes as _ctypes

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
ARTIFACTS  = Path(__file__).parents[2] / "artifacts" / "activations"
PAPER_FIGS = (
    Path(__file__).parents[4]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

# Pre-computed last-token SAE activations (L11)
LAST_SAE_DIR = ARTIFACTS / "run_20260218_134529_vulnerable_code_qwen_coder_standard_16384_10M"

# Pre-computed last-token raw residual activations (all layers)
RAW_RUN_DIR = ARTIFACTS / "raw_activations" / "vulnerable_code_qwen_coder_standard_16384_raw"
RAW_RUN_DIR_FALLBACK = ARTIFACTS / "run_20260216_231950_cl_secure_code_qwen_coder_topk_16384_raw"

# Source JSONL for code samples (any layer file works — we only need the code strings)
SOURCE_JSONL = RAW_RUN_DIR / "activations_layer_0_raw_component_hidden_state_last_token.jsonl"
if not SOURCE_JSONL.exists():
    SOURCE_JSONL = RAW_RUN_DIR_FALLBACK / "activations_layer_0_raw_component_hidden_state_last_token.jsonl"

MODEL_ID   = "Qwen/Qwen2.5-7B-Instruct"
SAE_REPO   = "rufimelo/vulnerable_code_qwen_coder_standard_16384_10M"
SAE_LAYER  = 11
MAX_TOKENS = 2048
SEED       = 42

# ── Style ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":     "serif",
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi":      150,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})


# ─────────────────────────────────────────────────────────────────────────────
# SAE loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sae_weights(repo_id: str, layer: int, device: str) -> dict[str, torch.Tensor]:
    """
    Download and load SAE weights from HuggingFace.
    Tries the SAELens layer-specific path first, then falls back to generic names.
    Returns dict with W_enc [d_model, d_sae], b_enc [d_sae], b_dec [d_model].
    """
    layer_prefix = f"blocks.{layer}.hook_resid_post"
    candidates = [
        f"{layer_prefix}/sae_weights.safetensors",
        "sae_weights.safetensors",
        "model.safetensors",
    ]

    repo_files = set(list_repo_files(repo_id))
    chosen = None
    for fname in candidates:
        if fname in repo_files:
            chosen = fname
            break
    if chosen is None:
        # Fallback: first .safetensors or .pt in repo
        for f in repo_files:
            if f.endswith(".safetensors") or f.endswith(".pt"):
                chosen = f
                break
    if chosen is None:
        raise FileNotFoundError(f"No weight file found in {repo_id}")

    print(f"  Loading SAE weights from {repo_id} / {chosen}")
    local_path = hf_hub_download(repo_id=repo_id, filename=chosen)

    if chosen.endswith(".safetensors"):
        raw = load_safetensors(local_path, device="cpu")
    else:
        raw = torch.load(local_path, map_location="cpu", weights_only=True)

    # Normalise key names
    key_map = {
        "W_enc": ["W_enc", "encoder.weight", "weight_enc"],
        "b_enc": ["b_enc", "encoder.bias",   "bias_enc"],
        "b_dec": ["b_dec", "decoder.bias",   "bias_dec", "pre_bias"],
    }
    weights = {}
    for canonical, aliases in key_map.items():
        for alias in aliases:
            if alias in raw:
                weights[canonical] = raw[alias].float().to(device)
                break

    # W_enc should be [d_model, d_sae]; transpose if stored as [d_sae, d_model]
    if "W_enc" in weights:
        W = weights["W_enc"]
        if W.shape[0] > W.shape[1]:   # stored as [d_sae, d_model] — transpose
            weights["W_enc"] = W.T

    d_model = weights["W_enc"].shape[0]
    if "b_dec" not in weights:
        weights["b_dec"] = torch.zeros(d_model, device=device)

    print(f"  W_enc: {weights['W_enc'].shape}  b_enc: {weights['b_enc'].shape}  "
          f"b_dec: {weights['b_dec'].shape}")
    return weights


def sae_encode_batch(residuals: torch.Tensor, weights: dict) -> torch.Tensor:
    """
    Encode a batch of residual vectors through the SAE.
    residuals: [batch, d_model]
    Returns:   [batch, d_sae]   (ReLU activations)
    """
    x = residuals - weights["b_dec"].unsqueeze(0)
    pre = x @ weights["W_enc"] + weights["b_enc"].unsqueeze(0)
    return F.relu(pre)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_source_records(jsonl_path: Path) -> list[dict]:
    records = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            try:
                secure_code = base64.b64decode(r["secure_code"]).decode("utf-8", errors="replace")
                vuln_code   = base64.b64decode(r["vulnerable_code"]).decode("utf-8", errors="replace")
            except Exception:
                secure_code = r.get("secure_code", "")
                vuln_code   = r.get("vulnerable_code", "")
            records.append({
                "vuln_id":         r["vuln_id"],
                "cwe":             r["cwe"],
                "file_extension":  r["file_extension"],
                "secure_code":     secure_code,
                "vulnerable_code": vuln_code,
            })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Extraction: all-token residuals at L11 → SAE encode → mean-pool
# ─────────────────────────────────────────────────────────────────────────────

def extract_mean_sae(
    code_str: str,
    tokenizer,
    model,
    sae_weights: dict,
    layer: int,
    device: str,
    max_tokens: int = MAX_TOKENS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      mean_sae_vec   : mean over token positions of per-token SAE features  [d_sae]
      enc_of_mean_vec: SAE encoding of the mean residual vector              [d_sae]

    Uses a forward hook to capture only layer `layer` activations, avoiding
    materialising all 28 hidden states simultaneously (saves ~400 MB VRAM).
    """
    inputs = tokenizer(
        code_str,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    ).to(device)

    # Hook captures post-resid output of transformer layer `layer` only
    captured: dict = {}

    def _hook(module, inp, out):
        # Qwen2 decoder layers return a tuple; element 0 is hidden states
        captured["h"] = out[0].detach().float().cpu()   # [1, seq_len, d_model]

    hook = model.model.layers[layer].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        hook.remove()

    h = captured["h"][0]   # [seq_len, d_model]

    # (4) mean-token SAE: encode each token, then average
    sae_feats = sae_encode_batch(h, sae_weights)        # [seq_len, d_sae]
    mean_sae  = sae_feats.mean(dim=0).numpy()           # [d_sae]

    # (5) encode-of-mean: average residuals first, then encode once
    mean_resid  = h.mean(dim=0, keepdim=True)           # [1, d_model]
    enc_of_mean = sae_encode_batch(mean_resid, sae_weights)[0].numpy()  # [d_sae]

    return mean_sae, enc_of_mean


# ─────────────────────────────────────────────────────────────────────────────
# Probe utilities
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_auc_ci(y_true, y_score, n_bootstrap=500, ci=0.95, seed=SEED):
    from sklearn.metrics import roc_auc_score
    rng  = np.random.default_rng(seed)
    n    = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    aucs = np.array(aucs)
    alpha = (1 - ci) / 2
    return aucs.mean(), np.quantile(aucs, alpha), np.quantile(aucs, 1 - alpha)


def probe_vuln_secure(safe_mat, vuln_mat, n_components=50, cv=5, n_bootstrap=500):
    n = len(safe_mat)
    X = np.vstack([safe_mat, vuln_mat]).astype(np.float32)
    y = np.array([0] * n + [1] * n, dtype=int)

    n_comp  = min(n_components, X.shape[1], X.shape[0] - 1)
    clf     = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=SEED)
    skf     = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)

    y_score = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        pca    = PCA(n_components=min(n_comp, len(train_idx) - 1), random_state=SEED)
        X_tr   = pca.fit_transform(scaler.fit_transform(X[train_idx]))
        X_te   = pca.transform(scaler.transform(X[test_idx]))
        clf.fit(X_tr, y[train_idx])
        y_score[test_idx] = clf.predict_proba(X_te)[:, 1]

    mean_auc, ci_lo, ci_hi = _bootstrap_auc_ci(y, y_score, n_bootstrap=n_bootstrap)
    return {"roc_auc": mean_auc, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": n}


# ─────────────────────────────────────────────────────────────────────────────
# Load pre-computed baselines (conditions 1-3)
# ─────────────────────────────────────────────────────────────────────────────

def _t2np(path) -> np.ndarray:
    """Load a .pt tensor to numpy via ctypes (avoids NumPy 2.x / PyTorch compat issue)."""
    t = torch.load(path, weights_only=True, map_location="cpu").float().contiguous()
    buf = (_ctypes.c_float * t.numel()).from_address(t.data_ptr())
    return np.ctypeslib.as_array(buf).reshape(t.shape).copy()


def load_last_token_sae() -> dict | None:
    """Load pre-computed last-token SAE activations at L11 and run probe."""
    safe_pt = LAST_SAE_DIR / "safe_layer_11.pt"
    vuln_pt = LAST_SAE_DIR / "vulnerable_layer_11.pt"
    if not (safe_pt.exists() and vuln_pt.exists()):
        print(f"  [SKIP] last-token SAE .pt not found at {LAST_SAE_DIR}")
        return None
    safe_mat = _t2np(safe_pt)
    vuln_mat = _t2np(vuln_pt)
    result   = probe_vuln_secure(safe_mat, vuln_mat)
    print(f"  Last-token SAE  L11: AUROC {result['roc_auc']:.3f} "
          f"[{result['ci_lo']:.3f}–{result['ci_hi']:.3f}]")
    return result


def load_last_token_raw() -> dict | None:
    """Load pre-computed last-token raw residual at L11 and run probe."""
    # Try .pt files first
    for run_dir in [RAW_RUN_DIR, RAW_RUN_DIR_FALLBACK]:
        safe_pt = run_dir / "safe_layer_11.pt"
        vuln_pt = run_dir / "vulnerable_layer_11.pt"
        if safe_pt.exists() and vuln_pt.exists():
            safe_mat = _t2np(safe_pt)
            vuln_mat = _t2np(vuln_pt)
            result   = probe_vuln_secure(safe_mat, vuln_mat)
            print(f"  Last-token raw  L11: AUROC {result['roc_auc']:.3f} "
                  f"[{result['ci_lo']:.3f}–{result['ci_hi']:.3f}]")
            return result
    # Try JSONL
    for run_dir in [RAW_RUN_DIR, RAW_RUN_DIR_FALLBACK]:
        matches = list(run_dir.glob("activations_layer_11_*.jsonl"))
        if matches:
            rows_safe, rows_vuln = [], []
            with matches[0].open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    rows_safe.append(r["secure"])
                    rows_vuln.append(r["vulnerable"])
            safe_mat = np.array(rows_safe, dtype=np.float32)
            vuln_mat = np.array(rows_vuln, dtype=np.float32)
            result   = probe_vuln_secure(safe_mat, vuln_mat)
            print(f"  Last-token raw  L11: AUROC {result['roc_auc']:.3f} "
                  f"[{result['ci_lo']:.3f}–{result['ci_hi']:.3f}]")
            return result
    print("  [SKIP] last-token raw L11: no data found")
    return None


def load_mean_token_raw() -> dict | None:
    """Load mean-token raw residual results from the most recent mean_pool run."""
    mean_pool_root = ARTIFACTS / "mean_pool"
    if not mean_pool_root.exists():
        print("  [SKIP] mean_pool dir not found")
        return None
    # Find most recent run with L11 tensors
    runs = sorted(mean_pool_root.iterdir(), reverse=True)
    for run_dir in runs:
        safe_pt = run_dir / "safe_layer_11.pt"
        vuln_pt = run_dir / "vulnerable_layer_11.pt"
        if safe_pt.exists() and vuln_pt.exists():
            safe_mat = _t2np(safe_pt)
            vuln_mat = _t2np(vuln_pt)
            result   = probe_vuln_secure(safe_mat, vuln_mat)
            print(f"  Mean-token raw  L11: AUROC {result['roc_auc']:.3f} "
                  f"[{result['ci_lo']:.3f}–{result['ci_hi']:.3f}]  (run: {run_dir.name})")
            return result
    print("  [SKIP] no mean_pool run with L11 tensors found")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def fig_comparison(results: dict[str, dict | None], out_path: Path):
    """
    Horizontal bar chart comparing AUROC across the four conditions at L11.
    Bars are ordered bottom-to-top by AUROC.
    """
    labels_order = [
        ("last_sae",       "Last-token SAE\n(existing result)"),
        ("last_raw",       "Last-token raw\n(existing result)"),
        ("mean_raw",       "Mean-token raw\n(existing result)"),
        ("enc_of_mean_sae","Encode-of-mean SAE\n(mean resid → SAE)"),
        ("mean_sae",       "Mean-token SAE\n(per-token encode → mean)"),
    ]

    # Filter to available results, keep order
    available = [(k, lbl) for k, lbl in labels_order if results.get(k)]
    keys   = [k   for k, _ in available]
    labels = [lbl for _, lbl in available]
    aucs   = [results[k]["roc_auc"] for k in keys]
    lo_err = [results[k]["roc_auc"] - results[k]["ci_lo"] for k in keys]
    hi_err = [results[k]["ci_hi"]   - results[k]["roc_auc"] for k in keys]

    # Colour: highlight new conditions
    colours = []
    for k in keys:
        if k in ("mean_sae", "enc_of_mean_sae"):
            colours.append("#d65f5f")
        elif k == "mean_raw":
            colours.append("#4878cf")
        else:
            colours.append("#888888")

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    y_pos = np.arange(len(keys))
    bars  = ax.barh(y_pos, aucs, xerr=[lo_err, hi_err],
                    color=colours, alpha=0.85, height=0.55,
                    error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": "black"})

    ax.axvline(0.5, color="grey", lw=0.8, ls=":", alpha=0.6, label="Chance (0.5)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("AUROC  (vuln vs. secure, L11, 95\\% bootstrap CI)")
    ax.set_title("Pooling strategy × representation at Layer 11", fontsize=9)
    ax.set_xlim(0.35, 0.75)

    for bar, auc in zip(bars, aucs):
        ax.text(auc + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{auc:.3f}", va="center", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _find_most_recent_incomplete_run() -> Path | None:
    """Return the most recent mean_pool_sae run dir that has a checkpoint but no probe_results."""
    root = ARTIFACTS / "mean_pool_sae"
    if not root.exists():
        return None
    for run_dir in sorted(root.iterdir(), reverse=True):
        ckpt = run_dir / "checkpoint.jsonl"
        done = run_dir / "probe_results.json"
        if ckpt.exists() and not done.exists():
            return run_dir
    return None


def _load_checkpoint(ckpt_path: Path) -> tuple[list[dict], set[str]]:
    """Load existing checkpoint records. Returns (records_list, seen_vuln_ids)."""
    records, seen = [], set()
    with ckpt_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records.append(r)
            seen.add(r["vuln_id"])
    return records, seen


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir", default=None,
        help="Path to an existing out_dir to resume from, or 'auto' to find most recent."
    )
    parser.add_argument(
        "--probe_only", action="store_true",
        help="Skip model inference; load latest cached mean_pool_sae .pt tensors and re-run probe only."
    )
    args = parser.parse_args()

    if args.probe_only:
        # ── Probe-only mode ───────────────────────────────────────────────────
        root = ARTIFACTS / "mean_pool_sae"
        run_dirs = sorted(
            (d for d in root.iterdir()
             if (d / f"safe_mean_sae_layer_{SAE_LAYER}.pt").exists()),
            reverse=True,
        )
        if not run_dirs:
            raise FileNotFoundError(f"No completed mean_pool_sae runs found under {root}")
        run_dir = run_dirs[0]
        print(f"[probe-only] Loading from {run_dir}")

        def _load(name):
            import ctypes
            t = torch.load(run_dir / name, weights_only=True).float().contiguous()
            buf = (ctypes.c_float * t.numel()).from_address(t.data_ptr())
            return np.ctypeslib.as_array(buf).reshape(t.shape).copy()

        safe_mean_sae_mat = _load(f"safe_mean_sae_layer_{SAE_LAYER}.pt")
        vuln_mean_sae_mat = _load(f"vulnerable_mean_sae_layer_{SAE_LAYER}.pt")
        safe_enc_mean_mat = _load(f"safe_enc_of_mean_layer_{SAE_LAYER}.pt")
        vuln_enc_mean_mat = _load(f"vulnerable_enc_of_mean_layer_{SAE_LAYER}.pt")

        result_mean_sae    = probe_vuln_secure(safe_mean_sae_mat, vuln_mean_sae_mat)
        result_enc_of_mean = probe_vuln_secure(safe_enc_mean_mat, vuln_enc_mean_mat)

        result_last_sae = load_last_token_sae()
        result_last_raw = load_last_token_raw()
        result_mean_raw = load_mean_token_raw()

        print("\n" + "=" * 72)
        print(f"AUROC comparison — Layer {SAE_LAYER} — vuln vs. secure (leakage-corrected)")
        print("=" * 72)
        for label, r in [
            ("Last-token  SAE",    result_last_sae),
            ("Last-token  raw",    result_last_raw),
            ("Mean-token  raw",    result_mean_raw),
            ("Encode-of-mean SAE", result_enc_of_mean),
            ("Mean-token  SAE",    result_mean_sae),
        ]:
            if r is None:
                print(f"  {label:<22}  —")
            else:
                marker = " ← NEW" if "SAE" in label and "Mean" in label else ""
                print(f"  {label:<22}  {r['roc_auc']:.3f} [{r['ci_lo']:.3f}–{r['ci_hi']:.3f}]{marker}")
        print("=" * 72)

        results_payload = {
            "probe_only": True,
            "run_dir": str(run_dir),
            "sae_layer": SAE_LAYER,
            "last_sae":        result_last_sae,
            "last_raw":        result_last_raw,
            "mean_raw":        result_mean_raw,
            "mean_sae":        result_mean_sae,
            "enc_of_mean_sae": result_enc_of_mean,
        }
        out_json = run_dir / "probe_results_corrected.json"
        with out_json.open("w") as f:
            json.dump(results_payload, f, indent=2)
        print(f"Results saved: {out_json}")

        all_results = {
            "last_sae":        result_last_sae,
            "last_raw":        result_last_raw,
            "mean_raw":        result_mean_raw,
            "mean_sae":        result_mean_sae,
            "enc_of_mean_sae": result_enc_of_mean,
        }
        fig_comparison(all_results, PAPER_FIGS / "fig_mean_pool_sae_comparison.pdf")
        print("\nDone.")
        return

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Source JSONL: {SOURCE_JSONL}")

    # ── 1. Load SAE weights ───────────────────────────────────────────────────
    print(f"\nLoading SAE weights (L{SAE_LAYER}) from {SAE_REPO}...")
    sae_weights = load_sae_weights(SAE_REPO, SAE_LAYER, device="cpu")

    # ── 2. Load LLM ───────────────────────────────────────────────────────────
    print(f"\nLoading model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(device)
    model.eval()
    print("  Model ready.")

    # ── 3. Load source records ────────────────────────────────────────────────
    print("\nLoading source records...")
    records = load_source_records(SOURCE_JSONL)
    print(f"  {len(records)} samples loaded")

    # ── 4. Set up output directory (new or resume) ────────────────────────────
    resume_dir = None
    if args.checkpoint_dir:
        if args.checkpoint_dir == "auto":
            resume_dir = _find_most_recent_incomplete_run()
            if resume_dir:
                print(f"\nAuto-resuming from: {resume_dir}")
            else:
                print("\nNo incomplete run found; starting fresh.")
        else:
            resume_dir = Path(args.checkpoint_dir)
            if not resume_dir.exists():
                print(f"[WARN] --checkpoint_dir {resume_dir} not found; starting fresh.")
                resume_dir = None

    if resume_dir:
        out_dir = resume_dir
        ts      = out_dir.name
    else:
        ts      = time.strftime("%Y%m%d_%H%M%S")
        out_dir = ARTIFACTS / "mean_pool_sae" / ts
        out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "checkpoint.jsonl"
    print(f"Output dir: {out_dir}")

    # Load existing checkpoint if resuming
    ckpt_records: list[dict] = []
    seen_ids: set[str]       = set()
    if ckpt_path.exists():
        ckpt_records, seen_ids = _load_checkpoint(ckpt_path)
        print(f"  Resuming: {len(ckpt_records)} samples already processed, "
              f"{len(records) - len(ckpt_records)} remaining.")

    # ── 5. Extract mean-token SAE activations (with checkpointing) ───────────
    print(f"\nExtracting mean-token SAE activations at L{SAE_LAYER} "
          f"({len(records)} samples × 2 sides)...")
    t0 = time.time()

    d_sae = sae_weights["b_enc"].shape[0]

    with ckpt_path.open("a") as ckpt_f:
        for i, rec in enumerate(records):
            if rec["vuln_id"] in seen_ids:
                continue  # already done in a previous run

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                done_so_far = len(ckpt_records) + (i - len(seen_ids))
                if done_so_far > 0:
                    eta = elapsed / done_so_far * (len(records) - done_so_far)
                    print(f"  [{i+1:>4}/{len(records)}]  elapsed {elapsed:.0f}s  ETA {eta:.0f}s")

            try:
                s_ms, s_em = extract_mean_sae(rec["secure_code"],    tokenizer, model,
                                              sae_weights, SAE_LAYER, device)
                v_ms, v_em = extract_mean_sae(rec["vulnerable_code"], tokenizer, model,
                                              sae_weights, SAE_LAYER, device)
            except Exception as e:
                print(f"  [WARN] sample {i} ({rec['vuln_id']}): {e}")
                s_ms = s_em = v_ms = v_em = np.zeros(d_sae, dtype=np.float32)

            row = {
                "vuln_id":        rec["vuln_id"],
                "cwe":            rec["cwe"],
                "file_extension": rec["file_extension"],
                "safe_mean_sae":  s_ms.tolist(),
                "vuln_mean_sae":  v_ms.tolist(),
                "safe_enc_mean":  s_em.tolist(),
                "vuln_enc_mean":  v_em.tolist(),
            }
            ckpt_f.write(json.dumps(row) + "\n")
            ckpt_f.flush()
            ckpt_records.append(row)

    print(f"  Checkpoint complete: {len(ckpt_records)} records in {ckpt_path.name}")

    # Reconstruct matrices from checkpoint
    safe_mean_sae_mat = np.array([r["safe_mean_sae"] for r in ckpt_records], dtype=np.float32)
    vuln_mean_sae_mat = np.array([r["vuln_mean_sae"] for r in ckpt_records], dtype=np.float32)
    safe_enc_mean_mat = np.array([r["safe_enc_mean"] for r in ckpt_records], dtype=np.float32)
    vuln_enc_mean_mat = np.array([r["vuln_enc_mean"] for r in ckpt_records], dtype=np.float32)

    # ── 5. Save tensors ───────────────────────────────────────────────────────
    print(f"\nSaving tensors to {out_dir} ...")
    torch.save(torch.tensor(safe_mean_sae_mat), out_dir / f"safe_mean_sae_layer_{SAE_LAYER}.pt")
    torch.save(torch.tensor(vuln_mean_sae_mat), out_dir / f"vulnerable_mean_sae_layer_{SAE_LAYER}.pt")
    torch.save(torch.tensor(safe_enc_mean_mat), out_dir / f"safe_enc_of_mean_layer_{SAE_LAYER}.pt")
    torch.save(torch.tensor(vuln_enc_mean_mat), out_dir / f"vulnerable_enc_of_mean_layer_{SAE_LAYER}.pt")
    print("  Done.")

    # ── 6. Run probes — new conditions ────────────────────────────────────────
    print("\nRunning probes — new conditions...")
    result_mean_sae     = probe_vuln_secure(safe_mean_sae_mat, vuln_mean_sae_mat)
    result_enc_of_mean  = probe_vuln_secure(safe_enc_mean_mat, vuln_enc_mean_mat)
    print(f"  Mean-token SAE  L{SAE_LAYER}: AUROC {result_mean_sae['roc_auc']:.3f} "
          f"[{result_mean_sae['ci_lo']:.3f}–{result_mean_sae['ci_hi']:.3f}]")
    print(f"  Encode-of-mean  L{SAE_LAYER}: AUROC {result_enc_of_mean['roc_auc']:.3f} "
          f"[{result_enc_of_mean['ci_lo']:.3f}–{result_enc_of_mean['ci_hi']:.3f}]")

    # ── 7. Load pre-computed baselines ────────────────────────────────────────
    print("\nLoading pre-computed baselines...")
    result_last_sae = load_last_token_sae()
    result_last_raw = load_last_token_raw()
    result_mean_raw = load_mean_token_raw()

    # ── 8. Print summary table ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"AUROC comparison — Layer {SAE_LAYER} — vuln vs. secure")
    print("=" * 72)
    rows = [
        ("Last-token  SAE",   result_last_sae),
        ("Last-token  raw",   result_last_raw),
        ("Mean-token  raw",   result_mean_raw),
        ("Encode-of-mean SAE",result_enc_of_mean),
        ("Mean-token  SAE",   result_mean_sae),
    ]
    for label, r in rows:
        if r is None:
            print(f"  {label:<22}  —")
        else:
            marker = " ← NEW" if "SAE" in label and "Mean" in label else ""
            print(f"  {label:<22}  {r['roc_auc']:.3f} [{r['ci_lo']:.3f}–{r['ci_hi']:.3f}]{marker}")
    print("=" * 72)

    # ── 9. Save results JSON ──────────────────────────────────────────────────
    results_payload = {
        "run_ts":        ts,
        "sae_layer":     SAE_LAYER,
        "sae_repo":      SAE_REPO,
        "last_sae":      result_last_sae,
        "last_raw":      result_last_raw,
        "mean_raw":      result_mean_raw,
        "mean_sae":      result_mean_sae,
        "enc_of_mean_sae": result_enc_of_mean,
    }
    results_json = out_dir / "probe_results.json"
    with results_json.open("w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Results saved: {results_json}")

    # ── 10. Save figure ───────────────────────────────────────────────────────
    all_results = {
        "last_sae":        result_last_sae,
        "last_raw":        result_last_raw,
        "mean_raw":        result_mean_raw,
        "mean_sae":        result_mean_sae,
        "enc_of_mean_sae": result_enc_of_mean,
    }
    fig_path = PAPER_FIGS / "fig_mean_pool_sae_comparison.pdf"
    fig_comparison(all_results, fig_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
