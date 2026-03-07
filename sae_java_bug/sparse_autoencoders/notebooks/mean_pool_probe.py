"""
Mean-token pooling probe — reviewer Q1/Q4 response.

Extracts residual-stream activations using MEAN pooling over all token positions
(instead of last-token pooling) at layers 0, 3, 7, 11, 15, 19, 23, 27.
Runs the same vulnerable-vs-secure logistic-regression probe and reports AUROC
with 95% bootstrap CIs, enabling a direct comparison with last-token results.

Usage:
    conda run -n sae python mean_pool_probe.py

Saves:
    artifacts/activations/mean_pool/<run_ts>/
        safe_layer_{L}.pt           shape [N, 3584]
        vulnerable_layer_{L}.pt     shape [N, 3584]
        meta.json
    Paper figures dir / fig_mean_vs_last_token_pool.pdf
"""

import base64
import json
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
ARTIFACTS  = Path(__file__).parents[2] / "artifacts" / "activations"
PAPER_FIGS = (
    Path(__file__).parents[4]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

# Raw run directory — has all 8 layers as JSONL (no .pt files)
RAW_RUN_DIR = (
    ARTIFACTS / "raw_activations" / "vulnerable_code_qwen_coder_standard_16384_raw"
)
# Fallback to the other run dir (has .pt files, but only layers 0/14/27)
RAW_RUN_DIR_FALLBACK = (
    ARTIFACTS / "run_20260216_231950_cl_secure_code_qwen_coder_topk_16384_raw"
)

# Source JSONL: use layer-0 file from the run with all 8 layers
SOURCE_JSONL = RAW_RUN_DIR / "activations_layer_0_raw_component_hidden_state_last_token.jsonl"
if not SOURCE_JSONL.exists():
    SOURCE_JSONL = RAW_RUN_DIR_FALLBACK / "activations_layer_0_raw_component_hidden_state_last_token.jsonl"

MODEL_ID   = "Qwen/Qwen2.5-7B-Instruct"
LAYERS     = [0, 3, 7, 11, 15, 19, 23, 27]   # matches existing analysis
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
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_source_records(jsonl_path: Path):
    """Load (vuln_id, secure_code, vulnerable_code, cwe, file_extension) from JSONL."""
    records = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            try:
                secure_code   = base64.b64decode(r["secure_code"]).decode("utf-8", errors="replace")
                vuln_code     = base64.b64decode(r["vulnerable_code"]).decode("utf-8", errors="replace")
            except Exception:
                secure_code   = r.get("secure_code", "")
                vuln_code     = r.get("vulnerable_code", "")
            records.append({
                "vuln_id":        r["vuln_id"],
                "cwe":            r["cwe"],
                "file_extension": r["file_extension"],
                "secure_code":    secure_code,
                "vulnerable_code": vuln_code,
            })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Activation extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_mean_pool(
    code_str: str,
    tokenizer,
    model,
    target_layers: list,
    device,
    max_tokens: int = MAX_TOKENS,
) -> dict[int, np.ndarray]:
    """
    Run code_str through model, return {layer_idx: mean_pooled_vector [d_model]}
    using the mean over all non-padding token positions.
    """
    inputs = tokenizer(
        code_str,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states: tuple of (n_layers+1) tensors, each [1, seq_len, d_model]
    # index 0 = embedding output; index i = output of layer i-1
    result = {}
    for layer_idx in target_layers:
        # layer_idx+1 because hidden_states[0] is embeddings
        h = outputs.hidden_states[layer_idx + 1]   # [1, seq_len, d_model]
        mean_vec = h[0].float().mean(dim=0).cpu().numpy()  # [d_model]
        result[layer_idx] = mean_vec
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Probe utilities (same setup as within_language_baseline.py)
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
    """
    Binary probe: vulnerable (1) vs secure (0) over all paired samples.
    Returns dict with roc_auc, ci_lo, ci_hi, n.
    """
    n = len(safe_mat)
    X = np.vstack([safe_mat, vuln_mat]).astype(np.float32)
    y = np.array([0] * n + [1] * n, dtype=int)

    scaler = StandardScaler()
    pca    = PCA(n_components=min(n_components, X.shape[1], X.shape[0] - 1), random_state=SEED)
    X_pca  = pca.fit_transform(scaler.fit_transform(X))

    clf    = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=SEED)
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)

    y_score = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in skf.split(X_pca, y):
        clf.fit(X_pca[train_idx], y[train_idx])
        y_score[test_idx] = clf.predict_proba(X_pca[test_idx])[:, 1]

    mean_auc, ci_lo, ci_hi = _bootstrap_auc_ci(y, y_score, n_bootstrap=n_bootstrap)
    return {"roc_auc": mean_auc, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": n}


# ─────────────────────────────────────────────────────────────────────────────
# Load last-token results for comparison
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl_activations(jsonl_path: Path):
    safe_rows, vuln_rows = [], []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            safe_rows.append(r["secure"])
            vuln_rows.append(r["vulnerable"])
    return (
        np.array(safe_rows, dtype=np.float32),
        np.array(vuln_rows, dtype=np.float32),
    )


def load_last_token_results():
    """
    Load pre-computed last-token activations and run the same probe for comparison.
    Prefers .pt tensors; falls back to JSONL. Uses RAW_RUN_DIR (all 8 layers).
    """
    results = {}
    for layer in LAYERS:
        # Try .pt first (faster), fall back to JSONL
        safe_pt = RAW_RUN_DIR / f"safe_layer_{layer}.pt"
        vuln_pt = RAW_RUN_DIR / f"vulnerable_layer_{layer}.pt"
        if safe_pt.exists() and vuln_pt.exists():
            safe_mat = torch.load(safe_pt, weights_only=True).numpy().astype(np.float32)
            vuln_mat = torch.load(vuln_pt, weights_only=True).numpy().astype(np.float32)
        else:
            matches = list(RAW_RUN_DIR.glob(f"activations_layer_{layer}_*.jsonl"))
            if not matches:
                # try fallback dir
                matches = list(RAW_RUN_DIR_FALLBACK.glob(f"activations_layer_{layer}_*.jsonl"))
            if not matches:
                print(f"  [SKIP] last-token L{layer}: no data found")
                continue
            safe_mat, vuln_mat = _load_jsonl_activations(matches[0])

        results[layer] = probe_vuln_secure(safe_mat, vuln_mat)
        print(f"  Last-token L{layer}: AUROC {results[layer]['roc_auc']:.3f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def fig_mean_vs_last(last_results, mean_results, out_path: Path):
    """
    Single panel: AUROC vs layer for last-token and mean-token pooling.
    Both methods on the same axes with shaded 95% CI bands.
    """
    layers_last = sorted(last_results.keys())
    layers_mean = sorted(mean_results.keys())

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    def _plot(res_dict, layers, color, label, marker):
        aucs = [res_dict[l]["roc_auc"] for l in layers]
        los  = [res_dict[l]["ci_lo"]   for l in layers]
        his  = [res_dict[l]["ci_hi"]   for l in layers]
        xs   = [LAYERS.index(l) for l in layers]
        ax.plot(xs, aucs, f"{marker}-", color=color, label=label, lw=1.5, ms=5)
        ax.fill_between(xs, los, his, color=color, alpha=0.18)

    _plot(last_results, layers_last, "#333333", "Last-token",  "o")
    _plot(mean_results, layers_mean, "#d65f5f", "Mean-token",  "s")

    ax.axhline(0.5, color="grey", lw=0.7, ls=":", alpha=0.5, label="Chance (0.5)")
    if 11 in LAYERS:
        x11 = LAYERS.index(11)
        ax.axvline(x11, color="black", lw=1.0, ls="--", alpha=0.35)
        ax.text(x11 + 0.12, 0.51, "L11", fontsize=7, color="black", alpha=0.6)

    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels([f"L{l}" for l in LAYERS], rotation=45)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC  (vuln vs.\ secure)")
    ax.set_title("Last-token vs.\ mean-token pooling\n(raw residual stream, shaded = 95\\% bootstrap CI)",
                 fontsize=8)
    ax.set_ylim(0.38, 0.65)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Source JSONL: {SOURCE_JSONL}")

    # ── 1. Load source code ───────────────────────────────────────────────────
    print("Loading source records...")
    records = load_source_records(SOURCE_JSONL)
    print(f"  {len(records)} samples loaded")

    # ── 2. Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(device)
    model.eval()
    print("  Model ready.")

    # ── 3. Extract mean-pooled activations ───────────────────────────────────
    ts      = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ARTIFACTS / "mean_pool" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accumulators: {layer_idx: [secure_vecs], [vuln_vecs]}
    safe_bufs = {l: [] for l in LAYERS}
    vuln_bufs = {l: [] for l in LAYERS}
    meta_rows = []

    print(f"\nExtracting activations ({len(records)} samples × 2 sides × {len(LAYERS)} layers)...")
    t0 = time.time()
    for i, rec in enumerate(records):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (len(records) - i - 1)
            print(f"  [{i+1:>4}/{len(records)}]  elapsed {elapsed:.0f}s  ETA {eta:.0f}s")

        try:
            s_acts = extract_mean_pool(rec["secure_code"],   tokenizer, model, LAYERS, device)
            v_acts = extract_mean_pool(rec["vulnerable_code"], tokenizer, model, LAYERS, device)
        except Exception as e:
            print(f"  [WARN] sample {i} ({rec['vuln_id']}): {e}")
            for l in LAYERS:
                safe_bufs[l].append(np.zeros(model.config.hidden_size, dtype=np.float32))
                vuln_bufs[l].append(np.zeros(model.config.hidden_size, dtype=np.float32))
            meta_rows.append({k: rec[k] for k in ("vuln_id", "cwe", "file_extension")})
            continue

        for l in LAYERS:
            safe_bufs[l].append(s_acts[l])
            vuln_bufs[l].append(v_acts[l])
        meta_rows.append({k: rec[k] for k in ("vuln_id", "cwe", "file_extension")})

    # ── 4. Save tensors ───────────────────────────────────────────────────────
    print(f"\nSaving tensors to {out_dir} ...")
    for l in LAYERS:
        torch.save(torch.tensor(np.array(safe_bufs[l]), dtype=torch.float32),
                   out_dir / f"safe_layer_{l}.pt")
        torch.save(torch.tensor(np.array(vuln_bufs[l]), dtype=torch.float32),
                   out_dir / f"vulnerable_layer_{l}.pt")

    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta_rows, f)
    print("  Done.")

    # ── 5. Run probes ─────────────────────────────────────────────────────────
    print("\nRunning probes (mean-token) ...")
    mean_results = {}
    for l in LAYERS:
        safe_mat = np.array(safe_bufs[l], dtype=np.float32)
        vuln_mat = np.array(vuln_bufs[l], dtype=np.float32)
        mean_results[l] = probe_vuln_secure(safe_mat, vuln_mat)

    print("\nRunning probes (last-token, from saved .pt) ...")
    last_results = load_last_token_results()

    # ── 6. Print comparison table ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Vuln vs. Secure AUROC: last-token vs. mean-token pooling (raw residual)")
    print("=" * 70)
    print(f"{'Layer':>6}  {'Last-token':^28}  {'Mean-token':^28}")
    print(f"{'':>6}  {'AUROC [95% CI]':^28}  {'AUROC [95% CI]':^28}")
    print("-" * 70)
    for l in LAYERS:
        def _fmt(d):
            if d is None:
                return "        —        "
            return f"{d['roc_auc']:.3f} [{d['ci_lo']:.3f}–{d['ci_hi']:.3f}]"
        print(f"  L{l:>2}   {_fmt(last_results.get(l)):^28}  {_fmt(mean_results.get(l)):^28}")
    print("=" * 70)

    # ── 7. Save figure ────────────────────────────────────────────────────────
    fig_path = PAPER_FIGS / "fig_mean_vs_last_token_pool.pdf"
    fig_mean_vs_last(last_results, mean_results, fig_path)

    # ── 8. Save probe results as JSON ─────────────────────────────────────────
    results_payload = {
        "run_ts": ts,
        "layers": LAYERS,
        "last_token": {str(l): last_results[l] for l in LAYERS if l in last_results},
        "mean_token": {str(l): mean_results[l] for l in LAYERS if l in mean_results},
    }
    results_json = out_dir / "probe_results.json"
    with results_json.open("w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Results saved: {results_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
