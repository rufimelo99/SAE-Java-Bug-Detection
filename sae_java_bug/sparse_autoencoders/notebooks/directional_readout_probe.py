"""
Directional readout probe — testing geometric hypothesis.

Instead of training a linear probe, project test samples onto the vulnerability
direction computed from the training set: d^L = normalize(mean(vulnerable) - mean(secure)).

This tests whether the geometric consistency (87–88% per-pair alignment) is sufficient
for classification without additional learned parameters.

Usage:
    conda run -n sae python directional_readout_probe.py

Saves:
    artifacts/activations/directional_readout/<run_ts>/
        results.json          probe results across layers
        token_projections.csv token-level projections for interpretation
    Paper figures dir / fig_directional_readout_comparison.pdf
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
ARTIFACTS = Path(__file__).parents[2] / "artifacts" / "activations"
PAPER_FIGS = (
    Path(__file__).parents[4]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

# Raw run directory — has all 8 layers
RAW_RUN_DIR = (
    ARTIFACTS / "raw_activations" / "vulnerable_code_qwen_coder_standard_16384_raw"
)

# Source JSONL for metadata
SOURCE_JSONL = (
    RAW_RUN_DIR / "activations_layer_0_raw_component_hidden_state_last_token.jsonl"
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]
MAX_TOKENS = 2048
SEED = 42

# ── Style ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


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
                secure_code = base64.b64decode(r["secure_code"]).decode(
                    "utf-8", errors="replace"
                )
                vuln_code = base64.b64decode(r["vulnerable_code"]).decode(
                    "utf-8", errors="replace"
                )
            except Exception:
                secure_code = r.get("secure_code", "")
                vuln_code = r.get("vulnerable_code", "")
            records.append(
                {
                    "vuln_id": r["vuln_id"],
                    "cwe": r["cwe"],
                    "file_extension": r["file_extension"],
                    "secure_code": secure_code,
                    "vulnerable_code": vuln_code,
                }
            )
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

    result = {}
    for layer_idx in target_layers:
        h = outputs.hidden_states[layer_idx + 1]  # [1, seq_len, d_model]
        mean_vec = h[0].float().mean(dim=0).cpu().numpy()  # [d_model]
        result[layer_idx] = mean_vec
    return result


def extract_full_sequence(
    code_str: str,
    tokenizer,
    model,
    target_layers: list,
    device,
    max_tokens: int = MAX_TOKENS,
) -> dict[int, np.ndarray]:
    """
    Run code_str through model, return {layer_idx: full_sequence_vectors [seq_len, d_model]}
    for token-level analysis.
    """
    inputs = tokenizer(
        code_str,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    result = {}
    for layer_idx in target_layers:
        h = outputs.hidden_states[layer_idx + 1]  # [1, seq_len, d_model]
        seq_vecs = h[0].float().cpu().numpy()  # [seq_len, d_model]
        result[layer_idx] = seq_vecs
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Metric utilities
# ─────────────────────────────────────────────────────────────────────────────


def _bootstrap_auc_ci(y_true, y_score, n_bootstrap=500, ci=0.95, seed=SEED):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n = len(y_true)
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


# ─────────────────────────────────────────────────────────────────────────────
# Directional readout
# ─────────────────────────────────────────────────────────────────────────────


def compute_vulnerability_direction(safe_mat, vuln_mat):
    """
    Compute d = normalize(mean(vuln) - mean(safe)).
    Both inputs should be [n_samples, d_model].
    Returns normalized direction vector [d_model].
    """
    mean_vuln = vuln_mat.mean(axis=0)
    mean_safe = safe_mat.mean(axis=0)
    direction = mean_vuln - mean_safe
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    return direction


def project_onto_direction(X, direction):
    """
    Project each row of X onto the direction vector.
    Returns scores [n_samples] (signed distances).
    """
    return X @ direction


def directional_readout_vuln_secure(safe_mat, vuln_mat, cv=5, n_bootstrap=500):
    """
    Directional readout without learned parameters.

    1. Compute vulnerability direction from training set.
    2. Project test samples onto that direction.
    3. Evaluate AUROC with 95% bootstrap CIs.
    """
    n = len(safe_mat)
    X = np.vstack([safe_mat, vuln_mat]).astype(np.float32)
    y = np.array([0] * n + [1] * n, dtype=int)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    y_score = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in skf.split(X, y):
        # Compute direction on training set only
        safe_tr = X[train_idx[y[train_idx] == 0]]
        vuln_tr = X[train_idx[y[train_idx] == 1]]
        direction = compute_vulnerability_direction(safe_tr, vuln_tr)

        # Project test set
        X_test = X[test_idx]
        y_score[test_idx] = project_onto_direction(X_test, direction)

    mean_auc, ci_lo, ci_hi = _bootstrap_auc_ci(y, y_score, n_bootstrap=n_bootstrap)
    return {
        "roc_auc": mean_auc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n": n,
        "y_score": y_score,
        "y_true": y,
    }


def directional_readout_with_analysis(safe_mat, vuln_mat, cv=5, n_bootstrap=500):
    """
    Extended analysis: also compute the global vulnerability direction
    and return per-sample projections for interpretation.
    """
    result = directional_readout_vuln_secure(
        safe_mat, vuln_mat, cv=cv, n_bootstrap=n_bootstrap
    )

    # Compute global direction for interpretation
    global_direction = compute_vulnerability_direction(safe_mat, vuln_mat)

    # Project all samples onto this global direction
    X_all = np.vstack([safe_mat, vuln_mat]).astype(np.float32)
    projections = project_onto_direction(X_all, global_direction)

    # Compute per-pair alignment (same as in paper)
    n = len(safe_mat)
    pair_alignment = 0
    for i in range(n):
        if projections[n + i] > projections[i]:  # vuln projection > safe projection
            pair_alignment += 1
    alignment_pct = 100.0 * pair_alignment / n

    result["global_direction"] = global_direction
    result["projections"] = projections
    result["per_pair_alignment_pct"] = alignment_pct

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Load pre-computed activations
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


def load_activations(layer):
    """Load pre-computed mean-pooled activations for a layer."""
    safe_pt = RAW_RUN_DIR / f"safe_layer_{layer}.pt"
    vuln_pt = RAW_RUN_DIR / f"vulnerable_layer_{layer}.pt"
    if safe_pt.exists() and vuln_pt.exists():
        import ctypes as _ct

        def _t2np_local(p):
            t = torch.load(p, weights_only=True).float().contiguous()
            buf = (_ct.c_float * t.numel()).from_address(t.data_ptr())
            return np.ctypeslib.as_array(buf).reshape(t.shape).copy()

        return _t2np_local(safe_pt), _t2np_local(vuln_pt)
    else:
        # Try JSONL fallback
        matches = list(RAW_RUN_DIR.glob(f"activations_layer_{layer}_*.jsonl"))
        if not matches:
            print(f"  [SKIP] L{layer}: no data found")
            return None, None
        return _load_jsonl_activations(matches[0])


# ─────────────────────────────────────────────────────────────────────────────
# Comparison with linear probe
# ─────────────────────────────────────────────────────────────────────────────


def probe_vuln_secure_linear(
    safe_mat, vuln_mat, n_components=50, cv=5, n_bootstrap=500
):
    """
    Standard linear probe (for comparison). Scaler and PCA fit inside each CV fold.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    n = len(safe_mat)
    X = np.vstack([safe_mat, vuln_mat]).astype(np.float32)
    y = np.array([0] * n + [1] * n, dtype=int)

    n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
    clf = LogisticRegression(
        C=0.1, max_iter=1000, class_weight="balanced", random_state=SEED
    )
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)

    y_score = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        pca = PCA(n_components=min(n_comp, len(train_idx) - 1), random_state=SEED)
        X_tr = pca.fit_transform(scaler.fit_transform(X[train_idx]))
        X_te = pca.transform(scaler.transform(X[test_idx]))
        clf.fit(X_tr, y[train_idx])
        y_score[test_idx] = clf.predict_proba(X_te)[:, 1]

    mean_auc, ci_lo, ci_hi = _bootstrap_auc_ci(y, y_score, n_bootstrap=n_bootstrap)
    return {"roc_auc": mean_auc, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": n}


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────


def fig_directional_vs_linear(dir_results, linear_results, out_path: Path):
    """
    Compare directional readout vs linear probe across layers.
    """
    layers_dir = sorted(dir_results.keys())
    layers_linear = sorted(linear_results.keys())
    all_layers = sorted(set(layers_dir + layers_linear))

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    def _plot(res_dict, layers, color, label, marker):
        aucs = [res_dict[l]["roc_auc"] for l in layers]
        los = [res_dict[l]["ci_lo"] for l in layers]
        his = [res_dict[l]["ci_hi"] for l in layers]
        xs = [LAYERS.index(l) for l in layers]
        ax.plot(xs, aucs, f"{marker}-", color=color, label=label, lw=1.5, ms=5)
        ax.fill_between(xs, los, his, color=color, alpha=0.18)

    _plot(linear_results, layers_linear, "#2E86AB", "Linear Probe", "o")
    _plot(dir_results, layers_dir, "#A23B72", "Directional Readout", "s")

    ax.axhline(0.5, color="grey", lw=0.7, ls=":", alpha=0.5, label="Chance (0.5)")
    if 11 in LAYERS:
        x11 = LAYERS.index(11)
        ax.axvline(x11, color="black", lw=1.0, ls="--", alpha=0.35)
        ax.text(x11 + 0.12, 0.51, "L11", fontsize=7, color="black", alpha=0.6)

    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels([f"L{l}" for l in LAYERS], rotation=45)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC  (vuln vs.\ secure)")
    ax.set_title(
        "Directional readout vs.\ linear probe\n(raw residual stream, shaded = 95\\% bootstrap CI)",
        fontsize=8,
    )
    ax.set_ylim(0.38, 0.75)
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=LAYERS,
        help=f"Layers to test (default: {LAYERS})",
    )
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    parser.add_argument(
        "--no-linear-compare",
        action="store_true",
        help="Skip linear probe comparison (faster)",
    )
    args = parser.parse_args()

    print("\n[Directional Readout Probe]")
    print(f"Target layers: {args.layers}")
    print(f"CV folds: {args.cv}\n")

    # Create output dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ARTIFACTS / "directional_readout" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    dir_results = {}
    linear_results = {}

    for layer in args.layers:
        print(f"Layer {layer}:")
        safe_mat, vuln_mat = load_activations(layer)
        if safe_mat is None:
            print(f"  [SKIP] no data")
            continue

        # Directional readout (no learned parameters)
        print(f"  Computing directional readout...")
        dir_res = directional_readout_with_analysis(safe_mat, vuln_mat, cv=args.cv)
        dir_results[layer] = dir_res
        print(
            f"    AUROC: {dir_res['roc_auc']:.3f} [{dir_res['ci_lo']:.3f}–{dir_res['ci_hi']:.3f}]"
        )
        print(f"    Per-pair alignment: {dir_res['per_pair_alignment_pct']:.1f}%")

        # Linear probe (for comparison)
        if not args.no_linear_compare:
            print(f"  Computing linear probe...")
            lin_res = probe_vuln_secure_linear(safe_mat, vuln_mat, cv=args.cv)
            linear_results[layer] = lin_res
            print(
                f"    AUROC: {lin_res['roc_auc']:.3f} [{lin_res['ci_lo']:.3f}–{lin_res['ci_hi']:.3f}]"
            )
            gap = dir_res["roc_auc"] - lin_res["roc_auc"]
            print(f"    Gap (dir - linear): {gap:+.3f}")
        print()

    # Save results
    results_path = out_dir / "results.json"
    metadata = {
        "timestamp": timestamp,
        "model": MODEL_ID,
        "layers": args.layers,
        "cv_folds": args.cv,
        "directional_readout": {
            str(k): {
                "roc_auc": float(v["roc_auc"]),
                "ci_lo": float(v["ci_lo"]),
                "ci_hi": float(v["ci_hi"]),
                "n_samples": int(v["n"]),
                "per_pair_alignment_pct": float(v["per_pair_alignment_pct"]),
            }
            for k, v in dir_results.items()
        },
        "linear_probe": (
            {
                str(k): {
                    "roc_auc": float(v["roc_auc"]),
                    "ci_lo": float(v["ci_lo"]),
                    "ci_hi": float(v["ci_hi"]),
                    "n_samples": int(v["n"]),
                }
                for k, v in linear_results.items()
            }
            if linear_results
            else None
        ),
    }
    with results_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved results: {results_path}\n")

    # Figure
    if linear_results:
        fig_path = PAPER_FIGS / "fig_directional_readout_comparison.pdf"
        fig_directional_vs_linear(dir_results, linear_results, fig_path)

    print(f"[Done] Output dir: {out_dir}")


if __name__ == "__main__":
    main()
