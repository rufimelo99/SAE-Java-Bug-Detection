"""
PCA sensitivity analysis — reviewer Q1 response.

Runs the vuln-vs-secure binary probe on mean-token raw residual activations
across all 8 layers, sweeping over PCA dimensionalities:
    no PCA (full 3584-dim), 10, 20, 50 (current paper setting), 100, 200

Uses pre-saved mean-pool .pt tensors; no model inference required.

Usage:
    conda run -n demeanor python pca_sensitivity_probe.py

Saves:
    artifacts/pca_sensitivity/pca_sensitivity_results.json
    <paper_figs>/fig_pca_sensitivity.pdf
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
ARTIFACTS  = Path(__file__).parents[2] / "artifacts" / "activations"
OUT_DIR    = Path(__file__).parents[2] / "artifacts" / "pca_sensitivity"
PAPER_FIGS = (
    Path(__file__).parents[4]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

LAYERS      = [0, 3, 7, 11, 15, 19, 23, 27]
PCA_DIMS    = [None, 10, 20, 50, 100, 200]   # None = no PCA (full dim)
SEED        = 42
N_BOOTSTRAP = 500

mpl.rcParams.update({
    "font.family":     "serif",
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi":      150,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})


# ── Utilities ──────────────────────────────────────────────────────────────────

def _bootstrap_auc_ci(y_true, y_score, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    from sklearn.metrics import roc_auc_score
    rng  = np.random.default_rng(seed)
    n    = len(y_true)
    aucs = [
        roc_auc_score(y_true[idx := rng.integers(0, n, n)], y_score[idx])
        for _ in range(n_bootstrap)
        if len(np.unique(y_true[rng.integers(0, n, n)])) > 1
    ]
    aucs = np.array(aucs)
    return float(aucs.mean()), float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))


def probe(safe_mat, vuln_mat, n_components, cv=5):
    n = len(safe_mat)
    X = np.vstack([safe_mat, vuln_mat]).astype(np.float32)
    y = np.array([0] * n + [1] * n, dtype=int)

    clf     = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=SEED)
    skf     = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    y_score = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y):
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X[tr])
        X_te_sc  = scaler.transform(X[te])
        if n_components is not None:
            nc      = min(n_components, X_tr_sc.shape[1], len(tr) - 1)
            pca     = PCA(n_components=nc, random_state=SEED)
            X_tr_in = pca.fit_transform(X_tr_sc)
            X_te_in = pca.transform(X_te_sc)
        else:
            X_tr_in = X_tr_sc
            X_te_in = X_te_sc
        clf.fit(X_tr_in, y[tr])
        y_score[te] = clf.predict_proba(X_te_in)[:, 1]

    mean_auc, ci_lo, ci_hi = _bootstrap_auc_ci(y, y_score)
    return {"roc_auc": mean_auc, "ci_lo": ci_lo, "ci_hi": ci_hi}


# ── Load activations ───────────────────────────────────────────────────────────

def _t2np(path):
    import ctypes
    t = torch.load(path, map_location="cpu", weights_only=True).float().contiguous()
    buf = (ctypes.c_float * t.numel()).from_address(t.data_ptr())
    return np.ctypeslib.as_array(buf).reshape(t.shape).copy()


def load_mean_pool(layer):
    run_root = ARTIFACTS / "mean_pool"
    for run_dir in sorted(run_root.iterdir(), reverse=True):
        sp = run_dir / f"safe_layer_{layer}.pt"
        vp = run_dir / f"vulnerable_layer_{layer}.pt"
        if sp.exists() and vp.exists():
            return _t2np(sp), _t2np(vp)
    raise FileNotFoundError(f"No mean_pool activations found for layer {layer}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    results = {}   # {layer: {n_components_label: {roc_auc, ci_lo, ci_hi}}}

    for layer in LAYERS:
        print(f"\nLayer {layer}")
        safe_mat, vuln_mat = load_mean_pool(layer)
        results[layer] = {}
        for nc in PCA_DIMS:
            label = "full" if nc is None else str(nc)
            r = probe(safe_mat, vuln_mat, nc)
            results[layer][label] = r
            marker = " ← paper" if nc == 50 else ""
            print(f"  PCA={label:>5}  AUROC {r['roc_auc']:.3f} [{r['ci_lo']:.3f}–{r['ci_hi']:.3f}]{marker}")

    # Save JSON
    out_json = OUT_DIR / "pca_sensitivity_results.json"
    with out_json.open("w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved: {out_json}")

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(9, 4.5), sharey=True)
    axes = axes.flatten()

    colours = plt.cm.viridis(np.linspace(0.1, 0.9, len(PCA_DIMS)))
    labels  = ["Full (3584)", "PCA-10", "PCA-20", "PCA-50 (paper)", "PCA-100", "PCA-200"]

    for ax, layer in zip(axes, LAYERS):
        for nc, colour, label in zip(PCA_DIMS, colours, labels):
            key = "full" if nc is None else str(nc)
            r   = results[layer][key]
            lw  = 2.0 if nc == 50 else 1.0
            ax.bar(
                labels.index(label), r["roc_auc"],
                color=colour, alpha=0.85, width=0.7,
                yerr=[[r["roc_auc"] - r["ci_lo"]], [r["ci_hi"] - r["roc_auc"]]],
                error_kw={"elinewidth": 1.0, "capsize": 2},
                linewidth=lw,
            )
        ax.axhline(0.5, color="grey", lw=0.8, ls=":", alpha=0.6)
        ax.set_title(f"L{layer}", fontsize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(["Full", "10", "20", "50*", "100", "200"], fontsize=7, rotation=45)
        ax.set_ylim(0.40, 0.72)

    axes[0].set_ylabel("AUROC  (vuln vs. secure)")
    axes[4].set_ylabel("AUROC  (vuln vs. secure)")
    fig.suptitle(
        "PCA sensitivity: mean-token pooling, raw residual stream\n"
        "(*50 = paper setting; error bars = 95% bootstrap CI)",
        fontsize=8,
    )
    fig.tight_layout()
    fig_path = PAPER_FIGS / "fig_pca_sensitivity.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {fig_path}")


if __name__ == "__main__":
    main()
