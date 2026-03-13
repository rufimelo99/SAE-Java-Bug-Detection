"""
Nonlinear probe comparison — reviewer response.

Tests whether the near-chance binary (vulnerable vs. secure) AUROC obtained with
a linear logistic-regression probe persists under nonlinear alternatives,
ruling out the possibility that the null result is an artefact of linear probing.

Three probes per layer and representation:
  - LogReg  : L2-regularised logistic regression (linear baseline)
  - MLP     : 2-layer MLP with early stopping  (nonlinear)
  - RBF-SVM : RBF-kernel SVM                  (nonlinear)

All probes operate on PCA(50)-reduced, z-scored last-token activations.
5-fold stratified cross-validation; 500-resample bootstrap 95% CIs.

Outputs:
  fig_nonlinear_probes.pdf  — AUROC comparison across layers (paper figures dir)
  printed numerical table

Run:
  conda run -n sae python nonlinear_probe.py
"""

import json
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
ARTIFACTS  = Path(__file__).parents[2] / "artifacts" / "activations"
PAPER_FIGS = (
    Path(__file__).parents[4]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

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

LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]

RUNS_REGISTRY = [
    {
        "run_dir": ARTIFACTS / "raw_activations/vulnerable_code_qwen_coder_standard_16384_raw",
        "label":   "Raw residual",
    },
    {
        "run_dir": ARTIFACTS / "TO_UPLOAD",
        "label":   "SAE features (STD)",
    },
    {
        "run_dir": ARTIFACTS / "TOPK",
        "label":   "SAE features (TopK)",
    },
]

PROBE_STYLES = {
    "LogReg":  {"color": "#333333", "marker": "o", "ls": "-",  "lw": 1.8},
    "MLP":     {"color": "#d62728", "marker": "s", "ls": "--", "lw": 1.6},
    "RBF-SVM": {"color": "#1f77b4", "marker": "^", "ls": ":",  "lw": 1.6},
}

N_PCA        = 50
N_CV_FOLDS   = 5
N_BOOTSTRAP  = 500
CI_LEVEL     = 0.95


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (same logic as within_language_baseline.py)
# ─────────────────────────────────────────────────────────────────────────────

def _discover_layers(run_dir):
    layers = set()
    for f in run_dir.glob("activations_layer_*_*.jsonl"):
        try:
            layers.add(int(f.name.split("_")[2]))
        except (IndexError, ValueError):
            pass
    return sorted(layers)


def _load_layer(run_dir, layer):
    """Return (safe_mat, vuln_mat) as float32 numpy arrays, or None."""
    safe_pt = run_dir / f"safe_layer_{layer}.pt"
    vuln_pt = run_dir / f"vulnerable_layer_{layer}.pt"
    if safe_pt.exists() and vuln_pt.exists():
        safe = torch.load(safe_pt, weights_only=True).numpy().astype(np.float32)
        vuln = torch.load(vuln_pt, weights_only=True).numpy().astype(np.float32)
        return safe, vuln
    matches = list(run_dir.glob(f"activations_layer_{layer}_*.jsonl"))
    if not matches:
        return None
    secure_rows, vuln_rows = [], []
    with matches[0].open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            secure_rows.append(r["secure"])
            vuln_rows.append(r["vulnerable"])
    return np.array(secure_rows, dtype=np.float32), np.array(vuln_rows, dtype=np.float32)


def load_data():
    """Return {label -> {layer -> (safe_mat, vuln_mat)}}."""
    data = {}
    for entry in RUNS_REGISTRY:
        run_dir = entry["run_dir"]
        label   = entry["label"]
        if not run_dir.exists():
            print(f"[SKIP] {label}: {run_dir} not found")
            continue
        available = _discover_layers(run_dir)
        layers    = [l for l in LAYERS if l in available]
        if not layers:
            print(f"[SKIP] {label}: no layers found in {run_dir}")
            continue
        data[label] = {}
        for layer in tqdm(layers, desc=f"Loading {label}", leave=False):
            result = _load_layer(run_dir, layer)
            if result is None:
                continue
            data[label][layer] = result
        print(f"Loaded {label}: {list(data[label].keys())}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Probe helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_auc_ci(y_true, y_score, n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL):
    from sklearn.metrics import roc_auc_score
    rng  = np.random.default_rng(SEED)
    n    = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    aucs  = np.array(aucs)
    alpha = (1 - ci) / 2
    return aucs.mean(), np.quantile(aucs, alpha), np.quantile(aucs, 1 - alpha)


def _make_probes():
    return {
        "LogReg": LogisticRegression(
            C=0.1, max_iter=1000, class_weight="balanced", random_state=SEED
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            alpha=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=SEED,
            verbose=False,
        ),
        "RBF-SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=SEED,
            max_iter=2000,
        ),
    }


def run_probe_cv(safe_mat, vuln_mat, probe_name, probe):
    """5-fold CV on stacked [safe; vuln] with PCA(50). Returns (mean, lo, hi).
    Scaler and PCA are fit inside each fold to avoid data leakage."""
    n = len(safe_mat)
    X = np.vstack([safe_mat, vuln_mat])
    y = np.array([0] * n + [1] * n, dtype=int)

    n_comp  = min(N_PCA, X.shape[1], X.shape[0] - 1)
    skf     = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
    y_score = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        pca    = PCA(n_components=n_comp, random_state=SEED)
        X_tr   = pca.fit_transform(scaler.fit_transform(X[train_idx]))
        X_te   = pca.transform(scaler.transform(X[test_idx]))
        p = _make_probes()[probe_name]   # fresh instance per fold
        p.fit(X_tr, y[train_idx])
        y_score[test_idx] = p.predict_proba(X_te)[:, 1]

    return _bootstrap_auc_ci(y, y_score)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_all(data):
    """
    Returns results[label][probe_name] = {layer: (mean, lo, hi)}.
    """
    results = {}
    for label, layer_dict in data.items():
        results[label] = {p: {} for p in PROBE_STYLES}
        for layer in tqdm(sorted(layer_dict.keys()), desc=label):
            safe_mat, vuln_mat = layer_dict[layer]
            for probe_name in PROBE_STYLES:
                mean, lo, hi = run_probe_cv(safe_mat, vuln_mat, probe_name, probe_name)
                results[label][probe_name][layer] = (mean, lo, hi)
                print(
                    f"  {label}  L{layer:2d}  {probe_name:8s}  "
                    f"AUROC={mean:.3f} [{lo:.3f}–{hi:.3f}]"
                )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(results):
    n_panels = len(results)
    if n_panels == 0:
        print("No results to plot.")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 3.2), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (label, probe_results) in zip(axes, results.items()):
        all_layers = sorted({
            l for pd_dict in probe_results.values() for l in pd_dict
        })
        xs          = list(range(len(all_layers)))
        layer_lbls  = [f"L{l}" for l in all_layers]

        for probe_name, style in PROBE_STYLES.items():
            pd_dict = probe_results[probe_name]
            aucs = [pd_dict[l][0] if l in pd_dict else np.nan for l in all_layers]
            los  = [pd_dict[l][1] if l in pd_dict else np.nan for l in all_layers]
            his  = [pd_dict[l][2] if l in pd_dict else np.nan for l in all_layers]
            ax.plot(
                xs, aucs,
                style["marker"] + style["ls"],
                color=style["color"],
                lw=style["lw"],
                ms=5,
                label=probe_name,
            )
            ax.fill_between(xs, los, his, color=style["color"], alpha=0.12)

        ax.axhline(0.5, color="grey", lw=0.7, ls=":", alpha=0.5, label="Chance")

        if 11 in all_layers:
            x11 = all_layers.index(11)
            ax.axvline(x11, color="black", lw=1.0, ls="--", alpha=0.35)
            ax.text(x11 + 0.1, 0.505, "L11", fontsize=7, color="black", alpha=0.55)

        ax.set_xticks(xs)
        ax.set_xticklabels(layer_lbls, rotation=45)
        ax.set_xlabel("Layer")
        ax.set_ylabel("ROC-AUC  (vuln vs. secure)")
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(0.40, 0.70)
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Nonlinear probe comparison: LogReg vs MLP vs RBF-SVM\n"
        "(last-token pooled; PCA-50; shaded = 95% bootstrap CI)",
        fontsize=9,
    )
    fig.tight_layout()
    out = PAPER_FIGS / "fig_nonlinear_probes.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results):
    rows = []
    for label, probe_results in results.items():
        all_layers = sorted({l for pd in probe_results.values() for l in pd})
        for layer in all_layers:
            row = {"Representation": label, "Layer": layer}
            for probe_name in PROBE_STYLES:
                entry = probe_results[probe_name].get(layer)
                if entry:
                    row[probe_name] = f"{entry[0]:.3f} [{entry[1]:.3f}–{entry[2]:.3f}]"
                else:
                    row[probe_name] = "—"
            rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("Nonlinear probe summary  (AUROC, 95% CI)")
    print("=" * 90)
    print(df.to_string(index=False))
    print()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Paper figures dir : {PAPER_FIGS}")
    print(f"Activations dir   : {ARTIFACTS}")
    print()

    data    = load_data()
    results = run_all(data)
    make_figure(results)
    print_table(results)
    print("Done.")
