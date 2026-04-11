"""
Pairwise CWE-type heatmaps for SVEN and PreciseBugs.

Mirrors the DeltaSecommits heatmaps (fig_cwe_pairwise_c.pdf) but reads
SAE feature JSONL activations instead of raw-residual .pt files.

For each dataset:
  - Pools secure + vulnerable activations per CWE
  - Runs all pairwise binary probes (PCA-50 + LogReg, 5-fold CV)
  - Plots peak AUROC (max over 8 layers) as a heatmap

Usage:
    python plot_multi_dataset_cwe_heatmap.py
    python plot_multi_dataset_cwe_heatmap.py --datasets SVEN
    python plot_multi_dataset_cwe_heatmap.py --min_n 15
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
GITHUB     = Path(__file__).parents[4]
SAE_REPO   = Path(__file__).parents[3]
ACTS_DIR   = SAE_REPO / "sae_java_bug" / "artifacts" / "activations"
FIGS_DIR   = SAE_REPO / "sae_java_bug" / "artifacts" / "figures"
PAPER_FIGS = GITHUB / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations" / "figures"

DATASETS: dict[str, str] = {
    "SVEN":        "sven_c_only",
    "PreciseBugs": "precisebugs_c_only",
}

LAYERS  = [0, 3, 7, 11, 15, 19, 23, 27]
SEED    = 42
MIN_N   = 15   # min pairs per CWE to include

mpl.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ── Data loading ───────────────────────────────────────────────────────────────

def load_layer(acts_subdir: str, layer: int) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Load (secure, vuln, cwes) from JSONL files for a given layer."""
    base = ACTS_DIR / acts_subdir
    secure_rows, vuln_rows, cwes = [], [], []
    found = False
    for split in ("train", "test"):
        path = base / f"layer_{layer}_{split}.jsonl"
        if not path.exists():
            continue
        found = True
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                cwe = r.get("cwe", "unknown")
                if cwe == "unknown":
                    continue
                secure_rows.append(r["secure"])
                vuln_rows.append(r["vulnerable"])
                cwes.append(cwe)
    if not found or not secure_rows:
        return None
    return (
        np.array(secure_rows, dtype=np.float32),
        np.array(vuln_rows,   dtype=np.float32),
        cwes,
    )


# ── Probe ──────────────────────────────────────────────────────────────────────

def _bootstrap_auc(y_true, y_score, n=500):
    rng  = np.random.default_rng(SEED)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    a = np.array(aucs)
    return float(a.mean()), float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))


def probe_pair(X_a: np.ndarray, X_b: np.ndarray, n_components: int = 50) -> float | None:
    """Binary probe distinguishing class A from class B. Returns AUROC."""
    X = np.vstack([X_a, X_b])
    y = np.array([0] * len(X_a) + [1] * len(X_b), dtype=int)

    n_splits = min(5, int(y.sum()), int((~y.astype(bool)).sum()))
    if n_splits < 2:
        return None

    skf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    y_score = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X[train_idx])
        X_te   = scaler.transform(X[test_idx])

        n_comp = min(n_components, X_tr.shape[1], len(train_idx) - 1)
        pca    = PCA(n_components=n_comp, random_state=SEED)
        X_tr   = pca.fit_transform(X_tr)
        X_te   = pca.transform(X_te)

        clf = LogisticRegression(C=0.1, max_iter=1000,
                                 class_weight="balanced", random_state=SEED)
        clf.fit(X_tr, y[train_idx])
        y_score[test_idx] = clf.predict_proba(X_te)[:, 1]

    auc, _, _ = _bootstrap_auc(y, y_score)
    return auc


# ── Main per-dataset logic ─────────────────────────────────────────────────────

def run_dataset(name: str, subdir: str, min_n: int, n_components: int = 50):
    """Run pairwise CWE probes across all layers, return {pair: {layer: auc}}."""
    print(f"\n{'='*55}")
    print(f"  {name}  ({subdir})")
    print(f"{'='*55}")

    # Discover eligible CWEs from L11 (representative layer)
    ref = load_layer(subdir, 11)
    if ref is None:
        print("  [SKIP] no data at L11")
        return None, []

    secure_ref, vuln_ref, cwes_ref = ref
    from collections import Counter
    counts = Counter(cwes_ref)
    eligible = sorted(c for c, n in counts.items() if n >= min_n)
    print(f"  Eligible CWEs (n≥{min_n}): {eligible}")
    if len(eligible) < 2:
        print("  [SKIP] fewer than 2 eligible CWEs")
        return None, []

    pairs = list(combinations(eligible, 2))
    print(f"  Pairs: {len(pairs)}")

    # Results: pair_key -> {layer: auroc}
    pair_results: dict[str, dict[int, float]] = {
        f"{a}_vs_{b}": {} for a, b in pairs
    }

    for layer in LAYERS:
        print(f"  L{layer:>2} loading...", flush=True)
        data = load_layer(subdir, layer)
        if data is None:
            print(f"  L{layer:>2} [SKIP]")
            continue

        secure, vuln, cwes = data
        cwe_arr = np.array(cwes)

        # Pool secure + vulnerable per CWE
        cwe_to_X: dict[str, np.ndarray] = {}
        for cwe in eligible:
            mask = cwe_arr == cwe
            if mask.sum() < min_n:
                continue
            cwe_to_X[cwe] = np.vstack([secure[mask], vuln[mask]])

        for cwe_a, cwe_b in pairs:
            if cwe_a not in cwe_to_X or cwe_b not in cwe_to_X:
                continue
            auc = probe_pair(cwe_to_X[cwe_a], cwe_to_X[cwe_b], n_components)
            if auc is not None:
                pair_results[f"{cwe_a}_vs_{cwe_b}"][layer] = round(auc, 4)

        done = sum(1 for v in pair_results.values() if layer in v)
        print(f"  L{layer:>2} done ({done} pairs computed)", flush=True)

    return pair_results, eligible


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_heatmap(pair_results: dict, cwes: list, name: str, out_path: Path):
    """Heatmap of peak AUROC (max over layers) for each CWE pair."""
    n   = len(cwes)
    mat = np.full((n, n), np.nan)

    for key, layer_aucs in pair_results.items():
        if not layer_aucs:
            continue
        cwe_a, cwe_b = key.split("_vs_")
        if cwe_a not in cwes or cwe_b not in cwes:
            continue
        peak = max(layer_aucs.values())
        i, j = cwes.index(cwe_a), cwes.index(cwe_b)
        mat[i, j] = peak
        mat[j, i] = peak

    df = pd.DataFrame(mat, index=cwes, columns=cwes)

    fig, ax = plt.subplots(figsize=(max(4.5, n * 0.65), max(3.5, n * 0.60)))

    sns.heatmap(
        df, ax=ax,
        mask=np.isnan(mat),
        vmin=0.4, vmax=1.0,
        cmap="RdYlGn",
        annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.4,
        square=True,
    )
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True,
                                   color="#cccccc", lw=0))

    ax.set_title(
        f"Pairwise CWE probe — {name} — peak AUROC across layers\n"
        f"(SAE feature mean-pool; green = separable, yellow = chance)",
        fontsize=8,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_layer_profiles(pair_results: dict, name: str, out_path: Path):
    """Line plot: one line per CWE pair, AUROC across layers."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    cmap = plt.get_cmap("tab20")

    for idx, (key, layer_aucs) in enumerate(pair_results.items()):
        if not layer_aucs:
            continue
        xs   = [LAYERS.index(l) for l in sorted(layer_aucs)]
        aucs = [layer_aucs[l] for l in sorted(layer_aucs)]
        cwe_a, cwe_b = key.split("_vs_")
        ax.plot(xs, aucs, "-", color=cmap(idx % 20), lw=1.0,
                alpha=0.75, label=f"{cwe_a} vs {cwe_b}")

    ax.axhline(0.5, color="grey", lw=0.7, ls=":", alpha=0.5)
    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels([f"L{l}" for l in LAYERS], rotation=45)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(
        f"Pairwise CWE-type probe — {name} — all pairs\n"
        f"(SAE feature mean-pool)",
        fontsize=8,
    )
    n_pairs = sum(1 for v in pair_results.values() if v)
    ncol = max(1, n_pairs // 12)
    ax.legend(fontsize=5, loc="upper left", ncol=ncol, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--min_n", type=int, default=MIN_N)
    parser.add_argument("--n_components", type=int, default=50)
    args = parser.parse_args()

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        if name not in DATASETS:
            print(f"[WARN] Unknown dataset '{name}'. Available: {list(DATASETS)}")
            continue

        pair_results, cwes = run_dataset(
            name, DATASETS[name], args.min_n, args.n_components
        )
        if pair_results is None:
            continue

        slug = name.lower().replace(" ", "_")
        for dest in (FIGS_DIR, PAPER_FIGS):
            plot_heatmap(pair_results, cwes, name,
                         dest / f"fig_cwe_pairwise_{slug}.pdf")
            plot_layer_profiles(pair_results, name,
                                dest / f"fig_cwe_pairwise_{slug}_layers.pdf")

        # Summary
        peak_aucs = [max(v.values()) for v in pair_results.values() if v]
        if peak_aucs:
            print(f"\n  {name} — {len(cwes)} CWEs, {len(peak_aucs)} pairs")
            print(f"  Median peak AUROC: {np.median(peak_aucs):.3f}")
            print(f"  Min / Max:         {min(peak_aucs):.3f} / {max(peak_aucs):.3f}")
            print(f"  Pairs above 0.80:  {sum(a > 0.80 for a in peak_aucs)} / {len(peak_aucs)}")


if __name__ == "__main__":
    main()
