"""
Pairwise CWE-type probe within each language in the dataset.

For every language with >=2 CWEs above MIN_N, runs all pairwise binary probes
(PCA-50 + logistic regression, 5-fold CV) on vulnerable-side mean-token
activations across all 8 layers and plots a heatmap of peak AUROC.

Usage:
    conda run -n sae python plot_within_language_cwe_pairwise.py
    conda run -n sae python plot_within_language_cwe_pairwise.py --run_dir PATH
    conda run -n sae python plot_within_language_cwe_pairwise.py --min_n 10
"""

import argparse
import ctypes
import json
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
ARTIFACTS = Path(__file__).parents[2] / "artifacts" / "activations"
PAPER_FIGS = (
    Path(__file__).parents[4]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]
SEED = 42
MIN_N = 10  # min samples per CWE per language to be included

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


# ── Tensor loading ─────────────────────────────────────────────────────────────
def _t2np(t):
    t = t.float().contiguous()
    buf = (ctypes.c_float * t.numel()).from_address(t.data_ptr())
    return np.ctypeslib.as_array(buf).reshape(t.shape).copy()


def find_latest_run() -> Path:
    runs = sorted((ARTIFACTS / "mean_pool").glob("*/meta.json"))
    if not runs:
        raise FileNotFoundError(f"No mean_pool runs under {ARTIFACTS}/mean_pool/")
    return runs[-1].parent


def load_layer(run_dir: Path, layer: int):
    vuln = _t2np(
        torch.load(run_dir / f"vulnerable_layer_{layer}.pt", weights_only=True)
    )
    return vuln


# ── Bootstrap CI ───────────────────────────────────────────────────────────────
def _bootstrap_auc_ci(y_true, y_score, n=500, ci=0.95, seed=SEED):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    aucs = np.array(aucs)
    alpha = (1 - ci) / 2
    return (
        float(np.mean(aucs)),
        float(np.quantile(aucs, alpha)),
        float(np.quantile(aucs, 1 - alpha)),
    )


# ── Probe ──────────────────────────────────────────────────────────────────────
def probe(X, y, n_components=50, cv=5):
    n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
    clf = LogisticRegression(
        C=0.1, max_iter=1000, class_weight="balanced", random_state=SEED
    )
    n_splits = min(cv, int(y.sum()), int((y == 0).sum()))
    if n_splits < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    y_score = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        pca = PCA(n_components=min(n_comp, len(tr) - 1), random_state=SEED)
        Xtr = pca.fit_transform(scaler.fit_transform(X[tr]))
        Xte = pca.transform(scaler.transform(X[te]))
        clf.fit(Xtr, y[tr])
        y_score[te] = clf.predict_proba(Xte)[:, 1]
    mean_auc, ci_lo, ci_hi = _bootstrap_auc_ci(y, y_score)
    return {"roc_auc": mean_auc, "ci_lo": ci_lo, "ci_hi": ci_hi}


# ── Per-language pairwise probing ──────────────────────────────────────────────
def run_language(run_dir: Path, meta: list, lang: str, min_n: int):
    from collections import Counter

    counts = Counter(r["cwe"] for r in meta if r["file_extension"] == lang)
    cwes = sorted(c for c, n in counts.items() if n >= min_n)
    print(f"\n{lang.upper()}: {len(cwes)} CWEs with n≥{min_n}: {cwes}")

    # Pre-load layer activations
    vuln_by_layer = {}
    for layer in LAYERS:
        pt = run_dir / f"vulnerable_layer_{layer}.pt"
        if pt.exists():
            vuln_by_layer[layer] = load_layer(run_dir, layer)

    # Build index masks for each CWE
    masks = {}
    for cwe in cwes:
        masks[cwe] = np.array(
            [r["file_extension"] == lang and r["cwe"] == cwe for r in meta]
        )

    results = {}
    pairs = list(combinations(cwes, 2))
    total = len(pairs) * len(vuln_by_layer)
    done = 0

    for cwe_a, cwe_b in pairs:
        key = f"{cwe_a}_vs_{cwe_b}"
        results[key] = {
            "cwe_a": cwe_a,
            "cwe_b": cwe_b,
            "n_a": int(masks[cwe_a].sum()),
            "n_b": int(masks[cwe_b].sum()),
            "layers": {},
        }
        combined = masks[cwe_a] | masks[cwe_b]
        y = masks[cwe_b][combined].astype(int)

        for layer, vuln in vuln_by_layer.items():
            X = vuln[combined]
            res = probe(X, y)
            if res:
                results[key]["layers"][layer] = res
            done += 1
            if done % 20 == 0:
                print(f"  {lang.upper()} {done}/{total} probes done…")

    return results, cwes


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_heatmap(results: dict, cwes: list, lang: str, out_path: Path):
    """Heatmap of peak AUROC (max over layers) for each CWE pair."""
    n = len(cwes)
    mat = np.full((n, n), np.nan)

    for key, v in results.items():
        if not v["layers"]:
            continue
        peak_auc = max(d["roc_auc"] for d in v["layers"].values())
        i = cwes.index(v["cwe_a"])
        j = cwes.index(v["cwe_b"])
        mat[i, j] = peak_auc
        mat[j, i] = peak_auc

    df = pd.DataFrame(mat, index=cwes, columns=cwes)

    fig, ax = plt.subplots(figsize=(max(4.5, n * 0.55), max(3.5, n * 0.50)))

    # Diverging from 0.5 (chance): below chance = blue, above = red
    sns.heatmap(
        df,
        ax=ax,
        mask=np.isnan(mat),
        vmin=0.4,
        vmax=1.0,
        cmap="RdYlGn",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.4,
        square=True,
        cbar=False,
    )
    # Grey diagonal
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color="#cccccc", lw=0))

    ax.set_title(
        f"Pairwise CWE probe within {lang.upper()} — peak AUROC across layers\n"
        f"(vulnerable-side mean-token; green = separable, yellow = chance)",
        fontsize=8,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_layer_profiles(results: dict, lang: str, out_path: Path):
    """Line plot: one line per CWE pair, AUROC across layers."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    cmap = plt.get_cmap("tab20")
    pairs = list(results.items())

    for idx, (key, v) in enumerate(pairs):
        if not v["layers"]:
            continue
        layers_present = sorted(v["layers"].keys())
        xs = list(range(len(layers_present)))
        aucs = [v["layers"][l]["roc_auc"] for l in layers_present]
        label = f"{v['cwe_a']} vs {v['cwe_b']}"
        ax.plot(xs, aucs, "-", color=cmap(idx % 20), lw=1.0, alpha=0.75, label=label)

    ax.axhline(0.5, color="grey", lw=0.7, ls=":", alpha=0.5, label="Chance (0.5)")
    if 11 in LAYERS:
        x11 = LAYERS.index(11)
        ax.axvline(x11, color="black", lw=0.8, ls="--", alpha=0.3)
        ax.text(x11 + 0.1, 0.03, "L11", fontsize=7, color="black", alpha=0.5)

    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels([f"L{l}" for l in LAYERS], rotation=45)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(
        f"Pairwise CWE-type probe within {lang.upper()} — all pairs\n"
        f"(vulnerable-side mean-token)",
        fontsize=8,
    )
    ncol = max(1, len(pairs) // 20)
    ax.legend(fontsize=5, loc="upper left", ncol=ncol, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument(
        "--min_n",
        type=int,
        default=MIN_N,
        help="Min samples per CWE to include (default: %(default)s)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run()
    min_n = args.min_n
    print(f"Using mean-pool run: {run_dir}")
    print(f"Min samples per CWE: {min_n}")

    with (run_dir / "meta.json").open() as f:
        meta = json.load(f)

    # Discover eligible languages: those with >=2 CWEs above min_n
    from collections import Counter

    lang_cwe_counts = {}
    for r in meta:
        lang_cwe_counts.setdefault(r["file_extension"], Counter())[r["cwe"]] += 1

    eligible_langs = sorted(
        lang
        for lang, counts in lang_cwe_counts.items()
        if sum(1 for n in counts.values() if n >= min_n) >= 2
    )
    print(f"\nEligible languages: {eligible_langs}")

    for lang in eligible_langs:
        results, cwes = run_language(run_dir, meta, lang, min_n)
        if len(cwes) < 2:
            continue

        # Save JSON
        out_json = run_dir / f"cwe_pairwise_{lang}_results.json"
        serialisable = {}
        for k, v in results.items():
            serialisable[k] = {
                **v,
                "layers": {str(l): d for l, d in v["layers"].items()},
            }
        with out_json.open("w") as f:
            json.dump(
                {"language": lang, "min_n": min_n, "cwes": cwes, "pairs": serialisable},
                f,
                indent=2,
            )
        print(f"Saved JSON: {out_json}")

        # Heatmap + layer profiles
        for dest in [PAPER_FIGS, run_dir]:
            plot_heatmap(results, cwes, lang, dest / f"fig_cwe_pairwise_{lang}.pdf")
            plot_layer_profiles(
                results, lang, dest / f"fig_cwe_pairwise_{lang}_layers.pdf"
            )


if __name__ == "__main__":
    main()
