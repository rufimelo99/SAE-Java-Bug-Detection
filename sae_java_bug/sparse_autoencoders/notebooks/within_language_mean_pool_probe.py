"""
Within-language mean-token pooling probe — reviewer Q2 response.

Loads pre-computed mean-pool .pt activations (from mean_pool_probe.py) and runs
the binary vulnerable-vs-secure probe restricted to each language stratum (C, PHP)
at all eight layers. Directly answers whether the global mean-token AUROC gain
(0.63--0.65) survives within a single programming language, ruling out residual
cross-language effects.

Usage:
    conda run -n sae python within_language_mean_pool_probe.py
    conda run -n sae python within_language_mean_pool_probe.py --run_dir PATH

Requires:
    artifacts/activations/mean_pool/<ts>/
        safe_layer_{L}.pt          [N, 3584]
        vulnerable_layer_{L}.pt    [N, 3584]
        meta.json                  list of {vuln_id, cwe, file_extension}

Outputs:
    Paper figures dir / fig_within_lang_mean_pool.pdf
    Console table: within-language AUROC per language × layer
"""

import argparse
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

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
ARTIFACTS = Path(__file__).parents[2] / "artifacts" / "activations"
PAPER_FIGS = (
    Path(__file__).parents[4]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]
SEED = 42

LANGUAGES = {
    "C": ["c", "cc", "cpp", "h"],
    "PHP": ["php"],
}

LANG_COLORS = {
    "C": "#4878cf",
    "PHP": "#e07b39",
}

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


def find_latest_mean_pool_run() -> Path:
    runs = sorted((ARTIFACTS / "mean_pool").glob("*/meta.json"))
    if not runs:
        raise FileNotFoundError(
            f"No mean_pool runs found under {ARTIFACTS}/mean_pool/. "
            "Run mean_pool_probe.py first."
        )
    return runs[-1].parent


def load_meta(run_dir: Path) -> list[dict]:
    with (run_dir / "meta.json").open() as f:
        return json.load(f)


def _tensor_to_numpy(t: "torch.Tensor") -> np.ndarray:
    """Fast tensor → numpy via ctypes (avoids Python-level bytes() iteration)."""
    import ctypes

    t = t.float().contiguous()
    buf = (ctypes.c_float * t.numel()).from_address(t.data_ptr())
    return np.ctypeslib.as_array(buf).reshape(t.shape).copy()


def load_layer(run_dir: Path, layer: int) -> tuple[np.ndarray, np.ndarray]:
    safe_pt = run_dir / f"safe_layer_{layer}.pt"
    vuln_pt = run_dir / f"vulnerable_layer_{layer}.pt"
    safe_mat = _tensor_to_numpy(torch.load(safe_pt, weights_only=True))
    vuln_mat = _tensor_to_numpy(torch.load(vuln_pt, weights_only=True))
    return safe_mat, vuln_mat


# ─────────────────────────────────────────────────────────────────────────────
# Probe
# ─────────────────────────────────────────────────────────────────────────────


def _bootstrap_auc_ci(y_true, y_score, n_bootstrap=1000, ci=0.95, seed=SEED):
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
    return (
        float(np.mean(aucs)),
        float(np.quantile(aucs, alpha)),
        float(np.quantile(aucs, 1 - alpha)),
    )


def probe_subset(
    safe_mat: np.ndarray, vuln_mat: np.ndarray, n_components: int = 50, cv: int = 5
) -> dict:
    """Binary probe on a pre-filtered (safe, vuln) pair of matrices."""
    n = len(safe_mat)
    X = np.vstack([safe_mat, vuln_mat])
    y = np.array([0] * n + [1] * n, dtype=int)

    n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
    clf = LogisticRegression(
        C=0.1, max_iter=1000, class_weight="balanced", random_state=SEED
    )
    skf = StratifiedKFold(n_splits=min(cv, n), shuffle=True, random_state=SEED)

    y_score = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        pca = PCA(n_components=min(n_comp, len(train_idx) - 1), random_state=SEED)
        X_tr = pca.fit_transform(scaler.fit_transform(X[train_idx]))
        X_te = pca.transform(scaler.transform(X[test_idx]))
        clf.fit(X_tr, y[train_idx])
        y_score[test_idx] = clf.predict_proba(X_te)[:, 1]

    mean_auc, ci_lo, ci_hi = _bootstrap_auc_ci(y, y_score)
    return {"roc_auc": mean_auc, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": n}


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────


def make_figure(results: dict, global_mean: dict, out_path: Path):
    """
    One panel per language (C, PHP), showing:
      - within-language mean-token AUROC (coloured)
      - global mean-token AUROC (grey dashed, for reference)
      - chance line
    """
    langs = [l for l in LANGUAGES if results.get(l)]
    fig, axes = plt.subplots(
        1, len(langs), figsize=(3.5 * len(langs), 3.2), sharey=True
    )
    if len(langs) == 1:
        axes = [axes]

    for ax, lang in zip(axes, langs):
        res = results[lang]
        layers_ok = sorted(res.keys())
        xs = [LAYERS.index(l) for l in layers_ok]
        aucs = [res[l]["roc_auc"] for l in layers_ok]
        los = [res[l]["ci_lo"] for l in layers_ok]
        his = [res[l]["ci_hi"] for l in layers_ok]
        n = res[layers_ok[0]]["n"]

        ax.plot(
            xs,
            aucs,
            "s-",
            color=LANG_COLORS[lang],
            lw=1.5,
            ms=5,
            label=f"{lang} (n={n})",
        )
        ax.fill_between(xs, los, his, color=LANG_COLORS[lang], alpha=0.20)

        # Global mean-token for reference
        g_layers = sorted(global_mean.keys())
        g_xs = [LAYERS.index(l) for l in g_layers]
        g_aucs = [global_mean[l]["roc_auc"] for l in g_layers]
        ax.plot(
            g_xs,
            g_aucs,
            "o--",
            color="#888888",
            lw=1.0,
            ms=3,
            label="Global mean-token",
            alpha=0.6,
        )

        ax.axhline(0.5, color="grey", lw=0.7, ls=":", alpha=0.5)
        if 11 in LAYERS:
            x11 = LAYERS.index(11)
            ax.axvline(x11, color="black", lw=0.8, ls="--", alpha=0.25)

        ax.set_xticks(range(len(LAYERS)))
        ax.set_xticklabels([f"L{l}" for l in LAYERS], rotation=45)
        ax.set_xlabel("Layer")
        ax.set_title(f"Within-{lang}", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")

    axes[0].set_ylabel("AUROC  (vuln vs.\\ secure)")
    fig.suptitle(
        "Within-language mean-token pooling probe\n(shaded = 95\\% bootstrap CI)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Table
# ─────────────────────────────────────────────────────────────────────────────


def print_table(results: dict, global_mean: dict):
    langs = sorted(results.keys())
    print("\n" + "=" * 100)
    print("Within-language mean-token AUROC [95% CI]   (compare: global mean-token)")
    print("=" * 100)
    header = (
        f"{'Layer':>6}  " + "  ".join(f"{l:^27}" for l in langs) + f"  {'Global':^27}"
    )
    print(header)
    print("-" * 100)
    for layer in LAYERS:
        row = f"  L{layer:>2}   "
        for lang in langs:
            if layer in results.get(lang, {}):
                d = results[lang][layer]
                row += f"{d['roc_auc']:.3f} [{d['ci_lo']:.3f}–{d['ci_hi']:.3f}] (n={d['n']})   "
            else:
                row += f"{'—':^27}   "
        if layer in global_mean:
            g = global_mean[layer]
            row += f"{g['roc_auc']:.3f} [{g['ci_lo']:.3f}–{g['ci_hi']:.3f}]"
        print(row)
    print("=" * 100)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="Path to mean_pool run dir (default: latest)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_mean_pool_run()
    print(f"Using mean-pool run: {run_dir}")

    meta = load_meta(run_dir)
    extensions = [r["file_extension"] for r in meta]
    print(f"  {len(meta)} samples, extensions: {set(extensions)}")

    # Build language masks
    lang_masks = {}
    for lang, exts in LANGUAGES.items():
        mask = np.array([ext in exts for ext in extensions])
        n = mask.sum()
        if n >= 30:
            lang_masks[lang] = mask
            print(f"  {lang}: {n} samples")
        else:
            print(f"  {lang}: {n} samples — skipping (< 30)")

    # Load activations and probe per layer × language
    results = {lang: {} for lang in lang_masks}
    global_mean = {}

    for layer in LAYERS:
        safe_pt = run_dir / f"safe_layer_{layer}.pt"
        if not safe_pt.exists():
            print(f"  [SKIP] L{layer}: no .pt files")
            continue

        print(f"\nL{layer}:")
        safe_mat, vuln_mat = load_layer(run_dir, layer)

        # Global probe (all languages)
        res_all = probe_subset(safe_mat, vuln_mat)
        global_mean[layer] = res_all
        print(
            f"  Global:  AUROC {res_all['roc_auc']:.3f} [{res_all['ci_lo']:.3f}–{res_all['ci_hi']:.3f}]  (n={res_all['n']})"
        )

        # Within-language probes
        for lang, mask in lang_masks.items():
            s = safe_mat[mask]
            v = vuln_mat[mask]
            res = probe_subset(s, v)
            results[lang][layer] = res
            print(
                f"  {lang:4s}: AUROC {res['roc_auc']:.3f} [{res['ci_lo']:.3f}–{res['ci_hi']:.3f}]  (n={res['n']})"
            )

    # Remove languages with no results
    results = {l: v for l, v in results.items() if v}

    print_table(results, global_mean)

    out_path = PAPER_FIGS / "fig_within_lang_mean_pool.pdf"
    make_figure(results, global_mean, out_path)

    results_path = run_dir / "within_lang_mean_pool_results.json"
    payload = {
        "run_dir": str(run_dir),
        "global_mean": {str(l): v for l, v in global_mean.items()},
        "within_language": {
            lang: {str(l): v for l, v in ldata.items()}
            for lang, ldata in results.items()
        },
    }
    with results_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
