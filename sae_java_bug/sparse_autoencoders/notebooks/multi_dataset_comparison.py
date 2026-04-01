"""
Multi-dataset replication of the binary vulnerability probe.

Runs the standard binary probe (5-fold stratified CV, LR + PCA-50) across
all four C/C++ datasets and a CWE-balanced ablation alongside each, testing
whether AUROC is driven by CWE composition rather than a genuine
vulnerable/secure signal.

Datasets (activations expected in code-security-probing sibling repo):
    DeltaSecommits  (c_only)
    SVEN            (sven_c_only)
    BigVul          (bigvul_c_only)       — requires extraction first
    PreciseBugs     (precisebugs_c_only)  — requires extraction first

Usage:
    python multi_dataset_comparison.py
    python multi_dataset_comparison.py --layers 11 15
    python multi_dataset_comparison.py --datasets DeltaSecommits SVEN
    python multi_dataset_comparison.py --min_cwe_samples 15

Saves:
    <SAE repo>/sae_java_bug/artifacts/results/multi_dataset_comparison.json
    <paper repo>/figures/fig_multi_dataset_comparison.pdf
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
GITHUB     = Path(__file__).parents[4]   # …/GitHub/

SAE_REPO   = Path(__file__).parents[3]   # SAE-Java-Bug-Detection/
RESULTS    = SAE_REPO / "sae_java_bug" / "artifacts" / "results"
FIGS_DIR   = SAE_REPO / "sae_java_bug" / "artifacts" / "figures"

CSP_ACTS   = GITHUB / "code-security-probing" / "artifacts" / "activations"

PAPER_FIGS = (
    GITHUB
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)

LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]
SEED   = 42

DATASETS: dict[str, str] = {
    "DeltaSecommits": "c_only",
    "SVEN":           "sven_c_only",
    "BigVul":         "bigvul_c_only",
    "PreciseBugs":    "precisebugs_c_only",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_splits(
    acts_subdir: str,
    layer: int,
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """
    Load (secure_mat, vuln_mat, cwes) from whichever of train/test splits exist.
    Returns None if no files are found for this layer.
    """
    secure_rows, vuln_rows, cwes = [], [], []
    found = False
    for split in ("train", "test"):
        path = CSP_ACTS / acts_subdir / f"layer_{layer}_{split}.jsonl"
        if not path.exists():
            continue
        found = True
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                secure_rows.append(r["secure"])
                vuln_rows.append(r["vulnerable"])
                cwes.append(r.get("cwe", "unknown"))

    if not found or not secure_rows:
        return None

    return (
        np.array(secure_rows, dtype=np.float32),
        np.array(vuln_rows,   dtype=np.float32),
        cwes,
    )


# ── CWE balancing ─────────────────────────────────────────────────────────────

def apply_cwe_balance(
    secure: np.ndarray,
    vuln: np.ndarray,
    cwes: list[str],
    min_cwe_samples: int,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """
    Subsample each CWE to the minimum qualifying count.
    Returns (secure_balanced, vuln_balanced, n_cwes, min_count) or None
    if fewer than 2 CWEs qualify.
    """
    rng     = np.random.default_rng(SEED)
    cwe_arr = np.array(cwes)
    unique, counts = np.unique(cwe_arr, return_counts=True)

    qualifying = {
        cwe: np.where(cwe_arr == cwe)[0]
        for cwe, n in zip(unique, counts)
        if n >= min_cwe_samples
    }
    if len(qualifying) < 2:
        return None

    min_count = min(len(idx) for idx in qualifying.values())
    selected: list[int] = []
    for idx_list in qualifying.values():
        selected.extend(
            rng.choice(idx_list, size=min_count, replace=False).tolist()
        )

    selected = sorted(selected)
    return (
        secure[selected],
        vuln[selected],
        len(qualifying),
        min_count,
    )


# ── Probe ─────────────────────────────────────────────────────────────────────

def _bootstrap_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n: int = 500,
) -> tuple[float, float, float]:
    from sklearn.metrics import roc_auc_score
    rng  = np.random.default_rng(SEED)
    aucs = []
    sz   = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, sz, size=sz)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    a = np.array(aucs)
    return float(a.mean()), float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))


def run_probe(
    secure: np.ndarray,
    vuln: np.ndarray,
    n_components: int = 50,
) -> dict:
    from sklearn.metrics import roc_auc_score

    n  = len(secure)
    X  = np.vstack([secure, vuln]).astype(np.float32)
    y  = np.array([0] * n + [1] * n, dtype=int)

    skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
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

    mean_auc, ci_lo, ci_hi = _bootstrap_auc(y, y_score)
    return {
        "auroc": round(mean_auc, 4),
        "ci_lo": round(ci_lo,    4),
        "ci_hi": round(ci_hi,    4),
        "n":     n,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-dataset binary vulnerability probe comparison."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--layers", nargs="+", type=int, default=LAYERS)
    parser.add_argument("--n_components", type=int, default=50)
    parser.add_argument("--min_cwe_samples", type=int, default=20)
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict = {}

    for name in args.datasets:
        if name not in DATASETS:
            print(f"[WARN] Unknown dataset '{name}'. Available: {list(DATASETS)}")
            continue

        subdir   = DATASETS[name]
        sentinel = CSP_ACTS / subdir / f"layer_{args.layers[0]}_train.jsonl"
        if not sentinel.exists():
            print(f"\n[SKIP] {name}: activations not found at {CSP_ACTS / subdir}")
            print(f"       cd code-security-probing && python experiments/00b_extract_sven_acts.py "
                  f"--pairs_jsonl artifacts/data/... --out_subdir {subdir}")
            continue

        print(f"\n{'='*60}")
        print(f"  {name}  ({subdir})")
        print(f"{'='*60}")

        all_results[name] = {"subdir": subdir, "layers": {}}

        for layer in args.layers:
            data = load_all_splits(subdir, layer)
            if data is None:
                print(f"  L{layer:>2}  [SKIP] no data")
                continue

            secure, vuln, cwes = data
            n_pairs = len(secure)

            std = run_probe(secure, vuln, n_components=args.n_components)
            std_str = (f"AUROC={std['auroc']:.4f} "
                       f"[{std['ci_lo']:.3f}–{std['ci_hi']:.3f}]")

            bal_data = apply_cwe_balance(
                secure, vuln, cwes, min_cwe_samples=args.min_cwe_samples,
            )
            if bal_data is not None:
                s_bal, v_bal, n_cwes, min_count = bal_data
                bal     = run_probe(s_bal, v_bal, n_components=args.n_components)
                bal_str = (f"balanced: {bal['auroc']:.4f} "
                           f"[{bal['ci_lo']:.3f}–{bal['ci_hi']:.3f}] "
                           f"({n_cwes} CWEs × {min_count})")
            else:
                bal     = None
                bal_str = f"balanced: N/A (<2 CWEs with ≥{args.min_cwe_samples} samples)"

            print(f"  L{layer:>2}  n={n_pairs:>5}  {std_str}   {bal_str}")

            all_results[name]["layers"][str(layer)] = {
                "n_pairs":  n_pairs,
                "standard": std,
                "balanced": bal,
            }

    if not all_results:
        print("\nNo datasets available. Run extraction scripts first.")
        return

    # ── Summary table (L11) ───────────────────────────────────────────────────
    pivot = "11"
    print(f"\n{'Dataset':<20} {'n':>6}  {'L11 standard':>14}  {'L11 balanced':>14}")
    print("-" * 62)
    for name, res in all_results.items():
        l   = res["layers"].get(pivot, {})
        std = l.get("standard", {})
        bal = l.get("balanced") or {}
        n   = l.get("n_pairs", 0)
        print(
            f"  {name:<18} {n:>6}  "
            f"{std.get('auroc', float('nan')):>14.4f}  "
            f"{bal.get('auroc', float('nan')):>14.4f}"
        )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = RESULTS / "multi_dataset_comparison.json"
    with out_json.open("w") as f:
        json.dump(
            {"layers": args.layers, "n_components": args.n_components,
             "min_cwe_samples": args.min_cwe_samples, "results": all_results},
            f, indent=2,
        )
    print(f"\nResults → {out_json}")

    _plot(all_results, args.layers)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot(results: dict, layers: list[int]) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    COLORS  = {"DeltaSecommits": "#2196F3", "SVEN": "#4CAF50",
               "BigVul": "#FF9800", "PreciseBugs": "#9C27B0"}
    MARKERS = {"DeltaSecommits": "o", "SVEN": "s",
               "BigVul": "^", "PreciseBugs": "D"}

    xs = list(range(len(layers)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)

    for ax, mode, title in zip(
        axes,
        ["standard", "balanced"],
        ["Standard (all CWEs pooled)", "CWE-balanced ablation"],
    ):
        for name, res in results.items():
            aurocs, ci_los, ci_his, xs_plot = [], [], [], []
            for i, layer in enumerate(layers):
                entry = res["layers"].get(str(layer), {}).get(mode)
                if entry is None:
                    continue
                aurocs.append(entry["auroc"])
                ci_los.append(entry["ci_lo"])
                ci_his.append(entry["ci_hi"])
                xs_plot.append(i)

            if not aurocs:
                continue

            col = COLORS.get(name, "gray")
            mrk = MARKERS.get(name, "o")
            ax.plot(xs_plot, aurocs, marker=mrk, ls="-", lw=1.5, ms=4,
                    label=name, color=col)
            ax.fill_between(xs_plot, ci_los, ci_his, alpha=0.12, color=col)

        ax.axhline(0.5, color="gray", lw=1, ls="--", label="chance")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"L{l}" for l in layers], rotation=45)
        ax.set_xlabel("Layer")
        if mode == "standard":
            ax.set_ylabel("AUROC (5-fold CV)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        "Binary vulnerability probe — multi-dataset replication (C/C++)",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()

    out = FIGS_DIR / "fig_multi_dataset_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure → {out}")

    if PAPER_FIGS.exists():
        shutil.copy(out, PAPER_FIGS / "fig_multi_dataset_comparison.pdf")
        print(f"Figure → {PAPER_FIGS / 'fig_multi_dataset_comparison.pdf'}  (paper repo)")


if __name__ == "__main__":
    main()
