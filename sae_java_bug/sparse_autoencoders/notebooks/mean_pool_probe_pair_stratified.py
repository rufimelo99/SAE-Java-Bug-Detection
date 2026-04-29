"""
Mean-token pooling probe with pair-stratified CV — reviewer response.

Same as mean_pool_probe.py but enforces pair-level grouping in cross-validation:
both members of a (vulnerable, secure) pair must be in the same fold.

This addresses concern about within-pair leakage and provides stricter validation
of the last-token vs mean-token AUROC gap.

Usage:
    conda run -n sae python mean_pool_probe_pair_stratified.py

Outputs:
    Console table: pair-stratified AUROC vs current (non-grouped) results
    Comparison JSON with both approaches
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
ARTIFACTS = Path(__file__).parents[2] / "artifacts" / "activations"
OUT_DIR = Path(__file__).parents[2] / "artifacts" / "pair_stratified_probes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pre-computed mean-pool activations from mean_pool_probe.py
MEAN_POOL_RUN = sorted((ARTIFACTS / "mean_pool").glob("*/meta.json"))[-1].parent

LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]
SEED = 42
N_BOOTSTRAP = 500


# ── Utilities ──────────────────────────────────────────────────────────────────


def _bootstrap_auc_ci(y_true, y_score, n_bootstrap=N_BOOTSTRAP, ci=0.95, seed=SEED):
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


def load_meta(run_dir: Path):
    with (run_dir / "meta.json").open() as f:
        return json.load(f)


def _tensor_to_numpy(t: "torch.Tensor") -> np.ndarray:
    import ctypes

    t = t.float().contiguous()
    buf = (ctypes.c_float * t.numel()).from_address(t.data_ptr())
    return np.ctypeslib.as_array(buf).reshape(t.shape).copy()


def load_layer(run_dir: Path, layer: int):
    safe_pt = run_dir / f"safe_layer_{layer}.pt"
    vuln_pt = run_dir / f"vulnerable_layer_{layer}.pt"
    safe_mat = _tensor_to_numpy(torch.load(safe_pt, weights_only=True))
    vuln_mat = _tensor_to_numpy(torch.load(vuln_pt, weights_only=True))
    return safe_mat, vuln_mat


def probe_pair_stratified(safe_mat, vuln_mat, pair_ids, n_components=50, cv=5):
    """
    Binary probe with pair-level stratification.

    Both members of each pair (safe and vulnerable) must be in the same fold.
    """
    n = len(safe_mat)
    X = np.vstack([safe_mat, vuln_mat])
    y = np.array([0] * n + [1] * n, dtype=int)

    # Pair group: [pair_0, pair_1, ..., pair_0, pair_1, ...]
    # (secure samples, then vulnerable samples in same order)
    pair_groups = np.concatenate([pair_ids, pair_ids])

    # Custom pair-stratified fold: unique pairs, shuffle by pair, group by fold
    unique_pairs = np.unique(pair_ids)
    n_pairs = len(unique_pairs)
    n_per_fold = max(1, n_pairs // cv)

    np.random.RandomState(SEED).shuffle(unique_pairs)

    y_score = np.zeros(len(y), dtype=float)

    for fold_idx in range(cv):
        # Pairs for this test fold
        test_pair_start = fold_idx * n_per_fold
        if fold_idx == cv - 1:
            test_pairs = unique_pairs[test_pair_start:]
        else:
            test_pairs = unique_pairs[test_pair_start : test_pair_start + n_per_fold]

        test_mask = np.isin(pair_groups, test_pairs)
        train_mask = ~test_mask

        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        scaler = StandardScaler()
        pca = PCA(
            n_components=min(n_components, X_tr.shape[1], X_tr.shape[0] - 1),
            random_state=SEED,
        )
        X_tr_pca = pca.fit_transform(scaler.fit_transform(X_tr))
        X_te_pca = pca.transform(scaler.transform(X_te))

        clf = LogisticRegression(
            C=0.1, max_iter=1000, class_weight="balanced", random_state=SEED
        )
        clf.fit(X_tr_pca, y_tr)
        y_score[test_mask] = clf.predict_proba(X_te_pca)[:, 1]

    mean_auc, ci_lo, ci_hi = _bootstrap_auc_ci(y, y_score)
    return {"roc_auc": mean_auc, "ci_lo": ci_lo, "ci_hi": ci_hi}


# ────────────────────────────────────────────────────────────────────────────────


def main():
    print("Loading mean-pool activations...")
    meta = load_meta(MEAN_POOL_RUN)
    pair_ids = np.array([r["vuln_id"] for r in meta])

    results = {}

    print(f"\nPair-stratified probing ({len(np.unique(pair_ids))} pairs):")
    print("=" * 80)
    print(f"{'Layer':<8} {'AUROC':<10} {'95% CI':<20}")
    print("=" * 80)

    for layer in LAYERS:
        safe_mat, vuln_mat = load_layer(MEAN_POOL_RUN, layer)

        res = probe_pair_stratified(safe_mat, vuln_mat, pair_ids)
        results[layer] = res

        print(
            f"L{layer:<7} {res['roc_auc']:.3f}     [{res['ci_lo']:.3f}–{res['ci_hi']:.3f}]"
        )

    print("=" * 80)

    # Save results
    out_file = OUT_DIR / "pair_stratified_results.json"
    with out_file.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_file}")


if __name__ == "__main__":
    main()
