"""
Pair-stratified cross-validation for vulnerability probing.

This script implements pair-stratified CV to ensure both members of each
vulnerable-secure pair stay in the same fold, eliminating potential leakage.
Compares standard StratifiedKFold vs pair-stratified splits.

Usage:
    conda run -n sae python pair_stratified_probe.py

Output:
    artifacts/analysis/pair_stratified_cv/
        pair_stratified_results.json    (main results)
        comparison.json                 (standard vs pair-stratified)
"""

import base64
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
ARTIFACTS = Path(__file__).parents[2] / "artifacts" / "activations"
RESULTS_DIR = (
    Path(__file__).parents[2] / "artifacts" / "analysis" / "pair_stratified_cv"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_RUN_DIR = (
    ARTIFACTS / "raw_activations" / "vulnerable_code_qwen_coder_standard_16384_raw"
)

SOURCE_JSONL = (
    RAW_RUN_DIR / "activations_layer_0_raw_component_hidden_state_last_token.jsonl"
)

LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]
SEED = 42


class PairStratifiedKFold:
    """
    Custom splitter ensuring both members of a pair stay in the same fold.

    Strategy: For each pair i, assign to fold f(i). Both secure[i] and vuln[i]
    go to the same fold. This ensures no leakage between pair members.
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def split(self, X, y=None, groups=None):
        """
        X: shape [2*n_pairs, d]  (secure samples stacked on top of vuln samples)
        y: shape [2*n_pairs]      (0 for secure, 1 for vuln)
        groups: optional, pair IDs

        Yields (train_idx, test_idx) where pairs stay together.
        """
        n_total = len(X)
        n_pairs = n_total // 2

        # Create pair indices: pair i has indices [i, n_pairs + i]
        pair_indices = np.arange(n_pairs)

        # Shuffle pairs if requested
        if self.shuffle:
            self.rng.shuffle(pair_indices)

        # Assign pairs to folds, ensuring balanced fold sizes
        fold_assignment = np.zeros(n_pairs, dtype=int)
        for i, pair_id in enumerate(pair_indices):
            fold_assignment[pair_id] = i % self.n_splits

        # Generate fold splits
        for fold in range(self.n_splits):
            # Pairs in this fold
            test_pair_indices = np.where(fold_assignment == fold)[0]
            # Expand to sample indices: both secure and vuln members
            test_idx = np.concatenate(
                [
                    test_pair_indices,  # secure members
                    test_pair_indices + n_pairs,  # vuln members
                ]
            )
            test_idx = np.sort(test_idx)

            # Training set: all other pairs
            train_idx = np.setdiff1d(np.arange(n_total), test_idx)

            yield train_idx, test_idx


def load_source_records(jsonl_path: Path):
    """Load records with pair structure."""
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
                    "pair_id": len(records),  # Unique pair identifier
                    "vuln_id": r["vuln_id"],
                    "cwe": r["cwe"],
                    "file_extension": r["file_extension"],
                    "secure_code": secure_code,
                    "vulnerable_code": vuln_code,
                }
            )
    return records


def load_last_token_activations(layer: int):
    """Load pre-computed last-token activations."""
    matches = list(RAW_RUN_DIR.glob(f"activations_layer_{layer}_*.jsonl"))
    if not matches:
        return None, None

    safe_rows, vuln_rows = [], []
    with matches[0].open() as f:
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


def _bootstrap_auc_ci(y_true, y_score, n_bootstrap=500, ci=0.95, seed=SEED):
    """Compute AUROC with bootstrap CI."""
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


def probe_with_splitter(safe_mat, vuln_mat, splitter, n_components=50, n_bootstrap=500):
    """
    Run probing with a custom splitter (standard or pair-stratified).

    Returns:
        dict with roc_auc, ci_lo, ci_hi, n_pairs, n_folds_used
    """
    n_pairs = len(safe_mat)
    X = np.vstack([safe_mat, vuln_mat]).astype(np.float32)
    y = np.array([0] * n_pairs + [1] * n_pairs, dtype=int)

    n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
    clf = LogisticRegression(
        C=0.1, max_iter=1000, class_weight="balanced", random_state=SEED
    )

    y_score = np.zeros(len(y), dtype=float)
    n_folds_used = 0

    for train_idx, test_idx in splitter.split(X, y):
        scaler = StandardScaler()
        pca = PCA(n_components=min(n_comp, len(train_idx) - 1), random_state=SEED)
        X_tr = pca.fit_transform(scaler.fit_transform(X[train_idx]))
        X_te = pca.transform(scaler.transform(X[test_idx]))
        clf.fit(X_tr, y[train_idx])
        y_score[test_idx] = clf.predict_proba(X_te)[:, 1]
        n_folds_used += 1

    mean_auc, ci_lo, ci_hi = _bootstrap_auc_ci(y, y_score, n_bootstrap=n_bootstrap)
    return {
        "roc_auc": float(mean_auc),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "n_pairs": n_pairs,
        "n_folds_used": n_folds_used,
    }


def main():
    print("\n[Pair-Stratified Cross-Validation Analysis]\n")

    results = {
        "description": "Comparison of standard StratifiedKFold vs pair-stratified CV",
        "n_bootstrap": 500,
        "n_components_pca": 50,
        "cv_folds": 5,
        "by_layer": {},
        "summary": {},
    }

    # Load pair information for summary
    records = load_source_records(SOURCE_JSONL)
    n_pairs = len(records)
    print(f"Loaded {n_pairs} pairs from {SOURCE_JSONL.name}")

    # Run analysis for each layer
    for layer in LAYERS:
        print(f"\nLayer {layer}:")

        # Load activations
        safe_mat, vuln_mat = load_last_token_activations(layer)
        if safe_mat is None:
            print(f"  [SKIP] No data found")
            continue

        # Standard StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        std_result = probe_with_splitter(safe_mat, vuln_mat, skf, n_bootstrap=500)
        print(
            f"  Standard CV:        AUROC {std_result['roc_auc']:.4f} [{std_result['ci_lo']:.4f}–{std_result['ci_hi']:.4f}]"
        )

        # Pair-stratified CV
        psf = PairStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        pair_result = probe_with_splitter(safe_mat, vuln_mat, psf, n_bootstrap=500)
        print(
            f"  Pair-stratified CV: AUROC {pair_result['roc_auc']:.4f} [{pair_result['ci_lo']:.4f}–{pair_result['ci_hi']:.4f}]"
        )

        # Compute delta
        delta = pair_result["roc_auc"] - std_result["roc_auc"]
        print(f"  Δ AUROC: {delta:+.4f}")

        results["by_layer"][layer] = {
            "standard_cv": std_result,
            "pair_stratified_cv": pair_result,
            "delta_auroc": float(delta),
            "delta_pct": float(100.0 * delta / std_result["roc_auc"]),
        }

    # Summary statistics
    all_deltas = [
        results["by_layer"][layer]["delta_auroc"]
        for layer in LAYERS
        if layer in results["by_layer"]
    ]
    all_pair_aurocs = [
        results["by_layer"][layer]["pair_stratified_cv"]["roc_auc"]
        for layer in LAYERS
        if layer in results["by_layer"]
    ]

    if all_deltas:
        results["summary"] = {
            "mean_delta_auroc": float(np.mean(all_deltas)),
            "std_delta_auroc": float(np.std(all_deltas)),
            "min_delta_auroc": float(np.min(all_deltas)),
            "max_delta_auroc": float(np.max(all_deltas)),
            "mean_pair_stratified_auroc": float(np.mean(all_pair_aurocs)),
            "max_pair_stratified_auroc": float(np.max(all_pair_aurocs)),
            "layer_with_max_auroc": int(LAYERS[np.argmax(all_pair_aurocs)]),
            "interpretation": (
                "If delta is close to zero, pair-stratified CV gives similar results to standard CV, "
                "suggesting no major leakage. If delta is substantially negative, pair-stratified CV "
                "is more conservative (harder problem), indicating potential leakage in standard CV."
            ),
        }

    # Save results
    results_path = RESULTS_DIR / "pair_stratified_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {results_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if results["summary"]:
        for key, val in results["summary"].items():
            if key != "interpretation":
                print(f"{key:35s}: {val}")
        print(f"\n{results['summary']['interpretation']}")
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    main()
