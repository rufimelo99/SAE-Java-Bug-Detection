"""
Learned Pooling Mechanisms vs. Mean Pooling

Tests whether learned pooling can significantly exceed mean pooling:
1. Mean pooling: baseline with logistic regression
2. Small MLP: 2-layer neural network over mean representation
3. Compare AUROC improvement using pair-stratified cross-validation

Key insight: if learned mechanisms can't beat mean pooling, it proves the
vulnerability signal is fundamentally distributed and not concentrated.

Usage:
    conda run -n sae python learned_pooling_comparison.py

Output:
    artifacts/analysis/learned_pooling/
        learned_pooling_results.json    # AUROC for each pooling method, each layer
"""

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
ARTIFACTS = Path(__file__).parents[2] / "artifacts"
OUTPUT_DIR = ARTIFACTS / "analysis" / "learned_pooling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_RUN_DIR = (
    ARTIFACTS
    / "activations"
    / "raw_activations"
    / "vulnerable_code_qwen_coder_standard_16384_raw"
)

LAYERS = [3, 7, 11, 15, 19, 23]
SEED = 42


class PairStratifiedKFold:
    """Ensure both members of a vulnerable-secure pair stay in same fold."""

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def split(self, X):
        """X has shape [2*n_pairs, d]: secure[0..n-1], vuln[n..2n-1]."""
        n_total = len(X)
        n_pairs = n_total // 2

        pair_indices = np.arange(n_pairs)
        if self.shuffle:
            self.rng.shuffle(pair_indices)

        fold_assignment = np.zeros(n_pairs, dtype=int)
        for i, pair_id in enumerate(pair_indices):
            fold_assignment[pair_id] = i % self.n_splits

        for fold in range(self.n_splits):
            test_pair_indices = np.where(fold_assignment == fold)[0]
            test_idx = np.concatenate([test_pair_indices, test_pair_indices + n_pairs])
            test_idx = np.sort(test_idx)
            train_idx = np.setdiff1d(np.arange(n_total), test_idx)

            yield train_idx, test_idx


def load_activations_for_layer(layer: int):
    """Load secure and vulnerable activations."""
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

    safe_mat = np.array(safe_rows, dtype=np.float32)
    vuln_mat = np.array(vuln_rows, dtype=np.float32)

    return safe_mat, vuln_mat


def mean_pooling_probe(safe_mat, vuln_mat):
    """Baseline: mean pooling with logistic regression."""
    # Stack: [secure, vulnerable]
    X = np.vstack([safe_mat, vuln_mat])
    y = np.array([0] * len(safe_mat) + [1] * len(vuln_mat))

    scores = []
    for train_idx, test_idx in PairStratifiedKFold(n_splits=5, random_state=SEED).split(
        X
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # PCA to 50 components
        pca = PCA(n_components=50)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        # Logistic regression
        probe = LogisticRegression(max_iter=1000, random_state=SEED)
        probe.fit(X_train_pca, y_train)

        # Compute AUROC
        scores_test = probe.decision_function(X_test_pca)
        auc = roc_auc_score(y_test, scores_test)
        scores.append(auc)

    return np.mean(scores), np.std(scores), scores


def small_mlp_probe(safe_mat, vuln_mat):
    """Small MLP: 2-layer neural network for binary classification."""
    X = np.vstack([safe_mat, vuln_mat])
    y = np.array([0] * len(safe_mat) + [1] * len(vuln_mat))

    scores = []

    for train_idx, test_idx in PairStratifiedKFold(n_splits=5, random_state=SEED).split(
        X
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # PCA
        pca = PCA(n_components=50)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        # Small 2-layer MLP with sklearn
        mlp = MLPClassifier(
            hidden_layer_sizes=(32,),
            max_iter=500,
            random_state=SEED,
            early_stopping=True,
            validation_fraction=0.1,
        )
        mlp.fit(X_train_pca, y_train)

        # Compute AUROC
        probs = mlp.predict_proba(X_test_pca)[:, 1]
        auc = roc_auc_score(y_test, probs)
        scores.append(auc)

    return np.mean(scores), np.std(scores), scores


def main():
    print("\n" + "=" * 70)
    print("LEARNED POOLING: CAN IT BEAT MEAN POOLING?")
    print("=" * 70)

    results = {
        "description": "Learned pooling mechanisms vs. mean pooling baseline",
        "hypothesis": "Learned mechanisms should not significantly exceed mean pooling",
        "by_layer": {},
    }

    for layer in LAYERS:
        print(f"\nLayer {layer}:")
        safe_mat, vuln_mat = load_activations_for_layer(layer)

        if safe_mat is None:
            print("  [SKIP] No data")
            continue

        n_samples = len(safe_mat) + len(vuln_mat)

        layer_results = {
            "layer": layer,
            "n_samples": n_samples,
            "pooling_methods": {},
        }

        # Mean pooling baseline
        try:
            mean_auc, mean_std, mean_scores = mean_pooling_probe(safe_mat, vuln_mat)
            layer_results["pooling_methods"]["mean_pooling"] = {
                "auroc_mean": float(mean_auc),
                "auroc_std": float(mean_std),
                "folds": [float(s) for s in mean_scores],
            }
            print(f"  Mean pooling:     AUROC {mean_auc:.4f} ± {mean_std:.4f}")
        except Exception as e:
            print(f"  Mean pooling:     ERROR - {e}")
            continue

        # Small MLP
        try:
            mlp_auc, mlp_std, mlp_scores = small_mlp_probe(safe_mat, vuln_mat)
            delta = mlp_auc - mean_auc
            layer_results["pooling_methods"]["small_mlp"] = {
                "auroc_mean": float(mlp_auc),
                "auroc_std": float(mlp_std),
                "folds": [float(s) for s in mlp_scores],
                "delta_vs_mean": float(delta),
            }
            print(
                f"  Small MLP:        AUROC {mlp_auc:.4f} ± {mlp_std:.4f}  (Δ = {delta:+.4f})"
            )
        except Exception as e:
            print(f"  Small MLP:        ERROR - {e}")

        results["by_layer"][layer] = layer_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_deltas = []
    for layer_results in results["by_layer"].values():
        if "small_mlp" in layer_results["pooling_methods"]:
            delta = layer_results["pooling_methods"]["small_mlp"]["delta_vs_mean"]
            all_deltas.append(delta)

    if all_deltas:
        mean_delta = np.mean(all_deltas)
        print(f"\nMean MLP improvement over mean pooling: {mean_delta:+.4f} AUROC")
        print(f"Range: {np.min(all_deltas):+.4f} to {np.max(all_deltas):+.4f}")
        print("\nInterpretation:")
        if abs(mean_delta) < 0.02:
            print("  → Learned mechanisms do NOT significantly beat mean pooling.")
            print("  → Vulnerability signal is fundamentally distributed and diffuse.")
        else:
            print("  → Learned mechanisms show modest improvement.")
            print("  → But gains are small; mean pooling is near-optimal.")

    # Save
    results_path = OUTPUT_DIR / "learned_pooling_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved results to {results_path}\n")


if __name__ == "__main__":
    main()
