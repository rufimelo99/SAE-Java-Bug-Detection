#!/usr/bin/env python3
"""
Paired Ranking Task: Validate the "relative vs. absolute" claim.

For each test pair (vulnerable, secure), computes projections onto the
vulnerability direction. Accuracy = (# pairs where vulnerable > secure) / total.

If the direction truly captures relative ordering, this should achieve ~87% accuracy
(matching the 87-88% per-pair alignment reported in the paper).

Usage:
    python scripts/paired_ranking_task.py [--layers 0,3,7,11,15,19,23,27] [--n_folds 5]
"""

import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Constants ──────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
ARTIFACTS = REPO_ROOT / "sae_java_bug" / "artifacts" / "activations"
LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]
SEED = 42


# ── Tensor I/O ────────────────────────────────────────────────────────────
def _t2np(t: torch.Tensor) -> np.ndarray:
    """Fast tensor → numpy via ctypes (avoids Python-level iteration)."""
    t = t.float().contiguous()
    buf = (ctypes.c_float * t.numel()).from_address(t.data_ptr())
    return np.ctypeslib.as_array(buf).reshape(t.shape).copy()


def find_latest_run(pooling_type: str = "mean_pool") -> Path:
    """Find the most recent activation run."""
    runs = sorted((ARTIFACTS / pooling_type).glob("*/meta.json"))
    if not runs:
        raise FileNotFoundError(
            f"No {pooling_type} runs under {ARTIFACTS}/{pooling_type}/"
        )
    return runs[-1].parent


def load_layer(run_dir: Path, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Load safe and vulnerable activations for a layer."""
    safe_pt = run_dir / f"safe_layer_{layer}.pt"
    vuln_pt = run_dir / f"vulnerable_layer_{layer}.pt"

    if not safe_pt.exists() or not vuln_pt.exists():
        raise FileNotFoundError(f"Missing activations for layer {layer}")

    safe = _t2np(torch.load(safe_pt, weights_only=True))
    vuln = _t2np(torch.load(vuln_pt, weights_only=True))

    return safe, vuln


def load_metadata(run_dir: Path) -> dict:
    """Load metadata (contains pair info, CWEs, etc.)."""
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {"n_pairs": len(load_layer(run_dir, 0)[0])}
    with open(meta_path) as f:
        return json.load(f)


# ── Pair-stratified CV ────────────────────────────────────────────────────
class PairStratifiedKFold:
    """K-fold splitter that keeps paired samples together."""

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, n_pairs: int):
        """
        Yields (train_indices, test_indices) ensuring pairs stay together.

        For paired data, train_indices and test_indices are pair IDs.
        """
        rng = np.random.RandomState(self.random_state)

        pair_ids = np.arange(n_pairs)
        if self.shuffle:
            rng.shuffle(pair_ids)

        fold_size = n_pairs // self.n_splits
        for fold_idx in range(self.n_splits):
            start = fold_idx * fold_size
            end = (
                n_pairs if fold_idx == self.n_splits - 1 else (fold_idx + 1) * fold_size
            )

            test_pair_ids = set(pair_ids[start:end])
            train_pair_ids = np.array(
                [pid for pid in pair_ids if pid not in test_pair_ids]
            )
            test_pair_ids_arr = np.array(sorted(list(test_pair_ids)))

            yield train_pair_ids, test_pair_ids_arr


# ── Direction computation ──────────────────────────────────────────────────
def compute_vulnerability_direction(
    safe_activations: np.ndarray, vuln_activations: np.ndarray, pca_dim: int = 50
) -> tuple[np.ndarray, PCA]:
    """
    Compute vulnerability direction as normalize(mean(vuln) - mean(safe)).

    Args:
        safe_activations: (n_pairs, d) secure samples
        vuln_activations: (n_pairs, d) vulnerable samples
        pca_dim: Apply PCA to this dimensionality

    Returns:
        direction: (pca_dim,) normalized unit vector
        pca: fitted PCA transformer
    """
    # Stack and apply PCA
    all_acts = np.vstack([safe_activations, vuln_activations])
    pca = PCA(n_components=pca_dim)
    acts_pca = pca.fit_transform(all_acts)

    safe_pca = acts_pca[: len(safe_activations)]
    vuln_pca = acts_pca[len(safe_activations) :]

    # Direction: vulnerable - secure
    direction = vuln_pca.mean(axis=0) - safe_pca.mean(axis=0)
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    return direction, pca


# ── Paired ranking evaluation ──────────────────────────────────────────────
def paired_ranking_accuracy(
    safe_activations: np.ndarray,
    vuln_activations: np.ndarray,
    test_pair_ids: np.ndarray,
    direction: np.ndarray,
    pca: PCA,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate pairwise ranking: for each pair, check if vulnerable > secure.

    Args:
        safe_activations: (n_pairs, d) full dataset of secure activations
        vuln_activations: (n_pairs, d) full dataset of vulnerable activations
        test_pair_ids: (n_test_pairs,) indices of pairs to evaluate
        direction: (pca_dim,) vulnerability direction
        pca: fitted PCA transformer

    Returns:
        accuracy: fraction of pairs where vulnerable > secure
        proj_vuln: projections of vulnerable samples
        proj_secure: projections of secure samples
    """
    all_acts = np.vstack([safe_activations, vuln_activations])
    acts_pca = pca.transform(all_acts)

    safe_pca = acts_pca[: len(safe_activations)]
    vuln_pca = acts_pca[len(safe_activations) :]

    correct = 0
    projections_vuln = []
    projections_secure = []

    for pair_id in test_pair_ids:
        proj_vuln = vuln_pca[pair_id] @ direction
        proj_secure = safe_pca[pair_id] @ direction

        projections_vuln.append(proj_vuln)
        projections_secure.append(proj_secure)

        if proj_vuln > proj_secure:
            correct += 1

    accuracy = correct / len(test_pair_ids)
    return accuracy, np.array(projections_vuln), np.array(projections_secure)


# ── Main analysis ──────────────────────────────────────────────────────────
def run_paired_ranking_task(
    run_dir: Path, layers: list[int], n_folds: int = 5, pca_dim: int = 50
) -> dict:
    """
    Run paired ranking task across specified layers.

    Returns:
        results: dict with per-layer accuracies and statistics
    """
    print(f"\n{'='*70}")
    print(f"PAIRED RANKING TASK — Validating 'Relative vs. Absolute' Claim")
    print(f"{'='*70}")
    print(f"Run directory: {run_dir}")
    print(f"Layers: {layers}")
    print(f"Folds: {n_folds}, PCA dim: {pca_dim}")
    print(f"{'='*70}\n")

    # Load metadata to get dataset size
    meta = load_metadata(run_dir)
    n_pairs = len(meta) if isinstance(meta, list) else meta.get("n_pairs", 2493)

    results = {}

    for layer in layers:
        print(f"\nLayer {layer}:")
        print("-" * 70)

        # Load activations
        safe_acts, vuln_acts = load_layer(run_dir, layer)
        assert len(safe_acts) == len(vuln_acts), "Mismatch in safe/vuln sample counts"

        # Run pair-stratified CV
        splitter = PairStratifiedKFold(n_splits=n_folds, random_state=SEED)
        fold_accuracies = []
        all_diffs = []

        for fold_idx, (train_pair_ids, test_pair_ids) in enumerate(
            splitter.split(len(safe_acts))
        ):
            # Extract fold data
            safe_train = safe_acts[train_pair_ids]
            vuln_train = vuln_acts[train_pair_ids]

            # Compute direction on training data
            direction, pca = compute_vulnerability_direction(
                safe_train, vuln_train, pca_dim
            )

            # Evaluate on test pairs
            accuracy, proj_vuln, proj_secure = paired_ranking_accuracy(
                safe_acts, vuln_acts, test_pair_ids, direction, pca
            )

            fold_accuracies.append(accuracy)
            all_diffs.extend(proj_vuln - proj_secure)

            print(f"  Fold {fold_idx}: {accuracy:.3f} ({len(test_pair_ids)} pairs)")

        # Summary statistics
        fold_accs = np.array(fold_accuracies)
        mean_acc = fold_accs.mean()
        std_acc = fold_accs.std()
        ci_95 = 1.96 * std_acc

        print(f"\n  Mean accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
        print(f"  95% CI: [{mean_acc - ci_95:.3f}, {mean_acc + ci_95:.3f}]")
        print(f"  Chance baseline: 0.500")
        print(f"  Paper claim (per-pair alignment): ~0.87")

        # Projection statistics
        diffs = np.array(all_diffs)
        frac_positive = np.mean(diffs > 0)
        print(f"  Fraction with vuln > secure: {frac_positive:.3f}")

        results[f"layer_{layer}"] = {
            "layer": layer,
            "mean_accuracy": float(mean_acc),
            "std_accuracy": float(std_acc),
            "ci_95": float(ci_95),
            "fold_accuracies": fold_accs.tolist(),
            "n_test_pairs": len(test_pair_ids),
            "frac_positive": float(frac_positive),
        }

    # ── Overall summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    layer_means = np.array([results[f"layer_{l}"]["mean_accuracy"] for l in layers])
    print(f"\nMean accuracy across all layers: {layer_means.mean():.3f}")
    print(f"Range: [{layer_means.min():.3f}, {layer_means.max():.3f}]")

    print(f"\nInterpretation:")
    print(f"  • 0.85–0.90: ✓ Relative ordering hypothesis STRONGLY SUPPORTED")
    print(f"  • 0.70–0.80: ⚠ Partial support; direction is informative but noisier")
    print(f"  • 0.50–0.60: ✗ Direction does NOT encode relative ordering")
    print(f"  • <0.50:     ✗ Direction is backwards or degenerate\n")

    return results


# ── CLI ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Paired ranking task for validating vulnerability direction"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="0,3,7,11,15,19,23,27",
        help="Comma-separated layer IDs to analyze",
    )
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--pca_dim", type=int, default=50, help="PCA dimensionality")
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="Path to activation run (auto-detected if not provided)",
    )

    args = parser.parse_args()

    # Parse layers
    layers = [int(x.strip()) for x in args.layers.split(",")]

    # Find run directory
    if args.run_dir is None:
        args.run_dir = find_latest_run()
        print(f"Auto-detected run: {args.run_dir}")
    else:
        args.run_dir = Path(args.run_dir)

    # Run analysis
    results = run_paired_ranking_task(
        args.run_dir, layers=layers, n_folds=args.n_folds, pca_dim=args.pca_dim
    )

    # Save results
    output_path = HERE.parent / "paired_ranking_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}\n")


if __name__ == "__main__":
    main()
