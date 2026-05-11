#!/usr/bin/env python3
"""
SAE feature stability analysis across seeds and hyperparameters.

Tests whether SAE-learned features are stable and interpretable:
1. Cross-seed stability (Jaccard overlap, rank correlations)
2. Monosemanticity metrics (polysemanticity, activation patterns)
3. Ablations on latent size and sparsity
4. Feature persistence across layers
5. Load onto vulnerability direction across random initializations
"""

import json
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import kendalltau, spearmanr


class SAEStabilityAnalysis:
    """Analyze stability of SAE-learned features."""

    @staticmethod
    def jaccard_overlap(
        features_a: List[int], features_b: List[int], top_k: int = 20
    ) -> float:
        """
        Compute Jaccard overlap between top-k features from two runs.

        Args:
            features_a: feature indices sorted by importance (run A)
            features_b: feature indices sorted by importance (run B)
            top_k: number of top features to compare

        Returns:
            jaccard: overlap / union
        """
        top_a = set(features_a[:top_k])
        top_b = set(features_b[:top_k])
        intersection = len(top_a & top_b)
        union = len(top_a | top_b)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def rank_correlation(
        ranks_a: np.ndarray, ranks_b: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute Spearman and Kendall rank correlations between two feature rankings.

        Args:
            ranks_a: (n_features,) - ranks from run A
            ranks_b: (n_features,) - ranks from run B

        Returns:
            spearman_rho, kendall_tau
        """
        rho, _ = spearmanr(ranks_a, ranks_b)
        tau, _ = kendalltau(ranks_a, ranks_b)
        return float(rho), float(tau)

    @staticmethod
    def compute_polysemanticity(
        feature_activations: np.ndarray,
        feature_interpretations: Dict[int, str],
        top_k: int = 20,
    ) -> Dict:
        """
        Measure polysemanticity: do top features have clearly separate semantic roles?

        Approximation: compute semantic diversity of top-k features by checking
        if interpretations cluster into distinct categories.

        Args:
            feature_activations: (n_samples, n_features) - activation patterns
            feature_interpretations: {feature_idx: interpretation_str}
            top_k: number of top features to analyze

        Returns:
            dict with polysemanticity metrics
        """
        # Sort features by activation variance (proxy for importance)
        variances = feature_activations.var(axis=0)
        top_features = np.argsort(-variances)[:top_k]

        # Count distinct interpretations among top features
        interpretations = [
            feature_interpretations.get(f, "unknown") for f in top_features
        ]
        unique_interpretations = len(set(interpretations))

        return {
            "top_k": top_k,
            "unique_interpretations": unique_interpretations,
            "diversity_ratio": unique_interpretations / top_k,
            "interpretations": interpretations,
        }

    @staticmethod
    def direction_loading_stability(
        sae_features_runs: List[np.ndarray],
        direction: np.ndarray,
        decoder_weights_runs: List[np.ndarray],
        top_k: int = 20,
    ) -> Dict:
        """
        Test whether top features loading onto direction are stable across seeds.

        Args:
            sae_features_runs: list of (n_samples, n_features) arrays from different seeds
            direction: (hidden_dim,) - vulnerability direction
            decoder_weights_runs: list of (n_features, hidden_dim) decoder matrices
            top_k: number of top features to track

        Returns:
            dict with stability metrics
        """
        num_runs = len(sae_features_runs)
        all_loadings = []

        for run_idx in range(num_runs):
            features = sae_features_runs[run_idx]
            decoder = decoder_weights_runs[run_idx]

            # Compute loading of each feature onto direction
            # loading[k] = feature_k's contribution to moving along direction
            loadings = decoder @ direction  # (n_features,)
            all_loadings.append(loadings)

        all_loadings = np.array(all_loadings)  # (num_runs, n_features)

        # Identify top-k features in each run
        top_features_per_run = [
            np.argsort(-all_loadings[i])[:top_k] for i in range(num_runs)
        ]

        # Compute pairwise Jaccard overlap
        jaccard_overlaps = []
        for i in range(num_runs):
            for j in range(i + 1, num_runs):
                overlap = SAEStabilityAnalysis.jaccard_overlap(
                    top_features_per_run[i].tolist(),
                    top_features_per_run[j].tolist(),
                    top_k=top_k,
                )
                jaccard_overlaps.append(overlap)

        # Compute rank correlation across runs
        rank_correlations = []
        for i in range(num_runs):
            for j in range(i + 1, num_runs):
                ranks_i = np.argsort(-all_loadings[i])
                ranks_j = np.argsort(-all_loadings[j])
                rho, tau = SAEStabilityAnalysis.rank_correlation(ranks_i, ranks_j)
                rank_correlations.append({"spearman": rho, "kendall": tau})

        return {
            "mean_jaccard_overlap": np.mean(jaccard_overlaps),
            "mean_spearman_rho": np.mean([rc["spearman"] for rc in rank_correlations]),
            "mean_kendall_tau": np.mean([rc["kendall"] for rc in rank_correlations]),
            "top_k": top_k,
            "num_runs": num_runs,
        }

    @staticmethod
    def ablate_latent_size(
        activations: np.ndarray,
        labels: np.ndarray,
        latent_sizes: List[int] = [4096, 8192, 16384, 32768],
        sparsity_coeff: float = 0.01,
    ) -> Dict:
        """
        Test SAE quality (reconstruction error, sparsity) across latent sizes.

        Args:
            activations: (n_samples, hidden_dim)
            labels: not used, but kept for consistency
            latent_sizes: list of latent dimensions to test
            sparsity_coeff: L1 sparsity coefficient

        Returns:
            dict with ablation results per latent size
        """
        results = {}

        for latent_size in latent_sizes:
            # Simulate SAE training (placeholder)
            encoder = np.random.randn(
                hidden_dim := activations.shape[1], latent_size
            ) / np.sqrt(hidden_dim)
            decoder = np.random.randn(latent_size, hidden_dim) / np.sqrt(latent_size)

            # Encode
            latents = activations @ encoder  # (n_samples, latent_size)
            latents_sparse = np.where(
                np.abs(latents) > np.percentile(np.abs(latents), 90), latents, 0
            )

            # Reconstruct
            reconstructed = latents_sparse @ decoder
            recon_mse = ((activations - reconstructed) ** 2).mean()

            # Sparsity
            sparsity = (latents_sparse == 0).mean()

            results[latent_size] = {
                "reconstruction_mse": float(recon_mse),
                "sparsity": float(sparsity),
                "mean_latent_activation": float(np.abs(latents).mean()),
            }

        return results

    @staticmethod
    def feature_persistence_across_layers(
        feature_importances_per_layer: Dict[int, np.ndarray],
        layers: List[int],
        top_k: int = 20,
    ) -> Dict:
        """
        Check whether top features persist across layers.

        Args:
            feature_importances_per_layer: {layer: (n_features,) importance scores}
            layers: list of layer indices
            top_k: number of top features per layer

        Returns:
            dict with persistence metrics
        """
        top_features_per_layer = {
            layer: set(np.argsort(-feature_importances_per_layer[layer])[:top_k])
            for layer in layers
        }

        # Compute pairwise Jaccard overlap
        overlaps = {}
        for i, l1 in enumerate(layers):
            for l2 in layers[i + 1 :]:
                overlap = len(
                    top_features_per_layer[l1] & top_features_per_layer[l2]
                ) / (
                    len(top_features_per_layer[l1] | top_features_per_layer[l2]) + 1e-8
                )
                overlaps[f"{l1}-{l2}"] = float(overlap)

        return {
            "layer_pairs": overlaps,
            "mean_overlap": float(np.mean(list(overlaps.values()))),
            "top_k": top_k,
        }


def main():
    """Run SAE stability analysis."""
    print("=" * 70)
    print("SAE FEATURE STABILITY ANALYSIS")
    print("=" * 70)

    print("\nTo use this script:")
    print("\n1. CROSS-SEED STABILITY:")
    print("   - Train SAE multiple times with different random seeds")
    print("   - Compute feature loadings onto vulnerability direction for each seed")
    print("   - Call direction_loading_stability() to measure consistency")
    print("   - Look for mean Jaccard overlap > 0.7 and Spearman rho > 0.8")

    print("\n2. MONOSEMANTICITY:")
    print(
        "   - For each SAE feature, manually inspect activations and assign semantic role"
    )
    print("   - Call compute_polysemanticity() to measure interpretation diversity")
    print(
        "   - Report diversity_ratio (should be close to 1 for monosemantic features)"
    )

    print("\n3. LATENT SIZE ABLATION:")
    print("   - Train SAEs with latent sizes: [4096, 8192, 16384, 32768]")
    print("   - Call ablate_latent_size() to compare reconstruction vs. sparsity")
    print("   - Look for sweet spot where reconstruction MSE is low and sparsity high")

    print("\n4. LAYER PERSISTENCE:")
    print(
        "   - For each layer, rank SAE features by importance (e.g., correlation with label)"
    )
    print("   - Call feature_persistence_across_layers() to check persistence")
    print("   - Report mean overlap across layer pairs")

    print("\nExample code:")
    print(
        """
    # Example for cross-seed stability
    import numpy as np
    from pathlib import Path

    sae_features_runs = [
        np.load(f'sae_features_seed_{seed}.npy') for seed in [0, 1, 2]
    ]
    direction = np.load('vulnerability_direction.npy')
    decoder_weights_runs = [
        np.load(f'sae_decoder_seed_{seed}.npy') for seed in [0, 1, 2]
    ]

    results = SAEStabilityAnalysis.direction_loading_stability(
        sae_features_runs, direction, decoder_weights_runs, top_k=20
    )
    print(f"Mean Jaccard overlap: {results['mean_jaccard_overlap']:.3f}")
    print(f"Mean Spearman rho: {results['mean_spearman_rho']:.3f}")
    """
    )


if __name__ == "__main__":
    main()
