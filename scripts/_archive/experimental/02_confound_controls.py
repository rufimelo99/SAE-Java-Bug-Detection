#!/usr/bin/env python3
"""
Confound control analysis: residualize activations for length and guard-token frequency.

Tests whether the high cross-layer cosine similarity (≥0.99) of the vulnerability
direction is driven by sequence length or guard-token frequency confounds.

Performs:
1. Length residualization via linear regression
2. Guard-token frequency residualization
3. Recomputes cross-layer cosine similarity after control
4. Null distribution via label permutation
5. Whitened direction comparison
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class ConfoundControl:
    """Residualize activations against confounds."""

    @staticmethod
    def extract_confounds(
        code_samples: List[str], tokenizer
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract confound features from code samples.

        Args:
            code_samples: list of code strings
            tokenizer: tokenizer to count tokens

        Returns:
            lengths: (n_samples,) - sequence length in tokens
            guard_token_freq: (n_samples,) - frequency of guard tokens
        """
        guard_tokens = {
            "if",
            "NULL",
            "sizeof",
            "malloc",
            "free",
            "assert",
            "!=",
            "==",
            "<",
            ">",
        }

        lengths = np.array([len(tokenizer.encode(code)) for code in code_samples])

        guard_freqs = []
        for code in code_samples:
            tokens = set(tokenizer.tokenize(code))
            guard_count = len(tokens & guard_tokens)
            guard_freqs.append(guard_count)

        return lengths, np.array(guard_freqs)

    @staticmethod
    def residualize(
        activations: np.ndarray, confounds: np.ndarray, alpha: float = 1.0
    ) -> np.ndarray:
        """
        Residualize activations against confounds via Ridge regression.

        Args:
            activations: (n_samples, hidden_dim)
            confounds: (n_samples, n_confounds)
            alpha: Ridge regularization strength

        Returns:
            residualized: (n_samples, hidden_dim) - confound-removed activations
        """
        ridge = Ridge(alpha=alpha)
        ridge.fit(confounds, activations)
        predictions = ridge.predict(confounds)
        residualized = activations - predictions
        return residualized

    @staticmethod
    def compute_direction(
        activations_secure: np.ndarray, activations_vulnerable: np.ndarray
    ) -> np.ndarray:
        """Compute mean-difference direction."""
        d = activations_secure.mean(axis=0) - activations_vulnerable.mean(axis=0)
        d = d / np.linalg.norm(d)
        return d


def test_confound_controls(
    activations_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
    confounds: Dict[str, np.ndarray],
    labels: np.ndarray,
    layers: List[int],
) -> Dict:
    """
    Test whether vulnerability direction similarity persists after confound removal.

    Args:
        activations_dict: {layer: (activations_secure, activations_vulnerable)}
        confounds: {'length': ..., 'guard_freq': ...}
        labels: binary vulnerability labels
        layers: list of layer indices

    Returns:
        results dict with similarities before/after control
    """
    results = {
        "original_similarity": {},
        "length_residualized_similarity": {},
        "guard_residualized_similarity": {},
        "both_residualized_similarity": {},
        "whitened_similarity": {},
        "null_permutation_similarity": {},
    }

    # Split by label
    secure_mask = labels == 0
    vulnerable_mask = labels == 1

    # 1. Original directions and similarities
    print("Computing original cross-layer similarities...")
    original_directions = {}
    for layer in layers:
        act_s, act_v = activations_dict[layer]
        d = ConfoundControl.compute_direction(act_s, act_v)
        original_directions[layer] = d

    # Store pairwise similarities
    original_sims = []
    for i, l1 in enumerate(layers):
        for l2 in layers[i + 1 :]:
            sim = np.dot(original_directions[l1], original_directions[l2])
            original_sims.append(sim)
            results["original_similarity"][f"{l1}-{l2}"] = float(sim)

    print(f"  Mean original similarity: {np.mean(original_sims):.6f}")
    print(f"  Min/Max: {np.min(original_sims):.6f} / {np.max(original_sims):.6f}")

    # 2. Length-residualized directions
    print("\nResidualizing for sequence length...")
    length_confounds = confounds["length"].reshape(-1, 1)
    length_res_directions = {}
    for layer in layers:
        act_s, act_v = activations_dict[layer]
        act_s_res = ConfoundControl.residualize(
            act_s[secure_mask], length_confounds[secure_mask]
        )
        act_v_res = ConfoundControl.residualize(
            act_v[vulnerable_mask], length_confounds[vulnerable_mask]
        )
        d = ConfoundControl.compute_direction(act_s_res, act_v_res)
        length_res_directions[layer] = d

    length_sims = []
    for i, l1 in enumerate(layers):
        for l2 in layers[i + 1 :]:
            sim = np.dot(length_res_directions[l1], length_res_directions[l2])
            length_sims.append(sim)
            results["length_residualized_similarity"][f"{l1}-{l2}"] = float(sim)

    print(f"  Mean length-residualized similarity: {np.mean(length_sims):.6f}")
    print(
        f"  Reduction from original: {(1 - np.mean(length_sims) / np.mean(original_sims)) * 100:.2f}%"
    )

    # 3. Guard-token-residualized directions
    print("\nResidualizing for guard-token frequency...")
    guard_confounds = confounds["guard_freq"].reshape(-1, 1)
    guard_res_directions = {}
    for layer in layers:
        act_s, act_v = activations_dict[layer]
        act_s_res = ConfoundControl.residualize(
            act_s[secure_mask], guard_confounds[secure_mask]
        )
        act_v_res = ConfoundControl.residualize(
            act_v[vulnerable_mask], guard_confounds[vulnerable_mask]
        )
        d = ConfoundControl.compute_direction(act_s_res, act_v_res)
        guard_res_directions[layer] = d

    guard_sims = []
    for i, l1 in enumerate(layers):
        for l2 in layers[i + 1 :]:
            sim = np.dot(guard_res_directions[l1], guard_res_directions[l2])
            guard_sims.append(sim)
            results["guard_residualized_similarity"][f"{l1}-{l2}"] = float(sim)

    print(f"  Mean guard-residualized similarity: {np.mean(guard_sims):.6f}")
    print(
        f"  Reduction from original: {(1 - np.mean(guard_sims) / np.mean(original_sims)) * 100:.2f}%"
    )

    # 4. Both confounds residualized
    print("\nResidualizing for both length and guard frequency...")
    both_confounds = np.hstack([length_confounds, guard_confounds])
    both_res_directions = {}
    for layer in layers:
        act_s, act_v = activations_dict[layer]
        act_s_res = ConfoundControl.residualize(
            act_s[secure_mask], both_confounds[secure_mask]
        )
        act_v_res = ConfoundControl.residualize(
            act_v[vulnerable_mask], both_confounds[vulnerable_mask]
        )
        d = ConfoundControl.compute_direction(act_s_res, act_v_res)
        both_res_directions[layer] = d

    both_sims = []
    for i, l1 in enumerate(layers):
        for l2 in layers[i + 1 :]:
            sim = np.dot(both_res_directions[l1], both_res_directions[l2])
            both_sims.append(sim)
            results["both_residualized_similarity"][f"{l1}-{l2}"] = float(sim)

    print(f"  Mean both-residualized similarity: {np.mean(both_sims):.6f}")
    print(
        f"  Reduction from original: {(1 - np.mean(both_sims) / np.mean(original_sims)) * 100:.2f}%"
    )

    # 5. Whitened directions
    print("\nComputing whitened direction similarities...")
    whitened_directions = {}
    for layer in layers:
        d = original_directions[layer]
        # Whiten by subtracting mean and dividing by std
        d_whitened = (d - d.mean()) / (d.std() + 1e-8)
        whitened_directions[layer] = d_whitened / np.linalg.norm(d_whitened)

    whitened_sims = []
    for i, l1 in enumerate(layers):
        for l2 in layers[i + 1 :]:
            sim = np.dot(whitened_directions[l1], whitened_directions[l2])
            whitened_sims.append(sim)
            results["whitened_similarity"][f"{l1}-{l2}"] = float(sim)

    print(f"  Mean whitened similarity: {np.mean(whitened_sims):.6f}")

    # 6. Null distribution via label permutation
    print("\nComputing null distribution via label permutation (10 permutations)...")
    null_sims = []
    for perm_idx in range(10):
        perm_labels = np.random.permutation(labels)
        perm_secure_mask = perm_labels == 0
        perm_vulnerable_mask = perm_labels == 1

        perm_directions = {}
        for layer in layers:
            act_s, act_v = activations_dict[layer]
            d = ConfoundControl.compute_direction(
                act_s[perm_secure_mask], act_v[perm_vulnerable_mask]
            )
            perm_directions[layer] = d

        for i, l1 in enumerate(layers):
            for l2 in layers[i + 1 :]:
                sim = np.dot(perm_directions[l1], perm_directions[l2])
                null_sims.append(sim)

    null_sims = np.array(null_sims)
    results["null_permutation_mean"] = float(null_sims.mean())
    results["null_permutation_std"] = float(null_sims.std())

    print(f"  Mean null similarity: {null_sims.mean():.6f} ± {null_sims.std():.6f}")
    print(
        f"  Original mean is {np.mean(original_sims) / null_sims.mean():.2f}x null mean"
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: High cross-layer similarity is driven by:")
    print("=" * 70)
    length_pct = (1 - np.mean(length_sims) / np.mean(original_sims)) * 100
    guard_pct = (1 - np.mean(guard_sims) / np.mean(original_sims)) * 100
    both_pct = (1 - np.mean(both_sims) / np.mean(original_sims)) * 100

    print(f"  Length confound: {length_pct:.1f}% of original similarity")
    print(f"  Guard-token confound: {guard_pct:.1f}% of original similarity")
    print(f"  Both confounds: {both_pct:.1f}% of original similarity")
    print(f"  Remaining after both: {np.mean(both_sims):.6f}")

    return results


def main():
    """Run confound control analysis."""
    print("=" * 70)
    print("CONFOUND CONTROL ANALYSIS")
    print("=" * 70)

    # TODO: Load pre-computed activations and code samples
    # Expected: activations_dict, confounds, labels, layers

    print("\nTo use this script:")
    print("1. Load your activation tensors and code samples")
    print("2. Call extract_confounds() to get length and guard-token frequency")
    print("3. Call test_confound_controls() to residualize and recompute similarities")
    print("\nExample:")
    print("  confounds = {")
    print("    'length': extract_confounds(code_samples, tokenizer)[0],")
    print("    'guard_freq': extract_confounds(code_samples, tokenizer)[1],")
    print("  }")
    print(
        "  results = test_confound_controls(activations_dict, confounds, labels, layers)"
    )


if __name__ == "__main__":
    main()
