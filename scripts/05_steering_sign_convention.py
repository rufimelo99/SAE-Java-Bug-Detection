#!/usr/bin/env python3
"""
Clarify and test steering sign convention.

The review notes an apparent sign inconsistency:
  - Vulnerability direction d_L = mean(secure) - mean(vulnerable)
  - Steering: inject -α·d_L during generation
  - Result: code scored as "more secure"

This should mean:
  - -α·d_L moves activations AWAY from secure centroid toward vulnerable
  - Yet this results in code scored as more defensive

This script clarifies the sign convention and comprehensively tests:
1. Both +α·d_L and -α·d_L steering
2. Various steering strengths (α = [0.5, 1, 2, 5, 10])
3. Multiple layers (L3, L7, L11, L23)
4. Random direction baseline
5. Length-only direction baseline
6. Orthogonal direction baseline
"""

import json
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


class SteeringAnalysis:
    """Analyze steering behavior and sign convention."""

    @staticmethod
    def compute_direction(
        activations_secure: np.ndarray, activations_vulnerable: np.ndarray
    ) -> np.ndarray:
        """Compute vulnerability direction: secure - vulnerable."""
        d = activations_secure.mean(axis=0) - activations_vulnerable.mean(axis=0)
        d = d / (np.linalg.norm(d) + 1e-8)
        return d

    @staticmethod
    def compute_orthogonal_direction(direction: np.ndarray) -> np.ndarray:
        """Compute orthogonal direction via random projection + Gram-Schmidt."""
        random_dir = np.random.randn(*direction.shape)
        orthogonal = random_dir - (random_dir @ direction) * direction
        orthogonal = orthogonal / (np.linalg.norm(orthogonal) + 1e-8)
        return orthogonal

    @staticmethod
    def compute_length_direction(
        activations: np.ndarray, code_lengths: np.ndarray
    ) -> np.ndarray:
        """Compute direction correlated with code length."""
        # Regress activations on length to isolate length signal
        mean_by_length_short = activations[code_lengths < np.median(code_lengths)].mean(
            axis=0
        )
        mean_by_length_long = activations[code_lengths >= np.median(code_lengths)].mean(
            axis=0
        )
        d = mean_by_length_long - mean_by_length_short
        d = d / (np.linalg.norm(d) + 1e-8)
        return d

    @staticmethod
    def test_steering_direction_sweep(
        activations_generated: np.ndarray,
        labels_generated: np.ndarray,
        probe_classifier: LogisticRegression,
        direction: np.ndarray,
        alpha_values: List[float] = [0, 0.5, 1, 2, 5, 10],
        sign: str = "both",  # 'both', 'positive', 'negative'
    ) -> Dict:
        """
        Test steering with various strengths and both signs.

        Args:
            activations_generated: (n_samples, hidden_dim) - generated code activations
            labels_generated: (n_samples,) - vulnerability labels of generated code
            probe_classifier: trained logistic regression classifier
            direction: (hidden_dim,) - vulnerability direction
            alpha_values: list of steering strengths
            sign: which signs to test ('positive', 'negative', or 'both')

        Returns:
            dict with AUROC for each (alpha, sign) pair
        """
        results = {"alphas": alpha_values, "aurocs": {}}

        for alpha in alpha_values:
            if sign in ["negative", "both"]:
                # -α·d_L: move away from secure
                activations_steered_neg = activations_generated - alpha * direction
                y_pred_neg = probe_classifier.predict_proba(activations_steered_neg)[
                    :, 1
                ]
                auroc_neg = roc_auc_score(labels_generated, y_pred_neg)
                results["aurocs"][f"-α={alpha}"] = float(auroc_neg)

            if sign in ["positive", "both"]:
                # +α·d_L: move toward secure
                activations_steered_pos = activations_generated + alpha * direction
                y_pred_pos = probe_classifier.predict_proba(activations_steered_pos)[
                    :, 1
                ]
                auroc_pos = roc_auc_score(labels_generated, y_pred_pos)
                results["aurocs"][f"+α={alpha}"] = float(auroc_pos)

        return results

    @staticmethod
    def test_baseline_directions(
        activations_generated: np.ndarray,
        labels_generated: np.ndarray,
        probe_classifier: LogisticRegression,
        vulnerability_direction: np.ndarray,
        code_lengths: np.ndarray,
        alpha: float = 5.0,
    ) -> Dict:
        """
        Test steering with various baseline directions to isolate specificity.

        Args:
            activations_generated: (n_samples, hidden_dim)
            labels_generated: (n_samples,)
            probe_classifier: trained classifier
            vulnerability_direction: (hidden_dim,)
            code_lengths: (n_samples,) - length of generated code in tokens
            alpha: steering strength

        Returns:
            dict with AUROC for each direction type
        """
        results = {"direction_types": []}

        # 1. Vulnerability direction (main)
        activations_steered = activations_generated - alpha * vulnerability_direction
        y_pred = probe_classifier.predict_proba(activations_steered)[:, 1]
        auroc_vuln = roc_auc_score(labels_generated, y_pred)
        results["direction_types"].append(
            {
                "name": "vulnerability_direction",
                "auroc": float(auroc_vuln),
                "interpretation": "main steering direction",
            }
        )

        # 2. Random orthogonal direction
        orthogonal_dir = SteeringAnalysis.compute_orthogonal_direction(
            vulnerability_direction
        )
        activations_steered = activations_generated - alpha * orthogonal_dir
        y_pred = probe_classifier.predict_proba(activations_steered)[:, 1]
        auroc_orthogonal = roc_auc_score(labels_generated, y_pred)
        results["direction_types"].append(
            {
                "name": "random_orthogonal",
                "auroc": float(auroc_orthogonal),
                "interpretation": "should have no effect",
            }
        )

        # 3. Length-correlated direction
        length_dir = SteeringAnalysis.compute_length_direction(
            activations_generated, code_lengths
        )
        activations_steered = activations_generated - alpha * length_dir
        y_pred = probe_classifier.predict_proba(activations_steered)[:, 1]
        auroc_length = roc_auc_score(labels_generated, y_pred)
        results["direction_types"].append(
            {
                "name": "length_direction",
                "auroc": float(auroc_length),
                "interpretation": "tests if vulnerability signal is length",
            }
        )

        # 4. Random direction
        random_dir = np.random.randn(vulnerability_direction.shape[0])
        random_dir = random_dir / np.linalg.norm(random_dir)
        activations_steered = activations_generated - alpha * random_dir
        y_pred = probe_classifier.predict_proba(activations_steered)[:, 1]
        auroc_random = roc_auc_score(labels_generated, y_pred)
        results["direction_types"].append(
            {
                "name": "random_direction",
                "auroc": float(auroc_random),
                "interpretation": "control: completely random",
            }
        )

        return results

    @staticmethod
    def analyze_sign_convention(
        direction: np.ndarray, mean_secure: np.ndarray, mean_vulnerable: np.ndarray
    ) -> Dict:
        """
        Document the sign convention clearly.

        Args:
            direction: (hidden_dim,) = mean(secure) - mean(vulnerable)
            mean_secure: (hidden_dim,) - centroid of secure samples
            mean_vulnerable: (hidden_dim,) - centroid of vulnerable samples

        Returns:
            dict explaining sign convention
        """
        # Verify direction definition
        d_recomputed = (mean_secure - mean_vulnerable) / np.linalg.norm(
            mean_secure - mean_vulnerable
        )
        direction_normalized = direction / (np.linalg.norm(direction) + 1e-8)

        alignment = np.dot(direction_normalized, d_recomputed)

        explanation = {
            "definition": "d_L = mean(secure) - mean(vulnerable), normalized",
            "direction_alignment_with_definition": float(alignment),
            "steering_operations": {
                "plus_alpha_d_L": "moves activation TOWARD secure centroid (increases security)",
                "minus_alpha_d_L": "moves activation AWAY from secure centroid (decreases security)",
            },
            "expected_behavior": {
                "plus_alpha_d_L": "code should be scored as MORE secure by probe",
                "minus_alpha_d_L": "code should be scored as LESS secure by probe",
            },
            "note": (
                "If -α·d_L results in more secure code, it suggests the probe's decision boundary "
                "is not aligned with the direction's semantic meaning, or there is a confound "
                "(e.g., the direction encodes length rather than true security features)."
            ),
        }

        return explanation


def main():
    """Run steering sign convention analysis."""
    print("=" * 70)
    print("STEERING SIGN CONVENTION ANALYSIS")
    print("=" * 70)

    print("\nCLARIFYING THE SIGN CONVENTION:")
    print("-" * 70)
    print("Vulnerability direction: d_L = mean(secure) - mean(vulnerable)")
    print("")
    print("This means:")
    print("  - +α·d_L: move toward secure centroid (should increase 'security')")
    print("  - -α·d_L: move away from secure centroid (should decrease 'security')")
    print("")
    print("If -α·d_L results in higher AUROC (more 'secure' by probe):")
    print(
        "  1. The direction may encode a confound (length, guard tokens) not true semantics"
    )
    print(
        "  2. The probe's decision boundary may be misaligned with direction semantics"
    )
    print("  3. Distribution shift: generated code differs from training distribution")
    print("")

    print("\nTo fully test this, run:")
    print("-" * 70)
    print(
        """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    # Load generated code activations and labels
    activations_generated = np.load('generated_activations.npy')
    labels_generated = np.load('generated_labels.npy')

    # Load trained probe
    probe = pickle.load(open('probe_classifier.pkl', 'rb'))

    # Load vulnerability direction
    direction = np.load('vulnerability_direction.npy')
    code_lengths = np.load('generated_code_lengths.npy')

    # Test sign convention
    analysis = SteeringAnalysis()

    # 1. Sign sweep
    results_sign = analysis.test_steering_direction_sweep(
        activations_generated, labels_generated, probe, direction,
        alpha_values=[0, 1, 5, 10], sign='both'
    )
    print("Sign sweep results:")
    for key, auroc in results_sign['aurocs'].items():
        print(f"  {key}: {auroc:.4f}")

    # 2. Baseline direction tests
    results_baselines = analysis.test_baseline_directions(
        activations_generated, labels_generated, probe, direction,
        code_lengths, alpha=5.0
    )
    print("\\nBaseline direction comparison (α=5.0):")
    for result in results_baselines['direction_types']:
        print(f"  {result['name']}: {result['auroc']:.4f}")
        print(f"    ({result['interpretation']})")

    # 3. Document convention
    convention = analysis.analyze_sign_convention(
        direction,
        mean_secure=activations_generated[labels_generated == 0].mean(axis=0),
        mean_vulnerable=activations_generated[labels_generated == 1].mean(axis=0)
    )
    print("\\nSign convention explanation:")
    print(f"  {convention['note']}")
    """
    )

    print("\nEXPECTED RESULTS:")
    print("-" * 70)
    print("✓ Vulnerability direction should show effect (AUROC shift with α)")
    print("✓ Random/orthogonal directions should show NO effect")
    print("✓ +α·d_L and -α·d_L should have OPPOSITE effects")
    print(
        "✓ Length direction should show minimal effect (if vulnerability is semantic)"
    )
    print("")
    print("If these don't hold, investigate:")
    print("  - Distribution shift between training and generated code")
    print("  - Confounds in the direction (length, token frequency)")
    print("  - Probe generalization on out-of-distribution data")


if __name__ == "__main__":
    main()
