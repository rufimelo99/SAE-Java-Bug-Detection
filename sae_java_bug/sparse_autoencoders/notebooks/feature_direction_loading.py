"""
Feature-to-direction loading analysis.

For each of the 16,384 SAE features, computes Pearson correlation between:
- Feature activation delta (vulnerable - secure)
- Pair alignment with the vulnerability direction

Reports top 20 positively and negatively loaded features, with interpretations
of what they may represent.

Usage:
    python feature_direction_loading.py

Outputs:
    feature_direction_loading_results.json — Full correlation matrix
    Table of top loaded features (for Paper 1 Section 5)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
GITHUB = Path(__file__).parents[4]

ARTIFACTS = Path(__file__).parents[2] / "artifacts" / "activations"
PAPER_FIGS = (
    GITHUB
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

TARGET_LAYER = 11  # Layer with strongest vulnerability signal


def find_mean_pool_run() -> Path:
    """Find latest DeltaSecommits mean_pool run."""
    runs = sorted((ARTIFACTS / "mean_pool").glob("*/meta.json"))
    if not runs:
        raise FileNotFoundError("No mean_pool runs found")
    return runs[-1].parent


def find_sae_run() -> Path:
    """Find latest SAE features run."""
    runs = sorted((ARTIFACTS / "mean_pool_sae").glob("*/"))
    if not runs:
        raise FileNotFoundError("No SAE runs found")
    return runs[-1]


def load_activations(run_dir: Path, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Load raw and SAE activations."""
    safe = torch.load(run_dir / f"safe_layer_{layer}.pt", weights_only=True).numpy()
    vuln = torch.load(
        run_dir / f"vulnerable_layer_{layer}.pt", weights_only=True
    ).numpy()
    return safe, vuln


def load_sae_activations(sae_dir: Path, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Load SAE feature activations (dictionary output)."""
    # Load SAE feature activations if available
    safe_sae_file = sae_dir / f"safe_layer_{layer}_sae_features.pt"
    vuln_sae_file = sae_dir / f"vulnerable_layer_{layer}_sae_features.pt"

    if safe_sae_file.exists() and vuln_sae_file.exists():
        safe_sae = torch.load(safe_sae_file, weights_only=True).numpy()
        vuln_sae = torch.load(vuln_sae_file, weights_only=True).numpy()
        return safe_sae, vuln_sae

    raise FileNotFoundError("SAE activations not found")


def compute_direction(safe: np.ndarray, vuln: np.ndarray) -> np.ndarray:
    """Compute unit vulnerability direction."""
    d = vuln.mean(0) - safe.mean(0)
    d = d / (np.linalg.norm(d) + 1e-10)
    return d


def compute_alignment(delta: np.ndarray, direction: np.ndarray) -> float:
    """Compute cosine alignment."""
    norm_delta = np.linalg.norm(delta)
    norm_dir = np.linalg.norm(direction)
    if norm_delta < 1e-10 or norm_dir < 1e-10:
        return 0.0
    return np.dot(delta, direction) / (norm_delta * norm_dir)


def run_feature_loading_analysis() -> dict:
    """Analyze which features load onto vulnerability direction."""
    print("Loading activations...")
    run_dir = find_mean_pool_run()
    safe, vuln = load_activations(run_dir, TARGET_LAYER)

    # Compute direction
    direction = compute_direction(safe, vuln)
    print(f"Direction norm: {np.linalg.norm(direction):.3f}")

    # Per-pair deltas and alignments
    deltas = vuln - safe
    alignments = np.array(
        [compute_alignment(deltas[i], direction) for i in range(len(deltas))]
    )

    print(f"Mean alignment: {alignments.mean():.3f}")
    print(f"Fraction aligned: {(alignments > 0).mean():.1%}")

    # Try to load SAE features if available
    sae_dir = None
    try:
        sae_dir = find_sae_run()
        print("\nLoading SAE features...")
        safe_sae, vuln_sae = load_sae_activations(sae_dir, TARGET_LAYER)

        # Per-feature correlation with alignment
        correlations = []
        for feat_idx in range(safe_sae.shape[1]):
            feature_delta = vuln_sae[:, feat_idx] - safe_sae[:, feat_idx]
            corr, pval = pearsonr(feature_delta, alignments)
            correlations.append(
                {"feature": feat_idx, "correlation": corr, "pvalue": pval}
            )

        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values("correlation", ascending=False)

        results = {
            "layer": TARGET_LAYER,
            "n_features": safe_sae.shape[1],
            "top_positive": corr_df.head(20)[["feature", "correlation"]].to_dict(
                "records"
            ),
            "top_negative": corr_df.tail(20)[["feature", "correlation"]].to_dict(
                "records"
            ),
            "all_correlations": corr_df.to_dict("records"),
        }

        return results

    except FileNotFoundError:
        print("\nNote: SAE features not found. Using raw activations instead.")
        print(
            "For mechanistic analysis, extract SAE features with mean_pool_sae_probe.py"
        )

        # Fallback: analyze raw activation dimensions
        results = {
            "layer": TARGET_LAYER,
            "n_features": safe.shape[1],
            "note": "Using raw residual stream dimensions (no SAE disentanglement)",
            "n_positively_aligned": (alignments > 0).sum(),
            "n_negatively_aligned": (alignments < 0).sum(),
            "alignment_distribution": {
                "mean": float(alignments.mean()),
                "std": float(alignments.std()),
                "min": float(alignments.min()),
                "max": float(alignments.max()),
            },
        }

        return results


def print_results(results: dict):
    """Print feature loading results."""
    print("\n" + "=" * 70)
    print(f"FEATURE LOADING ANALYSIS — Layer {results['layer']}")
    print("=" * 70)

    if "note" in results:
        print(f"\nNote: {results['note']}")
        print(f"\nAlignment distribution:")
        for key, val in results["alignment_distribution"].items():
            print(f"  {key}: {val:.4f}")
    else:
        print(f"\nTotal features analyzed: {results['n_features']}")
        print(f"\nTOP 20 POSITIVELY LOADED FEATURES:")
        print("(highest correlation with alignment)")
        df_pos = pd.DataFrame(results["top_positive"])
        print(df_pos.to_string(index=False))

        print(f"\nTOP 20 NEGATIVELY LOADED FEATURES:")
        print("(lowest correlation with alignment)")
        df_neg = pd.DataFrame(results["top_negative"])
        print(df_neg.to_string(index=False))

        # Summary stats
        all_corrs = np.array([r["correlation"] for r in results["all_correlations"]])
        print(f"\nCorrelation statistics:")
        print(f"  Mean: {all_corrs.mean():.4f}")
        print(f"  Std: {all_corrs.std():.4f}")
        print(f"  Max: {all_corrs.max():.4f}")
        print(f"  Min: {all_corrs.min():.4f}")


def save_results(results: dict):
    """Save results to JSON."""
    out_file = PAPER_FIGS / "feature_direction_loading_results.json"
    with open(out_file, "w") as f:
        # Convert numpy types to native Python for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        json.dump(results, f, indent=2, default=convert)

    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    print("Feature-to-Direction Loading Analysis")
    print("=" * 70)
    print()

    results = run_feature_loading_analysis()
    print_results(results)
    save_results(results)

    print("\nDone.")
