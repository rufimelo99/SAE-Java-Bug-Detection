"""
Feature-to-direction loading analysis (C code only).

For each of the 16,384 SAE features, computes Pearson correlation between:
- Feature activation delta (vulnerable - secure)
- Pair alignment with the vulnerability direction

Reports top 20 positively and negatively loaded features.

Usage:
    python feature_direction_loading_c_only.py

Outputs:
    fig_feature_direction_loading.pdf
    feature_direction_loading_c_results.json
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
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

TARGET_LAYER = 11
C_EXTS = {"c", "cc", "cpp", "h"}

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
})


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
    """Load raw activations."""
    safe = torch.load(run_dir / f"safe_layer_{layer}.pt", weights_only=True).numpy()
    vuln = torch.load(run_dir / f"vulnerable_layer_{layer}.pt", weights_only=True).numpy()
    return safe, vuln


def main():
    print("Feature-to-direction loading analysis (C code only)")
    
    # Load metadata and apply C filter
    run_dir = find_mean_pool_run()
    with (run_dir / "meta.json").open() as f:
        meta = json.load(f)
    
    extensions = [r["file_extension"] for r in meta]
    c_mask = np.array([ext in C_EXTS for ext in extensions])
    n_c_pairs = c_mask.sum()
    
    print(f"Total pairs: {len(meta)}")
    print(f"C pairs: {n_c_pairs}")
    
    # Load activations
    safe, vuln = load_activations(run_dir, TARGET_LAYER)
    safe_c = safe[c_mask]
    vuln_c = vuln[c_mask]
    
    # Compute direction
    d_L = (safe_c.mean(0) - vuln_c.mean(0))
    d_L = d_L / np.linalg.norm(d_L)
    
    # Per-pair alignment
    delta_c = safe_c - vuln_c
    alignment = delta_c @ d_L
    
    # Top loaded features (correlation between feature delta and alignment)
    feature_deltas = vuln_c - safe_c  # vuln -> secure direction
    correlations = np.array([
        pearsonr(feature_deltas[:, i], alignment)[0] 
        for i in range(feature_deltas.shape[1])
    ])
    
    top_pos_idx = np.argsort(-correlations)[:20]
    top_neg_idx = np.argsort(correlations)[:20]
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top positive features
    ax = axes[0]
    y_pos = np.arange(len(top_pos_idx))
    ax.barh(y_pos, correlations[top_pos_idx], color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Feature {i}" for i in top_pos_idx], fontsize=8)
    ax.set_xlabel("Correlation with Vulnerability Direction", fontsize=9)
    ax.set_title("Top 20 Positively-Loaded SAE Features (Secure-Enriched)", fontsize=10, weight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    
    # Top negative features
    ax = axes[1]
    y_pos = np.arange(len(top_neg_idx))
    ax.barh(y_pos, correlations[top_neg_idx], color="coral")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Feature {i}" for i in top_neg_idx], fontsize=8)
    ax.set_xlabel("Correlation with Vulnerability Direction", fontsize=9)
    ax.set_title("Top 20 Negatively-Loaded SAE Features (Vulnerable-Enriched)", fontsize=10, weight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PAPER_FIGS / "fig_feature_direction_loading.pdf", dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {PAPER_FIGS / 'fig_feature_direction_loading.pdf'}")
    
    # Save results
    results = {
        "n_c_pairs": int(n_c_pairs),
        "target_layer": TARGET_LAYER,
        "top_positive_features": top_pos_idx.tolist(),
        "top_positive_correlations": correlations[top_pos_idx].tolist(),
        "top_negative_features": top_neg_idx.tolist(),
        "top_negative_correlations": correlations[top_neg_idx].tolist(),
    }
    
    with open(PAPER_FIGS / "feature_direction_loading_c_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved: {PAPER_FIGS / 'feature_direction_loading_c_results.json'}")


if __name__ == "__main__":
    main()
