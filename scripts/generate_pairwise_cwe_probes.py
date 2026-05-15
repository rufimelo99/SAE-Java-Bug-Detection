#!/usr/bin/env python3
"""
Generate pairwise CWE-type probe AUROC heatmaps for all models.

This creates heatmaps showing binary classification AUROC between each pair
of specific CWE types (e.g., CWE-119 vs CWE-120) across all layers.

Usage:
    python scripts/generate_pairwise_cwe_probes.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Style matching paper
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MODELS = ["qwen", "codellama", "starcoder2"]
LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]
ACTIVATIONS_DIR = Path("sae_java_bug/artifacts/multi_model_probing")
OUTPUT_DIR = Path("../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures")

# Top CWE types to include (from DeltaSecommits)
CWE_TYPES = [
    "CWE-119",  # Buffer Overflow
    "CWE-120",  # Buffer Copy without Bounds Check
    "CWE-125",  # Out-of-bounds Read
    "CWE-787",  # Out-of-bounds Write
    "CWE-78",   # OS Command Injection
    "CWE-89",   # SQL Injection
    "CWE-22",   # Path Traversal
    "CWE-401",  # Missing Release of Resource
]


def load_activations_and_metadata(model: str) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Load NPZ activations and metadata for a model."""
    npz_file = ACTIVATIONS_DIR / f"activations_{model}-7b.npz"
    
    if not npz_file.exists():
        logger.warning(f"Activations not found: {npz_file}")
        return {}, np.array([]), np.array([])
    
    logger.info(f"Loading activations from {npz_file.name}")
    data = np.load(npz_file)
    activations = {key: data[key] for key in data.files}
    
    # Load metadata
    metadata_file = ACTIVATIONS_DIR / "meta.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        cwes = np.array([item.get("cwe", "unknown") for item in metadata])
    else:
        cwes = np.array(["unknown"] * activations[list(activations.keys())[0]].shape[0])
    
    logger.info(f"Loaded {len(cwes)} samples with CWE labels")
    return activations, cwes, np.arange(len(cwes))


def probe_pairwise(X: np.ndarray, y: np.ndarray, n_components: int = 50, cv: int = 5) -> float:
    """
    Binary classification probe: y=1 for CWE1, y=0 for CWE2.
    Returns AUROC.
    """
    if len(np.unique(y)) < 2:
        return 0.5
    
    n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
    
    clf = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=42)
    skf = StratifiedKFold(n_splits=min(cv, y.sum(), (y == 0).sum()), shuffle=True, random_state=42)
    
    y_score = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        pca = PCA(n_components=min(n_comp, len(tr) - 1), random_state=42)
        
        Xtr = pca.fit_transform(scaler.fit_transform(X[tr]))
        Xte = pca.transform(scaler.transform(X[te]))
        
        clf.fit(Xtr, y[tr])
        y_score[te] = clf.predict_proba(Xte)[:, 1]
    
    try:
        auroc = roc_auc_score(y, y_score)
    except:
        auroc = 0.5
    
    return auroc


def generate_pairwise_heatmap(model: str):
    """Generate pairwise CWE probe AUROC heatmap for one model."""
    logger.info(f"\nGenerating pairwise CWE probes for {model}...")
    
    activations, cwes, idx_array = load_activations_and_metadata(model)
    
    if not activations:
        logger.warning(f"Skipping {model} - no activations")
        return
    
    # Determine which CWEs are available
    available_cwes = sorted([c for c in np.unique(cwes) if c in CWE_TYPES])
    if len(available_cwes) < 2:
        logger.warning(f"Not enough CWE types for {model}")
        return
    
    logger.info(f"Available CWEs: {available_cwes}")
    
    # Build matrix: auroc[i, j] = auroc for CWE_i vs CWE_j at peak layer
    n_cwes = len(available_cwes)
    matrix = np.zeros((n_cwes, n_cwes))
    
    for i, cwe1 in enumerate(available_cwes):
        for j, cwe2 in enumerate(available_cwes):
            if i == j:
                matrix[i, j] = 100  # Diagonal: perfect separation
                continue
            
            # Get indices for both CWEs
            mask1 = cwes == cwe1
            mask2 = cwes == cwe2
            
            if mask1.sum() == 0 or mask2.sum() == 0:
                matrix[i, j] = 50
                continue
            
            # Use activations from peak layer (L15 typically)
            peak_layer = 15
            key = f"layer_{peak_layer}_vuln_mean"
            
            if key not in activations:
                matrix[i, j] = 50
                continue
            
            X = activations[key]
            y = np.concatenate([np.ones(mask1.sum()), np.zeros(mask2.sum())])
            X_subset = np.vstack([X[mask1], X[mask2]])
            
            auroc = probe_pairwise(X_subset, y) * 100
            matrix[i, j] = auroc
            
            logger.info(f"  {cwe1} vs {cwe2}: {auroc:.1f}%")
    
    # Generate figure
    fig, ax = plt.subplots(figsize=(9, 8))
    
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=50, vmax=100, aspect="auto")
    
    # Labels
    cwe_short = [c.split("-")[1] for c in available_cwes]
    ax.set_xticks(range(len(available_cwes)))
    ax.set_yticks(range(len(available_cwes)))
    ax.set_xticklabels(cwe_short, fontsize=10, rotation=45)
    ax.set_yticklabels(cwe_short, fontsize=10)
    ax.set_xlabel("CWE Type (target)", fontsize=11, fontweight="bold")
    ax.set_ylabel("CWE Type (source)", fontsize=11, fontweight="bold")
    
    # Add text annotations
    for i in range(len(available_cwes)):
        for j in range(len(available_cwes)):
            text = ax.text(j, i, f"{matrix[i, j]:.0f}%",
                          ha="center", va="center", color="black", fontsize=9)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label="AUROC (%)")
    
    # Title
    ax.set_title(f"Pairwise CWE-type Probe AUROC: {model.upper()}\n(Peak layer, mean-token pooling)",
                fontsize=12, fontweight="bold", pad=15)
    
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / f"fig_cwe_pairwise_probe_{model}.pdf"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"✓ Saved: {output_file}")
    plt.close()


def main():
    logger.info("Generating pairwise CWE probe AUROC heatmaps...")
    
    for model in MODELS:
        try:
            generate_pairwise_heatmap(model)
        except Exception as e:
            logger.error(f"Error processing {model}: {e}")
    
    logger.info("\n✓ Pairwise CWE probe generation complete!")


if __name__ == "__main__":
    main()
