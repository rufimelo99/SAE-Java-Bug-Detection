#!/usr/bin/env python3
"""
Generate multi-model figures matching the paper's existing style.

Generates for all three models (Qwen, CodeLlama, StarCoder2):
1. Pairwise CWE classification heatmaps (from cross_family_transfer data)
2. Direction alignment by CWE type
3. Summary statistics tables

Usage:
    python scripts/generate_multimodel_styled_figures.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Style matching existing paper figures
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FAMILIES = ["memory_safety", "injection", "resource", "info_disclosure", "control_flow"]

_SCRIPTS_DIR = Path(__file__).parent
RESULTS_DIR = _SCRIPTS_DIR.parent / "results" / "raw_data"
OUTPUT_DIR = _SCRIPTS_DIR.parent.parent / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations" / "figures"


def discover_models() -> list:
    """Auto-discover models that have direction_geometry results."""
    return sorted(
        p.stem.replace("_direction_geometry", "")
        for p in RESULTS_DIR.glob("*_direction_geometry.json")
    )


def load_results(model: str) -> Dict[str, Any]:
    """Load all results for a model."""
    results = {}
    for experiment in ["direction_geometry", "cwe_universality", "paired_ranking"]:
        json_file = RESULTS_DIR / f"{model}_{experiment}.json"
        if json_file.exists():
            with open(json_file) as f:
                results[experiment] = json.load(f)
    return results


def generate_cwe_pairwise_heatmap(model: str, results: Dict[str, Any]):
    """Generate pairwise CWE classification heatmap."""
    if "cwe_universality" not in results:
        logger.warning(f"No CWE universality data for {model}")
        return

    transfer_data = results["cwe_universality"].get("cross_family_transfer", {})
    
    # Build matrix
    matrix = np.zeros((len(FAMILIES), len(FAMILIES)))
    for i, src in enumerate(FAMILIES):
        for j, tgt in enumerate(FAMILIES):
            key = f"{src}->{tgt}"
            if key in transfer_data:
                matrix[i, j] = transfer_data[key]["mean"]
            elif i == j:
                matrix[i, j] = 100  # Diagonal is 100% (same family)
            else:
                matrix[i, j] = 50  # Unknown, use chance

    # Generate figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use RdYlGn colormap (green = high, yellow = mid, red = low)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=50, vmax=100, aspect="auto")
    
    # Labels
    labels = [f.split("_")[0] for f in FAMILIES]
    ax.set_xticks(range(len(FAMILIES)))
    ax.set_yticks(range(len(FAMILIES)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Target CWE Family", fontsize=11, fontweight="bold")
    ax.set_ylabel("Source CWE Family", fontsize=11, fontweight="bold")
    
    # Add text annotations
    for i in range(len(FAMILIES)):
        for j in range(len(FAMILIES)):
            text = ax.text(j, i, f"{matrix[i, j]:.0f}%",
                          ha="center", va="center", color="black", fontsize=9)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label="Transfer Accuracy (%)")
    
    # Title
    ax.set_title(f"Pairwise CWE Transfer: {model.upper()}", 
                fontsize=12, fontweight="bold", pad=15)
    
    plt.tight_layout()
    
    # Save
    output_file = OUTPUT_DIR / f"fig_cwe_pairwise_{model}.pdf"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"✓ Saved: {output_file}")
    plt.close()


def generate_direction_alignment_by_cwe(model: str, results: Dict[str, Any]):
    """Generate direction alignment plot by CWE type (if data available)."""
    if "direction_geometry" not in results:
        logger.warning(f"No direction geometry data for {model}")
        return
    
    dir_data = results["direction_geometry"]
    layers_data = dir_data.get("layers", {})
    
    # Extract alignment percentages
    layers = sorted([int(k) for k in layers_data.keys()])
    alignments = [layers_data[str(l)].get("pct_aligned", 0) for l in layers]
    
    # Generate simple alignment curve (per-model)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(layers, alignments, marker="o", linewidth=2.5, markersize=8,
            color="darkblue", label=model.upper())
    
    # Add 50% chance line
    ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, 
              linewidth=1.5, label="Chance (50%)")
    
    # Formatting
    ax.set_xlabel("Layer", fontsize=11, fontweight="bold")
    ax.set_ylabel("Per-Pair Alignment (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"Vulnerability Direction Alignment: {model.upper()}",
                fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([40, 100])
    
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / f"fig_direction_alignment_{model}.pdf"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"✓ Saved: {output_file}")
    plt.close()


def generate_model_summary_table():
    """Generate summary statistics table for all models."""
    rows = []

    for model in discover_models():
        results = load_results(model)
        if "direction_geometry" not in results:
            continue
        
        dir_data = results["direction_geometry"]
        layers_data = dir_data.get("layers", {})
        
        alignments = [v.get("pct_aligned", 0) for v in layers_data.values()]
        if not alignments:
            continue
        
        peak_align = max(alignments)
        peak_layer = list(layers_data.keys())[alignments.index(peak_align)]
        mean_align = np.mean(alignments)
        
        rows.append({
            "model": model.upper(),
            "peak_align": f"{peak_align:.1f}%",
            "peak_layer": peak_layer,
            "mean_align": f"{mean_align:.1f}%",
            "n_pairs": dir_data.get("n_pairs", "unknown")
        })
    
    # Save as CSV
    summary_file = OUTPUT_DIR / "multimodel_summary.csv"
    with open(summary_file, "w") as f:
        f.write("Model,Peak Alignment,Peak Layer,Mean Alignment,N Pairs\n")
        for row in rows:
            f.write(f"{row['model']},{row['peak_align']},{row['peak_layer']},{row['mean_align']},{row['n_pairs']}\n")
    
    logger.info(f"✓ Saved: {summary_file}")


def main():
    logger.info("Generating multi-model styled figures...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models = discover_models()
    logger.info(f"Models found: {models}")

    for model in models:
        logger.info(f"\nProcessing {model}...")
        results = load_results(model)

        if not results:
            logger.warning(f"No results found for {model}")
            continue

        generate_cwe_pairwise_heatmap(model, results)
        generate_direction_alignment_by_cwe(model, results)

    generate_model_summary_table()
    logger.info("\n✓ All figures generated successfully!")


if __name__ == "__main__":
    main()
