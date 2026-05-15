#!/usr/bin/env python3
"""
Generate direction steering style plots for all models.

Creates visualization of steering effects based on direction magnitudes
and alignment data available in JSON results.

This serves as visualization pending full steering experiments.
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MODELS = ["qwen", "codellama", "starcoder2"]
LAYERS_TO_PLOT = [3, 7, 11, 15, 19, 23]
RESULTS_DIR = Path("results/raw_data")
OUTPUT_DIR = Path("../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures")


def load_direction_data(model: str):
    """Load direction geometry data for a model."""
    json_file = RESULTS_DIR / f"{model}_direction_geometry.json"
    if json_file.exists():
        with open(json_file) as f:
            return json.load(f)
    return None


def generate_model_comparison_figure():
    """Generate comparison figure showing alignment across all models."""
    fig, ax = plt.subplots(figsize=(11, 6))
    
    colors = {
        "qwen": "#1f77b4",
        "codellama": "#ff7f0e",
        "starcoder2": "#2ca02c",
    }
    
    for model in MODELS:
        data = load_direction_data(model)
        if not data:
            continue
        
        layers_data = data.get("layers", {})
        layers = sorted([int(k) for k in layers_data.keys()])
        alignments = [layers_data[str(l)].get("pct_aligned", 0) for l in layers]
        
        ax.plot(
            layers,
            alignments,
            marker="o",
            linewidth=2.5,
            markersize=8,
            label=model.upper(),
            color=colors[model],
        )
    
    # Add chance line
    ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, linewidth=1.5, label="Chance")
    
    # Formatting
    ax.set_xlabel("Layer", fontsize=11, fontweight="bold")
    ax.set_ylabel("Per-Pair Alignment (%)", fontsize=11, fontweight="bold")
    ax.set_title("Vulnerability Direction Alignment: Multi-Model Comparison",
                fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([40, 100])
    ax.set_xticks(LAYERS_TO_PLOT)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "fig_alignment_comparison_all_models.pdf"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_file}")
    plt.close()


def generate_direction_magnitude_comparison():
    """Compare paired distance magnitudes (proxy for steering effect size)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    
    colors = {
        "qwen": "#1f77b4",
        "codellama": "#ff7f0e",
        "starcoder2": "#2ca02c",
    }
    
    for model in MODELS:
        data = load_direction_data(model)
        if not data:
            continue
        
        layers_data = data.get("layers", {})
        layers = sorted([int(k) for k in layers_data.keys()])
        distances = [layers_data[str(l)].get("mean_paired_distance", 0) for l in layers]
        
        ax.plot(
            layers,
            distances,
            marker="s",
            linewidth=2.5,
            markersize=8,
            label=model.upper(),
            color=colors[model],
        )
    
    # Log scale to show differences
    ax.set_yscale("log")
    
    # Formatting
    ax.set_xlabel("Layer", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Paired Distance (L2, log scale)", fontsize=11, fontweight="bold")
    ax.set_title("Vulnerability Signal Magnitude: Multi-Model Comparison",
                fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xticks(LAYERS_TO_PLOT)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "fig_magnitude_comparison_all_models.pdf"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_file}")
    plt.close()


def generate_direction_stability_comparison():
    """Compare cross-layer cosine similarity (direction stability)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        data = load_direction_data(model)
        if not data:
            continue
        
        layers = sorted([int(k) for k in data.get("layers", {}).keys()])
        cosines = data.get("cross_layer_cosines", {})
        
        # Build matrix (L-to-L27 cosines)
        stability = []
        for l in layers:
            key = f"{l}-27"  # Compare to final layer
            if key in cosines:
                stability.append(cosines[key])
            elif l == 27:
                stability.append(1.0)  # Perfect similarity with itself
            else:
                stability.append(0.0)
        
        # Plot
        ax.plot(layers, stability, marker="o", linewidth=2.5, markersize=8,
               color="#1f77b4")
        ax.fill_between(layers, stability, alpha=0.3, color="#1f77b4")
        
        # Formatting
        ax.set_xlabel("Layer", fontsize=10, fontweight="bold")
        ax.set_ylabel("Cosine Similarity to L27", fontsize=10, fontweight="bold")
        ax.set_title(f"{model.upper()}", fontsize=11, fontweight="bold")
        ax.set_ylim([-1.1, 1.1])
        ax.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.axhline(y=1, color="green", linestyle="--", linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(LAYERS_TO_PLOT)
    
    fig.suptitle("Direction Stability: Cosine Similarity to Final Layer (L27)",
                fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / "fig_direction_stability_all_models.pdf"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_file}")
    plt.close()


def main():
    print("Generating direction steering-style plots...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    generate_model_comparison_figure()
    generate_direction_magnitude_comparison()
    generate_direction_stability_comparison()
    
    print("\n✓ All styling plots generated!")
    print(f"\nNOTE: Full steering experiments require running:")
    print("  python scripts/run_corrected_steering_experiment.py --models qwen,codellama,starcoder2")


if __name__ == "__main__":
    main()
