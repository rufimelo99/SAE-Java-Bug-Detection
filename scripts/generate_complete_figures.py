#!/usr/bin/env python3
"""
Generate complete figure set: pairwise CWE probes and steering experiments.

This integrates:
1. Pairwise CWE-type probe AUROC heatmaps (from existing Qwen runs)
2. Direction steering: causal validation plots (from steering experiments)

Usage:
    python scripts/generate_complete_figures.py --all
    python scripts/generate_complete_figures.py --steering-only
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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

OUTPUT_DIR = Path("../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures")
RESULTS_DIR = Path("results")


def generate_steering_plot_from_json(json_file: Path, model: str):
    """Generate steering plot from steering results JSON file."""
    if not json_file.exists():
        logger.warning(f"Steering results not found: {json_file}")
        return
    
    logger.info(f"Generating steering plot for {model} from {json_file.name}")
    
    with open(json_file) as f:
        results = json.load(f)
    
    if "layers" not in results:
        logger.warning(f"Invalid results format for {model}")
        return
    
    layers = sorted([int(k) for k in results["layers"].keys()])
    
    # ========================================================================
    # Figure: Direction Steering - Causal Validation
    # ========================================================================
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Prepare data
    effects = []
    for layer in layers:
        layer_data = results["layers"][str(layer)]["alpha_results"]
        effect = layer_data["20.0"] - layer_data["-20.0"]
        effects.append(effect)
    
    alphas = sorted([float(a) for a in results["layers"]["3"]["alpha_results"].keys()])
    
    # ====== Panel (A): Steering curves overlay ======
    color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    
    for layer_idx, layer in enumerate(layers):
        layer_data = results["layers"][str(layer)]["alpha_results"]
        prefs = [layer_data[str(float(a))] for a in alphas]
        
        if layer == 3:
            ax0.plot(alphas, prefs, marker="o", linewidth=2.5, markersize=8,
                    color=color_palette[0], label=f"Layer {layer}", zorder=10)
        else:
            ax0.plot(alphas, prefs, marker="o", linewidth=1.5, markersize=6,
                    color=color_palette[layer_idx], label=f"Layer {layer}", alpha=0.7)
    
    # Baseline
    baseline = results["baseline"]["mean_preference"]
    ax0.axhline(baseline, color="red", linestyle="--", linewidth=1.5, alpha=0.6, label="Baseline")
    ax0.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    
    ax0.set_xlabel("Steering Strength (α)", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Preference Score", fontsize=11, fontweight="bold")
    ax0.set_title("(Left) Steering Curves by Layer", fontsize=11, fontweight="bold")
    ax0.legend(fontsize=8, loc="best", ncol=2)
    ax0.grid(True, alpha=0.3)
    
    # ====== Panel (B): Effect magnitudes ======
    colors = ["darkblue" if layer == 3 else "steelblue" for layer in layers]
    ax1.bar([f"L{l}" for l in layers], effects, color=colors, alpha=0.7,
           edgecolor="black", linewidth=1)
    
    ax1.set_xlabel("Layer", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Effect Size (Δ preference)", fontsize=11, fontweight="bold")
    ax1.set_title("(Right) Causal Effect Strength", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Value labels
    for i, (layer, effect) in enumerate(zip(layers, effects)):
        ax1.text(i, effect, f"{effect:.4f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    
    # Main title
    fig.suptitle(
        f"Direction Steering: Causal Validation ({model.upper()}, n={results['n_samples']} pairs)",
        fontsize=12, fontweight="bold", y=0.98
    )
    
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / f"fig_causal_summary_{model}.pdf"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"✓ Saved: {output_file}")
    plt.close()


def generate_steering_plots_all_models():
    """Generate steering plots for all models if results exist."""
    models = ["qwen", "codellama", "starcoder2"]
    
    for model in models:
        # Try to find existing steering results
        json_file = RESULTS_DIR / f"steering_results_{model}_100samples.json"
        
        # Also check for the format from run_corrected_steering_experiment.py
        if not json_file.exists():
            json_file = Path(f"results_real_preference_steering_{model}_100samples.json")
        
        if json_file.exists():
            generate_steering_plot_from_json(json_file, model)
        else:
            logger.warning(f"No steering results found for {model}")
            logger.info(f"  To generate: python scripts/run_corrected_steering_experiment.py --model {model}")


def main():
    parser = argparse.ArgumentParser(description="Generate complete paper figures")
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    parser.add_argument("--steering-only", action="store_true", help="Generate steering plots only")
    args = parser.parse_args()
    
    if args.steering_only or args.all:
        logger.info("Generating steering plots...")
        generate_steering_plots_all_models()
    
    logger.info("\n✓ Figure generation complete!")
    logger.info("\nNOTE: To generate steering plots for CodeLlama and StarCoder2:")
    logger.info("  python scripts/run_corrected_steering_experiment.py --models codellama,starcoder2")


if __name__ == "__main__":
    main()
