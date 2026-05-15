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

_SCRIPTS_DIR = Path(__file__).parent
RESULTS_DIR = _SCRIPTS_DIR.parent / "results" / "raw_data"
OUTPUT_DIR = _SCRIPTS_DIR.parent.parent / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations" / "figures"

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def discover_models() -> list:
    return sorted(
        p.stem.replace("_direction_geometry", "")
        for p in RESULTS_DIR.glob("*_direction_geometry.json")
    )


def load_direction_data(model: str):
    """Load direction geometry data for a model."""
    json_file = RESULTS_DIR / f"{model}_direction_geometry.json"
    if json_file.exists():
        with open(json_file) as f:
            return json.load(f)
    return None


def generate_model_comparison_figure():
    """Generate comparison figure showing alignment across all models."""
    models = discover_models()
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, model in enumerate(models):
        data = load_direction_data(model)
        if not data:
            continue
        layers_data = data.get("layers", {})
        layers = sorted(int(k) for k in layers_data.keys())
        alignments = [layers_data[str(l)].get("pct_aligned", 0) for l in layers]
        ax.plot(layers, alignments, marker="o", linewidth=2.5, markersize=8,
                label=model.upper(), color=_PALETTE[i % len(_PALETTE)])

    ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, linewidth=1.5, label="Chance")
    ax.set_xlabel("Layer", fontsize=11, fontweight="bold")
    ax.set_ylabel("Per-Pair Alignment (%)", fontsize=11, fontweight="bold")
    ax.set_title("Vulnerability Direction Alignment: Multi-Model Comparison",
                fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([40, 100])

    plt.tight_layout()
    output_file = OUTPUT_DIR / "fig_alignment_comparison_all_models.pdf"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_file}")
    plt.close()


def generate_direction_magnitude_comparison():
    """Compare paired distance magnitudes (proxy for steering effect size)."""
    models = discover_models()
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, model in enumerate(models):
        data = load_direction_data(model)
        if not data:
            continue
        layers_data = data.get("layers", {})
        layers = sorted(int(k) for k in layers_data.keys())
        distances = [layers_data[str(l)].get("mean_paired_distance", 0) for l in layers]
        ax.plot(layers, distances, marker="s", linewidth=2.5, markersize=8,
                label=model.upper(), color=_PALETTE[i % len(_PALETTE)])

    ax.set_yscale("log")
    ax.set_xlabel("Layer", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Paired Distance (L2, log scale)", fontsize=11, fontweight="bold")
    ax.set_title("Vulnerability Signal Magnitude: Multi-Model Comparison",
                fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    output_file = OUTPUT_DIR / "fig_magnitude_comparison_all_models.pdf"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_file}")
    plt.close()


def generate_direction_stability_comparison():
    """Compare cross-layer cosine similarity (direction stability) vs final layer."""
    import math
    models = discover_models()
    n = len(models)
    if n == 0:
        return
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]
        data = load_direction_data(model)
        if not data:
            ax.set_visible(False)
            continue

        layers = sorted(int(k) for k in data.get("layers", {}).keys())
        cosines = data.get("cross_layer_cosines", {})
        last_layer = layers[-1]

        stability = []
        for l in layers:
            key = f"{l}-{last_layer}"
            if key in cosines:
                stability.append(cosines[key])
            elif l == last_layer:
                stability.append(1.0)
            else:
                stability.append(0.0)

        color = _PALETTE[idx % len(_PALETTE)]
        ax.plot(layers, stability, marker="o", linewidth=2.5, markersize=6, color=color)
        ax.fill_between(layers, stability, alpha=0.2, color=color)
        ax.set_xlabel("Layer", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"Cosine Sim to L{last_layer}", fontsize=9, fontweight="bold")
        ax.set_title(model.upper(), fontsize=11, fontweight="bold")
        ax.set_ylim([-1.1, 1.1])
        ax.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.axhline(y=1, color="green", linestyle="--", linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Direction Stability: Cosine Similarity to Final Layer",
                fontsize=12, fontweight="bold", y=1.01)
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
