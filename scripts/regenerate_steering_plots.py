#!/usr/bin/env python3
"""
Generate publication-quality steering effect visualizations.

Creates:
1. Per-model steering effect plots (left: curves, right: magnitude comparison)
2. Combined multi-model comparison figure

Usage:
    python scripts/regenerate_steering_plots.py [--models qwen,codellama,starcoder2] [--n_samples 100]
"""

import json
import logging
from pathlib import Path
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Styling
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "raw_data"
OUTPUT_DIR = (
    Path(__file__).parent.parent.parent
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)

MODEL_COLORS = {
    "qwen": "#000000",
    "codellama": "#1f77b4",
    "starcoder2": "#2ca02c",
}

MODEL_LABELS = {
    "qwen": "Qwen-7B",
    "codellama": "CodeLlama-7B",
    "starcoder2": "StarCoder2-7B",
}

LAYER_COLORS = {
    3: "#1f77b4",  # Blue
    7: "#ff7f0e",  # Orange
    11: "#2ca02c",  # Green
    15: "#d62728",  # Red
    19: "#9467bd",  # Purple
    23: "#8c564b",  # Brown
}


def load_steering_results(model: str, n_samples: int = 100) -> Dict:
    """Load steering results for a model."""
    json_file = RESULTS_DIR / f"steering_results_{model}_{n_samples}samples.json"

    if not json_file.exists():
        logger.warning(f"Results not found: {json_file}")
        return {}

    with open(json_file) as f:
        return json.load(f)


def generate_steering_plot_per_model(model: str, results: Dict, n_samples: int = 100):
    """Generate left-right plot: (left) steering curves, (right) effect magnitudes."""
    if not results or "layers" not in results:
        logger.warning(f"Invalid results for {model}")
        return

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

    layers = sorted(int(k) for k in results["layers"].keys())
    alpha_values = sorted(
        float(k) for k in results["layers"][str(layers[0])]["alpha_results"].keys()
    )

    # LEFT PANEL: Steering curves per layer
    for layer in layers:
        layer_data = results["layers"][str(layer)]
        alpha_results = layer_data["alpha_results"]

        alphas = sorted(float(k) for k in alpha_results.keys())
        preferences = [alpha_results[str(a)] for a in alphas]

        ax_left.plot(
            alphas,
            preferences,
            marker="o",
            linewidth=2.0,
            markersize=6,
            label=f"L{layer}",
            color=LAYER_COLORS.get(layer, "#cccccc"),
        )

    # Baseline line
    ax_left.axhline(
        y=0,
        color="#cccccc",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Baseline",
    )

    ax_left.set_xlabel("Steering strength (α)", fontsize=9)
    ax_left.set_ylabel("Preference shift\n[log P(secure) - log P(vuln)]", fontsize=9)
    ax_left.set_title(
        f"{MODEL_LABELS[model]} - Direction Steering Curves",
        fontsize=10,
        fontweight="normal",
    )
    ax_left.legend(fontsize=7, loc="best", ncol=2, frameon=True)
    ax_left.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax_left.set_axisbelow(True)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)

    # RIGHT PANEL: Effect magnitudes at max suppression
    effects = []
    layer_labels = []

    for layer in layers:
        layer_data = results["layers"][str(layer)]
        alpha_results = layer_data["alpha_results"]

        # Effect magnitude: difference between max suppression and baseline
        max_suppress = alpha_results.get("20.0", 0.0)
        baseline = alpha_results.get("0.0", 0.0)
        effect = max_suppress - baseline

        effects.append(effect)
        layer_labels.append(f"L{layer}")

    colors = [LAYER_COLORS.get(layer, "#cccccc") for layer in layers]
    bars = ax_right.bar(
        layer_labels, effects, color=colors, alpha=0.8, edgecolor="black", linewidth=1.0
    )

    # Add value labels on bars
    for bar, effect in zip(bars, effects):
        height = bar.get_height()
        ax_right.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{effect:.3f}",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontsize=7,
        )

    ax_right.set_ylabel("Effect magnitude\n(α=20 - α=0)", fontsize=9)
    ax_right.set_title(
        f"{MODEL_LABELS[model]} - Effect Sizes",
        fontsize=10,
        fontweight="normal",
    )
    ax_right.grid(True, alpha=0.2, axis="y", linestyle="-", linewidth=0.5)
    ax_right.set_axisbelow(True)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    fig.tight_layout()

    output_file = OUTPUT_DIR / f"fig_causal_summary_{model}_{n_samples}samples.pdf"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight", dpi=150)
    plt.close(fig)

    logger.info(f"✓ Generated: {output_file}")


def generate_multimodel_comparison(models: list, n_samples: int = 100):
    """Generate combined multi-model steering comparison figure."""
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax_idx, model in enumerate(models):
        ax = axes[ax_idx]
        results = load_steering_results(model, n_samples)

        if not results or "layers" not in results:
            ax.set_visible(False)
            continue

        layers = sorted(int(k) for k in results["layers"].keys())
        alpha_values = sorted(
            float(k) for k in results["layers"][str(layers[0])]["alpha_results"].keys()
        )

        # Plot curves for each layer
        for layer in layers:
            layer_data = results["layers"][str(layer)]
            alpha_results = layer_data["alpha_results"]

            alphas = sorted(float(k) for k in alpha_results.keys())
            preferences = [alpha_results[str(a)] for a in alphas]

            ax.plot(
                alphas,
                preferences,
                marker="o",
                linewidth=2.0,
                markersize=5,
                label=f"L{layer}",
                color=LAYER_COLORS.get(layer, "#cccccc"),
            )

        # Baseline
        ax.axhline(y=0, color="#cccccc", linestyle="--", linewidth=1.5, alpha=0.7)

        ax.set_xlabel("Steering strength (α)", fontsize=9)
        if ax_idx == 0:
            ax.set_ylabel("Preference shift", fontsize=9)
        ax.set_title(MODEL_LABELS[model], fontsize=10, fontweight="normal")
        ax.legend(fontsize=7, loc="best", ncol=1, frameon=True)
        ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Direction Steering: Preference Shift Across Models",
        fontsize=11,
        fontweight="normal",
        y=1.00,
    )
    fig.tight_layout()

    output_file = (
        OUTPUT_DIR / f"fig_steering_multimodel_comparison_{n_samples}samples.pdf"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight", dpi=150)
    plt.close(fig)

    logger.info(f"✓ Generated: {output_file}")


def regenerate_all_steering_plots(models: list, n_samples: int = 100):
    """Generate all steering visualization plots."""
    logger.info("=" * 70)
    logger.info("STEERING VISUALIZATION PIPELINE")
    logger.info("=" * 70)

    # Per-model plots
    for model in models:
        logger.info(f"\nGenerating plots for {model}...")
        results = load_steering_results(model, n_samples)

        if results:
            generate_steering_plot_per_model(model, results, n_samples)
        else:
            logger.warning(f"No results available for {model}")

    # Combined comparison
    available_models = [m for m in models if load_steering_results(m, n_samples)]
    if available_models:
        logger.info(
            f"\nGenerating combined comparison for {len(available_models)} models..."
        )
        generate_multimodel_comparison(available_models, n_samples)

    logger.info("\n" + "=" * 70)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate steering effect visualizations"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="qwen,codellama,starcoder2",
        help="Comma-separated list of models",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples used in experiments",
    )

    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")]

    regenerate_all_steering_plots(models, args.n_samples)
