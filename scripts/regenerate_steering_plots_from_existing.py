#!/usr/bin/env python3
"""
Generate steering plots from existing results (including legacy format).

This script handles both:
1. New format: steering_results_{model}_100samples.json
2. Legacy format: results_real_preference_steering_100samples.json (Qwen only)

Useful for regenerating plots after format changes or styling updates.
"""

import json
import logging
from pathlib import Path

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
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = (
    PROJECT_ROOT
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)

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


def load_steering_results_qwen_legacy() -> dict:
    """Load steering results from legacy Qwen format."""
    json_file = (
        Path(__file__).parent.parent
        / "results_real_preference_steering_100samples.json"
    )

    if not json_file.exists():
        logger.warning(f"Legacy Qwen results not found: {json_file}")
        return {}

    with open(json_file) as f:
        return json.load(f)


def convert_legacy_to_standard(legacy_data: dict) -> dict:
    """Convert legacy Qwen format to standard format."""
    standard = {
        "model": "qwen",
        "n_samples": legacy_data.get("n_prompts", 100),
        "layers": {},
    }

    for layer_str, layer_data in legacy_data.get("layers", {}).items():
        layer_num = int(layer_str)
        standard["layers"][layer_str] = {
            "direction_norm": 1.0,  # Not available in legacy format
            "alpha_results": layer_data.get("alpha_results", {}),
        }

    return standard


def load_steering_results(model: str, n_samples: int = 100) -> dict:
    """Load steering results, handling both new and legacy formats."""
    # Try new format first
    json_file = RESULTS_DIR / f"steering_results_{model}_{n_samples}samples.json"

    if json_file.exists():
        logger.info(f"Loading from standard format: {json_file}")
        with open(json_file) as f:
            return json.load(f)

    # For Qwen, try legacy format
    if model == "qwen":
        legacy_data = load_steering_results_qwen_legacy()
        if legacy_data:
            logger.info("Converting legacy Qwen format to standard")
            return convert_legacy_to_standard(legacy_data)

    logger.warning(f"No steering results found for {model}")
    return {}


def generate_steering_plot_per_model(model: str, results: dict, n_samples: int = 100):
    """Generate left-right plot: (left) steering curves, (right) effect magnitudes."""
    if not results or "layers" not in results:
        logger.warning(f"Invalid results for {model}")
        return

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

    layers = sorted(int(k) for k in results["layers"].keys())
    layer_data_first = results["layers"][str(layers[0])]
    alpha_values = sorted(float(k) for k in layer_data_first["alpha_results"].keys())

    # LEFT PANEL: Steering curves per layer
    for layer in layers:
        layer_data = results["layers"][str(layer)]
        alpha_results = layer_data["alpha_results"]

        alphas = sorted(float(k) for k in alpha_results.keys())
        preferences = [float(alpha_results[str(a)]) for a in alphas]

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
        max_suppress = float(alpha_results.get("20.0", 0.0))
        baseline = float(alpha_results.get("0.0", 0.0))
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


def regenerate_steering_plots(models: list = None, n_samples: int = 100):
    """Generate steering plots from existing results."""
    if models is None:
        models = ["qwen", "codellama", "starcoder2"]

    logger.info("=" * 70)
    logger.info("STEERING PLOT GENERATION (from existing results)")
    logger.info("=" * 70)

    for model in models:
        logger.info(f"\nGenerating plots for {model}...")
        results = load_steering_results(model, n_samples)

        if results and "layers" in results:
            generate_steering_plot_per_model(model, results, n_samples)
        else:
            logger.warning(f"Could not load results for {model}")

    logger.info("\n" + "=" * 70)
    logger.info("✓ Plotting complete")
    logger.info("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate steering plots from existing results"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="qwen,codellama,starcoder2",
        help="Comma-separated list of models",
    )
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples")

    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")]

    regenerate_steering_plots(models, args.n_samples)
