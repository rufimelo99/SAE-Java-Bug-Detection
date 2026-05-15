#!/usr/bin/env python3
"""
Regenerate paired distances figure with consistent layer label styling.

Uses layer labels (L0, L3, L7, L11, L15, L19, L23, L27) to match the main
direction alignment figure style.
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Match style of fig_direction_alignment.pdf
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

MODELS = ["qwen-7b", "codellama-7b", "starcoder2-7b"]
MODEL_COLORS = {
    "qwen-7b": "#000000",  # Black
    "codellama-7b": "#1f77b4",  # Blue
    "starcoder2-7b": "#2ca02c",  # Green
}
MODEL_LABELS = {
    "qwen-7b": "Qwen-7B",
    "codellama-7b": "CodeLlama-7B",
    "starcoder2-7b": "StarCoder2-7B",
}

# Standard layer labels matching fig_direction_alignment
LAYER_LABELS = ["L0", "L3", "L7", "L11", "L15", "L19", "L23", "L27"]
LAYER_INDICES = [0, 3, 7, 11, 15, 19, 23, 27]


def generate_paired_distances():
    """Generate paired distances figure with styled formatting."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for model in MODELS:
        json_file = RESULTS_DIR / f"{model}_direction_geometry.json"
        if not json_file.exists():
            print(f"⚠ Missing: {json_file}")
            continue

        with open(json_file) as f:
            data = json.load(f)

        layers_data = data.get("layers", {})

        # Extract mean paired distances for standard layers
        distances = []
        for layer_idx in LAYER_INDICES:
            layer_key = str(layer_idx)
            if layer_key in layers_data:
                dist = layers_data[layer_key].get("mean_paired_distance", 0)
                distances.append(dist)
            else:
                distances.append(None)

        # Plot with square markers
        ax.plot(
            range(len(LAYER_LABELS)),
            distances,
            marker="s",
            linewidth=2.0,
            markersize=6,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )

    # Set up x-axis with layer labels
    ax.set_xticks(range(len(LAYER_LABELS)))
    ax.set_xticklabels(LAYER_LABELS)
    ax.set_xlabel("Layer", fontsize=9)

    # Set up y-axis (log scale for magnitude)
    ax.set_ylabel("Mean Paired Distance (L2)", fontsize=9)
    ax.set_yscale("log")

    # Title
    ax.set_title(
        "Vulnerability signal magnitude across layers",
        fontsize=10,
        fontweight="normal",
    )

    # Legend
    ax.legend(fontsize=8, loc="lower left", frameon=True, ncol=1)

    # Grid
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5, which="both")
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    output_file = OUTPUT_DIR / "fig_paired_distances.pdf"
    fig.savefig(output_file, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"✓ Generated: {output_file}")


if __name__ == "__main__":
    print("Regenerating paired distances figure with consistent styling...\n")
    generate_paired_distances()
    print("\n✓ Done!")
