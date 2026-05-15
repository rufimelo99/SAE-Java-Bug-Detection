#!/usr/bin/env python3
"""
Generate combined direction alignment curves for all three models.

Shows per-pair alignment across layers for Qwen, CodeLlama, and StarCoder2
in a single figure.
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Match style of other paper figures
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


def generate_combined_alignment():
    """Generate combined direction alignment curves for all models."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for model in MODELS:
        json_file = RESULTS_DIR / f"{model}_direction_geometry.json"
        if not json_file.exists():
            print(f"⚠ Missing: {json_file}")
            continue

        with open(json_file) as f:
            data = json.load(f)

        layers_data = data.get("layers", {})
        layers = sorted(int(k) for k in layers_data.keys())
        alignments = [layers_data[str(l)].get("pct_aligned", 0) for l in layers]

        ax.plot(
            layers,
            alignments,
            marker="o",
            linewidth=2.0,
            markersize=6,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )

    # Add chance line
    ax.axhline(
        y=50,
        color="#cccccc",
        linestyle=":",
        linewidth=1.5,
        label="Chance (50%)",
    )

    # Styling
    ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel("Per-Pair Alignment (%)", fontsize=9)
    ax.set_title(
        "Vulnerability direction alignment across layers",
        fontsize=10,
        fontweight="normal",
    )
    ax.set_ylim([40, 100])
    ax.legend(fontsize=8, loc="lower left", frameon=True)
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    output_file = OUTPUT_DIR / "fig_direction_alignment_combined.pdf"
    fig.savefig(output_file, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"✓ Generated: {output_file}")


if __name__ == "__main__":
    print("Generating combined direction alignment curves...\n")
    generate_combined_alignment()
    print("\n✓ Done!")
