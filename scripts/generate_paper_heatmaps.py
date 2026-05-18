#!/usr/bin/env python3
"""
Generate all paper heatmaps with consistent styling.

This module generates:
1. Multi-model alignment heatmap
2. Ranking accuracy heatmap
3. Pairwise CWE-type probe heatmaps (via existing script)

All heatmaps use consistent styling: RdYlGn colormap, serif fonts, 0-1 scale.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def configure_style():
    """Configure matplotlib for paper-quality heatmaps."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def generate_multimodel_alignment_heatmap(output_dir):
    """Generate multi-model per-pair alignment heatmap."""
    layers = ["L3", "L7", "L11", "L15", "L19", "L23", "L27"]
    models = ["Qwen-7B", "CodeLlama-7B", "StarCoder2-7B"]

    # Per-pair alignment percentages
    data = np.array(
        [
            [86.88, 86.00, 86.22],
            [87.53, 87.18, 87.04],
            [87.36, 87.06, 87.28],
            [87.42, 87.03, 87.25],
            [87.40, 86.96, 87.05],
            [87.38, 86.65, 86.91],
            [70.36, 70.14, 70.08],
        ]
    )

    data_norm = data / 100.0

    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(data_norm, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    # Annotations with dynamic text color for contrast
    for i in range(len(layers)):
        for j in range(len(models)):
            # Use white text on dark backgrounds, black on light backgrounds
            val = data_norm[i, j]
            text_color = "white" if val > 0.6 else "black"
            ax.text(
                j,
                i,
                f"{data[i, j]:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="normal",
            )

    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_yticklabels(layers, fontsize=10)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)

    plt.tight_layout()

    output_file = output_dir / "fig_multimodel_alignment_heatmap.pdf"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ {output_file.name}")
    plt.close()


def generate_ranking_accuracy_heatmap(output_dir):
    """Generate ranking accuracy heatmap."""
    layers = ["L3", "L7", "L11", "L15", "L19", "L23", "L27"]
    models = ["Qwen-7B", "CodeLlama-7B", "StarCoder2-7B"]

    # Ranking accuracy percentages
    data = np.array(
        [
            [58.21, 57.89, 58.07],
            [58.22, 58.05, 58.11],
            [58.22, 58.03, 58.14],
            [58.22, 58.01, 58.13],
            [58.23, 57.98, 58.09],
            [58.37, 58.06, 58.12],
            [56.27, 55.98, 56.04],
        ]
    )

    data_norm = data / 100.0

    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(data_norm, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    # Annotations with dynamic text color for contrast
    for i in range(len(layers)):
        for j in range(len(models)):
            # Use white text on dark backgrounds, black on light backgrounds
            val = data_norm[i, j]
            text_color = "white" if val > 0.6 else "black"
            ax.text(
                j,
                i,
                f"{data[i, j]:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="normal",
            )

    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_yticklabels(layers, fontsize=10)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)

    plt.tight_layout()

    output_file = output_dir / "fig_ranking_accuracy_heatmap.pdf"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ {output_file.name}")
    plt.close()


def main():
    """Generate all paper heatmaps."""
    configure_style()

    output_dir = (
        Path(__file__).parent.parent
        / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
        / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating paper heatmaps...")
    print(f"  Output directory: {output_dir}")
    print()

    generate_multimodel_alignment_heatmap(output_dir)
    generate_ranking_accuracy_heatmap(output_dir)

    print()
    print("✓ All heatmaps generated successfully!")


if __name__ == "__main__":
    main()
