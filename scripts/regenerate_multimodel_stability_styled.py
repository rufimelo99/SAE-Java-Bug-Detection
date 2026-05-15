#!/usr/bin/env python3
"""
Regenerate multimodel direction stability with fig_direction_alignment.pdf style.

Uses layer labels (L0, L3, L7, L11, L15, L19, L23, L27) to match the main
direction alignment figure style.
"""

import json
import math
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


def generate_stability_comparison():
    """Generate multi-model direction stability with styled formatting."""
    models = MODELS
    n = len(models)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]
        json_file = RESULTS_DIR / f"{model}_direction_geometry.json"
        if not json_file.exists():
            print(f"⚠ Missing: {json_file}")
            ax.set_visible(False)
            continue

        with open(json_file) as f:
            data = json.load(f)

        layers_data = data.get("layers", {})
        all_layers = sorted(int(k) for k in layers_data.keys())
        last_layer = all_layers[-1]

        cosines = data.get("cross_layer_cosines", {})

        # Extract stability for standard layers
        stability = []
        for layer_idx in LAYER_INDICES:
            key = f"{layer_idx}-{last_layer}"
            if key in cosines:
                stability.append(cosines[key])
            elif layer_idx == last_layer:
                stability.append(1.0)
            else:
                stability.append(0.0)

        color = MODEL_COLORS[model]
        ax.plot(
            range(len(LAYER_LABELS)),
            stability,
            marker="o",
            linewidth=2.0,
            markersize=6,
            color=color,
        )
        ax.fill_between(range(len(LAYER_LABELS)), stability, alpha=0.2, color=color)

        # Set up x-axis with layer labels
        ax.set_xticks(range(len(LAYER_LABELS)))
        ax.set_xticklabels(LAYER_LABELS, fontsize=8)
        ax.set_xlabel("Layer", fontsize=9)

        ax.set_ylabel(f"Cosine Sim to L{last_layer}", fontsize=9, fontweight="normal")
        ax.set_title(model.upper(), fontsize=10, fontweight="normal")
        ax.set_ylim([-1.1, 1.1])
        ax.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.axhline(y=1, color="green", linestyle="--", linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Direction stability: cosine similarity to final layer",
        fontsize=11,
        fontweight="normal",
        y=1.00,
    )
    plt.tight_layout()

    output_file = OUTPUT_DIR / "fig_multimodel_direction_stability.pdf"
    fig.savefig(output_file, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"✓ Generated: {output_file}")


if __name__ == "__main__":
    print(
        "Regenerating multi-model direction stability with direction_alignment.pdf style...\n"
    )
    generate_stability_comparison()
    print("\n✓ Done!")
