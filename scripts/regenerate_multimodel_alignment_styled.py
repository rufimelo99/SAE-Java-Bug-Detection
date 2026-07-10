#!/usr/bin/env python3
"""
Regenerate multimodel alignment comparison with fig_direction_alignment.pdf style.

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


def generate_alignment_comparison():
    """Generate multi-model alignment comparison with styled formatting."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for model in MODELS:
        json_file = RESULTS_DIR / f"{model}_direction_geometry.json"
        if not json_file.exists():
            print(f"⚠ Missing: {json_file}")
            continue

        with open(json_file) as f:
            data = json.load(f)

        layers_data = data.get("layers", {})

        # Extract alignment percentages for standard layers (convert to fraction)
        alignments, ci_lo, ci_hi = [], [], []
        for layer_idx in LAYER_INDICES:
            layer_key = str(layer_idx)
            if layer_key in layers_data:
                ld = layers_data[layer_key]
                p = ld.get("pct_aligned", 0) / 100.0
                n = ld.get("n_pairs", 1)
                se = np.sqrt(max(p * (1 - p) / n, 0))
                alignments.append(p)
                ci_lo.append(max(0.0, p - 1.645 * se))
                ci_hi.append(min(1.0, p + 1.645 * se))
            else:
                alignments.append(None)
                ci_lo.append(None)
                ci_hi.append(None)

        x = range(len(LAYER_LABELS))
        color = MODEL_COLORS[model]
        ax.plot(
            x, alignments,
            marker="o", linewidth=2.0, markersize=6,
            label=MODEL_LABELS[model], color=color,
        )
        # 90% CI band
        valid = [(i, lo, hi) for i, (lo, hi) in enumerate(zip(ci_lo, ci_hi)) if lo is not None]
        if valid:
            xi, los, his = zip(*valid)
            ax.fill_between(xi, los, his, alpha=0.15, color=color, linewidth=0)

    # Add chance line (50%) as dotted, matching fig_direction_alignment.pdf
    ax.axhline(
        y=0.5, color="#cccccc", linestyle=":", linewidth=1.5, label="Chance (50%)"
    )

    # Set up x-axis with layer labels
    ax.set_xticks(range(len(LAYER_LABELS)))
    ax.set_xticklabels(LAYER_LABELS)
    ax.set_xlabel("Layer", fontsize=9)

    # Set up y-axis
    ax.set_ylabel("Fraction pairs: δ$_L$ · d$_L$ > 0", fontsize=9)
    ax.set_ylim([0.4, 1.0])

    # Title
    ax.set_title(
        "Vulnerability direction alignment: multi-model comparison",
        fontsize=10,
        fontweight="normal",
    )

    # Legend matching style of fig_direction_alignment
    ax.legend(fontsize=8, loc="lower left", frameon=True)

    # Grid matching the other figure
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    output_file = OUTPUT_DIR / "fig_multimodel_alignment_comparison.pdf"
    fig.savefig(output_file, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"✓ Generated: {output_file}")


if __name__ == "__main__":
    print(
        "Regenerating multi-model alignment comparison with direction_alignment.pdf style...\n"
    )
    generate_alignment_comparison()
    print("\n✓ Done!")
