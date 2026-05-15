#!/usr/bin/env python3
"""
Regenerate cross-family vulnerability direction transfer heatmap with all families.

Generates a heatmap showing how well a vulnerability direction trained on one
family transfers to other vulnerability families.
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

# Family display names
FAMILY_NAMES = {
    "memory_safety": "Memory Safety",
    "injection": "Injection",
    "resource": "Resource",
    "info_disclosure": "Info. Disclosure",
    "control_flow": "Control Flow",
}


def generate_transfer_heatmap(model: str):
    """Generate cross-family transfer heatmap for a model."""
    json_file = RESULTS_DIR / f"{model}_cwe_universality.json"
    if not json_file.exists():
        print(f"⚠ Missing: {json_file}")
        return

    with open(json_file) as f:
        data = json.load(f)

    transfer = data.get("cross_family_transfer", {})

    # Extract families from the data
    families = set()
    for key in transfer.keys():
        source, target = key.split("->")
        families.add(source)
    families = sorted(families)
    family_names = [FAMILY_NAMES.get(f, f) for f in families]

    # Build transfer matrix
    n = len(families)
    matrix = np.zeros((n, n))
    for i, source in enumerate(families):
        for j, target in enumerate(families):
            key = f"{source}->{target}"
            if key in transfer:
                matrix[i, j] = transfer[key]["mean"]
            else:
                matrix[i, j] = np.nan

    # Create DataFrame
    df = pd.DataFrame(matrix, index=family_names, columns=family_names)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        df,
        ax=ax,
        vmin=70,
        vmax=90,
        cmap="RdYlGn",
        annot=True,
        fmt=".1f",
        annot_kws={"size": 8},
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Transfer alignment (%)"},
    )

    # Grey diagonal for self-family
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color="#cccccc", lw=0))

    # Labels
    ax.set_xlabel("Test on (direction evaluated on)", fontsize=9, fontweight="normal")
    ax.set_ylabel("Train on (direction computed from)", fontsize=9, fontweight="normal")

    # Calculate mean transfer (off-diagonal)
    off_diag = matrix[~np.eye(n, dtype=bool)]
    mean_transfer = np.nanmean(off_diag)
    std_transfer = np.nanstd(off_diag)

    ax.set_title(
        f"Cross-family vulnerability direction transfer\n(mean: {mean_transfer:.1f}% ± {std_transfer:.1f}%)",
        fontsize=10,
        fontweight="normal",
    )

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()

    output_file = OUTPUT_DIR / f"fig_direction_transfer_{model}.pdf"
    fig.savefig(output_file, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(
        f"✓ Generated: {output_file} (mean: {mean_transfer:.1f}% ± {std_transfer:.1f}%)"
    )


if __name__ == "__main__":
    print("Generating cross-family vulnerability direction transfer heatmaps...\n")

    for model in MODELS:
        generate_transfer_heatmap(model)

    print("\n✓ All heatmaps generated!")
