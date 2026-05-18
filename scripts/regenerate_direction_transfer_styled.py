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

    # Calculate mean transfer (off-diagonal)
    off_diag = matrix[~np.eye(n, dtype=bool)]
    mean_transfer = np.nanmean(off_diag)
    std_transfer = np.nanstd(off_diag)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(7, 6))

    # Normalize matrix to [0, 1] for consistent color mapping
    matrix_norm = matrix / 100.0
    im = ax.imshow(matrix_norm, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="equal")

    # Grey diagonal for self-family
    for i in range(n):
        ax.add_patch(
            plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="#cccccc", lw=0)
        )

    # Cell annotations with dynamic text color for contrast
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if not np.isnan(val):
                # Use white text on dark backgrounds, black on light backgrounds
                val_norm = val / 100.0
                text_color = "white" if val_norm > 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                    fontweight="normal",
                )

    # Grid lines
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(family_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(family_names, fontsize=8)

    # Labels
    ax.set_xlabel("Test on (direction evaluated on)", fontsize=9, fontweight="normal")
    ax.set_ylabel("Train on (direction computed from)", fontsize=9, fontweight="normal")

    ax.set_title(
        f"Cross-family vulnerability direction transfer\n(mean: {mean_transfer:.1f}% ± {std_transfer:.1f}%)",
        fontsize=10,
        fontweight="normal",
    )

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
