#!/usr/bin/env python3
"""
Regenerate pairwise CWE probe heatmaps with clean styling.

Uses decimal format (0.77 instead of 77%), full "CWE-xxx" labels on axes,
and matches the style of within-language heatmaps.
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Style matching the within-language heatmaps
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
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


def regenerate_heatmap(model: str):
    """Regenerate heatmap for a model using clean styling."""
    json_file = RESULTS_DIR / f"{model}_cwe_pairwise_probe.json"
    if not json_file.exists():
        print(f"❌ {model}: Results file not found")
        return

    with open(json_file) as f:
        results = json.load(f)

    cwes = results["cwe_types"]
    peak_layer = results["peak_layer"]
    peak_data = results["layers"][str(peak_layer)]["auroc_matrix"]

    n_cwes = len(cwes)
    mat = np.array(peak_data, dtype=float)

    # Create DataFrame with full CWE labels
    df = pd.DataFrame(mat, index=cwes, columns=cwes)

    fig, ax = plt.subplots(figsize=(max(4.5, n_cwes * 0.55), max(3.5, n_cwes * 0.50)))

    # Plot heatmap with clean styling
    sns.heatmap(
        df,
        ax=ax,
        vmin=0.4,
        vmax=1.0,
        cmap="RdYlGn",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.4,
        square=True,
        cbar=False,
    )

    # Grey diagonal for self-pairs
    for i in range(n_cwes):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color="#cccccc", lw=0))

    ax.set_title(
        f"Pairwise CWE-type probe — {model.upper()} (Layer {peak_layer})\n"
        f"(DeltaSecommits, vulnerable-side mean-token; green = separable, yellow = chance)",
        fontsize=8,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)

    fig.tight_layout()

    output_file = OUTPUT_DIR / f"fig_cwe_pairwise_probe_{model}.pdf"
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Regenerated: {output_file}")


if __name__ == "__main__":
    print("Regenerating pairwise CWE probe heatmaps with clean styling...\n")

    for model in MODELS:
        regenerate_heatmap(model)

    print("\n✓ All heatmaps regenerated with clean styling")
