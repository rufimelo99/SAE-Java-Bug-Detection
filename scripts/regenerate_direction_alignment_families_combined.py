#!/usr/bin/env python3
"""
Generate direction alignment curves with per-family breakdown for all three models on a single plot.
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
        "legend.fontsize": 7,
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
    "qwen-7b": "#000000",
    "codellama-7b": "#1f77b4",
    "starcoder2-7b": "#2ca02c",
}
MODEL_LABELS = {
    "qwen-7b": "Qwen-7B",
    "codellama-7b": "CodeLlama-7B",
    "starcoder2-7b": "StarCoder2-7B",
}

# Family colors for per-family curves
FAMILY_COLORS = {
    "memory_safety": "#0173b2",  # Blue
    "injection": "#de8f05",  # Orange
    "resource": "#029e73",  # Green
    "info_disclosure": "#cc78bc",  # Magenta
    "control_flow": "#ca9161",  # Brown
}

FAMILY_LABELS = {
    "memory_safety": "Memory Safety",
    "injection": "Injection",
    "resource": "Resource",
    "info_disclosure": "Info. Disclosure",
    "control_flow": "Control Flow",
}


def generate_combined_family_alignment():
    """Generate direction alignment with per-family breakdown for all models on single plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model in MODELS:
        json_file = RESULTS_DIR / f"{model}_cwe_universality.json"
        if not json_file.exists():
            print(f"⚠ Missing: {json_file}")
            continue

        with open(json_file) as f:
            data = json.load(f)

        # Get transfer data which includes per-layer per-family information
        transfer_by_layer = data.get("transfer_by_layer", {})

        # Extract families
        families = set()
        first_layer_data = (
            list(transfer_by_layer.values())[0] if transfer_by_layer else {}
        )
        for key in first_layer_data.keys():
            source, target = key.split("->")
            families.add(source)
        families = sorted(families)

        # Extract layer indices
        layers = sorted(int(k) for k in transfer_by_layer.keys())

        # Plot global alignment first
        dir_json = RESULTS_DIR / f"{model}_direction_geometry.json"
        if dir_json.exists():
            with open(dir_json) as f:
                dir_data = json.load(f)

            layers_data = dir_data.get("layers", {})
            all_layers = sorted(int(k) for k in layers_data.keys())
            alignments, ci_lo, ci_hi = [], [], []
            for l in all_layers:
                ld = layers_data[str(l)]
                p = ld.get("pct_aligned", 0)
                pf = p / 100.0
                n = ld.get("n_pairs", 1)
                se_pct = 100.0 * np.sqrt(max(pf * (1 - pf) / n, 0))
                alignments.append(p)
                ci_lo.append(max(0.0, p - 1.645 * se_pct))
                ci_hi.append(min(100.0, p + 1.645 * se_pct))

            color = MODEL_COLORS[model]
            # Plot global alignment with bold line
            ax.plot(
                all_layers, alignments,
                marker="o", linewidth=2.5, markersize=5,
                label=f"{MODEL_LABELS[model]} (global)",
                color=color, zorder=3,
            )
            # 90% CI band on global line
            ax.fill_between(all_layers, ci_lo, ci_hi, alpha=0.12, color=color, linewidth=0, zorder=2)

        # Plot per-family alignment (diagonal of transfer matrix = within-family)
        for family in families:
            family_alignments = []
            for layer in layers:
                layer_key = str(layer)
                pair_key = f"{family}->{family}"
                if (
                    layer_key in transfer_by_layer
                    and pair_key in transfer_by_layer[layer_key]
                ):
                    family_alignments.append(transfer_by_layer[layer_key][pair_key])
                else:
                    family_alignments.append(np.nan)

            if not all(np.isnan(family_alignments)):
                ax.plot(
                    layers,
                    family_alignments,
                    marker="s",
                    linewidth=1.2,
                    markersize=3,
                    label=f"{MODEL_LABELS[model]} - {FAMILY_LABELS[family]}",
                    color=MODEL_COLORS[model],
                    alpha=0.5,
                    linestyle="--",
                )

    # Add chance line
    ax.axhline(
        y=50,
        color="#cccccc",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label="Chance (50%)",
    )

    # Styling
    ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel("Per-Pair Alignment (%)", fontsize=9)
    ax.set_title(
        "Vulnerability direction alignment across models and families",
        fontsize=10,
        fontweight="normal",
    )
    ax.set_ylim([40, 100])
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(fontsize=6, loc="lower left", frameon=True, ncol=2)

    fig.tight_layout()

    output_file = OUTPUT_DIR / "fig_direction_alignment_families_combined.pdf"
    fig.savefig(output_file, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"✓ Generated: {output_file}")


if __name__ == "__main__":
    print("Generating combined per-family direction alignment for all models...\n")
    generate_combined_family_alignment()
    print("\n✓ Done!")
