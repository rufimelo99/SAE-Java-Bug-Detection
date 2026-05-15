#!/bin/bash

##
# Generate missing figure types:
# 1. Pairwise CWE-type probe AUROC heatmaps (from existing data)
# 2. Direction steering: causal validation plots (requires running experiments)
#
# Usage:
#   ./scripts/generate_missing_figures.sh [--steering-only] [--models=qwen,codellama,starcoder2]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures"

echo "=========================================================="
echo "Generating Missing Figures for Paper"
echo "=========================================================="
echo ""

# Parse arguments
STEERING_ONLY=""
MODELS="qwen,codellama,starcoder2"

for arg in "$@"; do
    case $arg in
        --steering-only)
            STEERING_ONLY="yes"
            ;;
        --models=*)
            MODELS="${arg#*=}"
            ;;
        --models)
            # also accept space-separated form: --models value
            shift
            MODELS="$1"
            ;;
        *)
            echo "Unknown option: $arg"
            ;;
    esac
done

# ========================================================================
# Figure 1: Pairwise CWE-type probe AUROC heatmaps
# ========================================================================

if [ -z "$STEERING_ONLY" ]; then
    echo "Step 1: Generating pairwise CWE-type probe AUROC heatmaps..."
    echo ""

    cd "$PROJECT_DIR"
    python3 "$SCRIPT_DIR/generate_pairwise_cwe_probes.py" --figures-only

    echo "✓ Pairwise CWE-type probe heatmaps generated"
    echo ""
fi

# ========================================================================
# Figure 4: Direction steering - Causal validation
# ========================================================================

echo "Step 2: Checking/generating steering plots..."
echo ""

IFS=',' read -ra MODEL_LIST <<< "$MODELS"

for model in "${MODEL_LIST[@]}"; do
    model=$(echo "$model" | xargs)  # trim whitespace
    
    # Check if steering results exist
    results_file="results_real_preference_steering_${model}_100samples.json"
    
    if [ -f "$results_file" ]; then
        echo "  ✓ $model steering results found - generating plot..."
        
        python3 <<PYTHON_SCRIPT
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

output_dir = Path("../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures")
output_dir.mkdir(parents=True, exist_ok=True)

with open("$results_file") as f:
    results = json.load(f)

layers = sorted([int(k) for k in results["layers"].keys()])
baseline = results["baseline"]["mean_preference"]
n_samples = results["n_samples"]

# Generate figure
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))

# Prepare data
effects = []
for layer in layers:
    layer_data = results["layers"][str(layer)]["alpha_results"]
    effect = layer_data["20.0"] - layer_data["-20.0"]
    effects.append(effect)

alphas = sorted([float(a) for a in results["layers"]["3"]["alpha_results"].keys()])

# Panel A
color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

for layer_idx, layer in enumerate(layers):
    layer_data = results["layers"][str(layer)]["alpha_results"]
    prefs = [layer_data[str(float(a))] for a in alphas]
    
    if layer == 3:
        ax0.plot(alphas, prefs, marker="o", linewidth=2.5, markersize=8,
                color=color_palette[0], label=f"Layer {layer}", zorder=10)
    else:
        ax0.plot(alphas, prefs, marker="o", linewidth=1.5, markersize=6,
                color=color_palette[layer_idx], label=f"Layer {layer}", alpha=0.7)

ax0.axhline(baseline, color="red", linestyle="--", linewidth=1.5, alpha=0.6, label="Baseline")
ax0.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

ax0.set_xlabel("Steering Strength (α)", fontsize=11, fontweight="bold")
ax0.set_ylabel("Preference Score", fontsize=11, fontweight="bold")
ax0.set_title("(Left) Steering Curves by Layer", fontsize=11, fontweight="bold")
ax0.legend(fontsize=8, loc="best", ncol=2)
ax0.grid(True, alpha=0.3)

# Panel B
colors = ["darkblue" if layer == 3 else "steelblue" for layer in layers]
ax1.bar([f"L{l}" for l in layers], effects, color=colors, alpha=0.7, edgecolor="black", linewidth=1)

ax1.set_xlabel("Layer", fontsize=11, fontweight="bold")
ax1.set_ylabel("Effect Size (Δ preference)", fontsize=11, fontweight="bold")
ax1.set_title("(Right) Causal Effect Strength", fontsize=11, fontweight="bold")
ax1.grid(True, alpha=0.3, axis="y")

for i, (layer, effect) in enumerate(zip(layers, effects)):
    ax1.text(i, effect, f"{effect:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

fig.suptitle(
    f"Direction Steering: Causal Validation ($("$model".upper()), n={n_samples} real pairs)",
    fontsize=12, fontweight="bold", y=0.98
)

plt.tight_layout()

output_file = output_dir / f"fig_causal_summary_$model.pdf"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {output_file}")
plt.close()

PYTHON_SCRIPT
    else
        echo "  ⏳ $model steering results not found"
        echo "     To generate, run:"
        echo "       python scripts/run_corrected_steering_experiment.py --model $model"
    fi
done

echo ""
echo "=========================================================="
echo "✓ Missing figures generation complete!"
echo "=========================================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "To generate steering plots for missing models:"
echo "  python scripts/run_corrected_steering_experiment.py --models codellama,starcoder2"
echo ""

