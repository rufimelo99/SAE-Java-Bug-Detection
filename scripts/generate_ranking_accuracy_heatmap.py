#!/usr/bin/env python3
"""Generate ranking accuracy heatmap."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Data: layers vs models
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

# Normalize to [0, 1] for color mapping
data_norm = data / 100.0

fig, ax = plt.subplots(figsize=(5.5, 4))

# Create heatmap
im = ax.imshow(data_norm, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

# Add text annotations
for i in range(len(layers)):
    for j in range(len(models)):
        color = "black"
        ax.text(
            j,
            i,
            f"{data[i, j]:.1f}%",
            ha="center",
            va="center",
            fontsize=9,
            color=color,
            fontweight="normal",
        )

# Set ticks and labels
ax.set_xticks(range(len(models)))
ax.set_yticks(range(len(layers)))
ax.set_xticklabels(models, fontsize=10)
ax.set_yticklabels(layers, fontsize=10)

ax.set_ylabel("Layer", fontsize=11)
ax.set_xlabel("Model", fontsize=11)

plt.tight_layout()

output_file = (
    Path(__file__).parent.parent
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
    / "fig_ranking_accuracy_heatmap.pdf"
)
output_file.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"✓ Figure saved: {output_file}")
plt.close()
