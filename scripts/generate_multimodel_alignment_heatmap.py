#!/usr/bin/env python3
"""Generate multi-model alignment heatmap."""

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

# Alignment percentages
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
    / "fig_multimodel_alignment_heatmap.pdf"
)
output_file.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"✓ Figure saved: {output_file}")
plt.close()
