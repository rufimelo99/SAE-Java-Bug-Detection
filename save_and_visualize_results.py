#!/usr/bin/env python3
"""
Save direction steering results and generate visualization.
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Setup style
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Results from the experiment
results_json = """PASTE_RESULTS_HERE"""

# Parse results
results = json.loads(results_json)

# Save to file
output_dir = Path("sae_java_bug/artifacts/direction_steering_likelihood")
output_dir.mkdir(parents=True, exist_ok=True)
results_file = output_dir / "results.json"

with open(results_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"✓ Results saved to {results_file}")

# Generate figure
fig, axes = plt.subplots(6, 1, figsize=(8, 12))

layer_keys = sorted(results["layer_results"].keys(), key=lambda x: int(x))

for ax_idx, layer_key in enumerate(layer_keys):
    ax = axes[ax_idx]
    layer_int = int(layer_key)
    layer_data = results["layer_results"][layer_key]

    # Aggregate across all snippets
    alphas = []
    logprobs = []

    for snippet_result in layer_data["snippet_results"]:
        for alpha_str, alpha_data in snippet_result["alpha_results"].items():
            alphas.append(float(alpha_str))
            logprobs.append(alpha_data["mean_logprob"])

    # Group by alpha
    alphas_sorted = sorted(set(alphas))
    logprobs_by_alpha = {a: [] for a in alphas_sorted}

    for a, lp in zip(alphas, logprobs):
        logprobs_by_alpha[a].append(lp)

    means = [np.mean(logprobs_by_alpha[a]) for a in alphas_sorted]
    stds = [np.std(logprobs_by_alpha[a]) for a in alphas_sorted]

    # Plot with error bars
    ax.errorbar(
        alphas_sorted,
        means,
        yerr=stds,
        fmt="o-",
        capsize=5,
        linewidth=2,
        markersize=6,
        label="Mean ± Std",
    )

    # Add baseline reference
    ax.axvline(
        0, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Baseline (α=0)"
    )
    ax.axhline(
        np.mean(logprobs_by_alpha[0]),
        color="red",
        linestyle=":",
        alpha=0.3,
        linewidth=1,
    )

    # Labels and formatting
    ax.set_xlabel("Steering strength (α)", fontsize=10)
    ax.set_ylabel("Log-prob(vulnerable code)", fontsize=10)
    ax.set_title(f"Layer {layer_int}", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # Compute slope for annotation
    if len(alphas_sorted) > 1:
        slope = (means[-1] - means[0]) / (alphas_sorted[-1] - alphas_sorted[0])
        ax.text(
            0.98,
            0.02,
            f"slope={slope:.4f}",
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

plt.suptitle(
    "Direction Steering: Vulnerability Likelihood vs Steering Strength",
    fontsize=12,
    fontweight="bold",
    y=0.995,
)
plt.tight_layout()

# Save figure
figures_dir = Path("figures")
figures_dir.mkdir(parents=True, exist_ok=True)
fig_path = figures_dir / "fig_direction_steering_likelihood.pdf"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"✓ Figure saved to {fig_path}")

# Print summary
print("\n" + "=" * 80)
print("DIRECTION STEERING EXPERIMENT SUMMARY")
print("=" * 80)
print(f"\nLayers tested: {len(layer_keys)}")
print(f"Snippets: {len(results['layer_results']['3']['snippet_results'])}")
print(f"Alpha range: {min(alphas_sorted):.1f} to {max(alphas_sorted):.1f}")
print(f"Prompts per snippet: 3")

print("\n" + "=" * 80)
print("EFFECT SIZES (logprob difference: α=+20 vs α=-20)")
print("=" * 80)

for layer_key in layer_keys:
    layer_data = results["layer_results"][layer_key]
    print(f"\nLayer {layer_key}:")
    for snippet_result in layer_data["snippet_results"]:
        name = snippet_result["name"]
        alphas_dict = snippet_result["alpha_results"]
        if "-20.0" in alphas_dict and "20.0" in alphas_dict:
            low = alphas_dict["-20.0"]["mean_logprob"]
            high = alphas_dict["20.0"]["mean_logprob"]
            diff = high - low
            print(
                f"  {name:20s}: Δ = {diff:+.4f} (toward vuln: {high:.4f}, toward secure: {low:.4f})"
            )

plt.close()
