#!/usr/bin/env python3
"""
Generate all paper figures for all models from raw experiment results.

This script reads JSON results from run_all_experiments.py and generates:
- Direction alignment heatmaps
- Per-pair consistency plots
- Cross-family transfer heatmaps
- Paired ranking accuracy curves
- Comparison plots across models

Usage:
    python generate_all_figures.py --results-dir ./results --output-dir ./figures
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODELS = ["qwen", "codellama", "starcoder2"]
LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]
COLORS = {
    "qwen": "#1f77b4",  # blue
    "codellama": "#ff7f0e",  # orange
    "starcoder2": "#2ca02c",  # green
}


class FigureGenerator:
    """Generate publication-quality figures from experiment results."""

    def __init__(self, results_dir: Path, output_dir: Path):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load all results
        self.results = {}
        self._load_results()

        logger.info(f"Loaded results from {self.results_dir}")
        logger.info(f"Output directory: {self.output_dir}")

    def _load_results(self):
        """Load all JSON results from raw_data directory."""
        raw_data_dir = self.results_dir / "raw_data"

        if not raw_data_dir.exists():
            logger.warning(f"raw_data directory not found: {raw_data_dir}")
            return

        for json_file in raw_data_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    key = json_file.stem
                    self.results[key] = data
                    logger.info(f"Loaded: {json_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")

    def generate_direction_alignment_heatmap(self):
        """Generate figure: direction cosine similarity across layers and models."""
        logger.info("Generating direction alignment heatmap...")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, model in enumerate(MODELS):
            key = f"{model}_direction_geometry"
            if key not in self.results:
                logger.warning(f"No results for {key}")
                continue

            data = self.results[key]
            cosines = data.get("cross_layer_cosines", {})

            # Build matrix
            matrix = np.zeros((len(LAYERS), len(LAYERS)))
            for i, l1 in enumerate(LAYERS):
                for j, l2 in enumerate(LAYERS):
                    key_str = f"{l1}-{l2}"
                    matrix[i, j] = cosines.get(key_str, 0)

            # Plot heatmap
            im = axes[idx].imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)
            axes[idx].set_xticks(range(len(LAYERS)))
            axes[idx].set_yticks(range(len(LAYERS)))
            axes[idx].set_xticklabels(LAYERS)
            axes[idx].set_yticklabels(LAYERS)
            axes[idx].set_xlabel("Layer")
            axes[idx].set_ylabel("Layer")
            axes[idx].set_title(f"{model.capitalize()} Direction Alignment")

            # Add colorbar
            plt.colorbar(im, ax=axes[idx], label="Cosine Similarity")

        plt.tight_layout()
        output_file = self.output_dir / "fig_direction_alignment_heatmaps.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {output_file}")
        plt.close()

    def generate_per_pair_alignment_curves(self):
        """Generate figure: per-pair alignment percentage across layers."""
        logger.info("Generating per-pair alignment curves...")

        fig, ax = plt.subplots(figsize=(10, 6))

        for model in MODELS:
            key = f"{model}_direction_geometry"
            if key not in self.results:
                continue

            data = self.results[key]
            layers_data = data.get("layers", {})

            alignment = []
            for layer in LAYERS:
                layer_str = str(layer)
                if layer_str in layers_data:
                    alignment.append(layers_data[layer_str]["pct_aligned"])
                else:
                    alignment.append(None)

            ax.plot(
                LAYERS,
                alignment,
                marker="o",
                label=model.capitalize(),
                color=COLORS[model],
                linewidth=2,
                markersize=6,
            )

        ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="Chance (50%)")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Per-Pair Alignment (%)", fontsize=12)
        ax.set_title("Vulnerability Direction Alignment Across Layers", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([40, 100])

        output_file = self.output_dir / "fig_per_pair_alignment.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {output_file}")
        plt.close()

    def generate_paired_distance_curves(self):
        """Generate figure: mean paired distances across layers."""
        logger.info("Generating paired distance curves...")

        fig, ax = plt.subplots(figsize=(10, 6))

        for model in MODELS:
            key = f"{model}_direction_geometry"
            if key not in self.results:
                continue

            data = self.results[key]
            layers_data = data.get("layers", {})

            distances = []
            for layer in LAYERS:
                layer_str = str(layer)
                if layer_str in layers_data:
                    distances.append(layers_data[layer_str]["mean_paired_distance"])
                else:
                    distances.append(None)

            ax.plot(
                LAYERS,
                distances,
                marker="s",
                label=model.capitalize(),
                color=COLORS[model],
                linewidth=2,
                markersize=6,
            )

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Mean Paired Distance (L2)", fontsize=12)
        ax.set_title("Vulnerability Signal Magnitude Across Layers", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        output_file = self.output_dir / "fig_paired_distances.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {output_file}")
        plt.close()

    def generate_cwe_transfer_heatmaps(self):
        """Generate figure: cross-family CWE transfer rates."""
        logger.info("Generating CWE transfer heatmaps...")

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        families = [
            "memory_safety",
            "injection",
            "resource",
            "info_disclosure",
            "control_flow",
        ]

        for idx, model in enumerate(MODELS):
            key = f"{model}_cwe_universality"
            if key not in self.results:
                logger.warning(f"No CWE universality results for {model}")
                continue

            data = self.results[key]
            transfer = data.get("cross_family_transfer", {})

            # Build matrix
            n_fam = len(families)
            matrix = np.zeros((n_fam, n_fam))

            for i, src_fam in enumerate(families):
                for j, tgt_fam in enumerate(families):
                    key_str = f"{src_fam}->{tgt_fam}"
                    if key_str in transfer:
                        matrix[i, j] = transfer[key_str].get("mean", 50)
                    elif i == j:
                        matrix[i, j] = 100  # Diagonal is 100%

            # Plot heatmap
            im = axes[idx].imshow(matrix, cmap="RdYlGn", vmin=50, vmax=100)
            axes[idx].set_xticks(range(n_fam))
            axes[idx].set_yticks(range(n_fam))
            axes[idx].set_xticklabels([f.split("_")[0] for f in families], rotation=45)
            axes[idx].set_yticklabels([f.split("_")[0] for f in families])
            axes[idx].set_xlabel("Target Family")
            axes[idx].set_ylabel("Source Family")
            axes[idx].set_title(f"{model.capitalize()} CWE Transfer Rates")

            plt.colorbar(im, ax=axes[idx], label="Transfer Rate (%)")

        plt.tight_layout()
        output_file = self.output_dir / "fig_cwe_transfer_heatmaps.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {output_file}")
        plt.close()

    def generate_ranking_accuracy_comparison(self):
        """Generate figure: pairwise ranking accuracy across models."""
        logger.info("Generating ranking accuracy comparison...")

        fig, ax = plt.subplots(figsize=(10, 6))

        for model in MODELS:
            key = f"{model}_paired_ranking"
            if key not in self.results:
                continue

            data = self.results[key]
            ranking = data.get("ranking_accuracy", {})

            accuracies = []
            for layer in LAYERS:
                layer_str = str(layer)
                if layer_str in ranking:
                    accuracies.append(ranking[layer_str]["accuracy"])
                else:
                    accuracies.append(None)

            ax.plot(
                LAYERS,
                accuracies,
                marker="D",
                label=model.capitalize(),
                color=COLORS[model],
                linewidth=2,
                markersize=6,
            )

        ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="Chance (50%)")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Pairwise Ranking Accuracy (%)", fontsize=12)
        ax.set_title("Vulnerability Ranking Accuracy Across Layers", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([45, 100])

        output_file = self.output_dir / "fig_ranking_accuracy.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {output_file}")
        plt.close()

    def generate_summary_table(self):
        """Generate summary statistics table as text and LaTeX."""
        logger.info("Generating summary statistics...")

        summary = {
            "model": [],
            "n_pairs": [],
            "peak_alignment": [],
            "peak_layer": [],
            "mean_alignment": [],
        }

        for model in MODELS:
            key = f"{model}_direction_geometry"
            if key not in self.results:
                continue

            data = self.results[key]
            summary["model"].append(model.capitalize())
            summary["n_pairs"].append(data.get("n_pairs", 0))

            layers_data = data.get("layers", {})
            alignments = [v["pct_aligned"] for v in layers_data.values()]

            if alignments:
                peak_alignment = max(alignments)
                peak_layer = LAYERS[alignments.index(peak_alignment)]
                mean_alignment = np.mean(alignments)

                summary["peak_alignment"].append(f"{peak_alignment:.1f}")
                summary["peak_layer"].append(str(peak_layer))
                summary["mean_alignment"].append(f"{mean_alignment:.1f}")

        # Save as text
        summary_file = self.output_dir / "summary_statistics.txt"
        with open(summary_file, "w") as f:
            f.write("Model\tN Pairs\tPeak Alignment\tPeak Layer\tMean Alignment\n")
            for i in range(len(summary["model"])):
                f.write(
                    f"{summary['model'][i]}\t{summary['n_pairs'][i]}\t"
                    f"{summary['peak_alignment'][i]}\t{summary['peak_layer'][i]}\t"
                    f"{summary['mean_alignment'][i]}\n"
                )

        logger.info(f"Saved: {summary_file}")

    def generate_all_figures(self):
        """Generate all figures."""
        logger.info("Starting figure generation...")

        self.generate_direction_alignment_heatmap()
        self.generate_per_pair_alignment_curves()
        self.generate_paired_distance_curves()
        self.generate_cwe_transfer_heatmaps()
        self.generate_ranking_accuracy_comparison()
        self.generate_summary_table()

        logger.info("All figures generated successfully!")


def main():
    parser = argparse.ArgumentParser(description="Generate all paper figures")
    parser.add_argument(
        "--results-dir",
        default="/Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection/results",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/rmelo/Documents/GitHub/On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures",
        help="Output directory for figures",
    )

    args = parser.parse_args()

    generator = FigureGenerator(args.results_dir, args.output_dir)
    generator.generate_all_figures()


if __name__ == "__main__":
    main()
