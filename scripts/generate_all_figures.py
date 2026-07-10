#!/usr/bin/env python3
"""
Generate all paper figures for all models from raw experiment results.

This script reads JSON results from run_all_experiments_FIXED.py and generates:
- Direction alignment heatmaps
- Per-pair consistency plots
- Cross-family transfer heatmaps
- Paired ranking accuracy curves
- Comparison plots across models

Models and layers are discovered automatically from the JSON files present.

Usage:
    python generate_all_figures.py --results-dir ./results --output-dir ./figures
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Enough distinct colours for up to ~12 models
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78",
]


def _model_color(model: str, model_list: List[str]) -> str:
    return _PALETTE[model_list.index(model) % len(_PALETTE)]


def _subplot_grid(n: int):
    """Return (nrows, ncols) for n subplots, favouring landscape layout."""
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def _get_layers_from_result(data: dict) -> List[int]:
    """Extract sorted layer indices from a direction_geometry result dict."""
    return sorted(int(k) for k in data.get("layers", {}).keys())


class FigureGenerator:
    """Generate publication-quality figures from experiment results."""

    def __init__(self, results_dir: Path, output_dir: Path):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict[str, Any] = {}
        self._load_results()

        # Discover models that have at least direction_geometry results
        self.models = sorted({
            k.replace("_direction_geometry", "")
            for k in self.results
            if k.endswith("_direction_geometry")
        })

        logger.info(f"Loaded results from {self.results_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Models found: {self.models}")

    def _load_results(self):
        raw_data_dir = self.results_dir / "raw_data"
        if not raw_data_dir.exists():
            logger.warning(f"raw_data directory not found: {raw_data_dir}")
            return
        for json_file in raw_data_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    self.results[json_file.stem] = json.load(f)
                logger.info(f"Loaded: {json_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")

    # ── Figure 1: direction alignment heatmaps ─────────────────────────────────

    def generate_direction_alignment_heatmap(self):
        logger.info("Generating direction alignment heatmap...")
        n = len(self.models)
        if n == 0:
            logger.warning("No models with direction_geometry results — skipping.")
            return

        nrows, ncols = _subplot_grid(n)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

        im = None
        for idx, model in enumerate(self.models):
            key = f"{model}_direction_geometry"
            ax = axes[idx]
            if key not in self.results:
                ax.set_visible(False)
                continue

            data = self.results[key]
            layers = _get_layers_from_result(data)
            cosines = data.get("cross_layer_cosines", {})

            matrix = np.zeros((len(layers), len(layers)))
            for i, l1 in enumerate(layers):
                for j, l2 in enumerate(layers):
                    matrix[i, j] = cosines.get(f"{l1}-{l2}", 0)

            im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)
            tick_step = max(1, len(layers) // 8)
            ticks = list(range(0, len(layers), tick_step))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels([layers[i] for i in ticks], fontsize=11)
            ax.set_yticklabels([layers[i] for i in ticks], fontsize=11)
            ax.set_xlabel("Layer", fontsize=12)
            ax.set_ylabel("Layer", fontsize=12)
            ax.set_title(model, fontsize=14, fontweight="bold")

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        if im is not None:
            fig.subplots_adjust(right=0.88)
            cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
            fig.colorbar(im, cax=cbar_ax, label="Cosine Sim")
            cbar_ax.tick_params(labelsize=11)
            cbar_ax.yaxis.label.set_size(12)

        out = self.output_dir / "fig_direction_alignment_heatmaps.pdf"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {out}")
        plt.close()

    # ── Figure 2: per-pair alignment curves ───────────────────────────────────

    def generate_per_pair_alignment_curves(self):
        logger.info("Generating per-pair alignment curves...")
        fig, ax = plt.subplots(figsize=(10, 6))

        for model in self.models:
            key = f"{model}_direction_geometry"
            if key not in self.results:
                continue
            data = self.results[key]
            layers = _get_layers_from_result(data)
            layers_data = data.get("layers", {})
            alignment = [layers_data.get(str(l), {}).get("pct_aligned") for l in layers]

            ax.plot(layers, alignment, marker="o", label=model,
                    color=_model_color(model, self.models), linewidth=2, markersize=4)

        ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="Chance (50%)")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Per-Pair Alignment (%)", fontsize=12)
        ax.set_title("Vulnerability Direction Alignment Across Layers", fontsize=14)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([40, 100])

        out = self.output_dir / "fig_per_pair_alignment.pdf"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {out}")
        plt.close()

    # ── Figure 3: paired distance curves ──────────────────────────────────────

    def generate_paired_distance_curves(self):
        logger.info("Generating paired distance curves...")
        fig, ax = plt.subplots(figsize=(10, 6))

        for model in self.models:
            key = f"{model}_direction_geometry"
            if key not in self.results:
                continue
            data = self.results[key]
            layers = _get_layers_from_result(data)
            layers_data = data.get("layers", {})
            distances = [layers_data.get(str(l), {}).get("mean_paired_distance") for l in layers]

            ax.plot(layers, distances, marker="s", label=model,
                    color=_model_color(model, self.models), linewidth=2, markersize=4)

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Mean Paired Distance (L2)", fontsize=12)
        ax.set_title("Vulnerability Signal Magnitude Across Layers", fontsize=14)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        out = self.output_dir / "fig_paired_distances.pdf"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {out}")
        plt.close()

    # ── Figure 4: CWE transfer heatmaps ───────────────────────────────────────

    def generate_cwe_transfer_heatmaps(self):
        logger.info("Generating CWE transfer heatmaps...")
        models_with_cwe = [m for m in self.models if f"{m}_cwe_universality" in self.results]
        n = len(models_with_cwe)
        if n == 0:
            logger.warning("No CWE universality results — skipping.")
            return

        families = ["memory_safety", "injection", "resource", "info_disclosure", "control_flow"]
        nrows, ncols = _subplot_grid(n)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

        for idx, model in enumerate(models_with_cwe):
            ax = axes[idx]
            data = self.results[f"{model}_cwe_universality"]
            transfer = data.get("cross_family_transfer", {})

            matrix = np.zeros((len(families), len(families)))
            for i, src in enumerate(families):
                for j, tgt in enumerate(families):
                    key = f"{src}->{tgt}"
                    matrix[i, j] = transfer[key]["mean"] if key in transfer else (100 if i == j else 50)

            im = ax.imshow(matrix, cmap="RdYlGn", vmin=50, vmax=100)
            labels = [f.split("_")[0] for f in families]
            ax.set_xticks(range(len(families)))
            ax.set_yticks(range(len(families)))
            ax.set_xticklabels(labels, rotation=45, fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("Target", fontsize=9)
            ax.set_ylabel("Source", fontsize=9)
            ax.set_title(model, fontsize=10)
            plt.colorbar(im, ax=ax, label="Transfer (%)")

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("CWE Cross-Family Transfer Rates", fontsize=13, y=1.01)
        plt.tight_layout()
        out = self.output_dir / "fig_cwe_transfer_heatmaps.pdf"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {out}")
        plt.close()

    # ── Figure 5: ranking accuracy ─────────────────────────────────────────────

    def generate_ranking_accuracy_comparison(self):
        logger.info("Generating ranking accuracy comparison...")
        fig, ax = plt.subplots(figsize=(10, 6))

        for model in self.models:
            key = f"{model}_paired_ranking"
            if key not in self.results:
                continue
            data = self.results[key]
            ranking = data.get("ranking_accuracy", {})
            layers = sorted(int(k) for k in ranking.keys())
            accuracies = [ranking[str(l)]["accuracy"] for l in layers]

            ax.plot(layers, accuracies, marker="D", label=model,
                    color=_model_color(model, self.models), linewidth=2, markersize=4)

        ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="Chance (50%)")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Pairwise Ranking Accuracy (%)", fontsize=12)
        ax.set_title("Vulnerability Ranking Accuracy Across Layers", fontsize=14)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([45, 100])

        out = self.output_dir / "fig_ranking_accuracy.pdf"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {out}")
        plt.close()

    # ── Summary table ──────────────────────────────────────────────────────────

    def generate_summary_table(self):
        logger.info("Generating summary statistics...")
        rows = []
        for model in self.models:
            key = f"{model}_direction_geometry"
            if key not in self.results:
                continue
            data = self.results[key]
            layers_data = data.get("layers", {})
            alignments = [v["pct_aligned"] for v in layers_data.values()]
            if not alignments:
                continue
            peak = max(alignments)
            peak_layer = list(layers_data.keys())[alignments.index(peak)]
            rows.append((model, data.get("n_pairs", 0), f"{peak:.1f}", peak_layer, f"{np.mean(alignments):.1f}"))

        summary_file = self.output_dir / "summary_statistics.txt"
        with open(summary_file, "w") as f:
            f.write("Model\tN Pairs\tPeak Alignment\tPeak Layer\tMean Alignment\n")
            for row in rows:
                f.write("\t".join(str(x) for x in row) + "\n")
        logger.info(f"Saved: {summary_file}")

    def generate_all_figures(self):
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
        default="/work7/ruimelo/SAE-Java-Bug-Detection/results",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output-dir",
        default="/work7/ruimelo/SAE-Java-Bug-Detection/../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures",
        help="Output directory for figures",
    )
    args = parser.parse_args()
    generator = FigureGenerator(Path(args.results_dir), Path(args.output_dir))
    generator.generate_all_figures()


if __name__ == "__main__":
    main()
