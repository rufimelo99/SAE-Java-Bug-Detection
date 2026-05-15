#!/usr/bin/env python3
"""
Comprehensive pipeline to run all experiments for three models and store raw JSON data.

Models: Qwen2.5-7B-Instruct, CodeLlama-7B-Instruct, StarCoder2-7B
Experiments: direction geometry, CWE universality, paired ranking, cross-family transfer

Usage:
    python run_all_experiments.py --models qwen,codellama,starcoder2 --output-dir ./results
    python run_all_experiments.py --models qwen --skip-existing
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODELS = {
    "qwen": "Qwen2.5-7B-Instruct",
    "codellama": "CodeLlama-7B-Instruct",
    "starcoder2": "StarCoder2-7B",
}

LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]

CWE_FAMILIES = {
    "memory_safety": ["CWE-119", "CWE-120", "CWE-125", "CWE-476", "CWE-787", "CWE-416"],
    "injection": ["CWE-20", "CWE-22", "CWE-78", "CWE-89"],
    "resource": ["CWE-401", "CWE-399", "CWE-415", "CWE-362", "CWE-400"],
    "info_disclosure": ["CWE-200"],
    "control_flow": ["CWE-190", "CWE-264"],
}


class ExperimentPipeline:
    """Main pipeline for running all experiments across models."""

    def __init__(self, output_dir: Path, skip_existing: bool = False):
        self.output_dir = Path(output_dir)
        self.skip_existing = skip_existing
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.raw_data_dir = self.output_dir / "raw_data"
        self.results_dir = self.output_dir / "results"
        self.logs_dir = self.output_dir / "logs"

        for d in [self.raw_data_dir, self.results_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory: {self.output_dir}")

    def run_direction_geometry(
        self,
        model: str,
        activations: Dict[str, np.ndarray],
        labels: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Compute direction geometry across layers.

        Args:
            model: Model name (qwen, codellama, starcoder2)
            activations: Dict of layer activations
            labels: Dict of labels (vulnerable, file_extension, cwe)

        Returns:
            Dict with direction geometry results
        """
        logger.info(f"Computing direction geometry for {model}...")

        results = {
            "model": model,
            "experiment": "direction_geometry",
            "n_pairs": len(labels["vulnerable"]),
            "layers": {},
            "cross_layer_cosines": {},
        }

        layer_directions = {}

        for layer in LAYERS:
            if f"layer_{layer}" not in activations:
                logger.warning(f"Layer {layer} not found for {model}")
                continue

            acts = activations[f"layer_{layer}"]
            vuln_mask = labels["vulnerable"] == 1

            vuln_mean = acts[vuln_mask].mean(axis=0)
            secure_mean = acts[~vuln_mask].mean(axis=0)

            # Compute direction
            direction = vuln_mean - secure_mean
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            layer_directions[layer] = direction

            # Per-pair alignment
            deltas = acts[vuln_mask] - acts[~vuln_mask]
            dots = deltas @ direction
            pct_aligned = (dots > 0).mean() * 100

            # Paired distances
            distances = np.linalg.norm(acts[vuln_mask] - acts[~vuln_mask], axis=1)

            results["layers"][str(layer)] = {
                "pct_aligned": float(pct_aligned),
                "mean_paired_distance": float(distances.mean()),
                "std_paired_distance": float(distances.std()),
                "median_paired_distance": float(np.median(distances)),
            }

        # Cross-layer cosine similarities
        for l1 in LAYERS:
            if l1 not in layer_directions:
                continue
            for l2 in LAYERS:
                if l2 not in layer_directions:
                    continue
                cosine = np.dot(layer_directions[l1], layer_directions[l2])
                key = f"{l1}-{l2}"
                results["cross_layer_cosines"][key] = float(cosine)

        self._save_result(model, "direction_geometry", results)
        return results

    def run_cwe_universality(
        self,
        model: str,
        activations: Dict[str, np.ndarray],
        labels: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Test cross-family vulnerability direction transfer.

        Returns:
            Dict with transfer rates between CWE families
        """
        logger.info(f"Computing CWE universality for {model}...")

        results = {
            "model": model,
            "experiment": "cwe_universality",
            "cross_family_transfer": {},
            "family_stats": {},
        }

        cwes = labels["cwe"]

        # Build family masks
        family_masks = {}
        for family, cwe_list in CWE_FAMILIES.items():
            family_masks[family] = np.isin(cwes, cwe_list)
            count = family_masks[family].sum()
            results["family_stats"][family] = {"n_pairs": int(count)}
            logger.info(f"  {family}: {count} pairs")

        # For each layer, compute transfer rates
        transfer_by_layer = {}
        for layer in LAYERS:
            if f"layer_{layer}" not in activations:
                continue

            acts = activations[f"layer_{layer}"]
            vuln_mask = labels["vulnerable"] == 1

            layer_transfer = {}
            for src_family, src_mask in family_masks.items():
                src_acts_vuln = acts[src_mask & vuln_mask]
                src_acts_sec = acts[src_mask & ~vuln_mask]

                if len(src_acts_vuln) == 0 or len(src_acts_sec) == 0:
                    continue

                # Compute direction for this family
                direction = src_acts_vuln.mean(axis=0) - src_acts_sec.mean(axis=0)
                direction = direction / (np.linalg.norm(direction) + 1e-10)

                for tgt_family, tgt_mask in family_masks.items():
                    tgt_acts_vuln = acts[tgt_mask & vuln_mask]
                    tgt_acts_sec = acts[tgt_mask & ~vuln_mask]

                    if len(tgt_acts_vuln) == 0 or len(tgt_acts_sec) == 0:
                        continue

                    # Test transfer
                    deltas = tgt_acts_vuln - tgt_acts_sec
                    dots = deltas @ direction
                    pct_aligned = (dots > 0).mean() * 100

                    key = f"{src_family}->{tgt_family}"
                    layer_transfer[key] = float(pct_aligned)

            transfer_by_layer[str(layer)] = layer_transfer

        results["transfer_by_layer"] = transfer_by_layer

        # Aggregate across layers
        all_transfers = defaultdict(list)
        for layer_transfers in transfer_by_layer.values():
            for key, val in layer_transfers.items():
                all_transfers[key].append(val)

        for key, vals in all_transfers.items():
            results["cross_family_transfer"][key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

        self._save_result(model, "cwe_universality", results)
        return results

    def run_paired_ranking(
        self,
        model: str,
        activations: Dict[str, np.ndarray],
        labels: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Evaluate pairwise ranking accuracy: does vulnerable rank higher than secure?

        Returns:
            Dict with ranking accuracy by layer
        """
        logger.info(f"Computing paired ranking accuracy for {model}...")

        results = {
            "model": model,
            "experiment": "paired_ranking",
            "ranking_accuracy": {},
        }

        vuln_mask = labels["vulnerable"] == 1

        for layer in LAYERS:
            if f"layer_{layer}" not in activations:
                continue

            acts = activations[f"layer_{layer}"]

            # Compute global direction
            vuln_mean = acts[vuln_mask].mean(axis=0)
            secure_mean = acts[~vuln_mask].mean(axis=0)
            direction = vuln_mean - secure_mean
            direction = direction / (np.linalg.norm(direction) + 1e-10)

            # Project and rank
            vuln_proj = acts[vuln_mask] @ direction
            sec_proj = acts[~vuln_mask] @ direction

            # Pairwise ranking: for each pair, does vulnerable project higher?
            n_vuln = len(vuln_proj)
            n_sec = len(sec_proj)

            correct = 0
            total = n_vuln * n_sec

            for v_proj in vuln_proj:
                correct += (v_proj > sec_proj).sum()

            accuracy = correct / total * 100 if total > 0 else 0

            results["ranking_accuracy"][str(layer)] = {
                "accuracy": float(accuracy),
                "n_vulnerable": int(n_vuln),
                "n_secure": int(n_sec),
                "total_pairs": int(total),
            }

        self._save_result(model, "paired_ranking", results)
        return results

    def _save_result(self, model: str, experiment: str, data: Dict[str, Any]):
        """Save result to JSON file."""
        output_file = self.raw_data_dir / f"{model}_{experiment}.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved: {output_file}")

    def run_all_experiments(self, models: List[str], activations_dict: Dict[str, Dict]):
        """Run all experiments for specified models."""
        summary = {}

        for model in models:
            logger.info(f"\n{'='*70}")
            logger.info(f"Running experiments for {model}")
            logger.info(f"{'='*70}")

            model_name = MODELS.get(model, model)

            # Check if activations exist
            if model not in activations_dict:
                logger.error(f"No activations found for {model}")
                continue

            activations = activations_dict[model]["activations"]
            labels = activations_dict[model]["labels"]

            summary[model] = {}

            # Run experiments
            try:
                summary[model]["direction_geometry"] = self.run_direction_geometry(
                    model, activations, labels
                )
            except Exception as e:
                logger.error(f"Direction geometry failed for {model}: {e}")

            try:
                summary[model]["cwe_universality"] = self.run_cwe_universality(
                    model, activations, labels
                )
            except Exception as e:
                logger.error(f"CWE universality failed for {model}: {e}")

            try:
                summary[model]["paired_ranking"] = self.run_paired_ranking(
                    model, activations, labels
                )
            except Exception as e:
                logger.error(f"Paired ranking failed for {model}: {e}")

        # Save summary
        summary_file = self.results_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nSummary saved to: {summary_file}")

        return summary


def main():
    parser = argparse.ArgumentParser(description="Run all experiments across models")
    parser.add_argument(
        "--models",
        default="qwen,codellama,starcoder2",
        help="Comma-separated list of models (qwen, codellama, starcoder2)",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip existing results"
    )

    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")]

    logger.info(f"Running experiments for models: {models}")
    logger.info(f"Output directory: {args.output_dir}")

    # TODO: Load activations from cache or compute
    # For now, this is a template showing the structure
    activations_dict = {}  # Load from your activation cache

    pipeline = ExperimentPipeline(args.output_dir, args.skip_existing)
    # pipeline.run_all_experiments(models, activations_dict)


if __name__ == "__main__":
    main()
