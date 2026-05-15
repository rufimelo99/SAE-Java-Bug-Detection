#!/usr/bin/env python3
"""
Comprehensive pipeline to run all experiments for three models and store raw JSON data.

Models: Qwen2.5-7B-Instruct, CodeLlama-7B-Instruct, StarCoder2-7B
Experiments: direction geometry, CWE universality, paired ranking, cross-family transfer

Usage:
    python run_all_experiments.py --models qwen,codellama,starcoder2
    python run_all_experiments.py --models qwen --language c
    python run_all_experiments.py --activations-dir /path/to/activations
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]

CWE_FAMILIES = {
    "memory_safety": ["CWE-119", "CWE-120", "CWE-125", "CWE-476", "CWE-787", "CWE-416"],
    "injection": ["CWE-20", "CWE-22", "CWE-78", "CWE-89"],
    "resource": ["CWE-401", "CWE-399", "CWE-415", "CWE-362", "CWE-400"],
    "info_disclosure": ["CWE-200"],
    "control_flow": ["CWE-190", "CWE-264"],
}


def load_metadata(metadata_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load metadata from JSONL file."""
    logger.info(f"Loading metadata from {metadata_file}")
    cwes, vulns, file_exts = [], [], []

    with open(metadata_file) as f:
        for line in f:
            data = json.loads(line)
            cwes.append(data.get("cwe", "unknown"))
            vulns.append(int(data.get("vulnerable", 0)))
            file_exts.append(data.get("file_extension", "unknown"))

    logger.info(f"Loaded metadata for {len(cwes)} pairs")
    return np.array(cwes), np.array(vulns), np.array(file_exts)


def load_activations_npz(npz_file: Path) -> Dict[str, np.ndarray]:
    """Load activations from NPZ file."""
    logger.info(f"Loading activations from {npz_file.name}")
    data = np.load(npz_file)
    activations = {}
    for key in data.files:
        activations[key] = data[key]
    logger.info(f"Loaded {len(activations)} activation arrays")
    return activations


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
        """Compute direction geometry across layers."""
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
            key_vuln = f"layer_{layer}_vuln_mean"
            key_sec = f"layer_{layer}_secure_mean"

            if key_vuln not in activations or key_sec not in activations:
                logger.warning(f"Layer {layer} not found for {model}")
                continue

            vuln_acts = activations[key_vuln]
            sec_acts = activations[key_sec]

            # Compute global direction
            vuln_mean_global = vuln_acts.mean(axis=0)
            sec_mean_global = sec_acts.mean(axis=0)
            direction = vuln_mean_global - sec_mean_global
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            layer_directions[layer] = direction

            # Per-pair alignment
            deltas = vuln_acts - sec_acts
            dots = deltas @ direction
            pct_aligned = (dots > 0).mean() * 100

            # Paired distances
            distances = np.linalg.norm(deltas, axis=1)

            results["layers"][str(layer)] = {
                "pct_aligned": float(pct_aligned),
                "mean_paired_distance": float(distances.mean()),
                "std_paired_distance": float(distances.std()),
                "median_paired_distance": float(np.median(distances)),
                "n_pairs": int(len(vuln_acts)),
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
        logger.info(f"✓ Direction geometry complete for {model}")
        return results

    def run_cwe_universality(
        self,
        model: str,
        activations: Dict[str, np.ndarray],
        labels: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Test cross-family vulnerability direction transfer."""
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
            key_vuln = f"layer_{layer}_vuln_mean"
            key_sec = f"layer_{layer}_secure_mean"

            if key_vuln not in activations or key_sec not in activations:
                continue

            vuln_acts = activations[key_vuln]
            sec_acts = activations[key_sec]

            layer_transfer = {}

            for src_family, src_mask in family_masks.items():
                src_vuln = vuln_acts[src_mask]
                src_sec = sec_acts[src_mask]

                if len(src_vuln) == 0 or len(src_sec) == 0:
                    continue

                # Compute direction for this family
                direction = src_vuln.mean(axis=0) - src_sec.mean(axis=0)
                direction = direction / (np.linalg.norm(direction) + 1e-10)

                for tgt_family, tgt_mask in family_masks.items():
                    tgt_vuln = vuln_acts[tgt_mask]
                    tgt_sec = sec_acts[tgt_mask]

                    if len(tgt_vuln) == 0 or len(tgt_sec) == 0:
                        continue

                    # Test transfer
                    deltas = tgt_vuln - tgt_sec
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
        logger.info(f"✓ CWE universality complete for {model}")
        return results

    def run_paired_ranking(
        self,
        model: str,
        activations: Dict[str, np.ndarray],
        labels: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Evaluate pairwise ranking accuracy."""
        logger.info(f"Computing paired ranking accuracy for {model}...")

        results = {
            "model": model,
            "experiment": "paired_ranking",
            "ranking_accuracy": {},
        }

        for layer in LAYERS:
            key_vuln = f"layer_{layer}_vuln_mean"
            key_sec = f"layer_{layer}_secure_mean"

            if key_vuln not in activations or key_sec not in activations:
                continue

            vuln_acts = activations[key_vuln]
            sec_acts = activations[key_sec]

            # Compute global direction
            vuln_mean = vuln_acts.mean(axis=0)
            sec_mean = sec_acts.mean(axis=0)
            direction = vuln_mean - sec_mean
            direction = direction / (np.linalg.norm(direction) + 1e-10)

            # Project and rank
            vuln_proj = vuln_acts @ direction
            sec_proj = sec_acts @ direction

            # Pairwise ranking: for each pair, does vulnerable project higher?
            n_vuln = len(vuln_proj)
            n_sec = len(sec_proj)
            correct = (vuln_proj[:, None] > sec_proj[None, :]).sum()
            total = n_vuln * n_sec

            accuracy = correct / total * 100 if total > 0 else 0

            results["ranking_accuracy"][str(layer)] = {
                "accuracy": float(accuracy),
                "n_vulnerable": int(n_vuln),
                "n_secure": int(n_sec),
                "total_pairs": int(total),
            }

        self._save_result(model, "paired_ranking", results)
        logger.info(f"✓ Paired ranking complete for {model}")
        return results

    def _save_result(self, model: str, experiment: str, data: Dict[str, Any]):
        """Save result to JSON file."""
        output_file = self.raw_data_dir / f"{model}_{experiment}.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"  Saved: {output_file.name}")


def main():
    parser = argparse.ArgumentParser(description="Run all experiments across models")
    parser.add_argument(
        "--models",
        default="qwen,codellama,starcoder2",
        help="Comma-separated list of models",
    )
    parser.add_argument(
        "--output-dir",
        default="/work7/ruimelo/SAE-Java-Bug-Detection/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--activations-dir",
        default="/work7/ruimelo/SAE-Java-Bug-Detection/sae_java_bug/artifacts/multi_model_probing",
        help="Directory containing activation NPZ files",
    )
    parser.add_argument(
        "--metadata-file",
        default="/work7/ruimelo/SAE-Java-Bug-Detection/sae_java_bug/artifacts/activations/TO_UPLOAD/activations_layer_11_sae_blocks.11.hook_resid_post_component_hook_resid_post.hook_sae_acts_post.jsonl",
        help="Path to metadata JSONL file",
    )
    parser.add_argument("--language", default="c", help="Language to analyze (c, all)")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip existing results"
    )

    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")]
    activations_dir = Path(args.activations_dir)
    metadata_file = Path(args.metadata_file)
    output_dir = Path(args.output_dir)

    logger.info(f"{'='*70}")
    logger.info("Vulnerability Representation Analysis Pipeline")
    logger.info(f"{'='*70}")
    logger.info(f"Models: {models}")
    logger.info(f"Language: {args.language}")
    logger.info(f"Output: {output_dir}")

    # Validate paths
    if not metadata_file.exists():
        logger.error(f"Metadata file not found: {metadata_file}")
        return 1

    if not activations_dir.exists():
        logger.error(f"Activations directory not found: {activations_dir}")
        return 1

    # Load metadata once
    cwes, vulns, file_exts = load_metadata(metadata_file)

    # Filter by language if specified
    if args.language != "all":
        lang_mask = file_exts == args.language
        cwes = cwes[lang_mask]
        vulns = vulns[lang_mask]
        file_exts = file_exts[lang_mask]
        logger.info(f"Filtered to {len(cwes)} {args.language} pairs")

    pipeline = ExperimentPipeline(output_dir, args.skip_existing)

    # Run experiments for each model
    completed = 0
    for model_short in models:
        logger.info(f"\n{'='*70}")
        logger.info(f"Model: {model_short}")
        logger.info(f"{'='*70}")

        # Find NPZ file
        npz_files = list(activations_dir.glob(f"activations_{model_short}*.npz"))

        if not npz_files:
            logger.error(f"No NPZ files found for {model_short}")
            logger.info(f"Available files:")
            for f in sorted(activations_dir.glob("activations_*.npz")):
                logger.info(f"  {f.name}")
            continue

        npz_file = npz_files[0]

        try:
            activations = load_activations_npz(npz_file)
            labels = {
                "cwe": cwes,
                "vulnerable": vulns,
                "file_extension": file_exts,
            }

            # Run all three experiments
            pipeline.run_direction_geometry(model_short, activations, labels)
            pipeline.run_cwe_universality(model_short, activations, labels)
            pipeline.run_paired_ranking(model_short, activations, labels)

            completed += 1

        except Exception as e:
            logger.error(f"Error processing {model_short}: {e}", exc_info=True)

    logger.info(f"\n{'='*70}")
    logger.info(f"Completed {completed}/{len(models)} models")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*70}\n")

    return 0 if completed == len(models) else 1


if __name__ == "__main__":
    exit(main())
