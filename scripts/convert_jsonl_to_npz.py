#!/usr/bin/env python3
"""
Convert JSONL activation files to NPZ format for multi-dataset probing.

This script reads JSONL activation files from SVEN and PreciseBugs datasets
and converts them to NPZ format compatible with the probing pipeline.

Usage:
    python scripts/convert_jsonl_to_npz.py [--datasets sven,precisebugs]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent / "sae_java_bug" / "artifacts"
ACTIVATIONS_DIR = ARTIFACTS_DIR / "activations"
OUTPUT_DIR = ARTIFACTS_DIR / "multi_model_probing"

DATASETS = {
    "deltasecommits": "c_only",
    "sven": "sven_c_only",
    "precisebugs": "precisebugs_c_only",
}

LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]


def load_jsonl_activations(dataset_dir: Path, layer: int) -> np.ndarray:
    """Load activations from JSONL train+test files."""
    activations = []

    for split in ["train", "test"]:
        jsonl_file = dataset_dir / f"layer_{layer}_{split}.jsonl"
        if not jsonl_file.exists():
            logger.warning(f"Missing: {jsonl_file}")
            continue

        logger.info(f"Loading {jsonl_file.name}...")
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    activation = np.array(record.get("activation", []))
                    if activation.size > 0:
                        activations.append(activation)

    if not activations:
        raise ValueError(f"No activations loaded for layer {layer}")

    return np.array(activations)


def load_metadata(dataset_dir: Path) -> tuple:
    """Load metadata (CWE labels and file extensions)."""
    meta_file = dataset_dir / "split_meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_file}")

    logger.info(f"Loading metadata from {meta_file}...")
    with open(meta_file) as f:
        meta_list = json.load(f)

    # Extract CWEs from list of records
    cwes = np.array([record.get("cwe", "unknown") for record in meta_list])
    file_exts = np.array([record.get("file_ext", "c") for record in meta_list])

    logger.info(
        f"Loaded metadata: {len(cwes)} pairs, {len(np.unique(cwes))} unique CWEs"
    )
    return cwes, file_exts


def convert_dataset(dataset_name: str, dataset_dir: str) -> None:
    """Convert a single dataset from JSONL to NPZ."""
    dataset_path = ACTIVATIONS_DIR / dataset_dir
    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        return

    logger.info(f"\nConverting {dataset_name} ({dataset_dir})...")

    # Load metadata once
    cwes, file_exts = load_metadata(dataset_path)

    # Load and convert activations for each layer
    layer_data = {}
    for layer in LAYERS:
        try:
            activations = load_jsonl_activations(dataset_path, layer)
            layer_data[f"layer_{layer}_vuln_mean"] = activations
            logger.info(f"  Layer {layer}: {activations.shape}")
        except Exception as e:
            logger.warning(f"  Layer {layer}: {e}")

    if not layer_data:
        logger.error(f"No activations loaded for {dataset_name}")
        return

    # Save as NPZ
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"activations_{dataset_name}-7b.npz"

    np.savez(output_file, **layer_data)
    logger.info(f"✓ Saved: {output_file}")

    # Also save metadata for reference
    meta_output = OUTPUT_DIR / f"metadata_{dataset_name}.json"
    with open(meta_output, "w") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "n_samples": len(cwes),
                "n_cwes": len(np.unique(cwes)),
                "unique_cwes": sorted(np.unique(cwes).tolist()),
                "n_languages": len(np.unique(file_exts)),
                "languages": sorted(np.unique(file_exts).tolist()),
                "layers": list(layer_data.keys()),
            },
            f,
            indent=2,
        )
    logger.info(f"✓ Saved metadata: {meta_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL activations to NPZ format"
    )
    parser.add_argument(
        "--datasets",
        default="sven,precisebugs",
        help="Comma-separated list of datasets to convert (sven, precisebugs)",
    )
    args = parser.parse_args()

    datasets_to_convert = [d.strip().lower() for d in args.datasets.split(",")]

    logger.info("=" * 70)
    logger.info("Converting JSONL Activations to NPZ Format")
    logger.info("=" * 70)

    for dataset_name in datasets_to_convert:
        if dataset_name not in DATASETS:
            logger.error(f"Unknown dataset: {dataset_name}")
            continue

        dataset_dir = DATASETS[dataset_name]
        try:
            convert_dataset(dataset_name, dataset_dir)
        except Exception as e:
            logger.error(f"Failed to convert {dataset_name}: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("Conversion complete!")
    logger.info(f"NPZ files saved to: {OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
