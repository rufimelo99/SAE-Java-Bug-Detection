#!/usr/bin/env python3
"""
Compute cross-layer direction geometry for PreciseBugs security and non-security
subsets. This is the key control experiment: if the same directional alignment
appears for non-security bug-fix pairs, the signal reflects patch structure
rather than vulnerability-specific representation.

Saves results to results/raw_data/ under names that generate_all_figures.py
and regenerate_direction_alignment_*.py will auto-discover.

Usage:
    python scripts/compute_direction_geometry_precisebugs.py
    python scripts/compute_direction_geometry_precisebugs.py --subset security
    python scripts/compute_direction_geometry_precisebugs.py --subset nonsecurity
    python scripts/compute_direction_geometry_precisebugs.py --skip-existing
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).parent
_PROJECT_DIR = _SCRIPTS_DIR.parent

ACTIVATIONS_DIR = _PROJECT_DIR / "sae_java_bug" / "artifacts" / "multi_model_probing"
RESULTS_DIR = _PROJECT_DIR / "results" / "raw_data"


def get_layers(activations: Dict[str, np.ndarray]) -> List[int]:
    return sorted({
        int(k.split("_")[1])
        for k in activations
        if k.startswith("layer_") and k.endswith("_vuln_mean")
    })


def compute_direction_geometry(model_full: str, npz_path: Path) -> dict:
    logger.info(f"Loading {npz_path.name} ...")
    data = np.load(npz_path)
    activations = {k: data[k] for k in data.files}

    layers = get_layers(activations)
    if not layers:
        logger.warning(f"No layer keys found in {npz_path.name}")
        return {}

    n_pairs = activations[f"layer_{layers[0]}_vuln_mean"].shape[0]
    logger.info(f"  {n_pairs} pairs, {len(layers)} layers")

    results = {
        "model": model_full,
        "experiment": "direction_geometry",
        "n_pairs": n_pairs,
        "layers": {},
        "cross_layer_cosines": {},
    }

    layer_directions = {}
    for layer in layers:
        key_vuln = f"layer_{layer}_vuln_mean"
        key_sec = f"layer_{layer}_secure_mean"

        if key_vuln not in activations or key_sec not in activations:
            logger.warning(f"  Layer {layer} missing vuln/secure keys — skipping")
            continue

        vuln_acts = activations[key_vuln]
        sec_acts = activations[key_sec]

        direction = vuln_acts.mean(axis=0) - sec_acts.mean(axis=0)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        layer_directions[layer] = direction

        deltas = vuln_acts - sec_acts
        dots = deltas @ direction
        pct_aligned = float((dots > 0).mean() * 100)
        distances = np.linalg.norm(deltas, axis=1)

        results["layers"][str(layer)] = {
            "pct_aligned": pct_aligned,
            "mean_paired_distance": float(distances.mean()),
            "std_paired_distance": float(distances.std()),
            "median_paired_distance": float(np.median(distances)),
            "n_pairs": n_pairs,
        }
        logger.info(f"  Layer {layer}: pct_aligned={pct_aligned:.1f}%")

    for l1 in layer_directions:
        for l2 in layer_directions:
            cosine = float(np.dot(layer_directions[l1], layer_directions[l2]))
            results["cross_layer_cosines"][f"{l1}-{l2}"] = cosine

    return results


def save_results(model_full: str, results: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{model_full}_direction_geometry.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset", choices=["security", "nonsecurity", "both"], default="both",
        help="Which PreciseBugs subset to process (default: both)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip models that already have a direction_geometry JSON"
    )
    args = parser.parse_args()

    prefixes = []
    if args.subset in ("security", "both"):
        prefixes.append("precisebugs_security")
    if args.subset in ("nonsecurity", "both"):
        prefixes.append("precisebugs_nonsecurity")

    for prefix in prefixes:
        npz_files = sorted(ACTIVATIONS_DIR.glob(f"activations_{prefix}_*.npz"))
        if not npz_files:
            logger.warning(f"No NPZ files found for prefix '{prefix}' in {ACTIVATIONS_DIR}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Subset: {prefix} ({len(npz_files)} models)")
        logger.info(f"{'='*60}")

        for npz_path in npz_files:
            model_full = npz_path.stem.replace("activations_", "")
            out_path = RESULTS_DIR / f"{model_full}_direction_geometry.json"

            if args.skip_existing and out_path.exists():
                logger.info(f"Skipping {model_full} (already exists)")
                continue

            try:
                results = compute_direction_geometry(model_full, npz_path)
                if results:
                    save_results(model_full, results)
            except Exception as e:
                logger.error(f"Error processing {model_full}: {e}", exc_info=True)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
