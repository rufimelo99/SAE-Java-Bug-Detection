#!/usr/bin/env python3
"""
Multi-model C-only analysis: direction geometry and CWE universality.
Computes these analyses for CodeLlama-7B and StarCoder2-7B within C using cached .npz activations.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ARTIFACTS_DIR = Path(
    "/Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection/sae_java_bug/artifacts"
)
METADATA_FILE = (
    ARTIFACTS_DIR
    / "activations/TO_UPLOAD/activations_layer_11_sae_blocks.11.hook_resid_post_component_hook_resid_post.hook_sae_acts_post.jsonl"
)
MULTI_MODEL_DIR = ARTIFACTS_DIR / "multi_model_probing"

MODELS = ["codellama-7b", "starcoder2-7b"]
LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]

CWE_FAMILIES = {
    "memory_safety": ["CWE-119", "CWE-120", "CWE-125", "CWE-476", "CWE-787", "CWE-416"],
    "resource": ["CWE-401", "CWE-399", "CWE-415", "CWE-362", "CWE-400"],
    "info_disclosure": ["CWE-200"],
    "injection": ["CWE-20", "CWE-22", "CWE-78", "CWE-89"],
    "control_flow": ["CWE-190", "CWE-264"],
}

MIN_FAMILY_PAIRS = 30


def load_metadata():
    """Load file_extension and cwe from source JSONL."""
    exts, cwes = [], []
    with open(METADATA_FILE) as f:
        for line in f:
            data = json.loads(line)
            exts.append(data["file_extension"])
            cwes.append(data["cwe"])
    return np.array(exts), np.array(cwes)


def compute_direction_geometry_c(model: str):
    """Compute direction geometry within C for all layers."""
    print(f"\nComputing direction geometry for {model}...")

    file_exts, _ = load_metadata()
    c_mask = file_exts == "c"

    npz_file = MULTI_MODEL_DIR / f"activations_{model}.npz"
    data = np.load(npz_file)

    results = {
        "model": model,
        "n_pairs": int(c_mask.sum()),
        "layers": {},
        "cross_layer_cosines": {},
    }

    # For each layer, compute direction and per-pair metrics
    layer_directions = {}

    for layer in LAYERS:
        vuln_mean = data[f"layer_{layer}_vuln_mean"][c_mask]
        secure_mean = data[f"layer_{layer}_secure_mean"][c_mask]

        # Compute direction
        direction = np.mean(vuln_mean, axis=0) - np.mean(secure_mean, axis=0)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        layer_directions[layer] = direction

        # Per-pair alignment
        deltas = vuln_mean - secure_mean
        dots = np.dot(deltas, direction)
        pct_aligned = (dots > 0).mean() * 100

        # Mean paired distance
        distances = np.linalg.norm(deltas, axis=1)
        mean_distance = distances.mean()

        results["layers"][str(layer)] = {
            "pct_aligned": float(pct_aligned),
            "mean_paired_distance": float(mean_distance),
            "std_paired_distance": float(distances.std()),
        }

    # Cross-layer cosine similarities
    for l1 in LAYERS:
        for l2 in LAYERS:
            d1 = layer_directions[l1]
            d2 = layer_directions[l2]
            cosine = np.dot(d1, d2)
            key = f"{l1}-{l2}"
            results["cross_layer_cosines"][key] = float(cosine)

    output_file = MULTI_MODEL_DIR / f"direction_geometry_c_{model}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_file}")
    return results


def compute_cwe_universality_c(model: str):
    """Compute CWE universality / cross-family transfer within C."""
    print(f"\nComputing CWE universality for {model}...")

    file_exts, cwes = load_metadata()
    c_mask = file_exts == "c"
    c_cwes = cwes[c_mask]

    npz_file = MULTI_MODEL_DIR / f"activations_{model}.npz"
    data = np.load(npz_file)

    results = {
        "model": model,
        "n_pairs": int(c_mask.sum()),
        "families": {},
        "cross_family_transfer": {},
    }

    # For each layer, compute per-family directions and cross-family transfer
    for layer in LAYERS:
        vuln_mean = data[f"layer_{layer}_vuln_mean"][c_mask]
        secure_mean = data[f"layer_{layer}_secure_mean"][c_mask]

        # Build family masks and compute per-family directions
        family_directions = {}
        family_pair_counts = {}

        for family, cwe_list in CWE_FAMILIES.items():
            family_mask = np.array([cwe in cwe_list for cwe in c_cwes])
            n_family = family_mask.sum()

            if n_family < MIN_FAMILY_PAIRS:
                continue

            family_pair_counts[family] = int(n_family)

            # Compute family-specific direction
            vuln_family = vuln_mean[family_mask]
            secure_family = secure_mean[family_mask]
            direction = np.mean(vuln_family, axis=0) - np.mean(secure_family, axis=0)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            family_directions[family] = direction

        # Cross-family transfer matrix
        transfer_matrix = {}
        for src_family in family_directions:
            for tgt_family in family_directions:
                src_direction = family_directions[src_family]

                # Evaluate on target family
                tgt_mask = np.array([cwe in CWE_FAMILIES[tgt_family] for cwe in c_cwes])
                tgt_vuln = vuln_mean[tgt_mask]
                tgt_secure = secure_mean[tgt_mask]
                tgt_deltas = tgt_vuln - tgt_secure

                dots = np.dot(tgt_deltas, src_direction)
                pct_aligned = (dots > 0).mean() * 100

                key = f"{src_family}-{tgt_family}"
                transfer_matrix[key] = float(pct_aligned)

        results["families"][str(layer)] = {
            "pair_counts": family_pair_counts,
            "transfer_matrix": transfer_matrix,
        }

    output_file = MULTI_MODEL_DIR / f"cwe_universality_c_{model}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_file}")
    return results


def main():
    print("=== Multi-Model C-only Analysis ===")
    for model in MODELS:
        compute_direction_geometry_c(model)
        compute_cwe_universality_c(model)
    print("\nDone!")


if __name__ == "__main__":
    main()
