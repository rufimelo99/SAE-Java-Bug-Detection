"""
Per-CWE Vulnerability Direction Analysis and Transfer

Tests whether the vulnerability direction is universal or CWE-specific:
1. Compute direction for each CWE type separately
2. Test cross-CWE transfer (direction trained on CWE-A, tested on CWE-B)
3. Test family-level transfer (memory safety ↔ injection)
4. Shows whether signal generalizes across vulnerability types

Usage:
    conda run -n sae python per_cwe_directions.py

Output:
    artifacts/analysis/per_cwe/
        per_cwe_results.json
        cross_cwe_transfer_matrix.json
        family_transfer_results.json
"""

import base64
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HERE = Path(__file__).parent
ARTIFACTS = Path(__file__).parents[2] / "artifacts"
OUTPUT_DIR = ARTIFACTS / "analysis" / "per_cwe"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_RUN_DIR = (
    ARTIFACTS
    / "activations"
    / "raw_activations"
    / "vulnerable_code_qwen_coder_standard_16384_raw"
)

SOURCE_JSONL = (
    RAW_RUN_DIR / "activations_layer_0_raw_component_hidden_state_last_token.jsonl"
)

LAYERS = [3, 7, 11, 15, 19, 23]
SEED = 42

# CWE groupings into families
CWE_FAMILIES = {
    "memory_safety": {
        "CWE-119",  # Buffer Overflow
        "CWE-125",  # Out-of-bounds Read
        "CWE-787",  # Out-of-bounds Write
        "CWE-190",  # Integer Overflow
        "CWE-189",  # Numeric Errors
    },
    "injection": {
        "CWE-89",  # SQL Injection
        "CWE-79",  # Cross-site Scripting
        "CWE-94",  # Code Injection
        "CWE-78",  # OS Command Injection
    },
    "resource": {
        "CWE-400",  # Uncontrolled Resource Consumption
        "CWE-401",  # Missing Release of Memory
        "CWE-404",  # Improper Resource Validation
    },
    "information_disclosure": {
        "CWE-200",  # Exposure of Sensitive Information
        "CWE-327",  # Use of Broken Cryptography
        "CWE-295",  # Improper Certificate Validation
    },
    "access_control": {
        "CWE-284",  # Improper Access Control
        "CWE-269",  # Improper Privilege Management
    },
}

# Reverse mapping: CWE -> family
CWE_TO_FAMILY = {}
for family, cwes in CWE_FAMILIES.items():
    for cwe in cwes:
        CWE_TO_FAMILY[cwe] = family


def load_source_records(jsonl_path: Path):
    """Load records with CWE labels."""
    records = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            try:
                secure_code = base64.b64decode(r["secure_code"]).decode(
                    "utf-8", errors="replace"
                )
                vuln_code = base64.b64decode(r["vulnerable_code"]).decode(
                    "utf-8", errors="replace"
                )
            except Exception:
                secure_code = r.get("secure_code", "")
                vuln_code = r.get("vulnerable_code", "")

            records.append(
                {
                    "cwe": r["cwe"],
                    "secure": secure_code,
                    "vulnerable": vuln_code,
                }
            )
    return records


def load_activations_for_layer(layer: int):
    """Load pre-computed activations with CWE labels."""
    matches = list(RAW_RUN_DIR.glob(f"activations_layer_{layer}_*.jsonl"))
    if not matches:
        return None, None, None

    safe_rows, vuln_rows, cwes = [], [], []
    with matches[0].open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            safe_rows.append(r["secure"])
            vuln_rows.append(r["vulnerable"])
            cwes.append(r.get("cwe", "unknown"))

    return (
        np.array(safe_rows, dtype=np.float32),
        np.array(vuln_rows, dtype=np.float32),
        cwes,
    )


def compute_vulnerability_direction(safe_mat, vuln_mat):
    """Compute normalized vulnerability direction."""
    if len(safe_mat) == 0:
        return None
    mean_vuln = vuln_mat.mean(axis=0)
    mean_safe = safe_mat.mean(axis=0)
    direction = mean_vuln - mean_safe
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return None
    direction = direction / norm
    return direction


def compute_per_pair_alignment(safe_mat, vuln_mat, direction):
    """Compute % pairs aligned with direction."""
    if direction is None:
        return None
    alignments = []
    for i in range(len(safe_mat)):
        delta = vuln_mat[i] - safe_mat[i]
        alignment = np.dot(delta, direction)
        alignments.append(alignment > 0)
    return 100.0 * np.mean(alignments)


def analyze_per_cwe_directions(layer: int):
    """Compute direction for each CWE type."""
    print(f"\nLayer {layer}:")

    safe_mat, vuln_mat, cwes = load_activations_for_layer(layer)
    if safe_mat is None:
        print("  [SKIP] No data")
        return None

    # Group by CWE
    cwe_to_indices = {}
    for i, cwe in enumerate(cwes):
        if cwe not in cwe_to_indices:
            cwe_to_indices[cwe] = []
        cwe_to_indices[cwe].append(i)

    results = {
        "layer": layer,
        "n_pairs_total": len(safe_mat),
        "per_cwe_stats": {},
        "cross_cwe_transfer": {},
    }

    # Compute direction for each CWE with sufficient samples
    cwe_directions = {}
    min_samples = 20  # minimum pairs for reliable direction

    for cwe, indices in cwe_to_indices.items():
        if len(indices) < min_samples:
            continue

        safe_subset = safe_mat[indices]
        vuln_subset = vuln_mat[indices]
        direction = compute_vulnerability_direction(safe_subset, vuln_subset)

        if direction is None:
            continue

        cwe_directions[cwe] = direction

        pct_aligned = compute_per_pair_alignment(safe_subset, vuln_subset, direction)
        results["per_cwe_stats"][cwe] = {
            "n_pairs": len(indices),
            "pct_aligned_own": float(pct_aligned) if pct_aligned else None,
        }
        print(f"  {cwe}: n={len(indices):4d}, aligned={pct_aligned:.1f}%")

    # Test cross-CWE transfer: train direction on CWE-A, test on CWE-B
    cwe_list = list(cwe_directions.keys())
    for i, source_cwe in enumerate(cwe_list):
        for j, target_cwe in enumerate(cwe_list):
            if i == j:
                continue

            source_direction = cwe_directions[source_cwe]
            target_indices = cwe_to_indices[target_cwe]
            target_safe = safe_mat[target_indices]
            target_vuln = vuln_mat[target_indices]

            pct_aligned = compute_per_pair_alignment(
                target_safe, target_vuln, source_direction
            )
            if pct_aligned is not None:
                key = f"{source_cwe}->{target_cwe}"
                results["cross_cwe_transfer"][key] = float(pct_aligned)

    return results


def analyze_family_level_transfer(layer: int):
    """Test transfer between CWE families (memory safety ↔ injection, etc.)."""
    print(f"\nFamily-level analysis (Layer {layer}):")

    safe_mat, vuln_mat, cwes = load_activations_for_layer(layer)
    if safe_mat is None:
        return None

    # Group by family
    family_to_indices = {}
    for i, cwe in enumerate(cwes):
        family = CWE_TO_FAMILY.get(cwe, "other")
        if family not in family_to_indices:
            family_to_indices[family] = []
        family_to_indices[family].append(i)

    results = {
        "layer": layer,
        "per_family_stats": {},
        "cross_family_transfer": {},
    }

    min_samples = 30

    # Compute direction per family
    family_directions = {}
    for family, indices in family_to_indices.items():
        if len(indices) < min_samples:
            continue

        safe_subset = safe_mat[indices]
        vuln_subset = vuln_mat[indices]
        direction = compute_vulnerability_direction(safe_subset, vuln_subset)

        if direction is None:
            continue

        family_directions[family] = direction
        pct_aligned = compute_per_pair_alignment(safe_subset, vuln_subset, direction)

        results["per_family_stats"][family] = {
            "n_pairs": len(indices),
            "pct_aligned_own": float(pct_aligned) if pct_aligned else None,
        }
        print(f"  {family}: n={len(indices):4d}, aligned={pct_aligned:.1f}%")

    # Cross-family transfer
    family_list = list(family_directions.keys())
    for i, source_family in enumerate(family_list):
        for j, target_family in enumerate(family_list):
            if i == j:
                continue

            source_direction = family_directions[source_family]
            target_indices = family_to_indices[target_family]
            target_safe = safe_mat[target_indices]
            target_vuln = vuln_mat[target_indices]

            pct_aligned = compute_per_pair_alignment(
                target_safe, target_vuln, source_direction
            )
            if pct_aligned is not None:
                key = f"{source_family}->{target_family}"
                results["cross_family_transfer"][key] = float(pct_aligned)
                print(f"    {source_family} -> {target_family}: {pct_aligned:.1f}%")

    return results


def main():
    print("\n" + "=" * 70)
    print("PER-CWE VULNERABILITY DIRECTION ANALYSIS")
    print("=" * 70)

    all_results = {
        "description": "Per-CWE direction analysis: is the vulnerability signal universal or CWE-specific?",
        "hypothesis": "If universal: directions trained on one CWE transfer well to others",
        "by_layer_per_cwe": {},
        "by_layer_family": {},
        "summary": {},
    }

    # Analyze per-CWE and family-level for each layer
    for layer in LAYERS:
        per_cwe = analyze_per_cwe_directions(layer)
        if per_cwe:
            all_results["by_layer_per_cwe"][layer] = per_cwe

        family = analyze_family_level_transfer(layer)
        if family:
            all_results["by_layer_family"][layer] = family

    # Summary statistics
    all_transfer_pcts = []
    for layer_results in all_results["by_layer_per_cwe"].values():
        all_transfer_pcts.extend(layer_results["cross_cwe_transfer"].values())

    all_family_transfer = []
    for layer_results in all_results["by_layer_family"].values():
        all_family_transfer.extend(layer_results["cross_family_transfer"].values())

    if all_transfer_pcts:
        all_results["summary"] = {
            "mean_cross_cwe_transfer": float(np.mean(all_transfer_pcts)),
            "std_cross_cwe_transfer": float(np.std(all_transfer_pcts)),
            "min_cross_cwe_transfer": float(np.min(all_transfer_pcts)),
            "max_cross_cwe_transfer": float(np.max(all_transfer_pcts)),
        }

    if all_family_transfer:
        all_results["summary"]["mean_cross_family_transfer"] = float(
            np.mean(all_family_transfer)
        )
        all_results["summary"]["std_cross_family_transfer"] = float(
            np.std(all_family_transfer)
        )

    all_results["summary"]["interpretation"] = (
        "High cross-CWE transfer (>70%) indicates the vulnerability direction is universal. "
        "Low transfer (<60%) indicates CWE-specific signals. Family-level transfer shows "
        "whether memory-safety and injection attacks share a common encoding."
    )

    # Save results
    results_path = OUTPUT_DIR / "per_cwe_direction_analysis.json"
    with results_path.open("w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if all_results["summary"]:
        print(
            f"Mean cross-CWE transfer: {all_results['summary'].get('mean_cross_cwe_transfer', 'N/A')}"
        )
        print(
            f"Mean cross-family transfer: {all_results['summary'].get('mean_cross_family_transfer', 'N/A')}"
        )
        print(f"\n{all_results['summary']['interpretation']}")
    print("=" * 70 + "\n")

    print(f"✓ Saved results to {results_path}\n")

    return all_results


if __name__ == "__main__":
    main()
