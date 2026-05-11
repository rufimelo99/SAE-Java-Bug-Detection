#!/usr/bin/env python3
"""
Generate figures for the C-focused mechanistic paper.

Runs the essential figure-generation notebooks and copies outputs to paper directory.

Usage:
    conda activate sae
    python generate_c_paper_figures.py
"""

import subprocess
import sys
from pathlib import Path

# Paths
NOTEBOOK_DIR = (
    Path(__file__).parent / "sae_java_bug" / "sparse_autoencoders" / "notebooks"
)
PAPER_FIGS = (
    Path(__file__).parent.parent
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)

# Required notebooks (in order)
NOTEBOOKS = [
    "cross_layer_direction_probe.py",  # Generates fig_direction_cosine_sim.pdf, fig_alignment_trajectory.pdf
    "patch_length_analysis.py",  # Generates fig_patch_length_by_cwe.pdf
    "causal_patching.py",  # Generates fig_causal_patching.pdf
    "directional_readout_probe.py",  # Generates fig_directional_readout_comparison.pdf
    "feature_direction_loading.py",  # Generates fig_feature_direction_loading.pdf
    "direction_transfer_sven.py",  # Generates fig_direction_transfer_sven.pdf
    "generation_steering.py",  # Generates fig_generation_steering.pdf
]


def run_notebook(notebook_name: str) -> bool:
    """Run a single notebook."""
    notebook_path = NOTEBOOK_DIR / notebook_name
    if not notebook_path.exists():
        print(f"❌ Not found: {notebook_path}")
        return False

    print(f"\n{'='*70}")
    print(f"Running: {notebook_name}")
    print(f"{'='*70}")

    result = subprocess.run(
        ["python", str(notebook_path)], cwd=str(NOTEBOOK_DIR), capture_output=False
    )

    if result.returncode == 0:
        print(f"✅ {notebook_name} completed successfully")
        return True
    else:
        print(f"❌ {notebook_name} failed with code {result.returncode}")
        return False


def main():
    print("Generating figures for C-focused mechanistic paper...")
    print(f"Figures output to: {PAPER_FIGS}")
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for notebook in NOTEBOOKS:
        if run_notebook(notebook):
            success_count += 1

    print(f"\n{'='*70}")
    print(f"Summary: {success_count}/{len(NOTEBOOKS)} notebooks completed successfully")
    print(f"{'='*70}")

    # List generated figures
    existing_figs = list(PAPER_FIGS.glob("fig_*.pdf"))
    print(f"\n✓ {len(existing_figs)} figures now available in {PAPER_FIGS}")

    required_figs = [
        "fig_crosslayer_probe_auc.pdf",
        "fig_patch_length_by_cwe.pdf",
        "fig_direction_cosine_sim.pdf",
        "fig_alignment_trajectory.pdf",
        "fig_causal_patching.pdf",
        "fig_directional_readout_comparison.pdf",
        "fig_feature_direction_loading.pdf",
        "fig_direction_transfer_sven.pdf",
        "fig_generation_steering.pdf",
    ]

    print("\nRequired figures for paper:")
    for fig in required_figs:
        fig_path = PAPER_FIGS / fig
        status = "✅" if fig_path.exists() else "⏳"
        print(f"  {status} {fig}")

    return 0 if success_count == len(NOTEBOOKS) else 1


if __name__ == "__main__":
    sys.exit(main())
