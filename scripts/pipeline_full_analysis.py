#!/usr/bin/env python3
"""
Comprehensive Analysis Pipeline

Orchestrates:
1. Direction geometry analysis (alignment, stability, transfer)
2. CWE-type probing analysis
3. Direction steering experiments (causal validation)
4. Publication-quality figure generation

This is the main entry point for regenerating all paper figures and analyses.

Usage:
    python scripts/pipeline_full_analysis.py [--stage geometry|steering|figures|all]
    python scripts/pipeline_full_analysis.py --stage all --n_samples 100
"""

import argparse
import logging
import subprocess
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent


# ============================================================================
# STAGE 1: DIRECTION GEOMETRY ANALYSIS
# ============================================================================

GEOMETRY_SCRIPTS = [
    ("Direction alignment (combined)", "regenerate_direction_alignment_combined.py"),
    (
        "Direction alignment (families)",
        "regenerate_direction_alignment_families_combined.py",
    ),
    ("Multimodel alignment comparison", "regenerate_multimodel_alignment_styled.py"),
    (
        "Multimodel magnitude comparison",
        "regenerate_multimodel_magnitude_comparison.py",
    ),
    ("Paired distances analysis", "regenerate_paired_distances_styled.py"),
    ("Direction transfer (families)", "regenerate_direction_transfer_styled.py"),
    ("Multimodel direction stability", "regenerate_multimodel_stability_styled.py"),
]


def run_geometry_analysis():
    """Run direction geometry analysis for all models."""
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 1: DIRECTION GEOMETRY ANALYSIS")
    logger.info("=" * 70)

    for name, script in GEOMETRY_SCRIPTS:
        logger.info(f"\n[Geometry] {name}...")
        script_path = SCRIPTS_DIR / script

        if not script_path.exists():
            logger.warning(f"  Script not found: {script}")
            continue

        try:
            result = subprocess.run(
                ["python", str(script_path)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                logger.info(f"  ✓ {name}")
            else:
                logger.error(f"  ✗ {name}")
                if result.stderr:
                    logger.error(f"  Error: {result.stderr[:200]}")

        except subprocess.TimeoutExpired:
            logger.error(f"  ✗ {name} (timeout)")
        except Exception as e:
            logger.error(f"  ✗ {name}: {e}")

    logger.info("\n✓ Geometry analysis complete")


# ============================================================================
# STAGE 2: STEERING EXPERIMENTS (Causal Validation)
# ============================================================================


def run_steering_experiments(models: List[str] = None, n_samples: int = 100):
    """Run direction steering experiments for causal validation."""
    if models is None:
        models = ["qwen", "codellama", "starcoder2"]

    logger.info("\n" + "=" * 70)
    logger.info("STAGE 2: DIRECTION STEERING EXPERIMENTS")
    logger.info("=" * 70)

    script_path = SCRIPTS_DIR / "run_steering_pipeline.py"

    if not script_path.exists():
        logger.error(f"Steering pipeline script not found: {script_path}")
        return

    models_str = ",".join(models)

    logger.info(f"\nRunning steering experiments for: {models_str}")
    logger.info(f"Sample size: {n_samples}")

    try:
        result = subprocess.run(
            [
                "python",
                str(script_path),
                "--models",
                models_str,
                "--n_samples",
                str(n_samples),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout for steering experiments
        )

        if result.returncode == 0:
            logger.info("✓ Steering experiments complete")
            if result.stdout:
                logger.info(result.stdout[-500:])  # Last 500 chars
        else:
            logger.error("✗ Steering experiments failed")
            if result.stderr:
                logger.error(result.stderr[-500:])

    except subprocess.TimeoutExpired:
        logger.error("✗ Steering experiments timeout (exceeded 1 hour)")
    except Exception as e:
        logger.error(f"✗ Steering experiments error: {e}")


# ============================================================================
# STAGE 3: STEERING VISUALIZATION
# ============================================================================


def generate_steering_plots(models: List[str] = None, n_samples: int = 100):
    """Generate steering effect visualizations."""
    if models is None:
        models = ["qwen", "codellama", "starcoder2"]

    logger.info("\n" + "=" * 70)
    logger.info("STAGE 3: STEERING VISUALIZATION")
    logger.info("=" * 70)

    script_path = SCRIPTS_DIR / "regenerate_steering_plots.py"

    if not script_path.exists():
        logger.error(f"Steering plots script not found: {script_path}")
        return

    models_str = ",".join(models)

    logger.info(f"\nGenerating steering plots for: {models_str}")

    try:
        result = subprocess.run(
            [
                "python",
                str(script_path),
                "--models",
                models_str,
                "--n_samples",
                str(n_samples),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            logger.info("✓ Steering plots generated")
        else:
            logger.error("✗ Steering plot generation failed")
            if result.stderr:
                logger.error(result.stderr[-500:])

    except subprocess.TimeoutExpired:
        logger.error("✗ Steering plot generation timeout")
    except Exception as e:
        logger.error(f"✗ Steering plot generation error: {e}")


# ============================================================================
# STAGE 4: OTHER ANALYSIS (CWE, etc.)
# ============================================================================


def run_other_analyses():
    """Run other figure generation scripts (CWE probing, etc.)."""
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 4: OTHER ANALYSES")
    logger.info("=" * 70)

    other_scripts = [
        ("CWE pairwise probing", "regenerate_cwe_pairwise_probe_styled.py"),
    ]

    for name, script in other_scripts:
        logger.info(f"\n[Other] {name}...")
        script_path = SCRIPTS_DIR / script

        if not script_path.exists():
            logger.warning(f"  Script not found: {script}")
            continue

        try:
            result = subprocess.run(
                ["python", str(script_path)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                logger.info(f"  ✓ {name}")
            else:
                logger.warning(f"  Partial: {name}")

        except Exception as e:
            logger.warning(f"  Skipped {name}: {e}")

    logger.info("\n✓ Other analyses complete")


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def run_full_pipeline(
    stage: str = "all",
    models: List[str] = None,
    n_samples: int = 100,
):
    """Run the complete analysis pipeline."""
    if models is None:
        models = ["qwen", "codellama", "starcoder2"]

    logger.info("\n" + "=" * 70)
    logger.info("COMPREHENSIVE ANALYSIS PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Stage: {stage}")
    logger.info(f"Models: {', '.join(models)}")
    logger.info(f"Sample size: {n_samples}")
    logger.info("=" * 70)

    if stage in ["geometry", "all"]:
        run_geometry_analysis()

    if stage in ["steering", "all"]:
        run_steering_experiments(models, n_samples)
        generate_steering_plots(models, n_samples)

    if stage in ["figures", "all"]:
        run_other_analyses()

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nGenerated figures are in:")
    logger.info(
        f"  {PROJECT_ROOT}/On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures/"
    )
    logger.info(f"\nResults are in:")
    logger.info(f"  {PROJECT_ROOT}/SAE-Java-Bug-Detection/results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive analysis pipeline for vulnerability representation research"
    )
    parser.add_argument(
        "--stage",
        choices=["geometry", "steering", "figures", "all"],
        default="all",
        help="Which analysis stage to run",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="qwen,codellama,starcoder2",
        help="Comma-separated list of models",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples for steering experiments",
    )

    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")]

    run_full_pipeline(stage=args.stage, models=models, n_samples=args.n_samples)
