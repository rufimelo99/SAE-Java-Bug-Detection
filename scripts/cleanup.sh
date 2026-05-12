#!/usr/bin/env bash
# =============================================================================
# Repository Cleanup Script
#
# Moves experimental scripts to archive/ and optionally cleans up artifacts
#
# Usage:
#   ./cleanup.sh                 # Move scripts only (safe)
#   ./cleanup.sh --clean-artifacts [tier]  # Also clean artifacts (careful!)
#
# Tiers:
#   tier1   - Delete TOPK, TO_UPLOAD (1.6 GB, safe to delete)
#   tier2   - Archive large experimental sets (61 GB, check paper first)
#   all     - Do both
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="$REPO_ROOT/scripts"
ARTIFACTS="$REPO_ROOT/sae_java_bug/artifacts"

echo "=========================================================================="
echo "  Repository Cleanup"
echo "=========================================================================="

# ── Parse flags ───────────────────────────────────────────────────────────────
CLEAN_ARTIFACTS=false
ARTIFACT_TIER=""

for arg in "$@"; do
    case "$arg" in
        --clean-artifacts) CLEAN_ARTIFACTS=true ;;
        tier1|tier2|all) ARTIFACT_TIER="$arg" ;;
    esac
done

# ── Phase 1: Archive Scripts ──────────────────────────────────────────────────
echo ""
echo "PHASE 1: Archiving experimental scripts"
echo "─────────────────────────────────────────"

mkdir -p "$SCRIPTS/_archive/experimental"
mkdir -p "$SCRIPTS/_archive/reviewer_response"

# Create archive README if it doesn't exist
if [ ! -f "$SCRIPTS/_archive/README.md" ]; then
    cat > "$SCRIPTS/_archive/README.md" << 'EOF'
# Archived Scripts

This folder contains experimental scripts and reviewer response analyses not part of the main pipeline.

## Organization

- **experimental/** - One-off experiment runners
- **reviewer_response/** - Scripts addressing reviewer feedback

## How to Run

Individual experiments:
```bash
bash experimental/[script_name].sh
# or
python experimental/[script_name].py
```

Batch runners:
```bash
bash ../run_all_response_experiments.sh
bash ../run_global_baselines.sh
```

## Why Archived?

These scripts were moved during cleanup to:
1. Reduce clutter in main scripts/ folder
2. Separate main pipeline from experiments
3. Preserve all functionality while improving organization
EOF
fi

# Move shell scripts (23 experimental)
SHELL_SCRIPTS=(
    "advanced_pooling_probe.sh"
    "causal_patching.sh"
    "collect_per_token_sae.sh"
    "cross_layer_direction_probe.sh"
    "direction_transfer_sven.sh"
    "directional_readout_probe.sh"
    "feature_asymmetry_crosslayer.sh"
    "feature_direction_loading.sh"
    "generate_ablation_figures.sh"
    "generate_advanced_pooling_figure.sh"
    "generation_steering.sh"
    "hypothesis.sh"
    "length_controlled_probe.sh"
    "magnitude_asymmetry_crosslayer.sh"
    "mean_pool_probe.sh"
    "mean_pool_sae_probe.sh"
    "nonlinear_probe.sh"
    "paired_suppression_test.sh"
    "patch_length_analysis.sh"
    "position_stratified_probe.sh"
    "positional_probe_b.sh"
    "token_feature_viz.sh"
    "token_pca_3d.sh"
    "token_position_importance.sh"
    "token_trajectory_3d.sh"
    "within_language_mean_pool_probe.sh"
    "within_language_probe.sh"
)

moved_shell=0
for script in "${SHELL_SCRIPTS[@]}"; do
    if [ -f "$SCRIPTS/$script" ]; then
        mv "$SCRIPTS/$script" "$SCRIPTS/_archive/experimental/"
        ((moved_shell++))
    fi
done
echo "  ✓ Moved $moved_shell shell scripts"

# Move Python scripts (reviewer response - 4)
REVIEWER_SCRIPTS=(
    "issue_01_cv_leakage_minimal.py"
    "issue_02_confound_controls_standalone.py"
    "issue_02_confound_numpy_only.py"
    "reviewer_response_analysis.py"
)

moved_reviewer=0
for script in "${REVIEWER_SCRIPTS[@]}"; do
    if [ -f "$SCRIPTS/$script" ]; then
        mv "$SCRIPTS/$script" "$SCRIPTS/_archive/reviewer_response/"
        ((moved_reviewer++))
    fi
done
echo "  ✓ Moved $moved_reviewer reviewer response scripts"

# Move Python scripts (experimental - 14)
EXPERIMENTAL_SCRIPTS=(
    "01_stronger_pooling_baselines.py"
    "02_confound_controls.py"
    "02_confound_controls_example.py"
    "03_sae_feature_stability.py"
    "04_external_validation_steering.py"
    "05_steering_sign_convention.py"
    "05_steering_sign_convention_example.py"
    "06_cv_leakage_check.py"
    "07_length_guardtoken_controls.py"
    "08_sae_validation_metrics.py"
    "09_fundamental_difficulty_baselines.py"
    "run_reviewer_response_fixes.py"
    "upload_activations.py"
)

moved_exp=0
for script in "${EXPERIMENTAL_SCRIPTS[@]}"; do
    if [ -f "$SCRIPTS/$script" ]; then
        mv "$SCRIPTS/$script" "$SCRIPTS/_archive/experimental/"
        ((moved_exp++))
    fi
done
echo "  ✓ Moved $moved_exp experimental Python scripts"

# ── Phase 2: Clean Artifacts (Optional) ────────────────────────────────────
if [ "$CLEAN_ARTIFACTS" = "true" ]; then
    echo ""
    echo "PHASE 2: Cleaning artifacts (tier: $ARTIFACT_TIER)"
    echo "─────────────────────────────────────────"

    case "$ARTIFACT_TIER" in
        tier1|all)
            echo "  Deleting TOPK/ (560 MB)..."
            rm -rf "$ARTIFACTS/activations/TOPK" 2>/dev/null || true
            rm -rf "$ARTIFACTS/activations/TOPK_tensors" 2>/dev/null || true

            echo "  Deleting TO_UPLOAD/ (6.1 GB)..."
            rm -rf "$ARTIFACTS/activations/TO_UPLOAD" 2>/dev/null || true

            echo "  ✓ Freed ~7.7 GB (tier1)"
            ;;
    esac

    case "$ARTIFACT_TIER" in
        tier2|all)
            echo ""
            echo "  WARNING: Tier 2 cleaning removes large experimental sets"
            echo "  Verify these are NOT referenced in the paper:"
            echo ""

            if grep -r "bigvul_c_only" "$REPO_ROOT/../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/" 2>/dev/null | grep -q "figures"; then
                echo "  ✓ Paper references bigvul_c_only - NOT DELETING"
            else
                read -p "  Delete bigvul_c_only/ (24 GB)? [y/N] " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    rm -rf "$ARTIFACTS/activations/bigvul_c_only" 2>/dev/null || true
                fi
            fi

            read -p "  Delete per_token/ (26 GB)? [y/N] " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -rf "$ARTIFACTS/activations/per_token" 2>/dev/null || true
            fi

            echo "  ✓ Freed variable amount based on selections (up to 50 GB)"
            ;;
    esac
fi

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "=========================================================================="
echo "  Cleanup Summary"
echo "=========================================================================="
echo ""
echo "Scripts moved:"
echo "  • $moved_shell shell scripts → _archive/experimental/"
echo "  • $moved_reviewer reviewer response scripts → _archive/reviewer_response/"
echo "  • $moved_exp experimental Python scripts → _archive/experimental/"
echo ""
echo "Kept in scripts/:"
echo "  • run_all.sh (main pipeline)"
echo "  • run_all_response_experiments.sh"
echo "  • run_global_baselines.sh"
echo "  • run_multi_dataset.sh"
echo "  • run_sae_exploration.sh"
echo "  • paired_ranking_task.py"
echo ""

if [ "$CLEAN_ARTIFACTS" = "true" ]; then
    echo "Artifacts: Cleaned (see above)"
else
    echo "Artifacts: NOT cleaned (use --clean-artifacts to enable)"
fi

echo ""
echo "Next steps:"
echo "  1. git add -A && git commit -m 'chore: archive experimental scripts'"
echo "  2. Test: ./run_all.sh --no-gpu (should work normally)"
echo "  3. Deploy"
echo ""
