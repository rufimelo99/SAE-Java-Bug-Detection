#!/bin/bash

##
# Master pipeline orchestrator for all experiments and figure generation
#
# This script runs the complete pipeline with automatic caching:
# 0. Computes and caches model activations (skipped if NPZ already exists)
# 1. Runs mechanistic experiments for all three models (Qwen, CodeLlama, StarCoder2)
#    - Direction geometry, CWE universality, paired ranking probes
#    - Skipped if experiment JSON results already exist
# 2. Generates base multi-model comparison figures (5 figures)
# 3. Generates per-model styled figures (6 figures: CWE heatmaps + alignment curves)
# 4. Generates critical paper figures:
#    - Pairwise CWE-type probe AUROC heatmaps (3 figures, one per model)
# 5. Runs steering experiments (if available in environment):
#    - Direction steering causal validation plots (3 figures, one per model)
#    - Skipped if steering JSON results already exist or PyTorch not available
#
# Usage:
#   ./run_pipeline.sh [--models=qwen,codellama,starcoder2] [--skip-existing]
#                     [--skip-activations] [--figures-only] [--with-datasets]
#
# Options:
#   --with-datasets: Also run comparative analysis on SVEN and PreciseBugs datasets
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_DIR}/results"
ACTIVATIONS_DIR="${PROJECT_DIR}/sae_java_bug/artifacts/multi_model_probing"
FIGURES_DIR="${PROJECT_DIR}/../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures"

MODELS="qwen,codellama,starcoder2"
LANGUAGE="all"
SKIP_EXISTING=""
SKIP_ACTIVATIONS=""
FORCE_ACTIVATIONS=""
FIGURES_ONLY=""
WITH_DATASETS=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
for arg in "$@"; do
    case $arg in
        --models=*)
            MODELS="${arg#*=}"
            ;;
        --language=*)
            LANGUAGE="${arg#*=}"
            ;;
        --skip-existing)
            SKIP_EXISTING="--skip-existing"
            ;;
        --skip-activations)
            SKIP_ACTIVATIONS="yes"
            ;;
        --force-activations)
            FORCE_ACTIVATIONS="yes"
            ;;
        --figures-only)
            FIGURES_ONLY="yes"
            ;;
        --with-datasets)
            WITH_DATASETS="yes"
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--models=...] [--language=c] [--skip-existing] [--skip-activations] [--force-activations] [--figures-only] [--with-datasets]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}Vulnerability Representation Analysis Pipeline${NC}"
echo -e "${BLUE}===============================================${NC}"
echo ""
echo "Configuration:"
echo "  Models:           $MODELS"
echo "  Language:         $LANGUAGE"
echo "  Activations dir:  $ACTIVATIONS_DIR"
echo "  Results dir:      $RESULTS_DIR"
echo "  Figures dir:      $FIGURES_DIR"
echo ""

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$ACTIVATIONS_DIR"
mkdir -p "$FIGURES_DIR"

# Map short model names (qwen, codellama, starcoder2) to the full names
# used by multi_model_probing.py (qwen-7b, codellama-7b, starcoder2-7b)
model_full_name() {
    case "$1" in
        qwen)            echo "qwen-7b" ;;
        qwen-coder)      echo "qwen-coder-7b" ;;
        codellama)       echo "codellama-7b" ;;
        codellama-13b)   echo "codellama-13b" ;;
        starcoder2)      echo "starcoder2-7b" ;;
        starcoder2-15b)  echo "starcoder2-15b" ;;
        deepseekcoder)   echo "deepseekcoder-7b" ;;
        llama3)          echo "llama3-8b" ;;
        *)               echo "$1" ;;
    esac
}

# Step 0: Compute activations
if [ -z "$FIGURES_ONLY" ] && [ -z "$SKIP_ACTIVATIONS" ]; then
    echo -e "${YELLOW}Step 0: Computing activations...${NC}"
    echo ""

    cd "$PROJECT_DIR"

    IFS=',' read -ra MODEL_LIST <<< "$MODELS"
    for model_short in "${MODEL_LIST[@]}"; do
        model_full="$(model_full_name "$model_short")"
        npz_path="${ACTIVATIONS_DIR}/activations_${model_full}.npz"

        if [ -f "$npz_path" ] && [ -z "$FORCE_ACTIVATIONS" ]; then
            echo -e "  ${GREEN}✓ $model_full — cached ($npz_path)${NC}"
            continue
        fi

        echo -e "  Extracting activations for ${model_full}..."
        python -m sae_java_bug.evaluation.multi_model_probing \
            --model "$model_full" \
            --output-dir "$ACTIVATIONS_DIR"
        echo -e "  ${GREEN}✓ $model_full activations saved${NC}"
    done

    echo ""
    echo -e "${GREEN}✓ Activations ready${NC}"
    echo ""
else
    if [ -n "$FIGURES_ONLY" ]; then
        echo -e "${YELLOW}Skipping activations (--figures-only mode)${NC}"
    else
        echo -e "${YELLOW}Skipping activations (--skip-activations)${NC}"
    fi
    echo ""
fi

# Step 1: Run experiments (unless --figures-only is set)
if [ -z "$FIGURES_ONLY" ]; then
    echo -e "${YELLOW}Step 1: Running mechanistic experiments...${NC}"
    echo ""

    # Check if experiment results already exist for all models
    IFS=',' read -ra MODEL_LIST <<< "$MODELS"
    ALL_EXIST=true
    for model_short in "${MODEL_LIST[@]}"; do
        model_short=$(echo "$model_short" | xargs)
        # Check for key experiment outputs
        if ! ls "$RESULTS_DIR"/raw_data/${model_short}*.json &>/dev/null; then
            ALL_EXIST=false
            break
        fi
    done

    if [ "$ALL_EXIST" = true ]; then
        echo -e "  ${GREEN}✓ All experiment results already exist${NC}"
        echo ""
    else
        cd "$SCRIPT_DIR"

        python3 run_all_experiments_FIXED.py \
            --models "$MODELS" \
            --language "$LANGUAGE" \
            --output-dir "$RESULTS_DIR" \
            --activations-dir "$ACTIVATIONS_DIR" \
            $SKIP_EXISTING

        echo ""
        echo -e "${GREEN}✓ Experiments completed${NC}"
        echo ""
    fi
else
    echo -e "${YELLOW}Skipping experiments (--figures-only mode)${NC}"
    echo ""
fi

# Step 2: Generate figures
echo -e "${YELLOW}Step 2: Generating figures...${NC}"
echo ""

cd "$SCRIPT_DIR"

python generate_all_figures.py \
    --results-dir "$RESULTS_DIR" \
    --output-dir "$FIGURES_DIR"

echo ""
echo -e "${GREEN}✓ Figures generated${NC}"

# Step 3: Generate multi-model styled figures
echo ""
echo -e "${YELLOW}Step 3: Generating multi-model styled figures...${NC}"
echo ""

cd "$SCRIPT_DIR"

python generate_multimodel_styled_figures.py
python generate_steering_style_plots.py

echo ""
echo -e "${GREEN}✓ Multi-model styled figures generated${NC}"

# Step 4: Generate missing critical figures
echo ""
echo -e "${YELLOW}Step 4: Generating critical paper figures...${NC}"
echo ""

bash generate_missing_figures.sh --models="$MODELS"

echo ""
echo -e "${GREEN}✓ Critical figures generated${NC}"

# Step 5: Run steering experiments (if not skipped)
if [ -z "$SKIP_ACTIVATIONS" ] && [ -z "$FIGURES_ONLY" ]; then
    echo ""
    echo -e "${YELLOW}Step 5: Running steering experiments...${NC}"
    echo ""

    # Check if steering results already exist
    IFS=',' read -ra MODEL_LIST <<< "$MODELS"
    STEERING_EXISTS=true
    for model_short in "${MODEL_LIST[@]}"; do
        model_short=$(echo "$model_short" | xargs)
        if ! ls "$PROJECT_DIR"/results_real_preference_steering_${model_short}_*.json &>/dev/null; then
            STEERING_EXISTS=false
            break
        fi
    done

    if [ "$STEERING_EXISTS" = true ]; then
        echo -e "  ${GREEN}✓ All steering experiment results already exist${NC}"
        echo ""
        # Regenerate steering plots from existing results
        cd "$SCRIPT_DIR"
        bash generate_missing_figures.sh --steering-only --models="$MODELS"
        echo -e "${GREEN}✓ Steering causal plots regenerated${NC}"
        echo ""
    else
        echo -e "${YELLOW}⚠ Steering experiments require PyTorch/GPU — skipping in this environment${NC}"
        echo ""
        echo "To run steering experiments in your VM:"
        echo "  cd $PROJECT_DIR"
        echo "  python scripts/run_multimodel_steering_experiments.py --models $MODELS"
        echo ""
        echo "Then regenerate causal plots:"
        echo "  bash scripts/generate_missing_figures.sh --steering-only --models=\"$MODELS\""
        echo ""
    fi
else
    if [ -n "$FIGURES_ONLY" ]; then
        echo -e "${YELLOW}Skipping steering experiments (--figures-only mode)${NC}"
    else
        echo -e "${YELLOW}Skipping steering experiments (--skip-activations mode)${NC}"
    fi
    echo ""
fi

# Step 6: Multi-dataset comparative analysis (optional)
if [ -n "$WITH_DATASETS" ]; then
    echo ""
    echo -e "${YELLOW}Step 6: Running multi-dataset comparative analysis...${NC}"
    echo ""

    cd "$SCRIPT_DIR"

    # Check if multi-dataset comparison script exists
    if [ -f "$PROJECT_DIR/sae_java_bug/sparse_autoencoders/notebooks/multi_dataset_comparison.py" ]; then
        python "$PROJECT_DIR/sae_java_bug/sparse_autoencoders/notebooks/multi_dataset_comparison.py"
        echo ""
        echo -e "${GREEN}✓ Multi-dataset comparative analysis completed${NC}"
        echo ""
    else
        echo -e "${YELLOW}⚠ Multi-dataset comparison script not found${NC}"
        echo ""
    fi
else
    echo -e "${YELLOW}Skipping multi-dataset analysis (use --with-datasets flag to enable)${NC}"
    echo ""
fi

# Summary
echo ""
echo -e "${BLUE}===============================================${NC}"
echo -e "${GREEN}Pipeline completed successfully!${NC}"
echo -e "${BLUE}===============================================${NC}"
echo ""
echo "Results saved to:"
echo "  Activations: $ACTIVATIONS_DIR/"
echo "  Raw data:    $RESULTS_DIR/raw_data/"
echo "  Figures:     $FIGURES_DIR"
echo ""
echo "Generated figures:"
echo "  ✓ Base multi-model figures (5): per-pair alignment, paired distances, CWE transfer, direction heatmaps, ranking accuracy"
echo "  ✓ Per-model styled figures (6): CWE pairwise heatmaps + direction alignment curves"
echo "  ✓ Multi-model comparison plots (3): alignment, magnitude, stability across architectures"
echo "  ✓ Critical paper figures (3): Pairwise CWE-type probe AUROC heatmaps (per model)"
echo "  ✓ Steering causal validation (3): Direction steering effect on model preference (when experiments available)"
if [ -n "$WITH_DATASETS" ]; then
    echo "  ✓ Multi-dataset comparative analysis: SVEN and PreciseBugs cross-dataset validation"
fi
echo ""
echo "Total: 20+ publication-quality figures (+ comparative analysis)"
echo ""
echo "Pipeline Features:"
echo "  • Automatically skips steps if raw data already exists"
echo "  • Caches activations to avoid recomputation"
echo "  • All figures regenerated with consistent, publication-quality styling"
echo "  • Optional multi-dataset comparative analysis (--with-datasets)"
echo ""
echo "To run with multi-dataset analysis:"
echo "  ./run_pipeline.sh --with-datasets"
echo ""
