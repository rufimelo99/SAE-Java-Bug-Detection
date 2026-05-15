#!/bin/bash

##
# Master pipeline orchestrator for all experiments and figure generation
#
# This script runs the complete pipeline:
# 0. Computes and caches model activations (skipped if NPZ already exists)
# 1. Runs experiments for all three models (Qwen, CodeLlama, StarCoder2)
# 2. Stores raw JSON results
# 3. Generates publication-quality figures (base figures + multi-model styled)
# 4. Generates per-model heatmaps and comparison plots
#
# Usage:
#   ./run_pipeline.sh [--models=qwen,codellama,starcoder2] [--skip-existing]
#                     [--skip-activations] [--figures-only]
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
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--models=...] [--language=c] [--skip-existing] [--skip-activations] [--force-activations] [--figures-only]"
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
    echo -e "${YELLOW}Step 1: Running experiments...${NC}"
    echo ""

    cd "$SCRIPT_DIR"

    python3 run_all_experiments_FIXED.py \
        --models "$MODELS" \
        --language "$LANGUAGE" \
        --output-dir "$RESULTS_DIR" \
        --activations-dir "$ACTIVATIONS_DIR" \
        $SKIP_EXISTING

    echo ""
    echo -e "${GREEN}✓ Experiments completed${NC}"
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
echo "  ✓ Multi-model base figures (per-pair alignment, paired distances, etc.)"
echo "  ✓ Per-model CWE pairwise heatmaps (fig_cwe_pairwise_*.pdf)"
echo "  ✓ Per-model direction alignment (fig_direction_alignment_*.pdf)"
echo "  ✓ Multi-model comparison plots (alignment, magnitude, stability)"
echo ""
