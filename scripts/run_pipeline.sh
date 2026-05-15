#!/bin/bash

##
# Master pipeline orchestrator for all experiments and figure generation
#
# This script runs the complete pipeline:
# 1. Runs experiments for all three models (Qwen, CodeLlama, StarCoder2)
# 2. Stores raw JSON results
# 3. Generates publication-quality figures
#
# Usage:
#   ./run_pipeline.sh [--models qwen,codellama,starcoder2] [--skip-existing] [--figures-only]
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_DIR}/results"
FIGURES_DIR="${PROJECT_DIR}/../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures"

MODELS="${1:-qwen,codellama,starcoder2}"
SKIP_EXISTING="${2:-}"
FIGURES_ONLY="${3:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}Vulnerability Representation Analysis Pipeline${NC}"
echo -e "${BLUE}===============================================${NC}"
echo ""

# Parse arguments
for arg in "$@"; do
    case $arg in
        --models=*)
            MODELS="${arg#*=}"
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING="--skip-existing"
            shift
            ;;
        --figures-only)
            FIGURES_ONLY="--figures-only"
            shift
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Models: $MODELS"
echo "  Results directory: $RESULTS_DIR"
echo "  Figures directory: $FIGURES_DIR"
echo ""

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$FIGURES_DIR"

# Step 1: Run experiments (unless --figures-only is set)
if [ -z "$FIGURES_ONLY" ]; then
    echo -e "${YELLOW}Step 1: Running experiments...${NC}"
    echo ""

    cd "$SCRIPT_DIR"

    python3 run_all_experiments.py \
        --models "$MODELS" \
        --output-dir "$RESULTS_DIR" \
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

python3 generate_all_figures.py \
    --results-dir "$RESULTS_DIR" \
    --output-dir "$FIGURES_DIR"

echo ""
echo -e "${GREEN}✓ Figures generated${NC}"

# Step 3: Summary
echo ""
echo -e "${BLUE}===============================================${NC}"
echo -e "${GREEN}Pipeline completed successfully!${NC}"
echo -e "${BLUE}===============================================${NC}"
echo ""
echo "Results saved to:"
echo "  Raw data: $RESULTS_DIR/raw_data/"
echo "  Summary: $RESULTS_DIR/results/summary.json"
echo "  Figures: $FIGURES_DIR"
echo ""
