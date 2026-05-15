#!/bin/bash

##
# Run experiments across multiple C code datasets:
# - DeltaSecommits (main)
# - SVEN (cross-dataset validation)
# - PreciseBugs (cross-dataset validation)
#
# This script orchestrates running the probing pipeline on each dataset
# and collects results for comparative analysis.
#
# Usage:
#   bash scripts/run_multi_dataset_experiments.sh [--models=qwen,codellama,starcoder2]
#                                                 [--skip-existing]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
MODELS="qwen,codellama,starcoder2"
SKIP_EXISTING=""

for arg in "$@"; do
    case $arg in
        --models=*)
            MODELS="${arg#*=}"
            ;;
        --skip-existing)
            SKIP_EXISTING="--skip-existing"
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}Multi-Dataset Experiment Pipeline${NC}"
echo -e "${BLUE}===============================================${NC}"
echo ""

# Define datasets with their metadata files
declare -A DATASETS=(
    ["deltasecommits"]="/Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection/sae_java_bug/artifacts/activations/advanced_pool/20260308_214306/meta.json"
    ["sven"]="/Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection/sae_java_bug/artifacts/activations/sven_c_only/split_meta.json"
    ["precisebugs"]="/Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection/sae_java_bug/artifacts/activations/precisebugs_c_only/split_meta.json"
)

# Define activation directories
declare -A ACTIVATION_DIRS=(
    ["deltasecommits"]="/Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection/sae_java_bug/artifacts/multi_model_probing"
    ["sven"]="/Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection/sae_java_bug/artifacts/activations/sven_c_only"
    ["precisebugs"]="/Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection/sae_java_bug/artifacts/activations/precisebugs_c_only"
)

# Results directory
RESULTS_DIR="${PROJECT_DIR}/results"
mkdir -p "$RESULTS_DIR"

# Run experiments for each dataset
for dataset in deltasecommits sven precisebugs; do
    metadata_file="${DATASETS[$dataset]}"
    activations_dir="${ACTIVATION_DIRS[$dataset]}"

    if [ ! -f "$metadata_file" ]; then
        echo -e "${YELLOW}⚠ Metadata not found for $dataset: $metadata_file${NC}"
        echo ""
        continue
    fi

    echo -e "${YELLOW}Running experiments for: $dataset${NC}"
    echo ""

    # Check if results already exist for this dataset
    if ls "$RESULTS_DIR"/raw_data/${dataset}*.json &>/dev/null && [ -n "$SKIP_EXISTING" ]; then
        echo -e "${GREEN}✓ Results already exist, skipping${NC}"
        echo ""
        continue
    fi

    # Run experiments
    cd "$SCRIPT_DIR"

    python3 run_all_experiments_FIXED.py \
        --models "$MODELS" \
        --language "c" \
        --output-dir "$RESULTS_DIR" \
        --metadata-file "$metadata_file" \
        --activations-dir "$activations_dir" \
        $SKIP_EXISTING

    echo ""
    echo -e "${GREEN}✓ $dataset experiments completed${NC}"
    echo ""
done

echo -e "${BLUE}===============================================${NC}"
echo -e "${GREEN}Multi-dataset experiments complete!${NC}"
echo -e "${BLUE}===============================================${NC}"
echo ""
echo "Results saved to: $RESULTS_DIR/raw_data/"
echo ""
