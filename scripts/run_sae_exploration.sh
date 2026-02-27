#!/bin/bash
set -euo pipefail

# Available configs and their layers:
#   GEMMA3                              layers [0..24]
#   KODCODE_LLAMA_3_2_1B               layers [0]
#   KODCODE_CODE_LLAMA_7B              layers [0]
#   QWEN_CODER_7B_SECURE_CODE_TOPK     layers [0, 14, 27]
#   QWEN_CODER_7B_VULNEABLE_CODE_STD_10M  layers [11]
#   QWEN_CODER_7B_VULNEABLE_CODE_STD   layers [0, 3, 7, 11, 15, 19, 23, 27]

CONFIG="${1:-QWEN_CODER_7B_VULNEABLE_CODE_STD}"
HF_PATH="${2:-rufimelo/DeltaSecommits}"
OUTPUT_DIR="${3:-../artifacts/activations/}"
MAX_TOKENS="${4:-2000}"
MAX_SAMPLES="${5:-}"  # leave empty to use all samples

# Space-separated list of layers to run. Defaults per config:
case "$CONFIG" in
    GEMMA3)                              LAYERS="${6:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24}" ;;
    KODCODE_LLAMA_3_2_1B)               LAYERS="${6:-0}" ;;
    KODCODE_CODE_LLAMA_7B)              LAYERS="${6:-0}" ;;
    QWEN_CODER_7B_SECURE_CODE_TOPK)     LAYERS="${6:-0 14 27}" ;;
    QWEN_CODER_7B_VULNEABLE_CODE_STD_10M) LAYERS="${6:-11}" ;;
    QWEN_CODER_7B_VULNEABLE_CODE_STD)   LAYERS="${6:-0 3 7 11 15 19 23 27}" ;;
    *)                                  LAYERS="${6:-}" ;;
esac

if [[ -z "$LAYERS" ]]; then
    echo "ERROR: No layers defined for config '$CONFIG'. Pass them as the 6th argument." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

BASE_ARGS=(
    --config "$CONFIG"
    --hf_path "$HF_PATH"
    --output_dir "$OUTPUT_DIR"
    --max_tokens "$MAX_TOKENS"
)

if [[ -n "$MAX_SAMPLES" ]]; then
    BASE_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

echo "Running SAE exploration | config: $CONFIG | layers: $LAYERS"

for layer in $LAYERS; do
    echo "--- Layer $layer ---"
    python -m sae_java_bug.sparse_autoencoders.sae_exploration "${BASE_ARGS[@]}" --layer "$layer"
done
