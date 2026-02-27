#!/bin/bash
set -euo pipefail

# Available configs:
#   GEMMA3
#   KODCODE_LLAMA_3_2_1B
#   KODCODE_CODE_LLAMA_7B
#   QWEN_CODER_7B_SECURE_CODE_TOPK
#   QWEN_CODER_7B_VULNEABLE_CODE_STD_10M
#   QWEN_CODER_7B_VULNEABLE_CODE_STD      <- iterates layers [0,3,7,11,15,19,23,27]

CONFIG="${1:-QWEN_CODER_7B_VULNEABLE_CODE_STD}"
HF_PATH="${2:-rufimelo/DeltaSecommits}"
OUTPUT_DIR="${3:-../artifacts/activations/}"
MAX_TOKENS="${4:-2000}"
MAX_SAMPLES="${5:-}"  # leave empty to use all samples
SKIP_LAYERS="${6:-}"  # space-separated layer indices to skip, e.g. "0 3 7"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

ARGS=(
    --config "$CONFIG"
    --hf_path "$HF_PATH"
    --output_dir "$OUTPUT_DIR"
    --max_tokens "$MAX_TOKENS"
)

if [[ -n "$MAX_SAMPLES" ]]; then
    ARGS+=(--max_samples "$MAX_SAMPLES")
fi

if [[ -n "$SKIP_LAYERS" ]]; then
    # shellcheck disable=SC2086
    ARGS+=(--skip_layers $SKIP_LAYERS)
fi

echo "Running SAE exploration with config: $CONFIG"
python -m sae_java_bug.sparse_autoencoders.sae_exploration "${ARGS[@]}"
