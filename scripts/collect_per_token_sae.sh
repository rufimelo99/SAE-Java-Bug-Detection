#!/usr/bin/env bash
# collect_per_token_sae.sh
# Collects per-token SAE activations for each requested layer.
# Edit LAYERS below, then run:  bash collect_per_token_sae.sh

set -euo pipefail

LAYERS=(0 3 7 11 15 19 23 27)
OUT_BASE="artifacts/activations"
SCRIPT="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks/collect_per_token_sae.py"

for LAYER in "${LAYERS[@]}"; do
    OUT="${OUT_BASE}/per_token_sae_l${LAYER}.jsonl"
    echo "════════════════════════════════════════"
    echo "Layer ${LAYER}  →  ${OUT}"
    echo "════════════════════════════════════════"
    python "${SCRIPT}" --layer "${LAYER}" --out "${OUT}"
done

echo ""
echo "All layers done."
