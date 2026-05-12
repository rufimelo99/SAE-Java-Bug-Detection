#!/usr/bin/env bash
# Token position importance analysis — mechanistic deepening (c)
#
# This script computes which token positions carry vulnerability signal
# by analyzing per-token projections onto the vulnerability direction.
#
# Usage:
#   bash scripts/token_position_importance.sh
#
# Requires:
#   - Raw residual stream activations in artifacts/activations/
#   - Model weights for Qwen2.5-7B-Instruct
#   - GPU with ~14GB VRAM
#
# Output:
#   - artifacts/analysis/token_position/
#       - position_importance_L{layer}.json
#       - figures/
#           - token_position_importance_heatmap.pdf
#           - token_position_alignment_heatmap.pdf
#   - figures/ (paper repo)
#       - (same figures copied for paper)

set -euo pipefail

NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"

echo "[*] Running token position importance analysis..."
python "${NOTEBOOKS}/token_position_importance.py"

echo "[✓] Done!"
