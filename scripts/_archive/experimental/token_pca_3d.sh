#!/usr/bin/env bash
# 3-D PCA trajectory of per-token SAE activations across token positions.
# Requires per_token_sae_l{layer}.jsonl produced by collect_per_token_sae.sh.
# Edit --layers to match which JSONL files you have collected.
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/token_pca_3d.py" \
    --layers 11 \
    --n_bins 20
