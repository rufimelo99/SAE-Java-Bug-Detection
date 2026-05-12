#!/usr/bin/env bash
# Mean-token × SAE features probe — compares 4 pooling × representation combos at L11.
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/mean_pool_sae_probe.py"
