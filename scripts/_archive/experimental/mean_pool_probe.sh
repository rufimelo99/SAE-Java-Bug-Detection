#!/usr/bin/env bash
# Mean-token pooling probe across all 8 layers (vs last-token baseline).
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/mean_pool_probe.py"
