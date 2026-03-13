#!/usr/bin/env bash
# Per-token residual-stream 3-D trajectory in vulnerability-direction PCA space.
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/token_trajectory_3d.py"
