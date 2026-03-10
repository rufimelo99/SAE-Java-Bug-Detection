#!/usr/bin/env bash
# Cross-layer vulnerability direction — cosine similarity heatmap between d^L and d^L'.
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/cross_layer_direction_probe.py"
