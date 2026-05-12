#!/usr/bin/env bash
# Cross-layer magnitude asymmetry — repeats the magnitude analysis for all 8 SAE layers.
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/magnitude_asymmetry_crosslayer.py"
