#!/usr/bin/env bash
# Cross-layer feature asymmetry (Δf) — replicates the 3.65× result at L0, L11, L27.
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/feature_asymmetry_crosslayer.py"
