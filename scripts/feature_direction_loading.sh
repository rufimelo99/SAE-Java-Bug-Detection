#!/usr/bin/env bash
# Feature-to-direction loading analysis.
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/feature_direction_loading.py"
