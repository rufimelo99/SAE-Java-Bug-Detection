#!/usr/bin/env bash
# Per-token SAE feature visualisation — coloured token heatmaps for selected features.
# Edit --features to select which SAE features to visualise.
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/token_feature_viz.py" \
    --features 1185 1797 \
    --multi_family
