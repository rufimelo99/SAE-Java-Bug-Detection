#!/usr/bin/env bash
# Regenerate fig_advanced_pooling_comparison.pdf from saved probe_results.json.
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/generate_advanced_pooling_figure.py"
