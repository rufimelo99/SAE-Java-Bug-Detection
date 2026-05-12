#!/usr/bin/env bash
# Regenerate all publication ablation figures (CWE confound, ΔAUC heatmap, L11 bar chart).
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/generate_ablation_figures.py"
