#!/usr/bin/env bash
# Position-stratified SAE feature probe — activation vs normalised token position at L11.
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/position_stratified_probe.py"
