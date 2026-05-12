#!/usr/bin/env bash
# Patch length distribution analysis by CWE family.
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/patch_length_analysis.py"
