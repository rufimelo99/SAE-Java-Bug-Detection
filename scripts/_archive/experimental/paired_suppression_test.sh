#!/usr/bin/env bash
# Paired suppression test + activation magnitude asymmetry at L11.
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/paired_suppression_test.py"
