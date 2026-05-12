#!/usr/bin/env bash
# Length-controlled probe — residualises / stratifies by token count.
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/length_controlled_probe.py"
