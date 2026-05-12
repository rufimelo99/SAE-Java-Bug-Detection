#!/usr/bin/env bash
# Generation-level steering experiment.
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/generation_steering.py"
