#!/usr/bin/env bash
# Activation patching experiment — body vs last-token effects.
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/causal_patching.py"
