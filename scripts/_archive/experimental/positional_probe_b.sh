#!/usr/bin/env bash
# Positional probe B — checks if discriminative signal survives after dropping bin 0.
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/positional_probe_b.py"
