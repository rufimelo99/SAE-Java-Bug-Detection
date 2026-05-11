#!/usr/bin/env bash
# Directional readout probe — linear readout along vulnerability direction.
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/directional_readout_probe.py"
