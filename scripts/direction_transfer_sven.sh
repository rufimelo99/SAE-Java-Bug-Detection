#!/usr/bin/env bash
# Direction transfer analysis: DeltaSecommits → SVEN.
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/direction_transfer_sven.py"
