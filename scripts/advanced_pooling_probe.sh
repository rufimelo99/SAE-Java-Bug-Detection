#!/usr/bin/env bash
# Advanced pooling strategies (last-token / mean / attention-weighted / diff-restricted).
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/advanced_pooling_probe.py"
