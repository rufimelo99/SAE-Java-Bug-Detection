#!/usr/bin/env bash
# Within-language mean-token pooling probe — checks if mean-pool AUROC gain
# survives inside a single programming language.
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/within_language_mean_pool_probe.py"
