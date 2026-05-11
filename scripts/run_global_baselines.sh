#!/usr/bin/env bash
# Global anomaly detection baseline evaluation.
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/run_global_baselines.py"
