#!/usr/bin/env bash
# Nonlinear probe comparison (LogReg vs MLP vs RF) — rules out linear-probing artefact.
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/nonlinear_probe.py"
