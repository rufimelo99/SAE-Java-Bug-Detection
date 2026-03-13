#!/usr/bin/env bash
# Within-language (C / PHP / JS) vulnerability probe — controls for language confound.
set -euo pipefail
NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
python "${NOTEBOOKS}/within_language_baseline.py"
