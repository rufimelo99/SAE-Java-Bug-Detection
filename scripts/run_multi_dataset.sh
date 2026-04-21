#!/usr/bin/env bash
# Multi-dataset vulnerability probing replication.
#
# SVEN activations are pre-extracted in code-security-probing.
# BigVul and PreciseBugs require a GPU (≥24 GB VRAM) for extraction.
#
# Usage:
#   bash scripts/run_multi_dataset.sh                    # full run
#   bash scripts/run_multi_dataset.sh --skip-extraction  # skip GPU step
#   bash scripts/run_multi_dataset.sh --sven-only        # SVEN + DeltaSecommits only
set -euo pipefail

NOTEBOOKS="$(dirname "$0")/../sae_java_bug/sparse_autoencoders/notebooks"
CSP="$(dirname "$0")/../../code-security-probing"

SKIP_EXTRACTION=0
SVEN_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --skip-extraction) SKIP_EXTRACTION=1 ;;
    --sven-only)       SVEN_ONLY=1; SKIP_EXTRACTION=1 ;;
  esac
done

echo "==================================================================="
echo " Multi-dataset vulnerability probing"
echo "==================================================================="

# ── Step 1: Extract activations for BigVul and PreciseBugs ──────────────────

if [[ $SKIP_EXTRACTION -eq 0 ]]; then

  echo "--- BigVul extraction (GPU ≥24 GB VRAM required) ---"
  if [[ -f "${CSP}/artifacts/activations/bigvul_c_only/layer_0_train.jsonl" ]]; then
    echo "  [SKIP] BigVul activations already present."
  else
    python "${CSP}/experiments/00b_extract_sven_acts.py" \
      --pairs_jsonl "${CSP}/artifacts/data/bigvul_raw/bigvul_c_pairs.jsonl" \
      --out_subdir  bigvul_c_only
  fi

  echo ""
  echo "--- PreciseBugs extraction (GPU ≥24 GB VRAM required) ---"
  if [[ -f "${CSP}/artifacts/activations/precisebugs_c_only/layer_0_train.jsonl" ]]; then
    echo "  [SKIP] PreciseBugs activations already present."
  else
    python "${CSP}/experiments/00b_extract_sven_acts.py" \
      --pairs_jsonl "${CSP}/artifacts/data/precisebugs_raw/precisebugs_c_pairs.jsonl" \
      --out_subdir  precisebugs_c_only
  fi

fi

# ── Step 2: Multi-dataset comparison ────────────────────────────────────────

echo ""
if [[ $SVEN_ONLY -eq 1 ]]; then
  echo "--- Running multi-dataset comparison (DeltaSecommits + SVEN) ---"
  python "${NOTEBOOKS}/multi_dataset_comparison.py" \
    --datasets DeltaSecommits SVEN
else
  echo "--- Running multi-dataset comparison (all available datasets) ---"
  python "${NOTEBOOKS}/multi_dataset_comparison.py"
fi

echo ""
echo "==================================================================="
echo " Done."
echo "   sae_java_bug/artifacts/results/multi_dataset_comparison.json"
echo "   sae_java_bug/artifacts/figures/fig_multi_dataset_comparison.pdf"
echo "==================================================================="
