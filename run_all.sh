#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Reproduce all experiments for
#   "On the Absence of Global Anomalies in Vulnerable Code Representations"
#
# Usage
# -----
#   ./run_all.sh              # full pipeline (GPU phases will be slow on MPS)
#   ./run_all.sh --no-gpu     # skip GPU-required phases (use pre-computed data)
#   ./run_all.sh --gpu-only   # run only GPU-required phases
#
# Prerequisites
# -------------
#   conda activate sae   (or conda run -n sae ... wraps every call)
#   Pre-computed activation files in sae_java_bug/artifacts/activations/
#   Qwen2.5-7B-Instruct accessible (HuggingFace cache or local)
#   SAE weights accessible (rufimelo/vulnerable_code_qwen_coder_standard_16384)
#
# Dependency graph
# ----------------
#   PHASE 1 (GPU)
#     1a  mean_pool_probe.py            → mean_pool/<ts>/*.pt + fig_mean_vs_last_token_pool.pdf
#     1b  position_stratified_probe.py  → positional_profiles_raw.jsonl
#     1c  token_feature_viz.py          → fig_token_feature_1797.pdf + token_acts_*.jsonl
#     1d  mean_pool_sae_probe.py        → mean_pool_sae/<ts>/*.pt + fig_mean_pool_sae_comparison.pdf
#
#   PHASE 2 (no GPU — depends on pre-computed .pt files)
#     2a  layer_cwe_ablation.ipynb      → main probe AUROCs, CWE family probe, residualisation
#     2b  generate_ablation_figures.py  → fig_delta_auc_heatmap.pdf, fig_cwe_language_confound.pdf
#     2c  within_language_baseline.py   → fig_within_lang_by_layer_*.pdf, fig_within_vs_resid.pdf
#     2d  length_controlled_probe.py    → fig_length_controlled.pdf
#     2e  nonlinear_probe.py            → fig_nonlinear_probes.pdf
#     2f  positional_probe_b.py         → fig_f1185_positional_profile.pdf (needs 1b checkpoint)
# =============================================================================

set -euo pipefail

# ── Parse flags ───────────────────────────────────────────────────────────────
RUN_GPU=true
RUN_NO_GPU=true

for arg in "$@"; do
    case "$arg" in
        --no-gpu)   RUN_GPU=false ;;
        --gpu-only) RUN_NO_GPU=false ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
NOTEBOOKS="$REPO_ROOT/sae_java_bug/sparse_autoencoders/notebooks"
ARTIFACTS="$REPO_ROOT/sae_java_bug/artifacts"
LOGS="$REPO_ROOT/logs"
PAPER_FIGS="$REPO_ROOT/../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures"

mkdir -p "$LOGS"

# ── Helpers ───────────────────────────────────────────────────────────────────
CONDA="conda run -n sae --no-capture-output"
START_TOTAL=$(date +%s)

run_step() {
    local label="$1"
    local logfile="$LOGS/${label}.log"
    shift
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ▶  $label"
    echo "     Log: $logfile"
    local t0=$(date +%s)
    if "$@" >"$logfile" 2>&1; then
        local elapsed=$(( $(date +%s) - t0 ))
        echo "  ✓  Done in ${elapsed}s"
    else
        local elapsed=$(( $(date +%s) - t0 ))
        echo "  ✗  FAILED after ${elapsed}s  (see $logfile)"
        tail -20 "$logfile"
        exit 1
    fi
}

# ── Environment check ─────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Checking environment"
$CONDA python -c "
import torch, sklearn, transformers, safetensors
print(f'  torch     {torch.__version__}')
print(f'  sklearn   {sklearn.__version__}')
print(f'  transformers {transformers.__version__}')
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'  device    {device}')
"

# Check required activation files exist
SAE_L11_JSONL="$ARTIFACTS/activations/run_20260218_134529_vulnerable_code_qwen_coder_standard_16384_10M/activations_layer_11_sae_blocks.11.hook_resid_post_component_hook_resid_post.hook_sae_acts_post.jsonl"
RAW_PT="$ARTIFACTS/activations/raw_activations/vulnerable_code_qwen_coder_standard_16384_raw"

echo ""
echo "  Checking required data files:"
for f in "$SAE_L11_JSONL"; do
    if [ -f "$f" ]; then echo "    ✓ $(basename $(dirname $f))/$(basename $f)"
    else              echo "    ✗ MISSING: $f"; fi
done


# =============================================================================
# PHASE 1 — GPU-required experiments
# =============================================================================
if [ "$RUN_GPU" = true ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  PHASE 1 — GPU experiments"
    echo "════════════════════════════════════════════════════════════════════"

    # ── 1a: Mean-token pooling probe ──────────────────────────────────────────
    # Paper: Fig. fig_mean_vs_last_token_pool.pdf + Table tab:mean_vs_last (App. L)
    # Output: artifacts/activations/mean_pool/<ts>/{safe,vulnerable}_layer_*.pt
    #         paper figures/fig_mean_vs_last_token_pool.pdf
    # Runtime: ~30 min on A100, ~90 min on MPS
    run_step "1a_mean_pool_probe" \
        $CONDA python "$NOTEBOOKS/mean_pool_probe.py"

    # ── 1b: Position-stratified probe ─────────────────────────────────────────
    # Paper: App. O (Table tab:sign_test, Fig. fig_f1185_profile)
    # Output: artifacts/token_viz/figures/positional_profiles_raw.jsonl  (checkpoint)
    #         artifacts/token_viz/figures/{fig_positional_profiles,fig_position_delta_heatmap}.pdf
    # Runtime: ~30 min on A100, ~32 min on MPS (observed: 1891s)
    # Note: run with -1 for full dataset; use --load_from to skip if checkpoint exists
    POSITIONAL_CKPT="$ARTIFACTS/token_viz/figures/positional_profiles_raw.jsonl"
    if [ -f "$POSITIONAL_CKPT" ]; then
        echo ""
        echo "  [1b] Checkpoint found — skipping position_stratified_probe.py inference"
    else
        run_step "1b_position_stratified_probe" \
            $CONDA python "$NOTEBOOKS/position_stratified_probe.py" \
                --n_sample -1 \
                --out_dir "$ARTIFACTS/token_viz/figures"
    fi

    # ── 1c: Token-level feature visualisation (Feature 1797) ──────────────────
    # Paper: Fig. fig_token_feature_1797.pdf (App. F)
    # Output: paper figures/fig_token_feature_1797.pdf
    # Runtime: ~5 min on A100 (3 forward passes)
    # Requires: SAE run JSONL (artifacts_study/ on VM, or SAE_L11_JSONL locally)
    run_step "1c_token_feature_viz" \
        $CONDA python "$NOTEBOOKS/token_feature_viz.py" \
            --features 1797 \
            --multi_family \
            --sae_run "$(dirname "$SAE_L11_JSONL")" \
            --out_dir "$PAPER_FIGS"

    # ── 1d: Mean-token SAE probe ───────────────────────────────────────────────
    # Paper: referenced in Discussion (mean-token × SAE combination)
    # Output: artifacts/activations/mean_pool_sae/<ts>/*.pt
    #         paper figures/fig_mean_pool_sae_comparison.pdf
    # Runtime: ~45 min on A100
    run_step "1d_mean_pool_sae_probe" \
        $CONDA python "$NOTEBOOKS/mean_pool_sae_probe.py"

fi  # END PHASE 1


# =============================================================================
# PHASE 2 — No-GPU experiments (use pre-computed .pt files)
# =============================================================================
if [ "$RUN_NO_GPU" = true ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  PHASE 2 — CPU experiments (pre-computed activations)"
    echo "════════════════════════════════════════════════════════════════════"

    # ── 2a: Main CWE ablation notebook ────────────────────────────────────────
    # Paper: Fig. fig_vuln_secure_by_layer.pdf, fig_cwe_family_classification.pdf,
    #         fig_feature_divergence_analysis.pdf, fig_delta_auc_heatmap.pdf,
    #         fig_pairwise_delta_auc.pdf, tab:top-features, AUROC 0.993
    # Also: Appendix G (ablation_l11), Appendix H (within_c_pairwise)
    # Note: longest CPU experiment (~10 min)
    run_step "2a_layer_cwe_ablation_notebook" \
        $CONDA jupyter nbconvert \
            --to notebook \
            --execute \
            --ExecutePreprocessor.timeout=1800 \
            --inplace \
            "$NOTEBOOKS/layer_cwe_ablation.ipynb"

    # ── 2b: Ablation figures ───────────────────────────────────────────────────
    # Paper: fig_cwe_language_confound.pdf (App. C), fig_delta_auc_heatmap.pdf,
    #         fig_ablation_l11.pdf (App. G)
    # Note: hardcoded ΔAUC values from notebook — run 2a first
    run_step "2b_ablation_figures" \
        $CONDA python "$NOTEBOOKS/generate_ablation_figures.py"

    # ── 2c: Within-language baseline ──────────────────────────────────────────
    # Paper: fig_vuln_secure_by_layer.pdf (multi-language overlay),
    #         fig_within_lang_by_layer_*.pdf (App. I),
    #         fig_within_vs_resid.pdf (App. I), fig_within_c_pairwise_*.pdf (App. H)
    # Within-C AUROC 0.469, within-PHP 0.319, within-JS 0.208
    run_step "2c_within_language_baseline" \
        $CONDA python "$NOTEBOOKS/within_language_baseline.py"

    # ── 2d: Length-controlled probe ────────────────────────────────────────────
    # Paper: fig_length_controlled.pdf (App. D)
    # Controls: Ridge-residualise log-token-count + within-quartile probing
    # Requires: tokenizer (downloads automatically, no GPU inference)
    run_step "2d_length_controlled_probe" \
        $CONDA python "$NOTEBOOKS/length_controlled_probe.py"

    # ── 2e: Nonlinear probes ───────────────────────────────────────────────────
    # Paper: fig_nonlinear_probes.pdf + Table tab:nonlinear (App. M)
    # MLP and RBF-SVM across all 8 layers × 2 representations
    # Note: SVM on 5000-sample dataset can be slow (~5 min)
    run_step "2e_nonlinear_probe" \
        $CONDA python "$NOTEBOOKS/nonlinear_probe.py"

    # ── 2f: Position-stratified analysis B ────────────────────────────────────
    # Paper: Table tab:sign_test, Fig. fig_f1185_positional_profile (App. O)
    # Depends on: 1b checkpoint (positional_profiles_raw.jsonl)
    # No GPU — pure sklearn on binned activations
    POSITIONAL_CKPT="$ARTIFACTS/token_viz/figures/positional_profiles_raw.jsonl"
    if [ -f "$POSITIONAL_CKPT" ]; then
        run_step "2f_positional_probe_b" \
            $CONDA python "$NOTEBOOKS/positional_probe_b.py"
        # Copy outputs to paper figures
        cp "$ARTIFACTS/token_viz/figures/fig_f1185_positional_profile.pdf" "$PAPER_FIGS/" 2>/dev/null || true
        cp "$ARTIFACTS/token_viz/figures/fig_perfeature_auroc_bins1_19.pdf" "$PAPER_FIGS/" 2>/dev/null || true
    else
        echo ""
        echo "  [2f] SKIPPED — positional_profiles_raw.jsonl not found (run phase 1b first)"
    fi

fi  # END PHASE 2


# =============================================================================
# Summary
# =============================================================================
ELAPSED_TOTAL=$(( $(date +%s) - START_TOTAL ))
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  All experiments complete in ${ELAPSED_TOTAL}s"
echo ""
echo "  Figures written to:"
echo "    $PAPER_FIGS"
echo ""
echo "  Paper figure inventory:"
for pdf in "$PAPER_FIGS"/*.pdf; do
    [ -f "$pdf" ] && echo "    $(basename $pdf)"
done
echo "════════════════════════════════════════════════════════════════════"
