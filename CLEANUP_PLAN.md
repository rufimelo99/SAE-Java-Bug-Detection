# Repository Cleanup Plan

## Executive Summary

- **27 shell scripts**: All experimental (moved to pipeline via run_all.sh)
- **14 experimental Python scripts**: One-off analyses, reviewer responses, debugging
- **4 reviewer-specific analysis scripts**: Issue investigations and response fixes
- **~76 GB artifacts**: Mostly raw activations that can be pruned

## Recommended Structure

```
scripts/
├── _archive/              # Move experimental/one-off scripts here
│   ├── experimental/
│   ├── reviewer_response/
│   └── debugging/
└── [keep main pipeline scripts only - currently wrapped via run_all.sh]
```

---

## SHELL SCRIPTS (.sh) - All 27 Can Be Organized

### Status: ALL are wrappers that call Python notebooks
All `.sh` scripts in `/scripts/` are just thin wrappers calling notebooks in `sae_java_bug/sparse_autoencoders/notebooks/`. They're organized for readability but could be archived.

### 🔧 Keep (Helper Scripts - 4)
These coordinate multiple experiments or the full pipeline:
```
scripts/
  run_all.sh                          ✓ Main pipeline orchestrator
  run_all_response_experiments.sh      ✓ Reviewer response batch runner
  run_global_baselines.sh             ✓ Global anomaly detection suite
  run_multi_dataset.sh                ✓ SVEN + PreciseBugs validation
  run_sae_exploration.sh              ✓ SAE feature analysis suite
```

### ⚠ Move to `scripts/_archive/experimental/` (23)
These are individual experiment runners, kept separate from main pipeline:
```
advanced_pooling_probe.sh
causal_patching.sh
collect_per_token_sae.sh
cross_layer_direction_probe.sh
direction_transfer_sven.sh
directional_readout_probe.sh
feature_asymmetry_crosslayer.sh
feature_direction_loading.sh
generate_ablation_figures.sh
generate_advanced_pooling_figure.sh
generation_steering.sh
hypothesis.sh
length_controlled_probe.sh
magnitude_asymmetry_crosslayer.sh
mean_pool_probe.sh
mean_pool_sae_probe.sh
nonlinear_probe.sh
paired_suppression_test.sh
patch_length_analysis.sh
position_stratified_probe.sh
positional_probe_b.sh
token_feature_viz.sh
token_pca_3d.sh
token_position_importance.sh
token_trajectory_3d.sh
within_language_mean_pool_probe.sh
within_language_probe.sh
```

---

## PYTHON SCRIPTS (.py) - 18 Total

### ✓ Keep in `scripts/` (1)
```
paired_ranking_task.py              ✓ New validation script for paper
```

### 📊 Move to `scripts/_archive/reviewer_response/` (4)
These address specific reviewer issues:
```
issue_01_cv_leakage_minimal.py         # Check for train/test leakage
issue_02_confound_controls_standalone.py # Verify confound control
issue_02_confound_numpy_only.py        # NumPy-only confound verification
reviewer_response_analysis.py          # Summarize responses
```

### 🔬 Move to `scripts/_archive/experimental/` (14)
These are one-off analyses and debugging scripts:
```
01_stronger_pooling_baselines.py        # Compare pooling methods
02_confound_controls.py                 # Confound analysis
02_confound_controls_example.py         # Example confound walkthrough
03_sae_feature_stability.py             # SAE feature persistence
04_external_validation_steering.py      # Cross-model steering validation
05_steering_sign_convention.py          # Direction sign investigation
05_steering_sign_convention_example.py  # Sign convention example
06_cv_leakage_check.py                  # CV contamination analysis
07_length_guardtoken_controls.py        # Length control verification
08_sae_validation_metrics.py            # SAE metric validation
09_fundamental_difficulty_baselines.py  # Establish performance floor
run_reviewer_response_fixes.py           # Apply reviewer response fixes
upload_activations.py                   # Artifact upload utility
```

---

## ARTIFACT CLEANUP - 76 GB Available

### ⚠ Large Artifact Sets - Prioritize for Cleanup

| Artifact | Size | Notes | Action |
|----------|------|-------|--------|
| `activations/` (total) | **76 GB** | Raw and pooled activations | Keep only `mean_pool` |
| `TO_UPLOAD/` | 6.1 GB | Upload staging area | Delete (already uploaded) |
| `bigvul_c_only/` | 24 GB | BigVul dataset activation set | Keep if paper needs it |
| `per_token/` | 26 GB | Per-token detailed analysis | Can delete if not in final paper |
| `mean_pool_sae/` | 3.9 GB | SAE-based pooling | Keep (paper figure) |
| `multi_model_probing/` | 1.5 GB | CodeLlama, StarCoder2 comparison | Keep (paper validation) |
| `precisebugs_c_only/` | 10 GB | Dataset validation | Keep (paper appendix) |
| `raw_activations/` | 2.1 GB | Qwen2.5 raw hidden states | Can compress or delete |
| `sven_c_only/` | 1.1 GB | SVEN dataset validation | Keep (paper validation) |
| `TOPK/` + `TOPK_tensors/` | 1.0 GB | Feature importance analysis | Delete (experimental) |

### Recommended Cleanup Strategy

**Tier 1 - Delete immediately (1.6 GB freed):**
```
sae_java_bug/artifacts/activations/TOPK/
sae_java_bug/artifacts/activations/TOPK_tensors/
sae_java_bug/artifacts/activations/TO_UPLOAD/
```

**Tier 2 - Archive to external storage if not in paper (61 GB conditional):**
```
# Check which are referenced in final paper before deleting:
sae_java_bug/artifacts/activations/bigvul_c_only/        (24 GB)
sae_java_bug/artifacts/activations/per_token/            (26 GB)
sae_java_bug/artifacts/activations/raw_activations/      (2.1 GB)
# Note: These have named backups in artifacts:
# - raw_activations/ ≈ vulnerable_code_qwen_coder_standard_16384_raw/
# - Keep only one copy
```

**Tier 3 - Keep (essential for paper reproducibility):**
```
sae_java_bug/artifacts/activations/mean_pool/            (546 MB) ✓
sae_java_bug/artifacts/activations/mean_pool_sae/        (3.9 GB) ✓
sae_java_bug/artifacts/activations/multi_model_probing/  (1.5 GB) ✓
sae_java_bug/artifacts/activations/precisebugs_c_only/   (10 GB) ✓
sae_java_bug/artifacts/activations/sven_c_only/          (1.1 GB) ✓
sae_java_bug/artifacts/figures/                          (< 100 MB) ✓
sae_java_bug/artifacts/causal_patching/                  (small) ✓
sae_java_bug/artifacts/results/                          (< 100 MB) ✓
```

---

## Implementation Steps

### Step 1: Create Archive Structure
```bash
mkdir -p scripts/_archive/experimental
mkdir -p scripts/_archive/reviewer_response
mkdir -p scripts/_archive/debugging
```

### Step 2: Move Shell Scripts (23)
```bash
cd scripts/
for f in advanced_pooling_probe.sh causal_patching.sh ... within_language_probe.sh; do
  mv "$f" _archive/experimental/
done
```

### Step 3: Move Python Scripts (18)
```bash
cd scripts/
# Reviewer response
mv issue_*.py reviewer_response_analysis.py _archive/reviewer_response/

# Experimental
mv 0*.py 05_steering*.py 06_*.py 07_*.py 08_*.py 09_*.py \
   run_reviewer_response_fixes.py upload_activations.py _archive/experimental/
```

### Step 4: Clean Up Artifacts
```bash
# Tier 1 (delete immediately)
rm -rf sae_java_bug/artifacts/activations/TOPK
rm -rf sae_java_bug/artifacts/activations/TOPK_tensors
rm -rf sae_java_bug/artifacts/activations/TO_UPLOAD

# Tier 2 (conditional - check paper references first)
# Before deleting, verify paper uses these:
grep -r "bigvul_c_only\|per_token\|raw_activations" \
  ../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/
```

### Step 5: Create Archive README
```bash
cat > scripts/_archive/README.md << 'EOF'
# Archived Scripts

This folder contains experimental scripts, reviewer response analyses, and debugging utilities not part of the main pipeline.

## Organization

- **experimental/** - One-off experiment runners and analysis scripts
- **reviewer_response/** - Scripts addressing specific reviewer issues
- **debugging/** - Utility scripts for validation and testing

## How to Run

To run individual experiments:
```bash
bash experimental/[script].sh
```

Or use the batch runners in the main scripts/ folder:
```bash
bash ../run_all_response_experiments.sh
```

## What Happened to These Scripts?

They were moved here during repository cleanup to:
1. Reduce clutter in main scripts/ folder
2. Clearly separate main pipeline from experiments
3. Preserve reproducibility while improving organization

All functionality is preserved. The main pipeline (`run_all.sh`) calls the core analysis through Python notebooks.
EOF
```

---

## Verification Checklist

Before implementing cleanup:

- [ ] Check that `run_all.sh` still finds all necessary notebooks (not moving .sh files breaks this)
- [ ] Verify paper only references figures in `sae_java_bug/artifacts/figures/`
- [ ] Confirm no external references to archived scripts
- [ ] Back up current state: `git commit -m "pre-cleanup snapshot"`
- [ ] Test that `./run_all.sh --no-gpu` still works after moving scripts

---

## Summary of Changes

| Category | Count | Action |
|----------|-------|--------|
| Shell scripts | 4 | Keep |
| Shell scripts | 23 | → `_archive/experimental/` |
| Python scripts | 1 | Keep |
| Python scripts | 4 | → `_archive/reviewer_response/` |
| Python scripts | 14 | → `_archive/experimental/` |
| Artifacts | 1.6 GB | Delete immediately |
| Artifacts | 61 GB | Archive if not in paper |
| **Net cleanup** | **~63 GB** | **Potential space savings** |

---

## Notes

1. **Run_all.sh compatibility**: The main pipeline uses Python notebooks directly, not the .sh wrappers. Moving .sh files won't break anything.

2. **Artifact storage**: The 6.1GB `TO_UPLOAD/` folder was likely a staging area for uploading to HuggingFace. Safe to delete once verified.

3. **Duplicate activations**: `raw_activations/` and `vulnerable_code_qwen_coder_standard_16384_raw/` appear to be the same (2.1GB each). Keep only one.

4. **Paper references**: Before deleting large artifacts (bigvul, per_token), verify they're not referenced in the final paper figures.
