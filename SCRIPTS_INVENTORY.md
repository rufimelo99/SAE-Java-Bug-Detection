# Scripts Inventory & Cleanup Guide

## Quick Reference

### Current State (Before Cleanup)
```
scripts/
├── *.sh (27 files)           ← All experimental wrappers
├── *.py (18 files)           ← Mix of pipeline, experimental, debugging
└── README.md
```

### After Cleanup
```
scripts/
├── run_all.sh                ✓ Main pipeline
├── run_all_response_experiments.sh  ✓ Reviewer response batch
├── run_global_baselines.sh   ✓ Global anomaly detection
├── run_multi_dataset.sh      ✓ External validation (SVEN, PreciseBugs)
├── run_sae_exploration.sh    ✓ SAE feature analysis
├── paired_ranking_task.py    ✓ Paper validation (NEW)
├── cleanup.sh                ✓ Cleanup automation
├── _archive/
│   ├── experimental/         ← 23 .sh + 14 .py (one-off analyses)
│   ├── reviewer_response/    ← 4 .py (issue investigations)
│   └── README.md
└── [other config files]
```

---

## Shell Scripts (27 Total)

All `.sh` files are thin wrappers around Python notebooks. They can safely be archived.

### Keep in `scripts/` - 5 Helper Scripts
These are NOT wrappers—they orchestrate multiple experiments:

| Script | Purpose | Artifact Output |
|--------|---------|-----------------|
| `run_all.sh` | **Main pipeline**: coordinates all GPU and CPU phases | `figures/`, `results/` |
| `run_all_response_experiments.sh` | Batch-run reviewer response experiments | Multiple analysis dirs |
| `run_global_baselines.sh` | Run global anomaly detection baselines | `results/global_anomalies.json` |
| `run_multi_dataset.sh` | Validate on SVEN + PreciseBugs | `sven_c_only/`, `precisebugs_c_only/` |
| `run_sae_exploration.sh` | SAE feature importance analysis | `mean_pool_sae/`, feature analysis |

### Archive to `_archive/experimental/` - 23 Wrapper Scripts
These call individual notebooks and are kept separate for clarity:

```
Pooling & Aggregation:
  advanced_pooling_probe.sh          (calls advanced_pooling_probe.py)
  mean_pool_probe.sh                 (calls mean_pool_probe.py)
  mean_pool_sae_probe.sh             (calls mean_pool_sae_probe.py)
  within_language_mean_pool_probe.sh (calls within_language_mean_pool_probe.py)

Position & Token Analysis:
  position_stratified_probe.sh       (calls position_stratified_probe.py)
  positional_probe_b.sh              (calls positional_probe_b.py)
  token_feature_viz.sh               (calls token_feature_viz.py)
  token_pca_3d.sh                    (calls token_pca_3d.py)
  token_trajectory_3d.sh             (calls token_trajectory_3d.py)
  token_position_importance.sh       (calls token_position_importance.py)

Direction & Geometry:
  cross_layer_direction_probe.sh     (calls cross_layer_direction_probe.py)
  directional_readout_probe.sh       (calls directional_readout_probe.py)
  direction_transfer_sven.sh         (calls direction_transfer_sven.py)
  magnitude_asymmetry_crosslayer.sh  (calls magnitude_asymmetry_crosslayer.py)
  feature_asymmetry_crosslayer.sh    (calls feature_asymmetry_crosslayer.py)
  feature_direction_loading.sh       (calls feature_direction_loading.py)

Causal & Intervention:
  causal_patching.sh                 (calls causal_patching.py)
  generation_steering.sh             (calls generation_steering.py)
  paired_suppression_test.sh         (calls paired_suppression_test.py)

Analysis & Probing:
  length_controlled_probe.sh         (calls length_controlled_probe.py)
  nonlinear_probe.sh                 (calls nonlinear_probe.py)
  within_language_probe.sh           (calls within_language_probe.py)

Data Collection:
  collect_per_token_sae.sh           (calls collect_per_token_sae.py)
  generate_ablation_figures.sh       (calls generate_ablation_figures.py)
  generate_advanced_pooling_figure.sh (calls generate_advanced_pooling_figure.py)

Other:
  hypothesis.sh                      (experimental hypothesis testing)
  patch_length_analysis.sh           (calls patch_length_analysis.py)
```

**Rationale**: These are individual experiment runners that were useful during development but are now superseded by the batch runners (`run_all*.sh`). The corresponding Python notebooks are preserved in `sae_java_bug/sparse_autoencoders/notebooks/` for reproducibility.

---

## Python Scripts (18 Total)

### Keep in `scripts/` - 1 NEW Script
```
paired_ranking_task.py               ✓ Validates "relative vs absolute" claim
                                     → Used for paper validation
                                     → Generated paired_ranking_results.json
```

### Archive to `_archive/reviewer_response/` - 4 Scripts
These address specific reviewer concerns raised during review:

```
issue_01_cv_leakage_minimal.py       ✓ Demonstrates pair-stratified CV prevents leakage
                                     → Responds to: "How do you prevent train/test contamination?"
                                     → Uses only NumPy, no dependencies

issue_02_confound_controls_standalone.py  ✓ Verifies confound controls work
                                     → Responds to: "Are language/CWE confounds properly controlled?"
                                     → Pure NumPy implementation

issue_02_confound_numpy_only.py      ✓ Minimal confound control verification
                                     → Lightweight version of above
                                     → Useful for quick validation

reviewer_response_analysis.py        ✓ Summarizes response status
                                     → Tracks which reviewer issues have been addressed
                                     → Generates response summary table
```

### Archive to `_archive/experimental/` - 14 Scripts
These are one-off explorations and debugging utilities:

```
Baseline Comparisons:
  01_stronger_pooling_baselines.py   # Compare token aggregation strategies
  09_fundamental_difficulty_baselines.py  # Establish performance floor with standard methods

Confound & Control Analyses:
  02_confound_controls.py            # Detailed confound analysis
  02_confound_controls_example.py    # Walkthrough example
  07_length_guardtoken_controls.py   # Control for sequence length artifacts

SAE Feature Analysis:
  03_sae_feature_stability.py        # How stable are SAE features across models?
  08_sae_validation_metrics.py       # Validate SAE-based metrics

Steering & Direction Analysis:
  04_external_validation_steering.py # Cross-model steering validation
  05_steering_sign_convention.py     # Investigate direction sign conventions
  05_steering_sign_convention_example.py  # Example walkthrough

Cross-Validation & Leakage:
  06_cv_leakage_check.py             # Comprehensive leakage detection

Utilities:
  run_reviewer_response_fixes.py      # Apply fixes from reviewer responses
  upload_activations.py              # Upload activation files to HuggingFace
```

---

## Artifact Artifacts - Size Breakdown

### Total: ~76 GB in `sae_java_bug/artifacts/`

#### Keep (Essential) - ~17 GB
```
mean_pool/               546 MB   ✓ Main activation set (PCA-50, mean pooled)
mean_pool_sae/          3.9 GB   ✓ SAE feature-based pooling (paper comparison)
multi_model_probing/    1.5 GB   ✓ CodeLlama, StarCoder2 validation
precisebugs_c_only/      10 GB   ✓ Dataset validation set (paper appendix)
sven_c_only/            1.1 GB   ✓ Dataset validation set (paper appendix)
figures/                <100 MB  ✓ All paper figures
causal_patching/        <100 MB  ✓ Activation patching analysis
results/                <100 MB  ✓ Summary results tables
```

#### Delete (Tier 1) - 7.7 GB [**SAFE**]
```
TOPK/                    560 MB   ✗ Feature importance (experimental)
TOPK_tensors/            441 MB   ✗ Feature tensors (experimental)
TO_UPLOAD/              6.1 GB   ✗ Upload staging area (already uploaded)
                       ─────────
Total: 7.7 GB savings (safe to delete)
```

#### Archive/Delete (Tier 2) - ~61 GB [**CHECK PAPER FIRST**]
```
bigvul_c_only/           24 GB   ⚠ BigVul dataset analysis (verify not in paper)
per_token/               26 GB   ⚠ Per-token detailed analysis (experimental)
raw_activations/         2.1 GB  ⚠ Duplicate of vulnerable_code_qwen_* (keep only one)
                       ─────────
Total: 52 GB potential savings (conditional)
```

Note: Before deleting Tier 2, verify these aren't referenced in paper figures:
```bash
grep -r "bigvul_c_only\|per_token\|raw_activations" \
  ../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures/
```

---

## Cleanup Execution

### Option 1: Manual (if you want to review)
```bash
# Just move scripts
mkdir -p scripts/_archive/{experimental,reviewer_response}
mv scripts/issue_*.py scripts/_archive/reviewer_response/
mv scripts/[0-9]*.py scripts/_archive/experimental/
# ... etc (see CLEANUP_PLAN.md for full list)

# Git commit
git add -A
git commit -m "chore: archive experimental scripts to reduce clutter"
```

### Option 2: Automated (recommended)
```bash
# Move all scripts
./scripts/cleanup.sh

# Test that pipeline still works
./run_all.sh --no-gpu

# Clean artifacts safely
./scripts/cleanup.sh --clean-artifacts tier1

# Git commit
git add -A && git commit -m "chore: cleanup experimental scripts and artifacts"
```

---

## Verification After Cleanup

```bash
# 1. Check directory structure
ls -la scripts/                           # Should show ~9 files
ls -la scripts/_archive/experimental/     # Should show ~37 files
ls -la scripts/_archive/reviewer_response/ # Should show 4 files

# 2. Verify run_all.sh still works
./run_all.sh --no-gpu                     # Should run without errors

# 3. Check Git status
git status                                # Should show script moves

# 4. Verify archive completeness
wc -l scripts/_archive/{experimental,reviewer_response}/*.py
# Should match: 4 + 14 = 18 Python files moved
```

---

## FAQ

**Q: Will cleanup break run_all.sh?**
A: No. run_all.sh calls Python notebooks directly, not the .sh wrappers. Moving .sh files has zero impact.

**Q: Can I restore archived scripts?**
A: Yes, they're all in Git history. Just `git checkout HEAD~1 scripts/[script].sh` or look in `scripts/_archive/`.

**Q: What if I need to run an archived script?**
A: They still work! Run from the archive:
```bash
bash scripts/_archive/experimental/advanced_pooling_probe.sh
# or
python scripts/_archive/experimental/01_stronger_pooling_baselines.py
```

**Q: How much space does cleanup save?**
A: Scripts only: minimal (~5 MB)
Artifacts tier1: ~7.7 GB
Artifacts tier2: ~52 GB (if paper doesn't reference them)

**Q: What about notebooks in `sae_java_bug/sparse_autoencoders/notebooks/`?**
A: Keep them all. They're referenced by run_all.sh and the archived scripts.

---

## Related Documentation

- `CLEANUP_PLAN.md` - Detailed cleanup strategy
- `scripts/cleanup.sh` - Automated cleanup script
- `scripts/_archive/README.md` - Archive contents (created during cleanup)
