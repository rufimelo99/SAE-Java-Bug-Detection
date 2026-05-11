# EMNLP 2026 Paper Pipeline — Setup & Execution Guide

**Status**: All experiments created and ready to run. Papers compile successfully.

**Deadline**: May 25, 2026 (ARR submission)

---

## Quick Start (3 commands)

```bash
# 1. Activate conda environment
conda activate sae

# 2. Navigate to repo
cd /Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection

# 3. Run full pipeline
./run_all.sh
```

---

## What This Pipeline Does

### Two Short Papers (4 pages each + appendix)

**Paper 0**: "Vulnerability Detection is Not Linear"
- Core claim: Vulnerability is semantically diffuse, not linearly separable
- Evidence: AUROC 0.5 with proper within-language/CWE controls
- New figure: Patch length distributions by CWE family

**Paper 1**: "The Vulnerability Direction as a Steering Target"
- Core claim: Vulnerability direction is a reliable mechanistic steering lever
- Evidence: Body-token activation patching; direction stability across layers
- Extension: Cross-dataset validation (SVEN) + feature loading analysis

---

## Full Pipeline: What Runs When

### PHASE 1 — GPU Experiments (~4 hours on A100, ~12 hours on MPS)

```
1a  mean_pool_probe              → mean_pool activations (3584-dim)
1b  position_stratified_probe    → token-position analysis
1c  token_feature_viz            → Feature 1797 visualization
1d  mean_pool_sae_probe          → SAE features analysis
1e  advanced_pooling_probe       → Attention-weighted pooling
```

**Output**: Pre-computed activation tensors in `artifacts/activations/mean_pool/`

### PHASE 2 — CPU Experiments (~30 min total)

```
Paper 0:
2a  layer_cwe_ablation           → CWE classification AUROCs
2b  ablation_figures             → fig_cwe_language_confound.pdf
2f  within_language_baseline     → Language controls

Paper 1:
2c  cross_layer_direction_probe  → fig_direction_cosine_sim.pdf ⭐
2d  causal_patching              → fig_causal_patching.pdf ⭐
2e  directional_readout_probe    → fig_directional_readout_comparison.pdf ⭐

Paper 0 (new):
2g  length_controlled_probe      → Length controls
2h  nonlinear_probe              → Nonlinear probe ceiling
2i  positional_probe_b           → Position analysis
2j  advanced_pooling_figure      → Pooling comparison

Paper 0 (new - requires run completion):
     patch_length_analysis       → fig_patch_length_by_cwe.pdf ⭐ [ALREADY RUN]
     run_global_baselines        → fig_global_anomaly_baselines.pdf ⭐
     direction_transfer_sven     → SVEN cross-dataset validation ⭐
     feature_direction_loading   → SAE feature correlations ⭐
```

⭐ = New experiments created for EMNLP

---

## Running Different Parts

### Run Full Pipeline
```bash
./run_all.sh
```

### Skip GPU Phase (use pre-computed tensors)
```bash
./run_all.sh --no-gpu
```

### GPU Phase Only
```bash
./run_all.sh --gpu-only
```

---

## Environment Setup

### 1. Verify conda environment exists

```bash
conda env list | grep sae
```

If `sae` environment doesn't exist, create it:

```bash
conda create -n sae python=3.11 pytorch::pytorch pytorch::pytorch-cuda=11.8 -y
conda activate sae
pip install -r requirements.txt
```

### 2. Verify required data files exist

The pipeline expects:
- `sae_java_bug/artifacts/activations/mean_pool/*/` — created by Phase 1a
- `code-security-probing/artifacts/activations/c_only/` — pre-existing dataset
- Qwen2.5-7B-Instruct cached (auto-downloads on first run)

Check:
```bash
ls sae_java_bug/artifacts/activations/
ls /Users/rmelo/Documents/GitHub/code-security-probing/artifacts/activations/c_only/ | head
```

### 3. Verify paper figures directory

```bash
mkdir -p ../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures
```

---

## New Experiments (Phase 2 — CPU Only)

These notebooks were created specifically for EMNLP and don't require GPU:

### Patch Length Analysis
- **What**: Analyzes patch length differences (secure - vulnerable) by CWE family
- **Run**: `bash scripts/patch_length_analysis.sh`
- **Output**: `figures/fig_patch_length_by_cwe.pdf`
- **Status**: ✅ Already run (865 pairs analyzed)

### Global Anomaly Detection Baselines
- **What**: Tests if vulnerability is detected as global anomaly (Isolation Forest, One-Class SVM, LOF)
- **Run**: `bash scripts/run_global_baselines.sh`
- **Output**: `figures/fig_global_anomaly_baselines.pdf`
- **Time**: ~2 min (CPU)

### SVEN Cross-Dataset Direction Transfer
- **What**: Validates vulnerability direction generalizes to independent SVEN dataset
- **Run**: `bash scripts/direction_transfer_sven.sh`
- **Output**: `figures/fig_direction_transfer_sven.pdf` + appendix table
- **Time**: ~5 min (CPU)
- **Requires**: Phase 1a completed (mean_pool activations)

### Feature-to-Direction Loading
- **What**: Identifies which SAE features load onto vulnerability direction
- **Run**: `bash scripts/feature_direction_loading.py`
- **Output**: `feature_direction_loading_results.json` (mechanistic analysis)
- **Time**: ~10 min (CPU)
- **Requires**: Phase 1d completed (SAE activations)

### Generation Steering (GPU)
- **What**: Steers code generation toward security via residual stream modification
- **Run**: `bash scripts/generation_steering.sh`
- **Output**: `figures/fig_generation_steering.pdf`
- **Time**: ~30 min (GPU)
- **Status**: Framework ready; needs full GPU execution

---

## File Structure

```
SAE-Java-Bug-Detection/
├── run_all.sh                      # Main pipeline orchestrator
├── EMNLP_PIPELINE_GUIDE.md         # This file
├── requirements.txt
├── scripts/
│   ├── patch_length_analysis.sh    # ⭐ NEW
│   ├── run_global_baselines.sh     # ⭐ NEW
│   ├── direction_transfer_sven.sh  # ⭐ NEW
│   ├── feature_direction_loading.sh # ⭐ NEW
│   ├── generation_steering.sh      # ⭐ NEW
│   ├── causal_patching.sh          # ⭐ NEW
│   ├── directional_readout_probe.sh # ⭐ NEW
│   ├── cross_layer_direction_probe.sh
│   └── [13 other experiment scripts]
│
├── sae_java_bug/
│   ├── sparse_autoencoders/notebooks/
│   │   ├── patch_length_analysis.py           # ⭐ NEW
│   │   ├── run_global_baselines.py            # ⭐ NEW
│   │   ├── direction_transfer_sven.py         # ⭐ NEW
│   │   ├── feature_direction_loading.py       # ⭐ NEW
│   │   ├── generation_steering.py             # ⭐ NEW
│   │   ├── causal_patching.py
│   │   ├── directional_readout_probe.py
│   │   ├── cross_layer_direction_probe.py
│   │   └── [20+ other notebooks]
│   │
│   ├── artifacts/activations/
│   │   ├── mean_pool/              # Phase 1a output
│   │   ├── mean_pool_sae/          # Phase 1d output
│   │   ├── advanced_pool/          # Phase 1e output
│   │   ├── raw_activations/        # Pre-existing
│   │   ├── c_only/                 # Pre-existing (DeltaSecommits)
│   │   ├── sven_c_only/            # Pre-existing (SVEN)
│   │   └── [other datasets]
│   └── evaluation/
│       ├── global_baselines.py      # Anomaly detection methods
│       └── [other evaluation code]
│
└── ../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/
    ├── paper_0_confounding/
    │   ├── paper_0.tex             # ✅ Compiles
    │   ├── sections/
    │   │   ├── 03_patch_structure.tex # ✅ Updated with fig_patch_length_by_cwe.pdf
    │   │   └── [7 other sections]
    │   └── figures/
    │       ├── fig_patch_length_by_cwe.pdf     # ⭐ NEW
    │       ├── fig_cwe_language_confound.pdf
    │       └── [7 other figures]
    │
    ├── paper_1_mechanistic/
    │   ├── paper_1.tex             # ✅ Compiles
    │   ├── sections/
    │   │   ├── 02_vulnerability_direction.tex
    │   │   ├── 04_activation_patching.tex
    │   │   └── [6 other sections]
    │   └── figures/
    │       ├── fig_direction_cosine_sim.pdf
    │       ├── fig_causal_patching.pdf
    │       ├── fig_directional_readout_comparison.pdf
    │       └── [5 other figures]
    │
    ├── figures/                    # Shared output dir
    ├── biblio.bib
    └── DATASET_ASSESSMENT.md       # Dataset landscape analysis
```

---

## Troubleshooting

### "No mean_pool runs found"
**Error**: Scripts that depend on Phase 1a complain no mean_pool activations exist.
**Fix**: Run Phase 1 first: `./run_all.sh` (don't use `--no-gpu`)

### "torch not found"
**Error**: Experiments import torch but environment isn't activated.
**Fix**: 
```bash
conda activate sae
python -c "import torch; print(torch.__version__)"
```

### "activation file not found"
**Error**: Scripts can't find `code-security-probing/artifacts/activations/c_only/`
**Fix**: Verify sibling repo exists:
```bash
ls /Users/rmelo/Documents/GitHub/code-security-probing/artifacts/activations/c_only/ | head
```

### Papers don't compile
**Error**: pdflatex fails with undefined references or missing figures.
**Fix**: 
```bash
cd paper_0_confounding && pdflatex -interaction=nonstopmode paper_0.tex
# Check for errors in output; verify figures/ directory exists and is readable
```

---

## Expected Outputs

### After Full Run

```
paper_0_confounding/paper_0.pdf              (218 KB, ✅ compiles)
paper_1_mechanistic/paper_1.pdf              (300 KB, ✅ compiles)

figures/
├── fig_patch_length_by_cwe.pdf             (28 KB) ⭐
├── fig_direction_cosine_sim.pdf            (31 KB)
├── fig_alignment_trajectory.pdf            (30 KB)
├── fig_causal_patching.pdf                 (18 KB)
├── fig_directional_readout_comparison.pdf  (15 KB)
├── fig_cwe_language_confound.pdf           (29 KB)
├── fig_nonlinear_probes.pdf                (22 KB)
├── fig_length_controlled.pdf               (18 KB)
└── [other figures from Phase 1-2]

logs/
├── 1a_mean_pool_probe.log
├── 1b_position_stratified_probe.log
├── 2c_cross_layer_direction_probe.log      ⭐
├── 2d_causal_patching.log                  ⭐
├── 2e_directional_readout_probe.log        ⭐
└── [other phase logs]
```

---

## Timing Estimates

| Phase | Duration | Hardware |
|-------|----------|----------|
| 1a (mean_pool_probe) | ~30 min | A100 / ~90 min on MPS |
| 1b (position) | ~30 min | A100 / ~32 min on MPS |
| 1c (token_feature_viz) | ~5 min | GPU |
| 1d (mean_pool_sae) | ~45 min | GPU |
| 1e (advanced_pooling) | ~90 min | GPU |
| **PHASE 1 Total** | **~3 hours** | **A100 (or ~4-5h on MPS)** |
| 2a-2j (CPU experiments) | ~30 min | CPU |
| **FULL PIPELINE** | **~4 hours** | **A100 (or ~5-6h on MPS)** |

---

## Post-Run Checklist

After `./run_all.sh` completes:

- [ ] Both papers compiled without errors
  ```bash
  ls -lh paper_0_confounding/paper_0.pdf paper_1_mechanistic/paper_1.pdf
  ```

- [ ] All figures exist
  ```bash
  ls ../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures/*.pdf | wc -l
  # Should be ≥ 15 figures
  ```

- [ ] Log files show no errors
  ```bash
  grep -i "error\|failed" logs/*.log
  # Should return nothing
  ```

- [ ] Papers are ≤ 4 pages (+ appendix)
  - Use `pdfinfo` or view in PDF reader to confirm page count

- [ ] Ready for submission on ARR
  ```bash
  cd ../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/
  # Submit paper_0.pdf and paper_1.pdf to ARR
  ```

---

## Command Reference

### Most Common

```bash
# Full pipeline
./run_all.sh

# Just CPU experiments (use pre-computed Phase 1)
./run_all.sh --no-gpu

# Just GPU experiments
./run_all.sh --gpu-only

# Individual experiment
bash scripts/patch_length_analysis.sh
bash scripts/causal_patching.sh
bash scripts/direction_transfer_sven.sh
bash scripts/run_global_baselines.sh
```

### Debugging

```bash
# Activate environment
conda activate sae

# Check single notebook
python sae_java_bug/sparse_autoencoders/notebooks/patch_length_analysis.py

# View logs
tail -100 logs/2c_cross_layer_direction_probe.log

# Recompile papers
cd paper_0_confounding && pdflatex -interaction=nonstopmode paper_0.tex
cd ../paper_1_mechanistic && pdflatex -interaction=nonstopmode paper_1.tex
```

---

## Questions?

Refer to:
1. **This file** — Setup and execution
2. **`DATASET_ASSESSMENT.md`** — Dataset choices and validation
3. **Paper sections** — Experiment rationale and results
4. **Individual notebook docstrings** — Experiment details

---

**Last Updated**: April 25, 2026  
**Status**: ✅ Ready for VM execution  
**EMNLP Deadline**: May 25, 2026
