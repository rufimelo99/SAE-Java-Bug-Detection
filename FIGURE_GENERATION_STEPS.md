# Figure Generation Steps for C-Focused Paper

## Current Status
- ✅ 6 figures already exist (from prior runs)
- ⏳ 3 figures need generation from existing activations

## Existing Figures (Ready to use)
```
✅ fig_crosslayer_probe_auc.pdf          → cross_layer_direction_probe.py (C-filtered)
✅ fig_patch_length_by_cwe.pdf           → patch_length_analysis.py
✅ fig_direction_cosine_sim.pdf          → cross_layer_direction_probe.py (C-filtered)
✅ fig_alignment_trajectory.pdf          → cross_layer_direction_probe.py (C-filtered)
✅ fig_causal_patching.pdf               → causal_patching.py
✅ fig_directional_readout_comparison.pdf → directional_readout_probe.py
```

## Figures to Generate

### 1. Feature Direction Loading (fig_feature_direction_loading.pdf)
**Purpose**: Show which SAE features load onto vulnerability direction  
**Script**: `feature_direction_loading_c_only.py` (created)  
**Command**:
```bash
conda activate sae
cd /Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection
python sae_java_bug/sparse_autoencoders/notebooks/feature_direction_loading_c_only.py
```

### 2. Direction Transfer SVEN (fig_direction_transfer_sven.pdf)
**Purpose**: Validate direction generalizes across datasets  
**Script**: `direction_transfer_sven.py` (already C-compatible)  
**Command**:
```bash
python sae_java_bug/sparse_autoencoders/notebooks/direction_transfer_sven.py
```
*Note*: SVEN is C-only dataset, so no filtering needed

### 3. Generation Steering (fig_generation_steering.pdf)
**Purpose**: Show AUROC shift with steering strength  
**Script**: `generation_steering.py` (check if needs C-only filtering)  
**Command**:
```bash
python sae_java_bug/sparse_autoencoders/notebooks/generation_steering.py
```

## Quick Run All

```bash
conda activate sae
cd /Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection

# Run the three missing figure generators
python sae_java_bug/sparse_autoencoders/notebooks/feature_direction_loading_c_only.py
python sae_java_bug/sparse_autoencoders/notebooks/direction_transfer_sven.py
python sae_java_bug/sparse_autoencoders/notebooks/generation_steering.py

# Verify all figures exist
ls -1 ../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures/fig_*.pdf | wc -l
# Should be ≥ 57 with our new ones
```

## After Figures Are Generated

1. Run the paper compilation:
```bash
cd ../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations
pdflatex -interaction=nonstopmode pre_print.tex
pdflatex -interaction=nonstopmode pre_print.tex  # Second pass for cross-refs
```

2. Check PDF compiles with all figures:
```bash
ls -lh pre_print.pdf
# Should be > 1.5 MB with embedded figures
```

3. Verify all paper references resolve (no "???" in PDF)

## Notes

- All activations are pre-computed; these scripts just generate visualizations
- Cross-layer analysis already has C filtering built-in
- Created C-only version of feature loading analysis
- SVEN dataset is C-only by definition
- Generation steering may need verification for C-only focus

## Timeline

- Figure generation: ~10-15 minutes
- Paper recompilation: ~5 minutes
- Total: ~20 minutes to ready state
