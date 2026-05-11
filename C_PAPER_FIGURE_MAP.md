# C-Focused Paper: Figure Generation Map

## Summary
The activations are already computed. We just need to run existing notebooks to generate the 9 figures required for the paper.

## Required Figures

| Figure | Purpose | Notebook | Status |
|--------|---------|----------|--------|
| `fig_crosslayer_probe_auc.pdf` | AUROC by layer (Section 4) | `cross_layer_direction_probe.py` | ✅ Exists |
| `fig_patch_length_by_cwe.pdf` | Patch structure asymmetry (Section 4) | `patch_length_analysis.py` | ✅ Exists |
| `fig_direction_cosine_sim.pdf` | Direction stability (Section 5) | `cross_layer_direction_probe.py` | ✅ Exists |
| `fig_alignment_trajectory.pdf` | Per-pair alignment (Section 5) | `cross_layer_direction_probe.py` | ✅ Exists |
| `fig_causal_patching.pdf` | Activation patching results (Methodology) | `causal_patching.py` | ✅ Exists |
| `fig_directional_readout_comparison.pdf` | Directional readout analysis (Methodology) | `directional_readout_probe.py` | ✅ Exists |
| `fig_feature_direction_loading.pdf` | Feature-to-direction correlations (Section 5) | `feature_direction_loading.py` | ⏳ May need C-only filter |
| `fig_direction_transfer_sven.pdf` | SVEN cross-dataset transfer (Section 5) | `direction_transfer_sven.py` | ⏳ May need C-only filter |
| `fig_generation_steering.pdf` | Generation steering results (Section 6) | `generation_steering.py` | ⏳ May need C-only filter |

## Quick Start

```bash
conda activate sae
cd /Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection

# Check which figures already exist
ls -1 ../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures/ | grep "fig_"

# Run specific notebooks to generate missing figures
python sae_java_bug/sparse_autoencoders/notebooks/feature_direction_loading.py
python sae_java_bug/sparse_autoencoders/notebooks/direction_transfer_sven.py
python sae_java_bug/sparse_autoencoders/notebooks/generation_steering.py
```

## C-Only Considerations

Some notebooks may run on multi-language data by default. For C-only analysis:
- Check if notebooks have a `C_EXTS` or language filter
- May need to add filtering to focus on C code only (1,827 pairs from 2,493 total)
- Look for flags like `--language C` or modify data loading

## File Locations

- **Notebooks**: `/Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection/sae_java_bug/sparse_autoencoders/notebooks/`
- **Activations**: `/Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection/sae_java_bug/artifacts/activations/`
- **Output figures**: `/Users/rmelo/Documents/GitHub/On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures/`

## Next Steps

1. ✅ Paper structure complete (`pre_print.pdf`)
2. ⏳ Run figure generation notebooks (minimal changes needed)
3. ⏳ Verify all 9 figures exist in figures/
4. ⏳ Recompile paper with actual figures (figures currently use graceful placeholders)
5. ⏳ Submit to EMNLP (May 25, 2026)

## Notes

- Most notebooks are already written and should work as-is
- Figures will be auto-saved to the paper figures directory
- If a notebook is language-agnostic, it will generate multi-language results; C-only filtering may need to be added to the notebooks if the paper requires strict C-only analysis
