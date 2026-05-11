# EMNLP Submission Verification Checklist

## Pre-Execution Checklist ✅

- [x] All shell scripts are executable
  ```bash
  ls -la scripts/*.sh | head -10
  # All should have -rwxr-xr-x permissions
  ```

- [x] run_all.sh has been updated with Paper 1 scripts (steps 2c-2e)
  ```bash
  grep "2c_cross_layer_direction_probe\|2d_causal_patching\|2e_directional_readout_probe" run_all.sh
  # Should show 3 matches
  ```

- [x] All new Python notebooks exist
  - [x] `sae_java_bug/sparse_autoencoders/notebooks/patch_length_analysis.py`
  - [x] `sae_java_bug/sparse_autoencoders/notebooks/generation_steering.py`
  - [x] `sae_java_bug/sparse_autoencoders/notebooks/direction_transfer_sven.py`
  - [x] `sae_java_bug/sparse_autoencoders/notebooks/run_global_baselines.py`
  - [x] `sae_java_bug/sparse_autoencoders/notebooks/feature_direction_loading.py`

- [x] All new shell scripts exist
  - [x] `scripts/patch_length_analysis.sh`
  - [x] `scripts/generation_steering.sh`
  - [x] `scripts/direction_transfer_sven.sh`
  - [x] `scripts/run_global_baselines.sh`
  - [x] `scripts/feature_direction_loading.sh`
  - [x] `scripts/causal_patching.sh`
  - [x] `scripts/directional_readout_probe.sh`

- [x] Paper 0 has been updated with patch length figure
  ```bash
  grep "fig_patch_length_by_cwe" paper_0_confounding/sections/03_patch_structure.tex
  # Should show figure reference
  ```

- [x] Both papers compile without errors
  ```bash
  cd paper_0_confounding && pdflatex paper_0.tex > /dev/null 2>&1 && echo "✓ Paper 0"
  cd ../paper_1_mechanistic && pdflatex paper_1.tex > /dev/null 2>&1 && echo "✓ Paper 1"
  ```

- [x] PDF files exist and are not empty
  ```bash
  ls -lh paper_0_confounding/paper_0.pdf paper_1_mechanistic/paper_1.pdf
  # Both should be > 100 KB
  ```

## Ready to Execute

Once you have:
1. Activated the `sae` conda environment
2. Verified GPU/MPS access (`nvidia-smi` or `python -c "import torch; print(torch.mps.is_available())"`)
3. Confirmed activation data exists (`ls code-security-probing/artifacts/activations/c_only/ | head`)

Then run:
```bash
./run_all.sh
```

## Post-Execution Checklist (Complete after running)

- [ ] run_all.sh completed without errors
  ```bash
  tail logs/*.log | grep -E "FAILED|Error|error"
  # Should return nothing
  ```

- [ ] All figures were generated
  ```bash
  ls ../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures/*.pdf | wc -l
  # Should be ≥ 15
  ```

- [ ] Key new figures exist
  - [ ] `fig_patch_length_by_cwe.pdf` (Paper 0 - patch length distributions)
  - [ ] `fig_global_anomaly_baselines.pdf` (Paper 0 - anomaly detection fails)
  - [ ] `fig_direction_transfer_sven.pdf` (Paper 1 appendix - SVEN validation)
  - [ ] `feature_direction_loading_results.json` (Paper 1 - SAE features)

- [ ] Both papers still compile after Phase 2
  ```bash
  cd paper_0_confounding && pdflatex paper_0.tex && echo "✓"
  cd ../paper_1_mechanistic && pdflatex paper_1.tex && echo "✓"
  ```

- [ ] Paper 0 is ≤ 4 pages (+ appendix)
  - [ ] Main content: 4 pages max
  - [ ] Appendix: separate section

- [ ] Paper 1 is ≤ 4 pages (+ appendix)
  - [ ] Main content: 4 pages max
  - [ ] Appendix: separate section

- [ ] All figure references in papers are valid
  - [ ] No "???" in PDF output
  - [ ] All `\ref{}` commands resolved

## Files for Final Submission

Ready to upload to ARR:

```
../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/
├── paper_0_confounding/paper_0.pdf        ← SUBMIT THIS
├── paper_1_mechanistic/paper_1.pdf        ← SUBMIT THIS
├── figures/                               ← All figures (for supplementary material)
│   ├── fig_patch_length_by_cwe.pdf
│   ├── fig_direction_cosine_sim.pdf
│   ├── fig_causal_patching.pdf
│   ├── fig_directional_readout_comparison.pdf
│   └── [11 more figures]
└── biblio.bib                            ← Reference file
```

## Timeline

- **Phase 1 (GPU)**: ~3-4 hours
  - `mean_pool_probe.py` (~30 min)
  - `position_stratified_probe.py` (~30 min)
  - `token_feature_viz.py` (~5 min)
  - `mean_pool_sae_probe.py` (~45 min)
  - `advanced_pooling_probe.py` (~90 min)

- **Phase 2 (CPU)**: ~30 min
  - 10 different CPU-based experiments
  - Generates all final figures
  - Papers compile to PDFs

- **Total**: ~4 hours (A100) or ~5-6 hours (MPS)

## Contact Information

For issues during execution, check:
1. `EMNLP_PIPELINE_GUIDE.md` — Comprehensive troubleshooting
2. `QUICK_START.txt` — Quick reference
3. Individual notebook docstrings — Experiment details
4. `logs/*.log` — Execution logs for debugging

---

**Status**: ✅ READY FOR IMMEDIATE VM EXECUTION  
**Deadline**: May 25, 2026 (ARR submission)
