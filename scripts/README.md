# Experiment Scripts

Each script runs one experiment end-to-end.
All scripts resolve paths relative to their own location, so they can be called from any directory.

All Python scripts run inside the `sae` conda environment via `conda run -n sae`.

---

## Data collection (requires GPU / VM)

| Script | Description |
|--------|-------------|
| `collect_per_token_sae.sh` | Collect per-token SAE feature activations for all 8 layers. Edit `LAYERS=(...)` to select which layers. Outputs `artifacts/activations/per_token_sae_l{L}.jsonl`. Must run on a machine with the Qwen2.5-7B model and SAE weights. |

---

## Probing — does the model encode vulnerability?

| Script | Description | Key output |
|--------|-------------|------------|
| `mean_pool_probe.sh` | Mean-token pooling probe across all 8 layers. Compares AUROC of mean-token vs last-token pooling. Main evidence that vulnerability signal is diffuse across positions. | Appendix L table + figure |
| `within_language_probe.sh` | Within-language (C / PHP / JS) probe. Controls for programming-language confound by running the binary probe within each language stratum separately. | Appendix N figure |
| `within_language_mean_pool_probe.sh` | Same as above but with mean-token pooling. Checks if the AUROC gain survives within a single language. | Appendix Q table |
| `nonlinear_probe.sh` | Compares linear (LogReg) vs nonlinear (MLP, Random Forest) probes at all layers. Rules out the possibility that near-chance AUROC is an artefact of linear probing. | Appendix M figure |
| `length_controlled_probe.sh` | Length-residualised and length-stratified probes. Controls for token-count differences between secure and vulnerable code. | Appendix figure |
| `advanced_pooling_probe.sh` | Compares four pooling strategies: last-token, mean-token, attention-weighted, and diff-restricted (tokens on changed lines). | Appendix P figure |
| `mean_pool_sae_probe.sh` | Compares 4 pooling × representation combos at L11: last/mean-token × raw/SAE features. | Appendix table |
| `cross_layer_direction_probe.sh` | Computes cosine similarity between the vulnerability direction `d^L` across all layer pairs. Shows whether the direction rotates or stays stable with depth. | Cosine heatmap figure |

---

## SAE feature asymmetry — negative-space encoding evidence

| Script | Description | Key output |
|--------|-------------|------------|
| `paired_suppression_test.sh` | Two analyses at L11: (1) within-pair total activation comparison; (2) activation magnitude scatter — 95.1% of secure-enriched features have higher mean activation on secure code. | `fig_paired_suppression.pdf`, Appendix figure |
| `magnitude_asymmetry_crosslayer.sh` | Repeats the magnitude asymmetry analysis across all 8 standard SAE layers (from `TO_UPload/` JONSLs). | Appendix S table + 2×4 scatter grid |
| `feature_asymmetry_crosslayer.sh` | Replicates the Δf feature-count asymmetry (e.g. 3.65× at L11) at L0 and L27 using standard SAE activations. | `fig_feature_asymmetry_crosslayer.pdf` |

---

## Positional analyses — where in the sequence is the signal?

| Script | Description | Key output |
|--------|-------------|------------|
| `position_stratified_probe.sh` | Mean SAE feature activation as a function of normalised token position (0→1). Shows the signal is distributed, not last-token only. | Appendix O figure |
| `positional_probe_b.sh` | Drops the first position bin and checks if discriminative signal persists. Uses `positional_profiles_raw.jsonl` (no GPU needed). | Appendix figure |
| `token_feature_viz.sh` | Per-token coloured heatmaps for selected SAE features. Edit `--features` to choose which features to visualise. | `token_viz/figures/` PDFs |
| `token_trajectory_3d.sh` | Per-token residual-stream trajectory in vulnerability-direction PCA space (x = `d^L`, y/z = top orthogonal PCs). | 3-D trajectory PDF |
| `token_pca_3d.sh` | 3-D PCA trajectory of per-token SAE activations coloured by position. Requires JSONL from `collect_per_token_sae.sh`. Edit `--layers` to match collected files. | `token_pca_3d/token_pca_3d_l{L}.pdf` + `.html` |

---

## Figure generation (post-hoc, no new computation)

| Script | Description |
|--------|-------------|
| `generate_ablation_figures.sh` | Regenerates publication ablation figures: CWE-language co-occurrence, ΔAUC heatmap, L11 bar chart. |
| `generate_advanced_pooling_figure.sh` | Regenerates `fig_advanced_pooling_comparison.pdf` from a saved `probe_results.json`. |

---

## Other

| Script | Description |
|--------|-------------|
| `hypothesis.sh` | Runs `feature_hypothesis.py` for all 7 non-L11 run IDs to generate LLM hypotheses for top SAE features. Requires AWS credentials (`--region us-east-2`). |
| `run_sae_exploration.sh` | Early exploration script for SAE feature analysis. |
