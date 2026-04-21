# Mechanistic Interpretability of Code LLMs: Vulnerability Encoding Analysis

**Paper**: "What Code LLMs Know: Functional Identity Without Security Status" (preprint)

---

## Overview

This repository contains the experimental pipeline for analyzing how code LLMs encode vulnerability information. The core finding: LLMs strongly represent **what code does** (functional identity) but weakly and diffusely represent **whether code is safe** (security status). Vulnerability information is present in the residual stream but distributed across token positions, inaccessible to standard last-token readout.

### Key Empirical Results

- **Dataset**: 2,493 paired vulnerable/secure samples from DeltaSecommits (25 CWEs, 14 languages)
- **Model**: Qwen2.5-7B-Instruct with Sparse Autoencoders (16,384 features, trained on vulnerable code)
- **CWE-type probing** (within-language): median AUROC 0.877–0.992 (strong functional encoding)
- **Binary vulnerability probing** (last-token): AUROC 0.447–0.538 (near chance)
- **Mean-token pooling**: AUROC 0.600–0.647 (+0.13–0.16 gap over last-token)
- **Cross-layer vulnerability direction**: 87–88% per-pair alignment, cosine similarity 0.986–0.999 (L3–L23)
- **Activation patching**: body patching shifts representation toward secure; last-token patching has no effect

---

## Experiment Structure

### Main Paper Results (Sections 3–5)

#### **Section 3: What the Model Knows (Ablation)**

**3.1 CWE Family × Language Confound**
- Tests whether model classifies CWE families (e.g., memory-corruption vs. injection)
- Identifies Feature 1797 as a language detector, not vulnerability detector
- **Code**: `sae_java_bug/sparse_autoencoders/notebooks/token_feature_viz.py`, `per_cwe_analysis.py`
- **Output**: Language-CWE co-occurrence heatmaps, Feature 1797 token-level activations

**3.2 Within-Language Pairwise CWE Probes**
- All pairwise CWE-type classifications within single languages (C, PHP, C++, Python, JavaScript)
- Uses mean-token pooling on per-pair delta vectors
- **Code**: 
  - `plot_within_language_cwe_pairwise.py` — C and PHP heatmaps
  - `plot_cwe79_vs_cwe89.py` — detailed CWE-79 vs CWE-89 analysis (C, PHP, JavaScript)
  - `plot_multi_dataset_cwe_heatmap.py` — C++, JavaScript heatmaps
- **Output**: AUROC heatmaps by CWE pair and layer

**3.3 File-Type Residualisation and Within-Language Control**
- Ridge regression to remove language signal from delta vectors
- Re-runs CWE-family and CWE-type probes on residualised features
- **Code**: `length_controlled_probe.py`, `lang_persistence_verify.py`
- **Output**: Delta-AUC tables showing language confound quantification

---

#### **Section 4: How Vulnerability Is Encoded (Main Results)**

**4.1 Mean-Token vs. Last-Token Pooling**
- Binary vulnerability probing (vulnerable vs. secure) at all 8 layers
- Compares last-token pooling (near chance) vs. mean-token pooling (above chance)
- **Code**:
  - `mean_pool_probe.py` — binary probing on raw residual stream
  - `mean_pool_sae_probe.py` — binary probing on SAE features
  - `plot_vuln_secure_by_layer.py` — visualization across layers
- **Output**: AUROC curves with 95% bootstrap CIs, mean vs. last-token comparison plot

**4.2 Why Mean-Token Pooling Is Optimal: Attention-Weighted Pooling**
- Tests attention-weighted pooling (expected to interpolate between strategies)
- Shows it performs *below* chance — final token attends to defensive-construct-rich positions
- **Code**: 
  - `advanced_pooling_probe.py` — computes attention-weighted, uniform, max, min pooling strategies
  - `generate_advanced_pooling_figure.py` — visualization
- **Output**: AUROC comparison across pooling strategies

**4.3 Cross-Layer Vulnerability Direction Geometry**
- Computes mean vulnerability direction $\mathbf{d}^L = \text{normalize}(\bar{v}^L - \bar{s}^L)$ at each layer
- Measures per-pair alignment: % of pairs where vulnerable representation lies further in direction
- **Code**:
  - `cross_layer_direction_probe.py` — direction computation and per-layer alignment
  - `directional_readout_probe.py` — directional readout baseline (project onto direction, no learning)
  - `visualize_paired_direction.py` — 3D visualization of representative pairs
- **Output**: Direction alignment statistics, 3D pair scatter plots

**4.4 SAE Feature Analysis**
- Analyzes which features activate on vulnerable vs. secure code
- Quantifies feature suppression asymmetry (95% of secure-enriched features stronger on secure)
- Position-stratified analysis confirms signal spans full sequence
- **Code**:
  - `mean_pool_sae_probe.py` — feature activation frequency and magnitude by class
  - `feature_asymmetry_crosslayer.py` — per-layer asymmetry statistics
  - `magnitude_asymmetry_crosslayer.py` — activation magnitude comparison
  - `paired_suppression_test.py` — statistical tests for feature suppression
  - `position_stratified_probe.py`, `positional_probe_b.py` — token-level signal analysis
- **Output**: Feature activation heatmaps, suppression asymmetry tables, position-stratified AUROC

**4.5 Within-Language Last-Token Probing Failure**
- Binary vulnerability probing restricted to single languages (C, PHP, JavaScript)
- Demonstrates last-token AUROC falls at or below chance in all strata
- **Code**:
  - `cwe79_vs_cwe89_php_probe.py` — detailed within-language analysis for PHP
  - `within_language_mean_pool_probe.py` — mean-token comparison within C
  - `plot_vuln_secure_by_layer.py` — layer-by-layer breakdown
- **Output**: Per-language AUROC tables, stratified comparisons

---

#### **Causal Analysis: Activation Patching**

- Replaces residual stream activations from vulnerable code with secure code activations
- Tests which positions are causally necessary for vulnerability representation
- **Code**:
  - `causal_patching.py` — residual stream substitution analysis
  - `causal_patching_behavioral.py` — measures downstream behavioral effects (log-probability shifts)
  - `causal_steering.py` — directional steering experiments
- **Output**: Layer-wise patching effect sizes, behavioral consequence tables

---

### Appendix Experiments

| Appendix | Topic | Code Files |
|----------|-------|-----------|
| A | CWE Projection & Language-CWE Co-occurrence | `per_cwe_analysis.py`, `token_feature_viz.py` |
| B | Length-Controlled Re-analysis | `length_controlled_probe.py` |
| C | Within-Language AUROC Across Layers | `cwe79_vs_cwe89_php_probe.py`, `within_language_mean_pool_probe.py` |
| D | SAE Architecture & Training Config | Config files (Appendix D.1–D.4) |
| E | Mean-Token Pooling Probe | `mean_pool_probe.py`, `mean_pool_sae_probe.py` |
| F | Position-Stratified Feature Analysis | `position_stratified_probe.py`, `positional_probe_b.py` |
| G | Within-Language Mean-Token AUROC | `plot_vuln_secure_by_layer.py`, `within_language_mean_pool_probe.py` |
| H | Cross-Layer Magnitude Asymmetry | `magnitude_asymmetry_crosslayer.py`, `feature_asymmetry_crosslayer.py` |
| I | Causal Interventions: Activation Patching | `causal_patching.py`, `causal_patching_behavioral.py` |
| J | Nonlinear Probe Comparison | `nonlinear_probe.py`, `nonlinear_probe_compute.py`, `nonlinear_probe_plot.py` |
| K | Advanced Pooling Strategies | `advanced_pooling_probe.py`, `generate_advanced_pooling_figure.py` |
| L | Cross-Layer Direction Geometry | `cross_layer_direction_probe.py`, `directional_readout_probe.py` |
| M | CWE-Specific Feature Enrichment | (part of `per_cwe_analysis.py`, `mean_pool_sae_probe.py`) |
| N | Token-Level Feature Analysis | `token_feature_viz.py`, `token_pca_3d.py`, `token_trajectory_3d.py` |
| O | Language Persistence After Residualisation | `lang_persistence_verify.py` |
| P | Per-Token PCA Trajectories | `token_pca_3d.py`, `token_trajectory_3d.py` |
| Q | PCA Dimensionality Sensitivity | `pca_sensitivity_probe.py` |
| R | Token Scaling | `collect_per_token_sae.py`, `convert_topk_to_pt.py` |
| S | Interpretation Pipeline | (Claude Opus labeling of top/bottom features) |
| T | Cross-Layer CWE-Family Probe | `per_cwe_analysis.py` |
| U | Multi-Model Probing Replication | `multi_model_probing.py` (CodeLlama-7B, StarCoder2-7B) |
| V | Within-Language CWE Pairwise (C++, JS) | `plot_multi_dataset_cwe_heatmap.py`, `plot_cwe79_vs_cwe89.py` |
| W | Multi-Dataset Replication | `multi_dataset_comparison.py`, `plot_multi_dataset_cwe_heatmap.py` (SVEN, PreciseBugs) |

---

## Repository Structure

```
sae_java_bug/
├── sparse_autoencoders/
│   ├── notebooks/                    # All experiment scripts
│   │   ├── Pooling & Probing
│   │   │   ├── mean_pool_probe.py
│   │   │   ├── mean_pool_sae_probe.py
│   │   │   ├── advanced_pooling_probe.py
│   │   │   ├── nonlinear_probe.py
│   │   │   ├── position_stratified_probe.py
│   │   │   ├── positional_probe_b.py
│   │   │   └── pca_sensitivity_probe.py
│   │   ├── CWE Analysis
│   │   │   ├── plot_within_language_cwe_pairwise.py
│   │   │   ├── plot_cwe79_vs_cwe89.py
│   │   │   ├── plot_multi_dataset_cwe_heatmap.py
│   │   │   └── per_cwe_analysis.py (also in evaluation/)
│   │   ├── Direction & Geometry
│   │   │   ├── cross_layer_direction_probe.py
│   │   │   ├── directional_readout_probe.py
│   │   │   └── visualize_paired_direction.py
│   │   ├── Features & Activation
│   │   │   ├── feature_asymmetry_crosslayer.py
│   │   │   ├── magnitude_asymmetry_crosslayer.py
│   │   │   ├── paired_suppression_test.py
│   │   │   ├── token_feature_viz.py
│   │   │   ├── token_pca_3d.py
│   │   │   └── token_trajectory_3d.py
│   │   ├── Causal Interventions
│   │   │   ├── causal_patching.py
│   │   │   ├── causal_patching_behavioral.py
│   │   │   └── causal_steering.py
│   │   ├── Control & Replication
│   │   │   ├── length_controlled_probe.py
│   │   │   ├── lang_persistence_verify.py
│   │   │   ├── within_language_baseline.py
│   │   │   └── within_language_mean_pool_probe.py
│   │   ├── Multi-Model & Multi-Dataset
│   │   │   ├── multi_model_probing.py
│   │   │   └── multi_dataset_comparison.py
│   │   ├── Utilities
│   │   │   ├── collect_per_token_sae.py
│   │   │   ├── convert_topk_to_pt.py
│   │   │   ├── generate_advanced_pooling_figure.py
│   │   │   ├── generate_ablation_figures.py
│   │   │   ├── plot_vuln_secure_by_layer.py
│   │   │   └── topk_feature_discriminability.py
│   ├── utils.py                    # SAE utilities (indexing, activation handling)
│   ├── schemas.py                  # Data structures
│   └── feature_semantics.py        # Feature interpretation helpers
├── evaluation/
│   ├── multi_model_probing.py      # Replication on CodeLlama, StarCoder2
│   ├── per_cwe_analysis.py         # CWE family & pairwise classification
│   ├── activation_extractor.py     # SAE activation collection
│   └── [other analysis modules]
├── logger.py                       # Logging utilities
├── __init__.py
└── sparse_autoencoders.egg-info/

scripts/                             # Slurm/shell job submission
├── mean_pool_probe.sh
├── advanced_pooling_probe.sh
├── cross_layer_direction_probe.sh
├── causal_patching.sh
├── [41 additional experiment scripts]
└── README.md                        # Script submission guide
```

---

## Installation & Setup

### Requirements
- Python 3.10+
- PyTorch 2.0+
- SAE dictionaries trained on Qwen2.5-7B-Instruct (vulnerability code only)
- DeltaSecommits dataset (2,493 paired samples across 25 CWEs, 14 languages)
- Activation cache from Qwen2.5-7B-Instruct on DeltaSecommits

### Installation

```bash
conda create -n lm-vulnerability-encoding python=3.10
conda activate lm-vulnerability-encoding
pip install -e /path/to/SAE-Java-Bug-Detection
```

### Data Preparation

1. **Activation Extraction**: Collect residual stream activations from Qwen2.5-7B-Instruct at layers [0, 3, 7, 11, 15, 19, 23, 27]
   ```bash
   # (Activation extraction script not included in this repo; uses separate inference pipeline)
   ```

2. **SAE Training**: Train 8 sparse autoencoders (one per layer) on vulnerable-code activations
   ```bash
   # (SAE training not included; uses jblakely/sae package)
   ```

3. **Dataset Location**: Place DeltaSecommits, SVEN, and PreciseBugs datasets in a standard location accessible to scripts

---

## Running Experiments

### Local Execution (Single Experiment)

```bash
cd sae_java_bug/sparse_autoencoders/notebooks/
python mean_pool_probe.py       # Binary vulnerability probe (vulnerable vs. secure)
python advanced_pooling_probe.py  # Compare pooling strategies
python cross_layer_direction_probe.py  # Compute vulnerability direction at each layer
```

### Batch Submission (Slurm Cluster)

```bash
cd scripts/
sbatch mean_pool_probe.sh
sbatch advanced_pooling_probe.sh
sbatch cross_layer_direction_probe.sh
# ... etc for all experiments
```

Or submit all at once:
```bash
bash run_all.sh
```

### Output

Experiment outputs are saved to `results/` subdirectories with structure:
```
results/
├── mean_pool_probe/
│   ├── auroc_by_layer.csv
│   ├── bootstrap_ci.csv
│   └── figure_mean_vs_last_token.pdf
├── advanced_pooling_probe/
│   ├── pooling_comparison.csv
│   └── figure_pooling_strategies.pdf
└── [one subdirectory per experiment]
```

---

## Key Findings at a Glance

### 1. **Functional Encoding is Strong**
- CWE-type pairwise probing (within-language): median AUROC 0.877–0.992
- Model distinguishes code functional purpose with high fidelity

### 2. **Security Encoding is Weak**
- Binary vulnerability probing (last-token): AUROC 0.447–0.538 (near chance)
- Single-token readout cannot access vulnerability information

### 3. **Signal is Distributed**
- Mean-token pooling: AUROC 0.600–0.647 (+0.13–0.16 over last-token)
- Vulnerability information is spread across sequence, not concentrated at final position

### 4. **Geometry is Coherent but Not Separable**
- 87–88% of vulnerable–secure pairs align with vulnerability direction (L3–L23)
- But population-level AUROC remains ~0.50 (pair-level geometry ≠ population separability)

### 5. **Activation Patching Confirms Causal Role**
- Replacing sequence body with secure activations: **+8.5 log-prob to secure sequence**
- Replacing last token only: **no effect** (±1% of body effect)
- Behavioral zero-crossing at L27: security-relevant token logits drop from +4.1 to exactly 0.00

### 6. **Feature-Level Explanation**
- 12,588 secure-enriched features vs. 3,445 vuln-enriched features
- Vulnerability = reduced activation of defensive-construct features
- Asymmetry holds across all 8 layers (69–95% per-layer magnitude asymmetry)

---

## Reproducibility & Verification

### Sensitivity Analyses (Appendix)

All methodological choices are validated:

- **PCA Dimensionality** (Appendix Q): Full-dimensionality falls below chance; PCA-50 and PCA-100 agree to 0.01 AUROC
- **Cross-Validation Fold Grouping** (Section 3): Pair-stratified CV for directional analysis; no cross-pair leakage in binary probes
- **Length Control** (Appendix B): Ridge regression to isolate language signal; residual results confirm finding holds
- **Nonlinear Probes** (Appendix J): MLP and RBF-SVM ceilings match or fall below linear probe; gain from pooling strategy, not classifier
- **Multi-Model Replication** (Appendix U): Core finding replicates on CodeLlama-7B and StarCoder2-7B without SAEs
- **Multi-Dataset Replication** (Appendix W): SVEN and PreciseBugs (C only) confirm asymmetry holds across datasets

### Checklists for Paper Reproducibility

- [ ] Qwen2.5-7B-Instruct SAE weights (layer 0, 3, 7, 11, 15, 19, 23, 27)
- [ ] DeltaSecommits dataset (2,493 paired samples)
- [ ] SVEN and PreciseBugs datasets (validation only)
- [ ] Activation cache or re-extraction pipeline
- [ ] All experiment scripts in `notebooks/` directory
- [ ] Test run on small subset to verify pipeline integrity

---

## Contact

For questions about the experiments or data, contact: **Rui Melo** (rufimelo99@gmail.com)

---

## Acknowledgments

This work uses:
- **DeltaSecommits** dataset (OSV + NVD derived)
- **Qwen2.5-7B-Instruct** model (Alibaba)
- **Sparse Autoencoders** (jblakely/sae package)
- **CodeLlama** and **StarCoder2** for replication (Meta, BigCode)
