# Comprehensive Analysis Pipeline

This pipeline orchestrates all stages of vulnerability representation analysis, from direction geometry to causal steering validation.

## Quick Start

```bash
# Run complete pipeline (all stages)
python scripts/pipeline_full_analysis.py --stage all

# Run specific stages
python scripts/pipeline_full_analysis.py --stage geometry  # Direction geometry only
python scripts/pipeline_full_analysis.py --stage steering  # Steering experiments only
python scripts/pipeline_full_analysis.py --stage figures   # Figure generation only

# Run with custom parameters
python scripts/pipeline_full_analysis.py --stage steering --models qwen,codellama --n_samples 100
```

## Pipeline Stages

### Stage 1: Direction Geometry Analysis (5-10 min)
Analyzes how vulnerability signals are encoded across layers in all models.

**Scripts:**
- `regenerate_direction_alignment_combined.py` - Global alignment across models
- `regenerate_direction_alignment_families_combined.py` - Per-family alignment
- `regenerate_multimodel_alignment_styled.py` - Multi-model alignment comparison
- `regenerate_multimodel_magnitude_comparison.py` - Signal magnitude across models
- `regenerate_paired_distances_styled.py` - Paired distance analysis
- `regenerate_direction_transfer_styled.py` - Cross-family transfer rates
- `regenerate_multimodel_stability_styled.py` - Direction stability across layers

**Output:** Figures showing alignment, magnitude, stability, and transfer patterns

**Data Required:** Pre-computed direction geometry JSON files in `results/raw_data/`

### Stage 2: Direction Steering Experiments (2-4 hours on GPU)
Runs causal interventions to validate that vulnerability directions mechanistically affect model behavior.

**Scripts:**
- `run_steering_pipeline.py` - Computes directions and runs steering experiments
- `regenerate_steering_plots.py` - Generates publication-quality plots

**Process:**
1. Loads 100 vulnerable-secure code pairs from DeltaSecommits
2. Splits into train (80) and test (20) sets
3. For each model and layer:
   - Computes direction from training data
   - Runs steering at alpha values: [-20, -10, -5, 0, 5, 10, 20]
   - Measures preference shift: log P(secure) - log P(vulnerable)
4. Saves results to JSON and generates visualizations

**Output:**
- `fig_causal_summary_{model}_100samples.pdf` - Per-model steering effects (2-panel: curves + magnitudes)
- `fig_steering_multimodel_comparison_100samples.pdf` - All models on single plot

**Hardware:** GPU strongly recommended (~30 min per model on A100, 2-4 hours on CPU)

### Stage 3: Other Analyses (10-15 min)
Additional analysis including CWE-type probing.

**Scripts:**
- `regenerate_cwe_pairwise_probe_styled.py` - CWE pairwise separation analysis

**Output:** CWE separation heatmaps for all models and datasets

## File Organization

```
SAE-Java-Bug-Detection/
├── scripts/
│   ├── pipeline_full_analysis.py          # Main pipeline coordinator
│   ├── run_steering_pipeline.py            # Steering experiment runner
│   ├── regenerate_steering_plots.py        # Steering visualization
│   ├── regenerate_direction_*.py           # Geometry analysis scripts
│   └── ... (other analysis scripts)
├── results/
│   ├── raw_data/                          # Input JSON data
│   │   ├── qwen-7b_direction_geometry.json
│   │   ├── codellama-7b_direction_geometry.json
│   │   ├── starcoder2-7b_direction_geometry.json
│   │   ├── steering_results_qwen_100samples.json
│   │   ├── steering_results_codellama_100samples.json
│   │   └── steering_results_starcoder2_100samples.json
│   └── ... (generated figures)
├── results_real_preference_steering_100samples.json  # Qwen steering data
└── PIPELINE_README.md                     # This file

On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/
├── figures/                               # Output figures
│   ├── fig_direction_alignment_combined.pdf
│   ├── fig_causal_summary_qwen_100samples.pdf
│   ├── fig_causal_summary_codellama_100samples.pdf
│   ├── fig_causal_summary_starcoder2_100samples.pdf
│   └── ... (all other paper figures)
└── sections/
    ├── results.tex                        # Main results section
    └── appendix/
        ├── multi_model_probing.tex        # Multi-model appendix
        └── ... (other appendix sections)
```

## Steering Analysis Details

### What is Direction Steering?

Direction steering tests whether the vulnerability direction mechanistically influences model decisions:

1. **Compute direction**: For test pairs, compute d = mean(vulnerable) - mean(secure)
2. **Apply steering**: Perturb activations at layer L by adding α·d
3. **Measure effect**: Compare log-likelihood of secure vs. vulnerable code
4. **Quantify causal impact**: Preference shift = ΔP(secure)

### Interpretation

- **Negative α (amplify)**: Strengthens vulnerability signal → model prefers secure code ✓
- **Positive α (suppress)**: Weakens vulnerability signal → model less confident ✓
- **Layer 3 > Layer 7 > Layer 23**: Effect decays through network, showing information flow

## Data Generation

### Required Input Data

For steering experiments, you need:
- `sae_java_bug/artifacts/activations/raw_activations/vulnerable_code_qwen_coder_standard_16384_raw/`
  - Contains base64-encoded vulnerable/secure code pairs

### Existing Results

We have pre-computed results for:
- ✓ Qwen-7B (100 samples, steering experiments done)
- ✓ CodeLlama-7B (geometry analysis, steering experiments can be run)
- ✓ StarCoder2-7B (geometry analysis, steering experiments can be run)

## Running Full Analysis from Scratch

```bash
# 1. Ensure data is available
#    (check results/raw_data/ for *.json files)

# 2. Run geometry analysis (fast)
python scripts/pipeline_full_analysis.py --stage geometry

# 3. Run steering (slow - requires GPU)
python scripts/pipeline_full_analysis.py --stage steering --models qwen,codellama,starcoder2

# 4. Generate all figures
python scripts/pipeline_full_analysis.py --stage figures

# Or all at once:
python scripts/pipeline_full_analysis.py --stage all
```

## Monitoring and Troubleshooting

### Check Progress
Each stage logs to stdout. Look for:
- `✓ {name}` = successful
- `✗ {name}` = failed
- Partial output = intermediate results available

### Common Issues

1. **"Results not found" for steering**
   - Run steering experiments first: `python scripts/run_steering_pipeline.py`

2. **"Dataset file not found"**
   - Steering needs raw activation artifacts
   - Check path in `run_steering_pipeline.py`

3. **Memory errors on GPU**
   - Reduce batch size or use smaller models
   - Or run on CPU (much slower)

4. **Missing input data**
   - Geometry scripts need `results/raw_data/*.json`
   - Ensure data is pre-computed

## Output

All figures are saved to:
```
On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures/
```

PDF filenames match figure references in paper:
- `fig_direction_alignment_combined.pdf`
- `fig_causal_summary_qwen_100samples.pdf`
- `fig_causal_summary_codellama_100samples.pdf`
- `fig_causal_summary_starcoder2_100samples.pdf`
- etc.

## Citation

If using this pipeline, please cite the paper:
```
[Citation info here]
```
