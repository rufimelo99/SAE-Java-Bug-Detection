# Quick Start: Steering Analysis Pipeline

## TL;DR - 3 Commands

```bash
# 1. Generate figures from all existing analyses (5 min)
python scripts/pipeline_full_analysis.py --stage geometry

# 2. Generate steering plots from existing Qwen data (5 min)
python scripts/regenerate_steering_plots_from_existing.py --models qwen

# 3. Run steering experiments for all models [SLOW - skip if you just want figures]
python scripts/pipeline_full_analysis.py --stage all
```

## What Was Added

A production-grade **direction steering pipeline** that:

✓ **Validates causality** - Tests if vulnerability directions mechanistically affect model behavior  
✓ **Multi-model support** - Qwen-7B, CodeLlama-7B, StarCoder2-7B  
✓ **Publication-ready plots** - Two-panel steering effect figures  
✓ **Fully integrated** - Coordinates with existing geometry and CWE analysis  
✓ **Handles legacy data** - Works with existing Qwen steering experiments  

## The Steering Analysis

### What It Does

1. Loads 100 real vulnerable-secure code pairs
2. Computes vulnerability direction at each layer
3. Applies steering: manipulates direction strength (α) from -20 to +20
4. Measures effect: how much does preference shift?
5. Generates plots showing causal impact

### Why It Matters

> **Proof that vulnerability directions are real, not statistical artifacts.**
>
> Just because we can extract a direction doesn't mean it's mechanistically important. 
> Steering proves the model actually uses this signal in its decisions.

### What to Expect

**Good signs:**
- ✓ Preference increases monotonically with steering strength
- ✓ Early layers (L3) have stronger effects than late layers
- ✓ Effect pattern is consistent across models

**Example results (Qwen-7B):**
```
Layer 3:  effect = +0.126  [strongest]
Layer 7:  effect = +0.030
Layer 11: effect = +0.015
Layer 15: effect = +0.010
Layer 19: effect = +0.008
Layer 23: effect = +0.007  [weakest]
```

## Files Created

### Pipeline Scripts
```
scripts/
├── pipeline_full_analysis.py           # Main coordinator
├── run_steering_pipeline.py             # Experiment runner
├── regenerate_steering_plots.py         # Plot generator
└── regenerate_steering_plots_from_existing.py  # Legacy data support
```

### Documentation
```
├── PIPELINE_README.md                  # Full pipeline documentation
├── STEERING_ANALYSIS_GUIDE.md          # Steering methodology & usage
└── QUICK_START.md                      # This file
```

### Output Figures
```
On-the-Absence-of-Global-Anomalies.../figures/
├── fig_causal_summary_qwen_100samples.pdf
├── fig_causal_summary_codellama_100samples.pdf      [if run]
├── fig_causal_summary_starcoder2_100samples.pdf     [if run]
└── fig_steering_multimodel_comparison_100samples.pdf [if run]
```

## One-Time Setup

### Already Done ✓

- Qwen-7B steering results exist: `results_real_preference_steering_100samples.json`
- Can regenerate plots immediately
- All scripts created and tested

### To Add Other Models

Need to run steering experiments (slow, GPU recommended):

```bash
# Run for CodeLlama and StarCoder2 (2-4 hours on GPU)
python scripts/run_steering_pipeline.py --models codellama,starcoder2

# Then generate plots
python scripts/regenerate_steering_plots.py --models codellama,starcoder2
```

## Common Tasks

### Just Regenerate Qwen Plot
```bash
python scripts/regenerate_steering_plots_from_existing.py --models qwen
```
⏱️ 5 minutes

### Run All Geometry Analysis
```bash
python scripts/pipeline_full_analysis.py --stage geometry
```
⏱️ 10 minutes

### Full Pipeline (All Stages)
```bash
python scripts/pipeline_full_analysis.py --stage all
```
⏱️ 3+ hours on GPU, requires GPU memory

### Run Specific Models Only
```bash
python scripts/pipeline_full_analysis.py --stage steering --models qwen
python scripts/regenerate_steering_plots.py --models qwen
```

## In the Paper

### Results Section
The steering analysis appears in:
- Section 4 (Results) - "Causal validation: confirming the direction is mechanistically real"
- Fig 9 - Direction steering curves and effect magnitudes
- Discussion of layer-by-layer effects

### Supporting Materials
- Appendix - Full technical details of steering methodology
- Supplementary figures - Additional models and datasets (if run)

## Troubleshooting

**Q: Plot generation fails**  
A: Check if results exist:
```bash
ls results/raw_data/steering_results_*.json
```

**Q: Steering experiments are slow**  
A: Expected! GPU recommended, ~30 min per model on A100

**Q: "Dataset file not found"**  
A: Steering needs raw code pairs. May need to run geometry first or check data path.

**Q: Want to modify parameters**  
A: Edit constants in:
- `run_steering_pipeline.py`: `LAYERS_TO_TEST`, `ALPHA_VALUES`, `DEVICE`
- `regenerate_steering_plots.py`: color schemes, figure sizes

## Next Steps

1. **Verify it works:** Run one of the quick commands above
2. **Check output:** Look for PDFs in `figures/`
3. **Update paper:** Add steering plots to results/appendix
4. **Document:** Include command used in reproducibility statement

## Key Files to Know

| File | Purpose |
|------|---------|
| `pipeline_full_analysis.py` | Main entry point - run this |
| `PIPELINE_README.md` | Full documentation |
| `STEERING_ANALYSIS_GUIDE.md` | Detailed steering methodology |
| `run_steering_pipeline.py` | Compute steering experiments |
| `regenerate_steering_plots.py` | Create plots from results |
| `regenerate_steering_plots_from_existing.py` | Quick plot from legacy data |

---

**Questions?** Check the detailed guides:
- `PIPELINE_README.md` for architecture and stages
- `STEERING_ANALYSIS_GUIDE.md` for methodology details
- Script comments for implementation specifics
