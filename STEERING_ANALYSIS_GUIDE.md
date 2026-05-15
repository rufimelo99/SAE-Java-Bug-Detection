# Steering Analysis Integration Guide

## Overview

The direction steering analysis has been integrated as a key component of the full analysis pipeline. This guide explains:

1. What steering analysis is and why it matters
2. How to run steering experiments
3. How to generate publication-quality plots
4. How to integrate with the broader analysis pipeline

## What is Direction Steering?

Direction steering is a **causal validation technique** that tests whether vulnerability directions mechanistically influence model decisions.

### The Experiment

For each layer, we:

1. **Compute direction** from training data: d = mean(vulnerable) - mean(secure)
2. **Apply steering** at test time: h'ᵢ = hᵢ + α·d
3. **Measure effect** on model preference: ΔP = log P(secure) - log P(vulnerable)
4. **Vary strength** α ∈ [-20, -10, -5, 0, 5, 10, 20]

### Interpretation

- **Negative α** (amplify): Strengthens vulnerability signal
  - Expected: Model increasingly prefers secure code ✓
  - Shows: Direction truly encodes vulnerability info

- **Positive α** (suppress): Weakens vulnerability signal  
  - Expected: Model becomes less confident
  - Shows: Direction is mechanistically active

- **Layer effects**: Early layers have stronger effects than late layers
  - L3 > L7 > L15 > L23
  - Shows: Information flows through network

## Quick Start

### Option 1: Generate from Existing Data (Fast - 5 min)

If you have pre-computed steering results:

```bash
# Generate plots from existing Qwen steering results
python scripts/regenerate_steering_plots_from_existing.py --models qwen

# Or all models if results exist
python scripts/regenerate_steering_plots_from_existing.py --models qwen,codellama,starcoder2
```

**Output:**
- `fig_causal_summary_qwen_100samples.pdf`
- (+ CodeLlama, StarCoder2 if data available)

### Option 2: Run Full Pipeline (Slow - 2-4 hours on GPU)

To run steering experiments and generate plots from scratch:

```bash
# Just steering stage
python scripts/pipeline_full_analysis.py --stage steering --models qwen,codellama,starcoder2

# Or full pipeline
python scripts/pipeline_full_analysis.py --stage all --n_samples 100
```

**What happens:**
1. Loads 100 vulnerable-secure pairs from DeltaSecommits
2. Splits into train (80) and test (20)
3. For each model and layer:
   - Computes direction from training data
   - Runs steering at different α values
   - Measures preference shifts
4. Saves results to JSON
5. Generates visualizations

**Requirements:**
- GPU (strongly recommended)
- ~30 min per model on A100, 2-4 hours on CPU
- ~8GB memory per model

## File Organization

```
SAE-Java-Bug-Detection/
├── scripts/
│   ├── pipeline_full_analysis.py              # Main pipeline coordinator
│   ├── run_steering_pipeline.py               # Steering experiment runner
│   ├── regenerate_steering_plots.py           # Visualization generator
│   ├── regenerate_steering_plots_from_existing.py  # Plot from existing data
│   └── STEERING_ANALYSIS_GUIDE.md             # This file
├── results/
│   ├── raw_data/
│   │   ├── steering_results_qwen_100samples.json
│   │   ├── steering_results_codellama_100samples.json
│   │   └── steering_results_starcoder2_100samples.json
│   └── fig_*.pdf
└── results_real_preference_steering_100samples.json  # Legacy Qwen data
```

## Figure Interpretation

### fig_causal_summary_{model}_100samples.pdf

**Left panel: Steering curves**
- X-axis: Steering strength (α)
  - Negative: amplify vulnerability direction
  - Positive: suppress vulnerability direction
- Y-axis: Preference shift [log P(secure) - log P(vulnerable)]
- Each line: one layer (L3, L7, L11, L15, L19, L23)

**Reading the plot:**
- Slopes should be positive (stronger α → more preference for secure)
- Early layers (L3, L7) have steeper slopes
- Late layers (L23) have gentler slopes

**Right panel: Effect magnitudes**
- Bar chart showing max effect: α=20 minus baseline (α=0)
- L3 should be largest, decreasing through L23
- Example: L3 = 0.126, L7 = 0.030, L23 = 0.007

## Integration with Paper

### In Main Results (sections/results.tex)

The steering analysis validates the direction finding:

> "Causal validation: confirming the direction is mechanistically real.
> We applied directional steering to 100 real vulnerable–secure code pairs 
> from DeltaSecommits, varying steering strength from −20 (amplify direction) 
> to +20 (suppress). ... This decay mirrors activation patching results, 
> confirming information propagates through intermediate layers."

### In Appendix (sections/appendix/)

Reference the steering experiments for reproducibility and additional details.

## Advanced Usage

### Custom Parameters

```bash
# Fewer samples (faster)
python scripts/pipeline_full_analysis.py --stage steering --n_samples 50

# Specific models only
python scripts/pipeline_full_analysis.py --stage steering --models qwen,codellama

# Different layers (edit run_steering_pipeline.py)
LAYERS_TO_TEST = [3, 7, 11, 15, 19, 23, 27]

# Different alpha values (edit run_steering_pipeline.py)
ALPHA_VALUES = [-30, -20, -10, 0, 10, 20, 30]
```

### Monitoring Progress

The pipeline logs to stdout:

```
INFO:__main__:Running steering experiments for: qwen,codellama,starcoder2
INFO:__main__:Loading model: Qwen/Qwen2.5-7B-Instruct
INFO:__main__:Computing direction for layer 3...
...
✓ Steering experiments complete
✓ Steering plots generated
```

### Troubleshooting

**"Dataset file not found"**
- Steering needs: `sae_java_bug/artifacts/activations/.../`
- Check path in `run_steering_pipeline.py`
- Or run geometry analysis first (doesn't need this data)

**Out of memory**
- Reduce batch size in steering script
- Or run on CPU (much slower)
- Or use fewer samples

**"Results not found" for visualization**
- Run `python scripts/run_steering_pipeline.py` first
- Or use `regenerate_steering_plots_from_existing.py` if results exist

## Data Flow

```
Input Data
│
├─→ run_steering_pipeline.py
│   ├─ Load code pairs
│   ├─ Compute directions
│   ├─ Run steering experiments
│   └─ Save JSON results
│
├─→ steering_results_{model}_100samples.json
│
├─→ regenerate_steering_plots.py
│   ├─ Load results
│   ├─ Generate curves
│   └─ Generate magnitudes
│
└─→ fig_causal_summary_{model}_100samples.pdf
```

## Citation

When using steering analysis results, cite:

```
[Citation to your paper with steering validation]
```

Steering methodology based on:
- Activation patching and causal intervention techniques
- Applied to mechanistic interpretability of neural networks

## Next Steps

1. **First time:** Run `regenerate_steering_plots_from_existing.py` with Qwen data
2. **Add CodeLlama/StarCoder2:** Run full `pipeline_full_analysis.py --stage steering`
3. **Integrate results:** Add steering figures to paper
4. **Reproduce:** Document exact command used for reproducibility

---

For questions or issues with the pipeline, check:
- `PIPELINE_README.md` - Full pipeline documentation
- Script logging output - Detailed progress and errors
- Comments in scripts - Implementation details
