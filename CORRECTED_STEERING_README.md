# Corrected Direction Steering Experiment

## Overview

This script implements a methodologically rigorous version of the direction steering experiment from "Vulnerability Knowledge in Code LLMs: Comparison, Not Classification" with two critical fixes:

1. **Length Normalization**: Per-token averaged log-probabilities instead of raw sums
2. **Information Leakage Fix**: Train/test split for direction computation

## Motivation

The original steering experiment had two limitations:

### Issue 1: Unnormalized Preference Metric
**Problem**: Raw log-probability difference `log P(secure) - log P(vulnerable)` can be biased when code lengths differ.
- Longer code = larger absolute log-prob values
- Comparison becomes unfair across variable-length pairs

**Solution**: Use per-token averaged log-probability
```
preference = mean(log_probs_secure) - mean(log_probs_vulnerable)
```
This normalizes by token count, making comparisons length-independent.

### Issue 2: Information Leakage in Direction Computation
**Problem**: Directions computed on all data, then evaluated on overlapping data for transfer metrics.
- Direction = normalize(mean_vulnerable - mean_secure)
- If same data used for computation and evaluation, transfer alignment can be inflated

**Solution**: Train/test split
1. Compute directions on 80% training set only
2. Evaluate steering effects on held-out 20% test set
3. No data contamination

## Usage

### Basic Usage
```bash
python run_corrected_steering_experiment.py --n_samples 100
```

### With Custom Parameters
```bash
python run_corrected_steering_experiment.py \
    --n_samples 100 \
    --test_split 0.2 \
    --device cuda
```

### Parameters
- `--n_samples`: Number of test code pairs to evaluate (default: 100)
- `--test_split`: Fraction of data reserved for testing, rest used for direction computation (default: 0.2)
- `--device`: Compute device, "cuda" or "cpu" (default: auto-detect)

## Output

Results saved to `results_corrected_steering_train_test_split.json`:
```json
{
  "n_test_samples": 100,
  "n_train_samples": 400,
  "methodology": "length-normalized + train/test split",
  "baseline": {
    "mean_preference": -0.1234,
    "std": 0.2456,
    "normalization": "per-token average (length-normalized)"
  },
  "layers": {
    "3": {
      "alpha_results": {
        "-20.0": -0.456,
        "0.0": -0.123,
        "20.0": 0.210
      }
    }
    ...
  }
}
```

## Key Differences from Original

| Aspect | Original | Corrected |
|--------|----------|-----------|
| Log-prob metric | Raw sum | Per-token average (normalized by token count) |
| Direction computation | All data | Training set only (80%) |
| Evaluation set | Overlapping with direction | Held-out test set (20%) |
| Length bias | Present (longer code → larger values) | Removed (normalized per token) |
| Direction leakage | Present (same-set evaluation) | Removed (train/test split) |

## Expected Results

The monotonic dose-response pattern should **persist or strengthen** under corrected methodology:
- As steering strength α increases (suppressing vulnerability direction), preference for secure code should increase monotonically
- Effect sizes may differ slightly due to normalization, but the direction and consistency should be robust
- The ~87-88% per-pair alignment claim is about representation geometry, not preference metrics, so should remain stable

## Requirements
- PyTorch
- Transformers
- NumPy
- Model: Qwen2.5-7B-Instruct (auto-downloaded)
- Dataset: DeltaSecommits (expects path: `sae_java_bug/artifacts/activations/...`)

## Interpretation

**Length-normalized preference metric**: 
- Positive values indicate model prefers secure code
- Can now be fairly compared across code pairs of different lengths
- Effect sizes are per-token, making them interpretable independent of code length

**Train/test split for directions**:
- Directions learned from one set, tested on unseen data
- Prevents double-dipping / information leakage
- Transfer alignment metrics become more trustworthy

## Next Steps

1. Run experiment: `python run_corrected_steering_experiment.py --n_samples 100`
2. Compare results to original results
3. Update paper with corrected results if monotonic pattern persists
4. Generate corrected figure with per-token normalized preference scores
