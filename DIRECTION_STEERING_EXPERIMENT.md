# Direction Steering: Vulnerability Likelihood Experiment

## Overview

Tests whether the **vulnerability direction** (identified in the paper as 87-88% aligned across pairs) is **causally active** in controlling vulnerability generation.

### The Question
When we steer model activations *away* from the vulnerability direction (toward secure), does the model become *less* likely to generate vulnerable code?

### Hypothesis
If the vulnerability direction is causally used by the model:
- Steering **toward vulnerable** (α > 0): Likelihood of vulnerable code increases
- Steering **toward secure** (α < 0): Likelihood of vulnerable code decreases  
- **Baseline** (α = 0): Original likelihood

## Experiment Design

### Step 1: Collect Vulnerable Code Snippets
- Load real vulnerable C code from DeltaSecommits
- Filter by: length 50-500 tokens, single CWE type
- Example: buffer overflows, SQL injection, etc.

### Step 2: Create Prompts
For each vulnerable snippet, generate 3-5 different prompts that could elicit it:
```python
prompts = [
    "// Write C code for: reading from a buffer\n",
    "// Write C code for: memory access\n",
    "// Write C code for: string handling\n",
]
```

### Step 3: Apply Direction Steering
For each layer L ∈ {3, 7, 11, 15, 19, 23}:
```python
# Baseline: run model normally
logprob_baseline = measure_logprob(prompt, vulnerable_code, layer=L, alpha=0)

# Steered toward secure (α < 0)
activations_steered = activations - alpha * direction[L]
logprob_steered = measure_logprob(prompt, vulnerable_code, layer=L, alpha=alpha)
```

### Step 4: Measure Log-Likelihood
Compute log-softmax probability of the vulnerable code following the prompt:
```
log_prob = mean(log_softmax(logits[target_positions]))
```

### Step 5: Plot Dose-Response
For each layer, plot α on x-axis vs log-likelihood of vulnerable code on y-axis.
Expected: **negative slope** (steering away → less likely).

## Running the Experiment

### Prerequisites
1. **Direction vectors**: Must be loaded from `cross_layer_direction_probe.py` output
   - File: `artifacts/direction_cache/directions_c_only.pt`
   - Contains: `d[layer]` for each layer L

2. **Vulnerable snippets**: Already in dataset
   - File: `artifacts/activations/raw_activations/.../activations_layer_0_raw_component_hidden_state_last_token.jsonl`

### Quick Start
```bash
cd /Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection

# Run with default settings (5 snippets, 2 prompts each, 7 alpha values)
python sae_java_bug/sparse_autoencoders/notebooks/direction_steering_vulnerability_likelihood.py

# Or customize
python sae_java_bug/sparse_autoencoders/notebooks/direction_steering_vulnerability_likelihood.py \
  --n_snippets 10 \
  --n_prompts_per 3 \
  --n_alphas 9
```

### Output
- **Results**: `artifacts/direction_steering_likelihood/results.json`
- **Figure**: `figures/fig_direction_steering_likelihood.pdf`

## Expected Results

### If hypothesis is correct (direction is causal):
```
Layer 3:  Negative slope from α=-20 to α=+20
          log_prob(-20) ≈ -∞ (very unlikely to generate vulnerable code)
          log_prob(0)   ≈ baseline
          log_prob(+20) ≈ baseline or higher

Layer 23: Shallower slope (fewer layers to propagate effect)
          Still shows trend but smaller magnitude
```

### Pattern
- **Early layers (L3)**: Steep slope (patch propagates through 24 layers)
- **Late layers (L23)**: Shallow slope (patch propagates through 4 layers)
- **Across all layers**: Consistent negative correlation

## Mechanistic Interpretation

If the pattern holds, it proves:

1. **Causality**: The direction is not epiphenomenal but actively used
2. **Signal propagation**: Effect of layer-L steering flows through L+1...L27 via attention
3. **Behavioral relevance**: The geometric signal translates to actual generation behavior
4. **Quantifiable steering**: We can reduce vulnerability generation by controlled direction steering

## Troubleshooting

### Issue: Direction vectors not loaded
**Solution**: Run `cross_layer_direction_probe.py` first to compute and save directions
```bash
python sae_java_bug/sparse_autoencoders/notebooks/cross_layer_direction_probe.py
```

### Issue: Model OOM on long sequences
**Solution**: Reduce `MAX_LENGTH` or `--n_snippets`

### Issue: No clear pattern
**Possible causes**:
- Direction not causally active (challenges the hypothesis)
- Prompts don't elicit the target vulnerability
- Steering magnitude (α) too small or too large
- Not enough samples for stable estimates

### Issue: Slope in wrong direction
**This would be surprising!** Would suggest:
- Direction is inverted in computation
- Hook registration is wrong
- Logprob measurement is flipped

## Comparison with Prior Work

| Experiment | Measures | Evidence | Limitation |
|-----------|----------|----------|-----------|
| Probing | Alignment | Correlation | Observational |
| Mean-pooling | Signal distribution | Correlation | Observational |
| **Direction steering** | **Behavior change** | **Causation** | **Direct evidence** |

This experiment provides the strongest form of evidence: behavioral causality.

## Next Steps After Validation

1. **Quantify steering effectiveness**: What's the minimum α needed to prevent vulnerability?
2. **Test on other CWE types**: Does the effect generalize?
3. **Compare with other steering methods**: Is direction-based steering better than random perturbation?
4. **Downstream application**: Use this for actual code security tools
