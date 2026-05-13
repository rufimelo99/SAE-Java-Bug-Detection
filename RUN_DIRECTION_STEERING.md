# Running the Direction Steering Experiment

## Overview

This experiment tests whether the vulnerability direction is **causally active** in controlling vulnerable code generation. By applying steering interventions at different layers and strengths, we measure the log-likelihood of vulnerable code sequences.

## Quick Start

### Option 1: Test Setup First (Recommended)

```bash
cd /Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection

# Test that everything is configured correctly
python test_direction_steering.py
```

This will:
- ✓ Check that the model/tokenizer can load
- ✓ Verify all activation files exist
- ✓ Try to compute a direction vector
- ✓ Report any issues

### Option 2: Run the Full Experiment

```bash
cd /Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection

# Minimal test (5 snippets, 2 prompts, 7 alpha values)
python sae_java_bug/sparse_autoencoders/notebooks/direction_steering_vulnerability_likelihood.py

# Or customize:
python sae_java_bug/sparse_autoencoders/notebooks/direction_steering_vulnerability_likelihood.py \
  --n_snippets 10 \
  --n_prompts_per 4 \
  --n_alphas 9
```

## What Happens

### 1. Load Curated Vulnerable Code Snippets
Five hand-picked simple examples:
- `strcpy_overflow` (CWE-120: Buffer Overflow)
- `sql_injection` (CWE-89: SQL Injection)  
- `null_check` (CWE-476: Null Pointer Dereference)
- `array_bounds` (CWE-119: Array Out of Bounds)
- `use_after_free` (CWE-416: Use After Free)

Each has 3 different prompts that could naturally elicit it.

### 2. Compute Vulnerability Directions
For each layer L ∈ {3, 7, 11, 15, 19, 23}:

```
d[L] = normalize(mean_vulnerable[L] - mean_secure[L])
```

Computed from C-only pairs in the activations dataset.

### 3. Apply Steering and Measure Log-Likelihood
For each snippet, prompt, layer, and steering strength α:

```
a_steered = a[L] - α * d[L]
log_prob = measure_log_probability(prompt, vulnerable_code, steered_activations)
```

Steering strengths range from α = -20 (far secure) to α = +20 (far vulnerable).

### 4. Generate Results and Figure
Outputs:
- `artifacts/direction_steering_likelihood/results.json` - Raw data
- `figures/fig_direction_steering_likelihood.pdf` - Dose-response curves

## Expected Results

If the hypothesis is correct (direction is causally active):

### Pattern
```
For each layer, plot should show negative slope:
  
  log_prob(vulnerable code)
         ^
         |     ╱ Steep at early layers (L3)
       0 +────•────────────
         |     ╲
        -2 |     ╲ Shallow at late layers (L23)
         |      •
         +──────────────► Steering strength (α)
            -20  0  +20
```

### Interpretation
- **α < 0** (toward secure): Log-prob **decreases** (fewer vulnerable sequences)
- **α = 0** (baseline): Normal likelihood
- **α > 0** (toward vulnerable): Log-prob **increases** (more vulnerable sequences)
- **Early layers (L3)**: Steep slope (24 layers of propagation)
- **Late layers (L23)**: Shallow slope (4 layers of propagation)

## Output Files

### `artifacts/direction_steering_likelihood/results.json`

Structure:
```json
{
  "snippets": [...],
  "layer_results": {
    "3": {
      "layer": 3,
      "snippet_results": [
        {
          "name": "strcpy_overflow",
          "cwe": "CWE-120",
          "prompts": 3,
          "alpha_results": {
            "-20.0": {"mean_logprob": -4.2, "n_prompts": 3},
            "-10.0": {"mean_logprob": -3.8, "n_prompts": 3},
            ...
            "+20.0": {"mean_logprob": -2.1, "n_prompts": 3}
          }
        },
        ...
      ]
    },
    ...
  }
}
```

### `figures/fig_direction_steering_likelihood.pdf`

One subplot per layer showing steering strength (α) on x-axis vs log-prob on y-axis.

## Parameters

```bash
--n_snippets N        # How many vulnerable snippets (default: 5, max: 5)
--n_prompts_per N     # Prompts per snippet (default: 2, max: 3)
--n_alphas N          # Number of alpha values to test (default: 7)
--device DEVICE       # 'cuda' or 'cpu' (auto-detected)
```

## Troubleshooting

### Issue: "Could not load direction for layer X"
**Cause**: Activation file missing or no C-only pairs found
**Fix**: Verify activation files exist:
```bash
ls sae_java_bug/artifacts/activations/raw_activations/vulnerable_code_qwen_coder_standard_16384_raw/
```

### Issue: All logprobs identical across alpha values
**Cause**: Direction not being applied (steering hook not registering)
**Check**:
1. Direction loaded successfully? (Should print "✓ Computed direction")
2. Direction shape correct? (Should be [3584] for Qwen2.5-7B)
3. Direction norm ≈ 1.0? (Should be normalized)

### Issue: Model out of memory
**Fix**: 
1. Reduce `--n_snippets` or `--n_prompts_per`
2. Use `--device cpu` for testing (slower but uses RAM)
3. Reduce `MAX_LENGTH` in script (default: 512)

### Issue: Very slow or hanging
**Check**:
1. Is Python using significant CPU/GPU? (Use `Activity Monitor`)
2. Try with `--n_snippets 1` to isolate the issue
3. Kill and restart: `Ctrl+C` then retry

## Understanding the Code

### `direction_steering_vulnerability_likelihood.py`

Key methods:
- `compute_vulnerability_direction(layer)`: Load or compute d[L]
- `measure_sequence_logprob(prompt, target, layer, alpha, direction)`: Core measurement
  - Registers steering hook at specified layer
  - Applies: `h = h - alpha * direction`
  - Measures log-prob of target sequence
- `run_experiment()`: Orchestrates full experiment across snippets/layers/alphas
- `plot_results()`: Generates dose-response curves

### How Steering Works

```python
def steering_hook(module, inp, output):
    h = output[0]  # Get hidden state at layer L
    h = h - alpha * direction  # Subtract α × direction (steering toward secure)
    return (h,) + output[1:]

model.layers[layer].register_forward_hook(steering_hook)
```

During forward pass:
1. Input tokens processed through layers 0 to L-1
2. At layer L, hook intercepts activations
3. Steering applied: h → h - α*d[L]
4. Modified activations flow through layers L+1 to L27
5. Output logits used to measure log-prob of target

## Next Steps

### If successful:
1. Commit results to paper repository
2. Write "Appendix E.1: Causal Steering via Direction Interpolation"
3. Update paper text with findings

### If unsuccessful:
1. Check direction computation (inspect mean_vulnerable vs mean_secure)
2. Verify hook is registered (add debug print statements)
3. Try different prompts (current ones are very simple)
4. Increase sample size (more C pairs → better direction estimate)

## Files and Paths

```
SAE-Java-Bug-Detection/
├── sae_java_bug/sparse_autoencoders/notebooks/
│   ├── direction_steering_vulnerability_likelihood.py  (Main experiment)
│   └── direction_loader.py                             (Direction I/O utility)
├── sae_java_bug/artifacts/
│   ├── activations/raw_activations/vulnerable_code_qwen_coder_standard_16384_raw/
│   │   └── activations_layer_*_raw_component_hidden_state_last_token.jsonl
│   └── direction_steering_likelihood/
│       └── results.json  (Generated)
├── figures/
│   └── fig_direction_steering_likelihood.pdf  (Generated)
├── test_direction_steering.py  (Setup test)
└── RUN_DIRECTION_STEERING.md  (This file)
```

---

**Questions?** Check `DIRECTION_STEERING_EXPERIMENT.md` for detailed methodology and expected results interpretation.
