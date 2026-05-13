# Direction Steering: Quick Start Guide

## What We're Testing

**Hypothesis**: The vulnerability direction (87-88% alignment) is **causally active** in controlling whether the model generates vulnerable code.

**Test**: When we steer activations *away* from vulnerability (toward secure), does the model generate vulnerable code *less often*?

## The Three Key Scripts

### 1. `direction_loader.py`
Loads the vulnerability direction vectors needed for steering.
- Computes or loads: `d[L] = normalize(mean_vulnerable[L] - mean_secure[L])`
- One direction per layer L ∈ {0, 3, 7, 11, 15, 19, 23, 27}

### 2. `direction_steering_vulnerability_likelihood.py`
Main experiment: measures vulnerability generation likelihood under steering.
- Loads vulnerable code snippets
- Creates prompts that could elicit them
- Applies steering at varying strengths (α)
- Measures log-probability of generating vulnerable code

### 3. `cross_layer_direction_probe.py` (already exists)
Pre-requisite: computes and caches the vulnerability directions.

## Full Workflow

### Step 0: Ensure Directions Are Computed
```bash
cd /Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection

# Compute vulnerability directions (if not already cached)
python sae_java_bug/sparse_autoencoders/notebooks/cross_layer_direction_probe.py \
  --language c \
  --output_dir sae_java_bug/artifacts/direction_cache
```

This creates: `artifacts/direction_cache/directions_c_only.pt`

### Step 1: Run the Steering Experiment
```bash
# Minimal test (5 snippets, 2 prompts each, 7 alpha values)
python sae_java_bug/sparse_autoencoders/notebooks/direction_steering_vulnerability_likelihood.py

# Full experiment (more data, more robust)
python sae_java_bug/sparse_autoencoders/notebooks/direction_steering_vulnerability_likelihood.py \
  --n_snippets 20 \
  --n_prompts_per 4 \
  --n_alphas 9
```

### Step 2: Check the Output
```bash
# Results JSON
cat sae_java_bug/artifacts/direction_steering_likelihood/results.json | python -m json.tool

# Figure
open figures/fig_direction_steering_likelihood.pdf
```

## Expected Output

### Figure: `fig_direction_steering_likelihood.pdf`

**One subplot per layer (L3, L7, L11, L15, L19, L23)**

Each shows:
```
     Log-Prob(vulnerable code)
              ^
              |      ↗ Steep at early layers (L3)
            0 +-----•────────────
              |      ↘
         -2   |       ╲
              |        ╲ Shallow at late layers (L23)
         -4   |         •
              |
              +----+----+----+----► Steering strength (α)
             -20   -10   0  +10  +20

KEY:
α < 0  : Steer toward SECURE (← should reduce vulnerability)
α = 0  : BASELINE (no steering)
α > 0  : Steer toward VULNERABLE (↗ should increase vulnerability)
```

### If Hypothesis is Correct

**Pattern across layers:**
- **L3**: Steep negative slope (-4 to 0 log-prob over α range)
  - Early layer, patch propagates through 24 layers → big effect
- **L7, L11, L15, L19**: Medium slopes
  - Moderate propagation distance
- **L23**: Shallow slope (-0.5 to 0 log-prob)
  - Late layer, patch propagates through 4 layers → small effect

**Key observations:**
1. **Negative slope everywhere**: Steering toward secure reduces vulnerability
2. **Slope decreases with layer**: Earlier steering has bigger effect
3. **Consistent dose-response**: Smooth relationship (not random)

### Interpretation

If you see this pattern:

✅ **Confirms causality**: The direction is not spurious
✅ **Confirms propagation**: Effect flows through remaining attention layers
✅ **Confirms mechanism**: Matches theory (more layers = bigger effect)
✅ **Paper contribution**: Direct behavioral evidence of mechanism

## Paper Integration

This experiment would become **Appendix E.1** in the paper:

### Section: "Causal Steering via Vulnerability Direction Interpolation"

**Question**: Is the vulnerability direction causally used by the model?

**Method**: Interpolate along direction at layer L with strength α
```
a_steered = a_vulnerable - α * d^L
```

**Measurement**: Log-probability of generating vulnerable code

**Results**: 
- Negative slope (steering away → less vulnerable)
- Layer-dependent magnitude (matches attention propagation)

**Significance**: Proves the direction is mechanistically active, not epiphenomenal

## Troubleshooting

### No clear pattern / random noise
**Check:**
- [ ] Direction vectors loaded correctly (verify in results JSON)
- [ ] Prompts are actually creating the vulnerable code
- [ ] Sample size sufficient (n_snippets ≥ 10, n_prompts_per ≥ 3)
- [ ] Alpha range appropriate for effect size

### Slope in wrong direction (steering toward vulnerable increases likelihood)
**This is surprising!** Suggests:
- [ ] Direction vectors might be inverted
- [ ] Hook registration order wrong
- [ ] Logprob measurement flipped

### OOM or slow
- Reduce `--n_snippets` or `MAX_LENGTH` in script
- Use smaller model (e.g., 3B) for testing

### Direction vectors not found
```bash
# Check cache exists
ls -lh sae_java_bug/artifacts/direction_cache/

# If missing, compute
python sae_java_bug/sparse_autoencoders/notebooks/cross_layer_direction_probe.py
```

## Files Generated

| File | Purpose |
|------|---------|
| `artifacts/direction_steering_likelihood/results.json` | Raw results (mean logprobs per layer/alpha/snippet) |
| `figures/fig_direction_steering_likelihood.pdf` | Dose-response curves |
| `artifacts/direction_cache/directions_c_only.pt` | Pre-computed direction vectors |

## Next Steps

### If successful:
1. Add to appendix as formal causal intervention
2. Quantify effect sizes for paper
3. Test generalization to other CWE types
4. Compare against other steering baselines

### If unsuccessful:
1. Check direction computation (run `cross_layer_direction_probe.py` with verbose output)
2. Try different prompt templates
3. Expand sample size
4. Debug hook registration (add print statements)

## Experiment Parameters to Tune

```python
# In direction_steering_vulnerability_likelihood.py

# Steering strength: how far along direction to move
ALPHAS = [-20, -10, -5, 0, 5, 10, 20]  # Default

# Which layers to test (earlier = bigger effect expected)
STEER_LAYERS = [3, 7, 11, 15, 19, 23]  # Default (skip 0, 27)

# How many vulnerable snippets to test
N_SNIPPETS = 10  # Default: 5

# How many different prompts per snippet
N_PROMPTS_PER = 3  # Default: 2

# Hidden dimension of Qwen2.5-7B (should be 3584 or 4096)
HIDDEN_DIM = 3584  # Verify from model config
```

## Contact / Questions

If results look good, prepare for paper submission:
```
Figure caption:
"Direction steering: Log-probability of vulnerable code as a function 
of steering strength α at layers 3–23. Negative slope confirms the 
vulnerability direction causally controls generation. Effect magnitude 
decreases with layer (L3 steep, L23 shallow) consistent with 
propagation through remaining transformer blocks."
```
