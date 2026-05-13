# Behavioral Patching Framework Explained

## The Question

**When we patch vulnerable code's body activations with secure activations at layer L, does the model's behavior shift toward predicting security-relevant tokens?**

## Why This Matters

We have two separate types of evidence about where vulnerability information lives:

1. **Representational Patching** (causal_patching.py): Shows that the vulnerability *direction* in activation space shifts when we patch
   - But do these shifts actually affect the model's outputs?
   - Does the model become more likely to generate secure code?

2. **Behavioral Patching** (causal_patching_behavioral_c_only.py): Tests if representational shifts translate to output behavior changes
   - Does the model predict security-relevant tokens more often?
   - Provides convergent evidence for the same mechanistic story

## The Experimental Design

```
For each pair (vulnerable_code, secure_code):

1. Cache Pass: Run secure_code → save activations at layers {0, 3, 7, 11, 15, 19, 23, 27}

2. Baseline Pass: Run vulnerable_code WITHOUT patching
   → Measure: log-softmax probability of security tokens at final position

3. Patched Passes (8 total): For each layer L in {0, 3, 7, 11, 15, 19, 23, 27}
   → Run vulnerable_code WITH body patch at layer L
   → Measure: log-softmax probability of security tokens at final position
   → Compute: Δ = patched_score - baseline_score

Result: Per-layer delta showing how much the patch shifted the model toward
        predicting security-relevant tokens
```

## What Are Security-Relevant Tokens?

Tokens that appear in secure, defensive C code:
- Memory safety: `NULL`, `nullptr`, `assert`, `sizeof`, `malloc`, `free`
- Input validation: `validate`, `check`, `verify`, `filter`
- Error handling: `error`, `errno`, `throw`, `catch`

These are manually curated to be single BPE tokens in Qwen2.5-7B-Instruct.

## The Expected Pattern

If vulnerability information is **distributed across body tokens** (not concentrated in final token):

```
Patching at L0  (early):   Patch propagates through 27 attention layers
                           → large effect on final logits
                           → Δ ≈ +4.0 (large positive)

Patching at L3  :          Patch propagates through 24 attention layers
                           → effect slightly smaller
                           → Δ ≈ +3.3

...

Patching at L23 (late):    Patch propagates through 4 attention layers
                           → effect much smaller
                           → Δ ≈ +0.4

Patching at L27 (final):   Final token NEVER patched by design
                           No transformer layers after L27
                           Patch has no pathway to affect final logits
                           → Δ ≈ 0.0 (exact zero)
```

**Key insight:** The monotonic decay to zero at L27 is mechanistically interpretable:
- Body patch excludes final token position
- At L27, final logits come directly from unpatched final token activations
- No subsequent layers to propagate the patch
- Therefore zero effect

## Why Not Just Patch the Final Token?

In the representational patching, we also measure "last_only" patches:
- Patching only the final token position
- Effect on mean-token representation: ~0

Why? Because the vulnerability signal is **distributed** across ~190 body tokens.
Changing 1 position barely affects the mean of 191 positions.

This validates the finding: the signal is in the body, not the final token.

## What Should the Results Look Like?

### If the hypothesis is correct (distributed in body):

```
Layer    Δ Security Vocabulary    Interpretation
─────────────────────────────────────────────────
L0       +4.06 ± 0.36           Baseline effect, 27 layers to propagate
L3       +3.25 ± 0.33           Slightly less, 24 layers to propagate
L7       +2.88 ± 0.30           Further decay, 20 layers to propagate
L11      +2.68 ± 0.30           ...
L15      +1.90 ± 0.27           ...
L19      +1.21 ± 0.22           ...
L23      +0.38 ± 0.09           Very small, only 4 layers to propagate
L27      +0.00 ± 0.00           Final token never patched → zero effect
```

**Pattern:** Smooth monotonic decay from L0 to L27, reaching exactly zero.

### What Would contradict the hypothesis?

- **All values identical** across layers (as we saw in the broken Exp A)
  → Suggests bug in implementation or wrong measurement
  
- **No decay pattern** (random or flat across layers)
  → Would suggest information is NOT distributed, contradicting representational results

- **Positive value at L27** (>0.1)
  → Would contradict the mechanistic interpretation (final token never patched)

## Running the Script

### C Language Only (Recommended)
```bash
cd /Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection
python sae_java_bug/sparse_autoencoders/notebooks/causal_patching_behavioral_c_only.py --n_pairs 100
```

Results: `sae_java_bug/artifacts/causal_patching/behavioral_results_c_only.json`

### All Languages (Original)
```bash
python sae_java_bug/sparse_autoencoders/notebooks/causal_patching_behavioral.py --n_pairs 100
```

Results: `sae_java_bug/artifacts/causal_patching/behavioral_results.json`

## Interpreting the Output

### Standard Error (s.e.)
- Shows variability across the 100 pairs
- Δ significant if |Δ| > 2 × s.e. (marked with *)
- At L27, expect SEM ≈ 0 if all values are exactly 0

### Effect Size
- Reported as "Δ / s.e." (how many standard errors the effect is)
- 4.0× s.e. = very strong effect
- <1× s.e. = noise level

### Layer-by-Layer Reading

**L0-L15:** Should see strong, decaying effects (4.0 → 1.9)
- Validates that patch is being applied
- Validates that information propagates through attention

**L19-L27:** Should see steep decay (1.2 → 0.38 → 0.0)
- Final layer should be exactly zero (mechanistically interpretable)
- Validates the distributed encoding hypothesis

## Relationship to Representational Patching

| Aspect | Representational (J.1) | Behavioral (Exp A) |
|--------|------------------------|-------------------|
| **Measures** | Shift in vulnerability direction | Shift in security token probability |
| **At layer** | Measured at L27 | Measured at final position |
| **Expected pattern** | Monotonic decay L0→L27 | Monotonic decay L0→L27 |
| **At L27** | ~-3.0 (shifted toward secure) | 0.0 (final token never patched) |
| **Interpretation** | Representations change | Behavior changes (direct evidence) |

Both should show the **same mechanistic story**: Layer-dependent propagation of the patch through attention layers.

## Common Questions

**Q: Why measure at the final position specifically?**
A: This is where the model predicts the next token. If we want to know if the patch affects what the model generates, we need to see how it affects next-token predictions.

**Q: Why does L27 = 0 exactly?**
A: By design, the body patch excludes the final token position. At L27 (the final layer), the final token activations come from the unpatched L27 output. There are no layers after L27 to propagate the patch. Therefore, the final logits are determined entirely by the unpatched final token, resulting in Δ = 0.

**Q: What if I see Δ = +8.50 at all layers?**
A: That's the broken Exp A result. This indicates the measurement (loss-based) is not layer-sensitive. The security vocabulary measurement (Exp A in the revised script) should NOT show this pattern—it should decay monotonically. If it does, there's a bug in the implementation.

**Q: Why run only on C?**
A: Eliminates language-as-confound. Qwen's final layer shows different behavior across languages (discussed in results.tex line 29-34). C-only analysis prevents this multi-language interference from obscuring the signal.

## Files Generated

- `behavioral_results_c_only.json`: Per-layer means and standard errors
- `fig_behavioral_patching_c_only.pdf`: Visualization of the decay pattern
- Console output: Per-pair detailed statistics

## Debugging Checklist

If results look wrong:

- [ ] Check that `n_pairs` is reasonable (≥50 for stable estimates)
- [ ] Check baseline score is reasonable (usually -20 to -10 log-softmax)
- [ ] Check L0 has positive value (patch should increase security token probability)
- [ ] Check L27 is ≈ 0.0 with very small SEM
- [ ] Check monotonic decay from L0 to L27
- [ ] Verify script is using `use_cache=False` in model calls
- [ ] Verify hook registration and removal isn't failing (check for error messages)
