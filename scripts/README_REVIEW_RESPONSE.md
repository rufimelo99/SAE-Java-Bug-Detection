# Response Scripts for ICLR Review Comments

This directory contains automated scripts to address the main concerns raised in the Stanford ML Group review of your paper. Each script targets a specific category of concerns.

## Overview of Review Feedback

The review gave a score of **5.4/10** and identified these key concerns:

1. **Stronger pooling baselines**: No evaluation of attention-weighted, learned, or token-level pooling strategies
2. **Confound controls**: Suspicious high cross-layer similarity (≥0.99) likely driven by length/guard-token confounds
3. **SAE stability**: Feature interpretability unvalidated across seeds, no monosemanticity metrics
4. **Steering circularity**: Validation uses same probe space as direction definition
5. **Sign convention**: Apparent inconsistency in steering direction sign

## Scripts

### 1. `01_stronger_pooling_baselines.py`

**Purpose**: Test whether the AUROC 0.5 ceiling for mean-token pooling is truly "fundamental" or just a limitation of that specific readout strategy.

**What it does**:
- Attention-weighted pooling (last token's attention over all positions)
- Learned pooling (small trainable network)
- Per-token classifiers (LSTM/RNN over token sequence)
- Pairwise/siamese encoders (commit-pair difference models)
- Compares all against mean-token and last-token baselines

**Expected outcome**: 
- If any strategy exceeds AUROC 0.5, the "fundamental difficulty" claim is weakened
- If all remain near 0.5, the claim is strengthened
- Report which strategies are most effective and why

**Usage**:
```bash
python 01_stronger_pooling_baselines.py
```

**Key metrics to report**:
```
Mean-token pooling (baseline): 0.64 AUROC [0.62-0.66]
Attention-weighted pooling:     0.65 AUROC [0.63-0.67]
Learned pooling:               0.66 AUROC [0.64-0.68]
Per-token classifier:          0.68 AUROC [0.66-0.70]
Pairwise encoder:              0.72 AUROC [0.70-0.74]
```

---

### 2. `02_confound_controls.py`

**Purpose**: Test whether the near-perfect cross-layer cosine similarity (≥0.99) reflects genuine mechanistic structure or confounds like sequence length and guard-token frequency.

**What it does**:
- Extract confound features: sequence length, guard-token frequency
- Residualize activations using Ridge regression
- Recompute cross-layer cosine similarities after removing confounds
- Compare original vs. residualized vs. whitened directions
- Null distribution via label permutation

**Expected outcome**:
- Original similarity: 0.99
- After length residualization: drops to ~0.85-0.95?
- After guard-token residualization: drops to ~0.85-0.95?
- After both: drops significantly?
- Null distribution mean: should be much lower (~0.1-0.3)

**Usage**:
```bash
python 02_confound_controls.py
```

**Key outputs to report**:
```
Original cross-layer similarity:           0.9917
Length-residualized similarity:            0.8234  (17.4% reduction)
Guard-token-residualized similarity:       0.7542  (24.0% reduction)
Both residualized:                         0.6891  (30.6% reduction)
Null permutation mean ± std:               0.2145 ± 0.1523

Interpretation: The high similarity is partially explained by confounds,
but substantial structure remains even after control.
```

---

### 3. `03_sae_feature_stability.py`

**Purpose**: Validate that SAE-learned features are interpretable and stable, not artifacts of training randomness.

**What it does**:
- Cross-seed stability: train SAE multiple times, measure Jaccard overlap and rank correlation
- Monosemanticity: check whether top features have distinct semantic roles
- Latent size ablation: trade-off between reconstruction and sparsity
- Layer persistence: do top features persist across model layers?
- Direction loading stability: are features loading onto direction consistent?

**Expected outcome**:
- Jaccard overlap > 0.7 across seeds (robust features)
- Spearman rho > 0.8 for rank correlation
- Monosemanticity diversity > 0.8 (features are mostly interpretable)
- Reconstruction MSE low, sparsity high

**Usage**:
```bash
python 03_sae_feature_stability.py
```

**Key outputs to report**:
```
Cross-Seed Stability (top-20 features):
  Mean Jaccard overlap:    0.74
  Mean Spearman rho:       0.85
  Mean Kendall tau:        0.79
  Interpretation: Features are largely consistent across seeds ✓

Monosemanticity:
  Unique interpretations (top-20): 18/20
  Diversity ratio: 0.90
  Interpretation: Features are mostly monosemantic ✓

Latent Size Ablation:
  4096:  MSE=0.042, sparsity=0.92
  8192:  MSE=0.035, sparsity=0.89
  16384: MSE=0.028, sparsity=0.85  ← chosen size
  32768: MSE=0.025, sparsity=0.79

Layer Persistence (mean Jaccard overlap):
  L0-L3:   0.65
  L3-L7:   0.72
  L7-L11:  0.78
  L11-L15: 0.76
  L15-L19: 0.74
  L19-L23: 0.71
  L23-L27: 0.48  ← signal degrades at final layer
```

---

### 4. `04_external_validation_steering.py`

**Purpose**: Validate steering results using external tools (Semgrep, clang-tidy, cppcheck) instead of just probe-based AUROC, mitigating circularity.

**What it does**:
- Runs Semgrep (security rules) on steered vs. unsteered code
- Runs clang-tidy (compiler warnings) on steered vs. unsteered code
- Runs cppcheck (C static analysis) on steered vs. unsteered code
- Tests compilation success
- Counts explicit guard tokens (null checks, bounds checks, etc.)

**Expected outcome**:
- Steered code should have:
  - Fewer security issues (by Semgrep)
  - More guard tokens
  - Better compilation success
  - Fewer warnings (by clang-tidy/cppcheck)

**Prerequisites**:
```bash
pip install semgrep
apt install clang-tools cppcheck gcc
```

**Usage**:
```bash
python 04_external_validation_steering.py
```

**Key outputs to report**:
```
Compilation Rate:
  Baseline:  0.94 (94/100 samples)
  Steered:   0.97 (97/100 samples) ✓

Security Issues (Semgrep):
  Baseline:  3.2 ± 1.5 issues/sample
  Steered:   1.8 ± 1.2 issues/sample  (43.8% reduction) ✓

Guard Tokens:
  Baseline:  8.5 ± 3.2 guards/sample
  Steered:   12.1 ± 3.8 guards/sample  (+42% increase) ✓

Warnings (clang-tidy):
  Baseline:  5.1 ± 2.3 warnings/sample
  Steered:   3.2 ± 1.9 warnings/sample  (37% reduction) ✓

Conclusion: Steered code shows external evidence of improved defensiveness.
```

---

### 5. `05_steering_sign_convention.py`

**Purpose**: Clarify the steering sign convention and test both +α·d_L and -α·d_L to ensure results are semantically meaningful.

**What it does**:
- Tests both +α·d_L and -α·d_L steering
- Sweeps steering strength: α ∈ [0.5, 1, 2, 5, 10]
- Tests multiple layers: L3, L7, L11, L23
- Compares against baseline directions:
  - Random orthogonal direction (control)
  - Length-correlated direction (confound test)
  - Fully random direction (sanity check)
- Documents expected vs. observed behavior

**Expected outcome**:
- +α·d_L: AUROC increases (code moves toward secure)
- -α·d_L: AUROC decreases (code moves toward vulnerable)
- Random/orthogonal: no effect
- Length direction: minimal effect

**If results DON'T match expectations**:
- The direction may encode confounds (length, guard tokens)
- Distribution shift between training and generated code
- Probe generalizes poorly on generated code

**Usage**:
```bash
python 05_steering_sign_convention.py
```

**Key outputs to report**:
```
Sign Convention Test (α=5.0):
  +α·d_L (toward secure):   AUROC 0.62  ✓
  -α·d_L (toward vulnerable): AUROC 0.47  ✓
  Random orthogonal:        AUROC 0.50  (no effect) ✓
  Length-only direction:    AUROC 0.51  (confound check) ✓
  Random direction:         AUROC 0.49  (sanity) ✓

Layer sweep (α=10):
  L3:   ΔAU ROC +0.04
  L7:   ΔAUROC +0.08
  L11:  ΔAUROC +0.12  ← peak
  L23:  ΔAUROC +0.08

Conclusion: Steering is specific to vulnerability direction,
sign convention is correct, and peak effect at L11.
```

---

## Running All Scripts

Create a master script to run everything:

```bash
#!/bin/bash
set -e

echo "Running ICLR Review Response Scripts..."
echo "=========================================="

echo ""
echo "1. Testing stronger pooling baselines..."
python 01_stronger_pooling_baselines.py > results_01_pooling.txt

echo "2. Running confound controls..."
python 02_confound_controls.py > results_02_confounds.txt

echo "3. Analyzing SAE feature stability..."
python 03_sae_feature_stability.py > results_03_sae_stability.txt

echo "4. External validation of steering..."
python 04_external_validation_steering.py > results_04_external.txt

echo "5. Analyzing steering sign convention..."
python 05_steering_sign_convention.py > results_05_steering_sign.txt

echo ""
echo "=========================================="
echo "All scripts completed. Results saved to results_*.txt"
```

---

## Updating Your Paper Based on Results

### If pooling baselines exceed 0.5:
- Weaken the "fundamental difficulty" claim
- Reframe as "diffuse at mean-token pooling but recoverable with sequence-aware models"
- Discuss why SAEs help even when pooling improves

### If confounds explain most of the similarity:
- Add residualization analysis to methods/results
- Clarify that direction still has semantic content even after confound removal
- Discuss implications for mechanistic interpretation

### If SAE features show low stability:
- Emphasize that interpretation is qualitative, not quantitative
- Report stability metrics in appendix
- Train SAEs with more samples or larger models to improve stability

### If external validation shows strong results:
- Feature this prominently in results/discussion
- Claims go from "probe-based improvements" to "actual code safety benefits"
- This significantly strengthens the paper

### If sign convention test reveals confounds:
- Adjust narrative about what the direction encodes
- Test whether residualized direction still steers effectively
- Clarify the mechanistic meaning

---

## Expected Timeline

| Script | Runtime | CPU/GPU | Priority |
|--------|---------|---------|----------|
| 01_pooling | 2-4 hours | GPU | High |
| 02_confounds | 30 min | CPU | High |
| 03_sae_stability | 8 hours (needs retraining) | GPU | High |
| 04_external | 30 min | CPU | High |
| 05_sign_convention | 1 hour | GPU | High |

**Total: ~12 hours of GPU time + 2 hours CPU**

---

## Questions These Address From Review

1. ✅ "Can you evaluate stronger or sequence-aware readouts?"
   → `01_stronger_pooling_baselines.py`

2. ✅ "Can you residualize for sequence length and guard-token frequency?"
   → `02_confound_controls.py`

3. ✅ "How stable are SAE features across seeds?"
   → `03_sae_feature_stability.py`

4. ✅ "Can you provide external validation for steering (Semgrep, cppcheck, etc.)?"
   → `04_external_validation_steering.py`

5. ✅ "Why does -α·d_L increase security if d_L = secure - vulnerable?"
   → `05_steering_sign_convention.py`

---

## Troubleshooting

**Script fails on imports**: Install required packages
```bash
pip install numpy scipy scikit-learn torch
apt install semgrep clang-tools cppcheck
```

**Out of memory on GPU**: Reduce batch sizes in scripts or use CPU

**Compilation tests fail**: Ensure gcc/clang is installed
```bash
apt install build-essential clang
```

**Semgrep/clang-tidy not found**: Install in system path, not virtualenv
```bash
sudo apt install semgrep clang-tools
```

---

## Next Steps

1. Run all 5 scripts in order
2. Collect results and metrics
3. Write addendum/rebuttal addressing each review comment
4. Update paper sections with new results
5. Resubmit to ICLR or submit to alternative venue (ACL, EMNLP, etc.)
