# Paired Ranking Task — Quick Start

This script validates the **"relative vs. absolute"** claim from the C paper by testing whether the vulnerability direction encodes relative ordering (i.e., vulnerable > secure for most pairs).

## What It Tests

**The paper's claim:**
- 87–88% of pairs align with the vulnerability direction: `(vulnerable - secure) · direction > 0`
- But binary classification only achieves ~50% AUROC
- Explanation: model learns *relative* ordering (B > A), not *absolute* categories (is B vulnerable?)

**This script checks:**
For each test pair, does `proj(vulnerable) > proj(secure)` onto the direction?
- If YES ~87% of the time → **Claim is validated**
- If NO most of the time → **Claim is refuted; direction doesn't encode relative ordering**

## Running the Script

### Option 1: Default (all layers, latest run)
```bash
cd /Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection
python scripts/paired_ranking_task.py
```

### Option 2: Specific layers only
```bash
python scripts/paired_ranking_task.py --layers 15,19,23
```

### Option 3: Custom fold count and PCA
```bash
python scripts/paired_ranking_task.py --layers 0,3,7,11,15,19,23,27 --n_folds 10 --pca_dim 100
```

### Option 4: Specify activation run directory
```bash
python scripts/paired_ranking_task.py --run_dir ./sae_java_bug/artifacts/activations/mean_pool/2025-05-10_xyz
```

## Output

The script will print:
```
======================================================================
PAIRED RANKING TASK — Validating 'Relative vs. Absolute' Claim
======================================================================

Layer 15:
----------------------------------------------------------------------
  Fold 0: 0.835 (312 pairs)
  Fold 1: 0.856 (315 pairs)
  Fold 2: 0.841 (314 pairs)
  Fold 3: 0.847 (313 pairs)
  Fold 4: 0.839 (316 pairs)

  Mean accuracy: 0.843 ± 0.008
  95% CI: [0.827, 0.859]
  Chance baseline: 0.500
  Paper claim (per-pair alignment): ~0.87
  Fraction with vuln > secure: 0.843

======================================================================
SUMMARY
======================================================================

Mean accuracy across all layers: 0.675

Interpretation:
  • 0.85–0.90: ✓ Relative ordering hypothesis STRONGLY SUPPORTED
  • 0.70–0.80: ⚠ Partial support; direction is informative but noisier
  • 0.50–0.60: ✗ Direction does NOT encode relative ordering
  • <0.50:     ✗ Direction is backwards or degenerate
```

And saves results to `paired_ranking_results.json`:
```json
{
  "layer_15": {
    "mean_accuracy": 0.843,
    "std_accuracy": 0.008,
    "ci_95": 0.016,
    "fold_accuracies": [0.835, 0.856, 0.841, 0.847, 0.839],
    "n_test_pairs": 1576,
    "frac_positive": 0.843
  },
  ...
}
```

## Interpreting Results

### Expected Outcome if Claim is True
- **Accuracy: 0.85–0.90** (close to the reported 87–88%)
- **Frac_positive: ~0.87** (matches "87% of pairs align")
- **Interpretation**: The vulnerability direction truly captures relative ordering.
  - **Rebuttal to reviewer**: "Direct pairwise ranking achieves 87% accuracy, confirming the direction encodes relative ordering as claimed."

### Unexpected Outcome if Claim is False
- **Accuracy: 0.50–0.60** (only slightly above chance)
- **Interpretation**: The direction does NOT reliably encode relative ordering.
  - **Action needed**: Investigate what the "87% alignment" metric actually measures.
  - **Possible explanations**:
    1. Alignment is computed differently than pairwise ranking
    2. Direction is unstable in test set
    3. Methodology issue in how direction is computed

### Partial Support (0.70–0.75)
- **Interpretation**: Direction is informative but not as strong as claimed.
- **Action**: Report actual accuracy and acknowledge gap vs. claimed 87%.

## Methodology Notes

1. **Pair-stratified CV**: Both members of each pair stay in the same fold to prevent leakage
2. **Direction computation**: Trained on folds' training data only (no test leakage)
3. **PCA**: Applied per fold to prevent information leakage
4. **Activation pooling**: Uses mean-token pooling (same as paper)

## Expected Runtime

- Per layer: ~10–30 seconds (depends on data size)
- All 8 layers: ~2–4 minutes
- With 10 folds: ~4–8 minutes

## Troubleshooting

**Error: "No mean_pool runs under..."**
- Check that activations are stored in `sae_java_bug/artifacts/activations/mean_pool/`
- If stored elsewhere, use `--run_dir PATH`

**Error: "Missing activations for layer X"**
- Layer X hasn't been computed yet
- Use `--layers` to skip it

**Very slow (>1 min per layer)?**
- Reduce `--pca_dim` from 50 to 20
- Reduce `--n_folds` from 5 to 3

## Next Steps

1. **Run the script** and note the mean accuracy
2. **Compare to 0.87**: Does it match, partially match, or contradict?
3. **Decide response**:
   - If ~0.87: "Directly validates claim; can be added to paper or rebuttal"
   - If 0.70–0.75: "Acknowledge gap; may require revising claims"
   - If <0.60: "Refutes claim; needs investigation into methodology"
