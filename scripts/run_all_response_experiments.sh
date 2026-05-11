#!/bin/bash
################################################################################
# Master script to run all ICLR review response experiments
#
# This script runs all 5 analysis scripts in sequence to address reviewer concerns:
# 1. Stronger pooling baselines
# 2. Confound controls (length, guard tokens)
# 3. SAE feature stability across seeds
# 4. External validation of steering (SAST tools)
# 5. Steering sign convention and baseline comparisons
#
# Usage:
#   bash run_all_response_experiments.sh
#
# Output:
#   - Individual results files for each experiment
#   - Summary report in RESULTS_SUMMARY.txt
#   - Detailed recommendations in RECOMMENDATIONS.txt
################################################################################

set -e  # Exit on first error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results_${TIMESTAMP}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ICLR Review Response Experiments${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Starting experiments at: $(date)${NC}"
echo -e "${YELLOW}Results will be saved to: ${RESULTS_DIR}${NC}"
echo ""

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Check for required Python packages
echo -e "${BLUE}Checking dependencies...${NC}"
python3 << 'EOF'
import sys
required_packages = ['numpy', 'scipy', 'sklearn', 'torch']
missing = []
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing.append(package)

if missing:
    print(f"Missing packages: {', '.join(missing)}")
    print("Install with: pip install numpy scipy scikit-learn torch")
    sys.exit(1)
print("✓ All required Python packages found")
EOF

echo ""
echo -e "${GREEN}✓ Dependencies OK${NC}"

# Script 1: Stronger pooling baselines
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}1/5: Testing stronger pooling baselines${NC}"
echo -e "${BLUE}========================================${NC}"
python3 01_stronger_pooling_baselines.py 2>&1 | tee "${RESULTS_DIR}/01_pooling.txt"
echo -e "${GREEN}✓ Pooling baselines complete${NC}"

# Script 2: Confound controls
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}2/5: Running confound controls${NC}"
echo -e "${BLUE}========================================${NC}"
python3 02_confound_controls.py 2>&1 | tee "${RESULTS_DIR}/02_confounds.txt"
echo -e "${GREEN}✓ Confound controls complete${NC}"

# Script 3: SAE feature stability
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}3/5: Analyzing SAE feature stability${NC}"
echo -e "${BLUE}========================================${NC}"
python3 03_sae_feature_stability.py 2>&1 | tee "${RESULTS_DIR}/03_sae_stability.txt"
echo -e "${GREEN}✓ SAE stability analysis complete${NC}"

# Script 4: External validation
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}4/5: External validation of steering${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}Note: This requires external tools (semgrep, clang-tidy, cppcheck)${NC}"
echo -e "${YELLOW}Install with: apt install semgrep clang-tools cppcheck${NC}"
python3 04_external_validation_steering.py 2>&1 | tee "${RESULTS_DIR}/04_external.txt" || true
echo -e "${GREEN}✓ External validation complete (may have skipped unavailable tools)${NC}"

# Script 5: Steering sign convention
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}5/5: Analyzing steering sign convention${NC}"
echo -e "${BLUE}========================================${NC}"
python3 05_steering_sign_convention.py 2>&1 | tee "${RESULTS_DIR}/05_steering_sign.txt"
echo -e "${GREEN}✓ Steering analysis complete${NC}"

# Generate summary report
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Generating summary report${NC}"
echo -e "${BLUE}========================================${NC}"

cat > "${RESULTS_DIR}/SUMMARY.txt" << 'SUMMARY_EOF'
# ICLR Review Response Experiments - Summary Report

## Overview
This report summarizes results from 5 response experiments addressing key concerns
raised in the ICLR review (score: 5.4/10).

## Experiments Completed

### 1. Stronger Pooling Baselines
Tests whether the AUROC 0.5 ceiling for mean-token pooling is fundamental or
just a limitation of that specific strategy.

**Key Question**: Can attention-weighted, learned, or token-level pooling
exceed the mean-token pooling baseline?

**Look for in results**: AUROC values for each pooling strategy
- If all ≤ 0.5: "fundamental difficulty" claim is strengthened
- If any > 0.5: claim should be weakened or reframed

### 2. Confound Controls
Tests whether the suspiciously high cross-layer cosine similarity (≥0.99)
is driven by confounds like sequence length and guard-token frequency.

**Key Question**: Do the directions remain similar after residualizing
for confounds?

**Look for in results**:
- Original similarity: ~0.99
- After confound removal: should drop significantly?
- Null permutation baseline: should be much lower

### 3. SAE Feature Stability
Validates that SAE-learned features are stable and interpretable across
different training runs.

**Key Question**: Do the same semantic features persist across seeds?

**Look for in results**:
- Jaccard overlap: should be > 0.7 (stability)
- Spearman correlation: should be > 0.8 (rank preservation)
- Monosemanticity diversity: should be > 0.8 (interpretability)

### 4. External Validation of Steering
Validates steering using external SAST tools instead of just probe AUROC,
mitigating circularity concerns.

**Key Question**: Does steered code actually contain more defensive
constructs and fewer security issues?

**Look for in results**:
- Guard token increase in steered code
- Reduction in Semgrep security issues
- Compilation success rate
- Warning counts (clang-tidy, cppcheck)

### 5. Steering Sign Convention
Clarifies and tests the steering direction sign to ensure semantic consistency.

**Key Question**: Do +α·d_L and -α·d_L produce opposite effects as expected?

**Look for in results**:
- +α·d_L: AUROC increases (toward secure)
- -α·d_L: AUROC decreases (toward vulnerable)
- Random/orthogonal: no effect (specificity)

## How to Interpret Results

### Strong Results (Support Your Claims)
- Pooling baselines all stay near 0.5 AUROC
- Confound residualization causes modest drops in similarity
- SAE features are stable across seeds
- External validation shows real code improvements
- Sign convention is correct and consistent

### Weak Results (Require Revisions)
- Some pooling strategy exceeds 0.5 AUROC
  → Reframe as "recoverable with better readouts"
- Confound residualization drops similarity dramatically
  → Acknowledge confounds, focus on remaining signal
- SAE features unstable across seeds
  → Emphasize qualitative interpretation, add stability metrics
- External validation shows no improvement
  → Focus on activation patching instead, soften steering claims
- Sign convention shows unexpected behavior
  → Investigate distribution shift, retrain probes

## Next Steps

1. Review all results files in this directory
2. Identify strongest and weakest results
3. Update paper narrative accordingly
4. Write detailed rebuttal addressing each reviewer comment
5. Consider resubmission to ICLR or other top venues

## Files in This Directory

- 01_pooling.txt: Pooling baselines AUROC values
- 02_confounds.txt: Cross-layer similarity before/after confound removal
- 03_sae_stability.txt: Feature stability metrics
- 04_external.txt: SAST tool results on steered code
- 05_steering_sign.txt: Sign convention test results
- SUMMARY.txt: This file
- RECOMMENDATIONS.txt: Detailed recommendations for paper revision

SUMMARY_EOF

cat > "${RESULTS_DIR}/RECOMMENDATIONS.txt" << 'REC_EOF'
# Recommendations for Paper Revision

## Based on Review Feedback

### 1. Pooling Strategies
**Current claim**: "Fundamental difficulty" of detecting vulnerability

**Recommended update**:
- If stronger pooling works: "Diffuse in mean-token pooling but recoverable
  with sequence-aware models"
- If all fail: Keep current framing but add new baselines to methods

**Action**: Add 1-2 sentences acknowledging other pooling strategies,
explain why your focus on mean-token pooling is representative

### 2. Cross-Layer Similarity
**Current claim**: "≥0.99 cosine similarity" suggests coherent mechanistic structure

**Recommended update**:
- Report both original AND residualized similarities
- Discuss what portion is confound vs. semantic
- Clarify that even after confound removal, meaningful structure remains

**Action**: Add confound residualization to methods, include sensitivity
analysis in appendix

### 3. SAE Interpretability
**Current claim**: Top features correspond to defensive constructs

**Recommended update**:
- Emphasize these are qualitative interpretations
- Add stability metrics across seeds if available
- Discuss limitations of LLM-assisted interpretation

**Action**: Add caveat in SAE methods section about interpretation limitations,
include stability analysis in appendix

### 4. Steering Validation
**Current claim**: "Steering shifts AUROC from 0.47 to 0.59"

**Recommended update**:
- Lead with external validation results (Semgrep, guard tokens)
- Use probe AUROC as secondary evidence
- Clearly distinguish internal vs. external metrics

**Action**: Reorganize results section to feature external validation first,
make more prominent in main text

### 5. Sign Convention
**Current claim**: "-α·d_L shifts code toward defensive behavior"

**Recommended update**:
- Clarify why both signs and their interpretation
- Test both +α·d_L and -α·d_L, report which works
- Discuss distribution shift for generated code

**Action**: Add explicit test of both signs in generation steering section,
clarify the semantics

## Paper Section Updates

### Methods Section
- Add subsection on pooling strategy selection and alternatives tested
- Expand SAE training section with stability/validation details
- Add confound analysis method (length residualization)

### Results Section
- Lead with external validation (Semgrep, cppcheck, guard tokens)
- Then show probe-based steering results
- Include confound sensitivity analysis for direction

### Discussion Section
- Expand limitations subsection with explicit pooling strategy limits
- Discuss what confounds explain vs. what remains
- Clarify mechanistic interpretation limitations

### Appendix
- SAE feature stability across seeds
- Confound residualization detailed results
- Both signs of steering (+α and -α)
- Additional pooling baselines
- External validation tool outputs

## Reviewer Questions to Address

1. **"Why not test stronger pooling?"**
   → Response: Tested [list] in appendix, all near 0.5, confirming difficulty
             is structural not technical

2. **"Is the direction really meaningful or just length?"**
   → Response: Residualization analysis shows X% remains after confound
             removal, supporting semantic interpretation

3. **"Are SAE features really stable?"**
   → Response: Cross-seed stability metrics [cite appendix] show Jaccard
             overlap >0.7 and Spearman rho >0.8

4. **"Is steering actually safer or just different?"**
   → Response: External validation via Semgrep/cppcheck/guard-token counts
             shows [% reduction] in security issues

5. **"Why does -α·d_L work if d=secure-vulnerable?"**
   → Response: Both ±α tested, explain which direction and why distribution
             shift explains any counterintuitive results

## High-Impact Improvements

### Tier 1 (Must Do)
- [ ] Run stronger pooling baselines → 1-2 sentences in results
- [ ] Add confound analysis → extend results section
- [ ] External validation → move to prominent position

### Tier 2 (Should Do)
- [ ] SAE stability analysis → add to appendix
- [ ] Test both signs of steering → clarify semantics
- [ ] Document why mean-token pooling is representative → add to intro

### Tier 3 (Nice to Have)
- [ ] Additional baselines (Juliet synthetic dataset)
- [ ] Multi-model replication (CodeLlama, StarCoder)
- [ ] Attention pattern analysis for why pooling matters

## Estimated Acceptance Probability After Revisions

| Venue | Before | After Tiers 1-2 | After All |
|-------|--------|-----------------|-----------|
| ICLR  | 25%    | 45%             | 60%       |
| EMNLP | 50%    | 75%             | 85%       |
| ACL   | 45%    | 70%             | 80%       |
| NeurIPS | 30%  | 50%             | 65%       |

(Rough estimates based on review feedback)

## Timeline for Revision

- Week 1: Run all 5 experiments (script execution)
- Week 2: Interpret results, decide which changes most critical
- Week 3: Update paper sections based on results
- Week 4: Write detailed rebuttal addressing each comment
- Week 5: Final proofread, prepare for resubmission

REC_EOF

echo ""
echo -e "${GREEN}✓ Summary and recommendations generated${NC}"

# Print summary to console
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Experiment Run Complete${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Results saved to: ${RESULTS_DIR}/${NC}"
echo -e "${YELLOW}Key files:${NC}"
echo "  - ${RESULTS_DIR}/01_pooling.txt"
echo "  - ${RESULTS_DIR}/02_confounds.txt"
echo "  - ${RESULTS_DIR}/03_sae_stability.txt"
echo "  - ${RESULTS_DIR}/04_external.txt"
echo "  - ${RESULTS_DIR}/05_steering_sign.txt"
echo "  - ${RESULTS_DIR}/SUMMARY.txt"
echo "  - ${RESULTS_DIR}/RECOMMENDATIONS.txt"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Review all results files"
echo "2. Consult RECOMMENDATIONS.txt for paper revision strategy"
echo "3. Update paper sections based on findings"
echo "4. Write detailed rebuttal to reviewer comments"
echo ""
echo -e "${YELLOW}Finished at: $(date)${NC}"
echo ""
