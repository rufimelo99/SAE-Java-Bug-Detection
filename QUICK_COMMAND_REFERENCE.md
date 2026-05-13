# Quick Command Reference

## Run the Experiment

```bash
cd /Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection

# Test it first (1 layer, ~5 min)
python run_preference_steering_experiment.py --quick

# Full experiment (6 layers, ~30-60 min)
python run_preference_steering_experiment.py

# If you want CPU only (slower but no GPU needed)
python run_preference_steering_experiment.py --quick --device cpu
```

## What Gets Generated

After running, you'll have:
- `results_preference_steering.json` — Raw measurement data
- `figures/fig_preference_steering.pdf` — Visualization (6 layers)

## Check Results Quickly

```bash
# View results summary
cat results_preference_steering.json | python -m json.tool | head -50

# Open the figure
open figures/fig_preference_steering.pdf
```

## Key Metric: Preference Score

```
preference = log_prob(secure_code) - log_prob(vulnerable_code)

Positive = model prefers SECURE code
Negative = model prefers VULNERABLE code
```

## Expected Pattern

If steering works (direction is causal):

```
Layer 3 (early):     Big effects, clearest trend
Layer 7:             Medium effects
Layer 11-15:         Smaller effects
Layer 19-23 (late):  Minimal effects, possible noise

Pattern: Effect size decreases with layer (early > late)
Reason: More transformer blocks downstream = more propagation
```

## Files You'll Use

```
Main script:
  run_preference_steering_experiment.py     ← RUN THIS

Documentation:
  RUN_PREFERENCE_EXPERIMENT.md              ← Full guide
  QUICK_COMMAND_REFERENCE.md                ← This file

Code pairs tested:
  - strcpy_overflow  (CWE-120: Buffer Overflow)
  - sql_injection    (CWE-89: SQL Injection)
  - null_check       (CWE-476: Null Pointer)
  - array_bounds     (CWE-119: Array OOB)
  - use_after_free   (CWE-416: Use After Free)

Output:
  results_preference_steering.json          ← Data
  figures/fig_preference_steering.pdf       ← Visualization
```

## Flowchart

```
1. Run: python run_preference_steering_experiment.py --quick
   ↓
2. Check output for errors
   ↓
3. If OK → open figures/fig_preference_steering.pdf
   ↓
4. If pattern looks good → run full experiment (without --quick)
   ↓
5. Review final results
   ↓
6. Decide: Include in paper? Which layers?
```

## If Something Goes Wrong

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: torch` | `pip install torch transformers` |
| Out of memory | Use `--device cpu` or `--quick` |
| Takes forever | Use `--quick` to test just 1 layer |
| File not found | Make sure you're in repo root |

## Next Steps

After getting results:

1. **Examine baseline preferences**
   - Open `results_preference_steering.json`
   - Check `"baseline"` section
   - Are all preferences pointing one direction? (Good sign)

2. **Look at visualization**
   - Open `figures/fig_preference_steering.pdf`
   - Does each layer show a trend?
   - Do early layers show bigger effects?

3. **Interpret**
   - If clean pattern → strong evidence, include in paper
   - If mixed → conditional evidence, note limitations
   - If noisy → preliminary evidence, skip or put in appendix

4. **Write up findings**
   - Update paper with results
   - Write figure caption
   - Add to Appendix E.1

---

**Start here:**
```bash
python run_preference_steering_experiment.py --quick
```
