# Running the Preference Steering Experiment

## What This Tests

Instead of measuring absolute vulnerability probability, we test whether steering changes the model's **preference between secure and vulnerable code**.

For each code pair:
```
preference = log_prob(secure_code) - log_prob(vulnerable_code)
```

**Expected behavior if steering works:**
- Steering toward secure (α < 0): preference should INCREASE (more positive)
- Steering toward vulnerable (α > 0): preference should DECREASE (more negative)
- Baseline (α = 0): some neutral preference

## Quick Start

### Option 1: Quick Test (Debug, ~5 min)
Test with just 1 layer to verify everything works:
```bash
cd /Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection
python run_preference_steering_experiment.py --quick
```

Output:
- `results_preference_steering.json` — Raw data
- `figures/fig_preference_steering.pdf` — Visualization

### Option 2: Full Experiment (All 6 layers, ~30-60 min)
```bash
python run_preference_steering_experiment.py
```

### Option 3: Force CPU (Slower but no GPU needed)
```bash
python run_preference_steering_experiment.py --device cpu --quick
```

## What You'll See

### Console Output

**Step 1: Baseline (no steering)**
```
▶ strcpy_overflow
  '// Write C code:': sec=-0.924, vuln=-1.045, pref=+0.121 → SECURE
  '// Function:': sec=-0.935, vuln=-1.052, pref=+0.117 → SECURE
  '// Code:': sec=-0.918, vuln=-1.038, pref=+0.120 → SECURE
  RESULT: PREFERS SECURE (Δ = +0.119)
```

**Step 2: Steering at each layer**
```
▶ Layer 3
  ✓ Computed from 1368 C pairs
  ... (preference measurements for each alpha)
```

### Output Files

1. **results_preference_steering.json**
   - Structured results with baseline and steering effects
   - Raw numbers for each code pair × layer × alpha

2. **figures/fig_preference_steering.pdf**
   - 6 subplots (one per layer)
   - X-axis: steering strength (α)
   - Y-axis: preference score
   - Each line = one code pair type

## Interpreting Results

### Baseline Preference
```json
"baseline": {
  "strcpy_overflow": {
    "preference": +0.119
  }
}
```
- **+0.119**: Model prefers SECURE code ✓ (good, room to steer either way)
- **-0.050**: Model prefers VULNERABLE code ✓ (also good, room to steer)
- **±0.005**: No clear preference ⚠️ (harder to see steering effect)

### Steering Effect
If the direction works:
- **Layer 7, α=-20**: preference should be MOST POSITIVE
- **Layer 7, α=0**: preference at baseline
- **Layer 7, α=+20**: preference should be MOST NEGATIVE (or least positive)

Pattern across layers:
- **Early layers (3, 7)**: Bigger steering effects (more propagation time)
- **Late layers (19, 23)**: Smaller steering effects (less propagation)

## Troubleshooting

### Memory Error
- Use `--quick` flag to test fewer layers
- Use `--device cpu` (slower but uses RAM instead of VRAM)
- Reduce `MAX_LENGTH` in the script (line 65)

### Takes too long
- Use `--quick` to test with just Layer 3
- Or reduce number of prompts (modify `PROMPTS` list in script)

### Model won't load
Make sure you have transformers installed:
```bash
pip install transformers torch
```

## Expected Runtime

| Config | Time | GPU Memory |
|--------|------|-----------|
| `--quick` (1 layer, 3 alphas) | 5-10 min | ~20GB |
| Full (6 layers, 5 alphas) | 30-60 min | ~20GB |
| `--device cpu` | 2-3x slower | ~8GB |

## What Happens Next

After running:

1. **Check baseline preferences** in `results_preference_steering.json`
   - Are they consistently pointing one direction?
   - Is the effect size reasonable (±0.05 to ±0.20)?

2. **Look at the PDF figure**
   - Do lines show clear trends?
   - Are early layers different from late layers?
   - Do they show expected layer decay?

3. **Decide on paper inclusion**
   - If clean pattern: include as main evidence
   - If mixed: include with caveats
   - If noisy: might skip or note as "preliminary evidence"

## Key Test Cases

The experiment tests 5 vulnerability types:

| Code | Secure Version | Vulnerable Version | CWE |
|------|---|---|---|
| **strcpy_overflow** | `strncpy + bounds check` | `strcpy(dest, src)` | CWE-120 |
| **sql_injection** | Prepared statements | `sprintf(sql, "...%s...")` | CWE-89 |
| **null_check** | `if (str == NULL)` | Direct `strlen(str)` | CWE-476 |
| **array_bounds** | `if (index >= 0 && ...)` | Unchecked `array[index]` | CWE-119 |
| **use_after_free** | Use then free | Free then use | CWE-416 |

---

**Ready to run? Start with:**
```bash
python run_preference_steering_experiment.py --quick
```
