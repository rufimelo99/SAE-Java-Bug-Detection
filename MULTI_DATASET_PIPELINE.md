# Multi-Dataset Mechanistic Interpretability Pipeline

Complete pipeline for extracting activations and running probing experiments on multiple C code vulnerability datasets (DeltaSecommits, SVEN, PreciseBugs) with flexible model and layer selection.

## Quick Start

### Basic Usage (DeltaSecommits only)
```bash
cd /Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection
./scripts/run_pipeline.sh
```

### Multi-Dataset Analysis
```bash
# Run on SVEN and PreciseBugs (in addition to DeltaSecommits)
./scripts/run_pipeline.sh --datasets=deltasecommits,sven,precisebugs

# Run figures only (cached activations)
./scripts/run_pipeline.sh --datasets=deltasecommits,sven,precisebugs --figures-only
```

### Custom Models and Layers
```bash
# Use different models
./scripts/run_pipeline.sh --models=qwen-7b,qwen-coder-7b --layers=3,7,11,15,19,23,27

# Extract only specific layers
./scripts/run_pipeline.sh --layers=7,15,23

# Combine dataset and model options
./scripts/run_pipeline.sh --datasets=sven,precisebugs --models=codellama-7b,starcoder2-7b --layers=3,7,11,15,19,23,27
```

## Configuration Parameters

### `--datasets`
Available datasets:
- `deltasecommits` (default) — DeltaSecommits C code pairs
- `sven` — SVEN C vulnerability dataset
- `precisebugs` — PreciseBugs vulnerability dataset

Specify as comma-separated list (no spaces):
```bash
--datasets=deltasecommits,sven,precisebugs  # All three
--datasets=sven                              # Just SVEN
--datasets=precisebugs,deltasecommits        # Custom order
```

### `--models`
Available models:
- `qwen-7b` — Qwen2.5-7B-Instruct (default)
- `qwen-coder-7b` — Qwen2.5-Coder-7B-Instruct
- `codellama-7b` — CodeLlama-7B-Instruct (default)
- `codellama-13b` — CodeLlama-13B-Instruct
- `starcoder2-7b` — StarCoder2-7B (default)
- `starcoder2-15b` — StarCoder2-15B

Default: `qwen-7b,codellama-7b,starcoder2-7b`

```bash
--models=qwen-7b,codellama-7b,starcoder2-7b  # All three 7B models
--models=codellama-13b                        # Just CodeLlama-13B
--models=qwen-7b,starcoder2-15b               # Mixed sizes
```

### `--layers`
Specify which transformer layers to extract activations from:
- Default: `3,7,11,15,19,23,27`
- Common choices:
  - `3,7,11,15,19,23,27` — Full range (8 layers, default)
  - `7,15,23` — Early, middle, late
  - `3,11,19,27` — Every 8 layers

```bash
--layers=3,7,11,15,19,23,27  # Full spectrum
--layers=7,15,23              # Three strategic layers
--layers=19,23,27             # Late layers only
```

### Other Options
- `--skip-existing` — Skip existing experiment results
- `--skip-activations` — Don't re-extract activations (use cached)
- `--figures-only` — Generate figures from cached data (fastest)
- `--language=c` — Filter to specific language (default: all)

## Pipeline Structure

The pipeline runs 5 steps for each dataset-model combination:

```
Step 0: Extract Activations (for all datasets/models/layers)
  ↓
Step 1: Run Mechanistic Experiments
  - Direction geometry: vulnerability direction in activation space
  - CWE universality: transfer across vulnerability types
  - Paired ranking: accuracy of paired vulnerability comparison
  ↓
Step 2: Generate Base Figures
  - Per-pair alignment curves
  - Paired distances
  - CWE transfer heatmaps
  ↓
Step 3: Generate Per-Model Styled Figures
  - CWE pairwise heatmaps (seaborn styled)
  - Direction alignment curves
  ↓
Step 4: Generate Critical Paper Figures
  - Pairwise CWE-type probe AUROC heatmaps
  - Direction steering causal validation plots (if steering data available)
  ↓
Step 5: Steering Experiments (optional, requires PyTorch)
  - Direction steering causal intervention
  - Effect size on model preference
```

## Output Structure

Results organized by dataset in `results/raw_data/`:
```
results/raw_data/
├── deltasecommits_cwe_universality.json
├── deltasecommits_direction_geometry.json
├── deltasecommits_paired_ranking.json
├── sven_cwe_universality.json
├── sven_direction_geometry.json
├── sven_paired_ranking.json
├── precisebugs_cwe_universality.json
├── precisebugs_direction_geometry.json
└── precisebugs_paired_ranking.json
```

Activations cached in `sae_java_bug/artifacts/multi_model_probing/`:
```
activations_deltasecommits_qwen-7b.npz
activations_deltasecommits_codellama-7b.npz
activations_deltasecommits_starcoder2-7b.npz
activations_sven_qwen-7b.npz
activations_sven_codellama-7b.npz
...
```

Figures in `On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures/`:
```
fig_per_pair_alignment.pdf
fig_cwe_pairwise_qwen.pdf
fig_cwe_pairwise_codellama.pdf
fig_direction_alignment_starcoder2.pdf
...
```

## Smart Caching

### Activation Caching
- Activations extracted once and cached as NPZ files
- Subsequent runs with `--skip-activations` or `--figures-only` reuse cached activations
- Use `--force-activations` to recalculate even if cached

### Experiment Result Caching
- Step 1 checks if JSON results exist for all models
- If complete, skips re-running experiments
- Use `--skip-existing` to skip models with existing results

### When Caching Helps
```bash
# First run: Extract activations and run experiments (~2-3 hours)
./run_pipeline.sh --datasets=deltasecommits,sven,precisebugs \
                  --models=qwen-7b,codellama-7b,starcoder2-7b

# Regenerate figures from cached data (~1 minute)
./run_pipeline.sh --datasets=deltasecommits,sven,precisebugs \
                  --figures-only

# Add new models to existing datasets (~1 hour)
./run_pipeline.sh --datasets=deltasecommits,sven,precisebugs \
                  --models=codellama-13b,starcoder2-15b \
                  --figures-only
```

## Common Workflows

### Scenario 1: Reproduce DeltaSecommits Results
```bash
# Extract with 7B models
./run_pipeline.sh --models=qwen-7b,codellama-7b,starcoder2-7b

# Or use defaults (same as above)
./run_pipeline.sh
```

### Scenario 2: Cross-Dataset Validation
```bash
# Extract activations for all three datasets with all 7B models
./run_pipeline.sh --datasets=deltasecommits,sven,precisebugs

# This generates separate results for each dataset
# Figures automatically organized by dataset/model
```

### Scenario 3: Layer Ablation Study
```bash
# Extract select layers only
./run_pipeline.sh --layers=3,7,11,15,19,23,27

# Or early vs. late layers
./run_pipeline.sh --datasets=deltasecommits --layers=3,7,11
./run_pipeline.sh --datasets=deltasecommits --layers=15,19,23,27
```

### Scenario 4: Model Scaling
```bash
# Compare 7B vs 13B/15B models across datasets
./run_pipeline.sh --datasets=deltasecommits,sven,precisebugs \
                  --models=qwen-7b,codellama-13b,starcoder2-15b
```

### Scenario 5: Fast Figure Regeneration
```bash
# After activations/experiments are cached:
./run_pipeline.sh --datasets=deltasecommits,sven,precisebugs --figures-only

# Takes ~1 minute, regenerates all publication figures
```

## Troubleshooting

### "Data file not found" (DeltaSecommits)
Check that activation data exists:
```bash
ls /Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection/sae_java_bug/artifacts/activations/TO_UPLOAD/
```

### "Metadata not found" (SVEN/PreciseBugs)
Verify dataset metadata files:
```bash
ls /Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection/sae_java_bug/artifacts/activations/sven_c_only/split_meta.json
ls /Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection/sae_java_bug/artifacts/activations/precisebugs_c_only/split_meta.json
```

### Model not found on HuggingFace
Ensure you have HuggingFace credentials configured:
```bash
huggingface-cli login
```

### Out of memory during activation extraction
Reduce batch size or use gradient checkpointing (edit `multi_dataset_activations.py`)

### CUDA out of memory
Try with CPU:
```python
# Edit multi_dataset_activations.py, line ~40:
DEVICE = "cpu"  # Force CPU instead of CUDA
```

## Advanced Usage

### Extract Activations Only (No Experiments)
```bash
# Just activation extraction, skip all experiments
python -m sae_java_bug.evaluation.multi_dataset_activations \
    --datasets=deltasecommits,sven,precisebugs \
    --models=qwen-7b,codellama-7b,starcoder2-7b \
    --layers=3,7,11,15,19,23,27
```

### Run Experiments Only (Skip Activation Extraction)
```bash
./run_pipeline.sh --skip-activations --figures-only
```

### Custom Data Sources
To add a new dataset, edit `sae_java_bug/evaluation/multi_dataset_activations.py`:

1. Add to `DATASETS` dict:
```python
DATASETS = {
    "mydataset": {
        "description": "My dataset description",
        "data_source": Path("path/to/data.json"),
        "output_dir": REPO_ROOT / "sae_java_bug/artifacts/multi_model_probing",
    }
}
```

2. Add loader function `load_mydataset_pairs()` following the pattern in the file

3. Call it in `load_pairs()` section

## Performance

Typical runtime estimates (per dataset × model):

| Operation | Runtime | Notes |
|-----------|---------|-------|
| Activation extraction | 1-2 hours | GPU required, 7B model, ~2.5K pairs |
| Experiments (1 dataset) | 30-45 min | Probing + direction geometry, all 8 layers |
| Figure generation | 1-2 min | From cached results |
| Full pipeline (1 dataset) | 2-3 hours | Everything from scratch |
| All 3 datasets + 3 models | ~20 hours | Full extraction + experiments |

## References

### Code Structure
- Activation extraction: `sae_java_bug/evaluation/multi_dataset_activations.py`
- Probing experiments: `sae_java_bug/evaluation/multi_model_probing.py` 
- Figure generation: `scripts/generate_all_figures.py`
- Pipeline orchestrator: `scripts/run_pipeline.sh`

### Data Sources
- DeltaSecommits: `sae_java_bug/artifacts/activations/TO_UPLOAD/`
- SVEN: `sae_java_bug/artifacts/activations/sven_c_only/`
- PreciseBugs: `sae_java_bug/artifacts/activations/precisebugs_c_only/`

## Next Steps

1. **Run full pipeline**: `./run_pipeline.sh --datasets=deltasecommits,sven,precisebugs`
2. **Monitor progress**: Check `results/raw_data/` for JSON files
3. **Generate figures**: `./run_pipeline.sh --datasets=deltasecommits,sven,precisebugs --figures-only`
4. **Analyze results**: Compare metrics across datasets in generated figures

---

**Last Updated**: 2026-05-15  
**Pipeline Version**: 2.0 (Multi-Dataset)
