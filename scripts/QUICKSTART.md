# Pipeline Quickstart

## One-liner: Run everything

```bash
cd /Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection/scripts && ./run_pipeline.sh
```

Done! Results in `results/` and figures in paper `figures/` directory.

## Common workflows

### Run experiments + generate figures
```bash
./run_pipeline.sh
```

### Just regenerate figures from existing results
```bash
./run_pipeline.sh --figures-only
```

### Run only for Qwen model
```bash
./run_pipeline.sh --models qwen
```

### Run for Qwen and CodeLlama
```bash
./run_pipeline.sh --models qwen,codellama
```

### Force regenerate (overwrite existing)
```bash
./run_pipeline.sh --skip-existing
```

## What gets generated

### Raw data (JSON)
```
results/raw_data/
├── qwen_direction_geometry.json         # Direction alignment stats
├── qwen_cwe_universality.json           # Cross-family transfer rates
├── qwen_paired_ranking.json             # Pairwise ranking accuracy
└── [same for codellama and starcoder2]
```

### Figures (PDF)
```
../On-the-Absence-of-Global-Anomalies.../figures/
├── fig_direction_alignment_heatmaps.pdf    # Layer alignment for all models
├── fig_per_pair_alignment.pdf              # Consistency curve across layers
├── fig_paired_distances.pdf                # Signal magnitude by layer
├── fig_cwe_transfer_heatmaps.pdf           # Cross-family generalization
└── fig_ranking_accuracy.pdf                # Ranking accuracy comparison
```

### Summary
```
results/results/
├── summary.json                         # Aggregated results all models
└── summary_statistics.txt               # Text table of key stats
```

## Key outputs to watch for

**Direction geometry** — Should show:
- Per-pair alignment: 87-88% at layers 3-23
- Cosine similarity: 0.99+ (very stable direction)
- Paired distance: 36x jump L0→L3, plateau L3-23, collapse at L27

**CWE universality** — Should show:
- Cross-family transfer: 80% average (transfers well across CWE families)
- All families show similar pattern (not family-specific)

**Paired ranking** — Should show:
- ~87% accuracy at mid-to-late layers
- Matches per-pair alignment percentages

## Troubleshooting

### "No results for [model]"
The experiment ran but found no data. Check that activation caches exist for that model.

### "ModuleNotFoundError: No module named 'matplotlib'"
```bash
pip install matplotlib numpy
```

### Figures not created
Check that output directory is writable:
```bash
mkdir -p ../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures
chmod 755 $_
```

## Files created

- `run_all_experiments.py` — Experiment runner (can run standalone)
- `generate_all_figures.py` — Figure generator (can run standalone)
- `run_pipeline.sh` — Master orchestrator (recommended entry point)
- `PIPELINE_README.md` — Full documentation
- `QUICKSTART.md` — This file

## Next steps

1. Run pipeline: `./run_pipeline.sh`
2. Check results: `cat ../results/results/summary_statistics.txt`
3. Review figures: Open PDFs in `figures/`
4. Copy to paper: `cp figures/*.pdf ../On-the-Absence-.../figures/`

## Full documentation

See `PIPELINE_README.md` for:
- Detailed experiment descriptions
- Figure explanations
- Configuration options
- How to extend pipeline
