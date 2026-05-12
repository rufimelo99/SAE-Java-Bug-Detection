# Archived Scripts

This folder contains experimental scripts and reviewer response analyses not part of the main pipeline.

## Organization

- **experimental/** - One-off experiment runners
- **reviewer_response/** - Scripts addressing reviewer feedback

## How to Run

Individual experiments:
```bash
bash experimental/[script_name].sh
# or
python experimental/[script_name].py
```

Batch runners:
```bash
bash ../run_all_response_experiments.sh
bash ../run_global_baselines.sh
```

## Why Archived?

These scripts were moved during cleanup to:
1. Reduce clutter in main scripts/ folder
2. Separate main pipeline from experiments
3. Preserve all functionality while improving organization
