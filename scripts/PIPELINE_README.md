# Vulnerability Representation Analysis Pipeline

This directory contains a complete pipeline for running mechanistic interpretability experiments on three 7B code language models (Qwen2.5-7B-Instruct, CodeLlama-7B-Instruct, StarCoder2-7B) and generating publication-quality figures.

## Overview

The pipeline consists of three main components:

1. **run_all_experiments.py** — Runs all mechanistic experiments for specified models
2. **generate_all_figures.py** — Generates publication-quality figures from experiment results
3. **run_pipeline.sh** — Master orchestration script that chains everything together

## Quick Start

### Run the complete pipeline:

```bash
cd /Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection/scripts
./run_pipeline.sh
```

This will:
1. Run experiments for all three models (Qwen, CodeLlama, StarCoder2)
2. Store raw JSON results in `results/raw_data/`
3. Generate summary statistics
4. Generate all publication figures in `../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures/`

### Run only for specific models:

```bash
./run_pipeline.sh --models qwen,codellama
```

### Skip existing results and regenerate:

```bash
./run_pipeline.sh --skip-existing
```

### Generate figures only (from existing results):

```bash
./run_pipeline.sh --figures-only
```

## Experiments

The pipeline runs three categories of experiments:

### 1. Direction Geometry Analysis

Analyzes how vulnerability is represented as a geometric direction in activation space across layers.

**Outputs:**
- `{model}_direction_geometry.json` containing:
  - Per-layer alignment percentages (% of pairs with vulnerable > secure projection)
  - Mean and std of paired distances
  - Cross-layer cosine similarities

**Key metrics:**
- Cosine similarity between directions at different layers (should be ~0.986-0.999)
- Per-pair alignment (should be ~87-88% at mid-to-late layers)
- Paired distance magnitude (should jump 36x from L0 to L3)

### 2. CWE Universality Analysis

Tests whether vulnerability direction trained on one CWE family transfers to other families.

**Outputs:**
- `{model}_cwe_universality.json` containing:
  - Cross-family transfer rates (mean, std, min, max)
  - Family-specific statistics

**Key metrics:**
- Transfer rate across families (should be ~80% on average)
- Whether signal is bug-family-specific or universal

### 3. Paired Ranking Task

Evaluates whether the vulnerability direction correctly ranks vulnerable code higher than secure code.

**Outputs:**
- `{model}_paired_ranking.json` containing:
  - Pairwise ranking accuracy by layer
  - Number of vulnerable and secure samples

**Key metrics:**
- Ranking accuracy (should match per-pair alignment, ~87-88% at mid-to-late layers)

## Output Structure

```
results/
├── raw_data/
│   ├── qwen_direction_geometry.json
│   ├── qwen_cwe_universality.json
│   ├── qwen_paired_ranking.json
│   ├── codellama_direction_geometry.json
│   ├── codellama_cwe_universality.json
│   ├── codellama_paired_ranking.json
│   ├── starcoder2_direction_geometry.json
│   ├── starcoder2_cwe_universality.json
│   └── starcoder2_paired_ranking.json
└── results/
    └── summary.json

figures/
├── fig_direction_alignment_heatmaps.pdf
├── fig_per_pair_alignment.pdf
├── fig_paired_distances.pdf
├── fig_cwe_transfer_heatmaps.pdf
├── fig_ranking_accuracy.pdf
└── summary_statistics.txt
```

## Generated Figures

### fig_direction_alignment_heatmaps.pdf
Cosine similarity between vulnerability directions at different layers, for all three models. Shows that the direction is stable across layers L3-L23 (cosine ~0.99) but collapses at L27.

### fig_per_pair_alignment.pdf
Percentage of vulnerable-secure pairs that align with the vulnerability direction across layers. Should show:
- ~50% alignment at L0 (random)
- ~87-88% alignment at L3-L23 (consistent)
- ~70% alignment at L27 (collapse)

### fig_paired_distances.pdf
Mean L2 distance between vulnerable and secure representations across layers. Log scale plot should show:
- 36x jump from L0 to L3
- Plateau from L3-L23
- Slight decrease at L27

### fig_cwe_transfer_heatmaps.pdf
Cross-family vulnerability direction transfer rates for all three models. Shows whether a direction learned on one CWE family (e.g., memory safety) transfers to other families (e.g., injection). Values should be in 70-90% range.

### fig_ranking_accuracy.pdf
Pairwise ranking accuracy across layers for all models. Should match per-pair alignment percentages and show the L0→L3 emergence and L27 collapse pattern.

### summary_statistics.txt
Text table with key statistics:
- Model name
- Number of vulnerable-secure pairs
- Peak alignment percentage and which layer
- Mean alignment across layers L3-L23

## Configuration

### Models

Three models are analyzed:
- **Qwen**: Qwen2.5-7B-Instruct
- **CodeLlama**: CodeLlama-7B-Instruct
- **StarCoder2**: StarCoder2-7B

### Layers

Standard 8 layers analyzed: `[0, 3, 7, 11, 15, 19, 23, 27]`

### CWE Families

Five vulnerability families are analyzed:
- **memory_safety**: CWE-119, CWE-120, CWE-125, CWE-476, CWE-787, CWE-416
- **injection**: CWE-20, CWE-22, CWE-78, CWE-89
- **resource**: CWE-401, CWE-399, CWE-415, CWE-362, CWE-400
- **info_disclosure**: CWE-200
- **control_flow**: CWE-190, CWE-264

## Extending the Pipeline

### Adding a new experiment:

1. Add experiment method to `ExperimentPipeline` class in `run_all_experiments.py`:

```python
def run_my_experiment(self, model: str, activations: Dict, labels: Dict) -> Dict:
    """Description of experiment."""
    results = {
        'model': model,
        'experiment': 'my_experiment',
        # ... compute results ...
    }
    self._save_result(model, 'my_experiment', results)
    return results
```

2. Call it in `run_all_experiments()` method

3. Add corresponding figure generation method to `FigureGenerator` class

### Adding a new figure:

1. Add method to `FigureGenerator` class in `generate_all_figures.py`:

```python
def generate_my_figure(self):
    """Generate my custom figure."""
    # Load results
    # Create plot
    # Save
    plt.savefig(self.output_dir / 'fig_myname.pdf', dpi=300, bbox_inches='tight')
```

2. Call it from `generate_all_figures()` method

## Troubleshooting

### No results generated

Check that activation data is properly loaded. You may need to modify the `ExperimentPipeline` to load activations from your cache. See the TODO in `run_all_experiments.py`.

### Figure generation fails

Ensure matplotlib is installed and writable to output directory:

```bash
pip install matplotlib
mkdir -p /Users/rmelo/Documents/GitHub/On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures
```

### Out of memory

The scripts process all layers in memory. For very large datasets, modify to process layers one at a time or use memory mapping.

## Performance Notes

- **Experiments**: ~5-10 minutes per model (depends on dataset size)
- **Figure generation**: < 1 minute for all figures
- **Total time**: ~20-30 minutes for full pipeline on all three models

## Output for Paper

All generated PDF figures are ready for publication:

```bash
# Copy to paper directory
cp figures/*.pdf ../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures/
```

Raw JSON data can be used for:
- Supplementary material tables
- Additional analyses
- Reproducibility/replication

## Related Files

- **run_all_experiments.py** — Main experiment orchestrator
- **generate_all_figures.py** — Figure generation
- **run_pipeline.sh** — Shell orchestration wrapper
- **multimodel_c_analysis.py** — Legacy multi-model analysis (superseded by this pipeline)
- **paired_ranking_task.py** — Original paired ranking implementation (reference)

## Citation

If you use this pipeline, please cite:

```
On the Absence of Global Anomalies in Vulnerable Code Representations
Rui Melo, André Catarino, Cláudia Mamede, Rui Abreu, Corina Păsăreanu
[Paper details TBD]
```
