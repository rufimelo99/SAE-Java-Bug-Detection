# Heatmap Generation Pipeline

## Overview

The paper heatmaps are now integrated into the main analysis pipeline via `generate_paper_heatmaps.py`. This unified script generates all analysis heatmaps with consistent styling matching the CWE pairwise probe heatmaps.

## Generated Heatmaps

### 1. Multi-model Alignment Heatmap
- **File**: `fig_multimodel_alignment_heatmap.pdf`
- **Data**: Per-pair alignment percentages across layers L3–L27 and models (Qwen-7B, CodeLlama-7B, StarCoder2-7B)
- **Color scale**: 0–1 (red = low alignment, yellow = medium, green = high alignment)
- **Usage**: Figure~\ref{fig:multimodel_alignment} in results.tex

### 2. Ranking Accuracy Heatmap
- **File**: `fig_ranking_accuracy_heatmap.pdf`
- **Data**: Ranking accuracy percentages (58% ± 0.4%) across layers and models
- **Color scale**: 0–1 (red = low accuracy, green = high accuracy)
- **Usage**: Figure~\ref{fig:ranking_accuracy} in results.tex

## Styling Consistency

All heatmaps use:
- **Colormap**: RdYlGn (red-yellow-green)
- **Font**: Serif, size 10pt
- **Color scale**: 0–1 (standardized across all visualizations)
- **DPI**: 150 (publication quality)
- **Axis labels**: Layer (rows) × Model (columns)
- **No colorbars**: Text annotations show exact values

This matches the styling of `generate_pairwise_cwe_probes.py` for visual consistency.

## Pipeline Integration

The heatmap generation is now Step 4 in `run_pipeline.sh`:

```bash
./scripts/run_pipeline.sh [--models=...] [--datasets=...] [--figures-only]
```

### Pipeline Steps
1. **Step 0**: Compute activations
2. **Step 1**: Run mechanistic experiments
3. **Step 2**: Generate base figures
4. **Step 3**: Generate multi-model styled figures
5. **Step 4**: Generate paper analysis heatmaps ← **NEW**
6. **Step 5**: Generate critical paper figures (CWE probes)
7. **Step 6**: Run steering experiments

## Usage

### Generate heatmaps standalone:
```bash
python scripts/generate_paper_heatmaps.py
```

### Generate as part of full pipeline:
```bash
./scripts/run_pipeline.sh --figures-only
```

### Generate heatmaps with specific models:
```bash
./scripts/run_pipeline.sh --models=qwen-7b,codellama-7b,starcoder2-7b --figures-only
```

## Output

All heatmaps are saved to:
```
On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures/
```

And automatically copied to the paper build directory.

## Customization

To modify heatmap styling, edit `generate_paper_heatmaps.py`:

```python
def configure_style():
    """Configure matplotlib for paper-quality heatmaps."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "figure.dpi": 150,
        # ... other settings
    })
```

To add new heatmaps:
1. Create a new function `generate_xxx_heatmap(output_dir)`
2. Follow the same data normalization (divide by 100) and styling
3. Call it from `main()`
4. Use vmin=0.0, vmax=1.0 for consistent color scaling

## LaTeX Integration

The heatmaps are referenced in `sections/results.tex`:

```latex
\begin{figure}[!htbp]
\centering
\includegraphics[width=0.55\linewidth]{figures/fig_multimodel_alignment_heatmap.pdf}
\caption{...}
\label{fig:multimodel_alignment}
\end{figure}
```

Update captions as needed for manuscript revisions.
