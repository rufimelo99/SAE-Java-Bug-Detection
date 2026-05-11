# Token Position Importance Analysis

**Purpose**: Deepen mechanistic understanding by identifying which token positions carry vulnerability signal.

**Key question**: If vulnerability signal is distributed across the sequence, which positions matter? Where does the model encode security-relevant information?

## What This Does

The analysis computes **per-token projections** onto the vulnerability direction, answering:

1. **Position Importance Heatmap**: Which tokens contribute most to the vulnerability direction at each layer?
2. **Position Alignment**: What % of vulnerable-secure pairs align positively at each position?
3. **Final Token Attention**: Which positions does the final token attend to (explaining why last-token readout fails)?

## Key Insight

If the vulnerability signal is diffuse (distributed across positions), this analysis identifies:
- **Concentrated vs. spread**: Is signal concentrated at a few critical positions, or truly spread?
- **Bottleneck positions**: Which tokens, if removed, would most degrade the signal?
- **Attention mismatch**: Does the final token attend to positions carrying vulnerability signal?

## Running the Analysis

```bash
# From repository root
bash scripts/token_position_importance.sh

# Or directly with Python
conda run -n sae python sae_java_bug/sparse_autoencoders/notebooks/token_position_importance.py
```

**Requirements**:
- GPU with ~14GB VRAM (loading model + extracting per-token activations)
- ~30-45 minutes for full analysis
- Pre-extracted raw residual stream activations (from existing runs)

## Output

### Files Generated

```
artifacts/analysis/token_position/
├── position_importance_summary.json          # Metadata
├── position_importance_L{layer}.json         # Per-layer results
│   ├── position_projections: [seq_len]       # Mean projection per position
│   ├── position_alignment: [seq_len]         # % pairs with positive alignment
│   └── position_std: [seq_len]               # Variability across pairs
├── figures/
│   ├── token_position_importance_heatmap.pdf
│   ├── token_position_alignment_heatmap.pdf
│   └── final_token_attention_heatmap.pdf
```

Also copies figures to the paper repository for direct inclusion.

### Interpretation

#### Token Position Importance Heatmap
- **Top subplot**: Per-token projection magnitude across layers L3–L23
  - Red = tokens that shift vulnerable representations toward vulnerable (positive vulnerability direction)
  - Blue = tokens that shift toward secure
  - White = no signal
- **Bottom subplot**: Collapsed across layers (mean importance per position)

**Read as**: High bars = positions where the vulnerability direction is strongest. If distributed, expect multiple moderate peaks. If concentrated, expect 1–2 sharp peaks.

#### Position Alignment Heatmap
- **Color intensity**: % of vulnerable-secure pairs with positive alignment at that position
- Values >0.5 indicate consistent alignment
- All-green pattern = truly distributed signal (consistent across pairs)
- Checkered pattern = signal concentrated in specific pairs

**Read as**: If we see green everywhere, signal is robustly distributed. If sparse/checkered, signal may be concentration-dependent.

#### Final Token Attention Heatmap
- **Color intensity**: Attention weight from final token to each position
- Explains the **inversion mystery**: final token attends strongly to positions that are more common in *secure* code (defensive constructs, keywords)

**Read as**: Bright spots show where final token looks. If these overlap with vulnerability-signal positions (from importance heatmap), we'd expect last-token readout to work — but the mismatch (attention goes to structural tokens, not vulnerability tokens) explains the failure.

## Mechanistic Insights

This analysis tests three mechanistic hypotheses:

### H1: Signal is Concentrated (Bottleneck at a Few Positions)
- **Prediction**: Importance heatmap shows 1–2 sharp peaks
- **Implication**: Signal is recoverable by identifying critical positions
- **Evidence against distributed hypothesis**

### H2: Signal is Distributed (Across Many Positions)
- **Prediction**: Importance heatmap shows broad/smooth profile
- **Implication**: Requires aggregation (mean-token pooling)
- **Supports main paper findings**

### H3: Attention Mismatch (Final Token Sees Wrong Positions)
- **Prediction**: Attention heatmap peaks ≠ position importance peaks
- **Implication**: Last-token readout fails because model attends to structural (safe-enriched) tokens
- **Explains**: Why attention-weighted pooling inverts the signal (results in Appendix~\ref{app:advanced_pooling})

## Expected Results

Based on the paper's claim that vulnerability is distributed:

1. **Position importance heatmap**: Should NOT show sharp peaks — expect broadly distributed projections
2. **Position alignment**: Should be consistently 0.5–0.6 across positions (not concentrated above 0.8)
3. **Final token attention**: Should have different peaks than position importance (explains the readout mismatch)

## Extending This Analysis

### Additional diagnostics (future work):

1. **Token type analysis**: Are specific syntactic types (keywords, operators, identifiers) more important?
   - Requires semantic tokenization or parsing
   
2. **Relative position importance**: Positions relative to the vulnerability site vs. absolute
   - Currently uses absolute; requires alignment to diff boundaries
   
3. **Position gradient**: Which positions, if ablated, most degrade the signal?
   - Requires ablation sweeps (expensive)

4. **Cross-pair consistency**: Is the same set of positions important for all vulnerable-secure pairs?
   - Currently averages; requires per-pair analysis

5. **Orthogonal structure**: Beyond the vulnerability direction, what else varies across positions?
   - Requires CCA or full PCA structure analysis

## Troubleshooting

**Out of Memory**: Reduce `MAX_TOKENS` or process in smaller batches

**No C samples found**: Verify raw activation JSONL has `file_extension: ".c"` records

**Missing model**: Ensure `Qwen/Qwen2.5-7B-Instruct` can be downloaded from HuggingFace

**Misaligned sequences**: Per-token activations are padded to match max length within each pair; check `position_importance_summary.json` for reported `max_seq_len`

## Citation

This analysis deepens the mechanistic understanding in:

> *What Code LLMs Know: Functional Identity Without Security Status*
> EMNLP 2026 submission
> 
> Token position importance analysis addresses mechanistic deepening (c):
> "Why is the signal distributed? Which positions carry vulnerability signal?"
