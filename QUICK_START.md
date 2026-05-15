# Multi-Dataset Pipeline: Quick Start Guide

## 🚀 Run Everything (All Datasets, All Models, All Layers)

```bash
cd /Users/rmelo/Documents/GitHub/SAE-Java-Bug-Detection
./run_pipeline.sh --datasets=deltasecommits,sven,precisebugs
```

**Time**: ~6-9 hours (GPU required)  
**Output**: Activations + Experiments + Figures for all datasets

## ⚡ Just Extract Activations (No Experiments)

```bash
./run_pipeline.sh --datasets=deltasecommits,sven,precisebugs --skip-existing
```

**Time**: ~2-3 hours  
**Output**: Cached activation NPZ files in `sae_java_bug/artifacts/multi_model_probing/`

## 🎨 Generate Figures Only (From Cached Data)

```bash
./run_pipeline.sh --datasets=deltasecommits,sven,precisebugs --figures-only
```

**Time**: ~1-2 minutes  
**Output**: PDF figures in `On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures/`

## 🧪 Custom Configuration

### Different Models
```bash
# Use CodeLlama 13B and StarCoder2 15B
./run_pipeline.sh --datasets=deltasecommits,sven,precisebugs \
                  --models=codellama-13b,starcoder2-15b
```

### Specific Layers Only
```bash
# Just layers 7, 15, 23
./run_pipeline.sh --datasets=deltasecommits,sven,precisebugs \
                  --layers=7,15,23
```

### Single Dataset
```bash
# Just SVEN
./run_pipeline.sh --datasets=sven --figures-only
```

## 📊 Available Options

| Parameter | Default | Example |
|-----------|---------|---------|
| `--datasets` | `deltasecommits` | `deltasecommits,sven,precisebugs` |
| `--models` | `qwen-7b,codellama-7b,starcoder2-7b` | `codellama-13b` |
| `--layers` | `3,7,11,15,19,23,27` | `7,15,23` |
| `--figures-only` | *(skip)* | Use for cached figures |
| `--skip-existing` | *(skip)* | Skip already-computed experiments |
| `--language` | `all` | `c` |

## 📁 What Gets Generated

| Location | Content |
|----------|---------|
| `results/raw_data/` | JSON experiment results |
| `sae_java_bug/artifacts/multi_model_probing/` | Cached activation NPZ files |
| `figures/` | PDF figures (20+) |

## ✅ Pipeline Steps

```
0️⃣ Extract activations (GPUs heavily used)
   ↓
1️⃣ Run experiments (CPU-intensive probing)
   ↓
2️⃣ Generate base figures (fast, ~1 min)
   ↓
3️⃣ Generate per-model figures (fast, ~1 min)
   ↓
4️⃣ Generate critical paper figures (fast, ~1 min)
```

## 💡 Smart Caching

```bash
# First run: Extract activations (2-3 hrs)
./run_pipeline.sh --datasets=sven,precisebugs

# Reuse cached activations for new models (1 hr)
./run_pipeline.sh --datasets=sven,precisebugs \
                  --models=codellama-13b \
                  --figures-only

# Fast figure updates (1 min)
./run_pipeline.sh --datasets=sven,precisebugs --figures-only
```

## 🔧 Troubleshooting

**Out of memory?**
- Reduce models: `--models=qwen-7b`
- Reduce datasets: `--datasets=sven`
- Use CPU: Edit `multi_dataset_activations.py`, set `DEVICE = "cpu"`

**Missing HuggingFace access?**
```bash
huggingface-cli login
```

**Want to skip long activation extraction?**
```bash
# Use --skip-activations to reuse cached data
./run_pipeline.sh --datasets=sven --skip-activations --figures-only
```

## 📖 Full Documentation

For comprehensive details, see: `MULTI_DATASET_PIPELINE.md`

---

**Status**: ✅ Ready to Use  
**Version**: 2.0 (Multi-Dataset)
