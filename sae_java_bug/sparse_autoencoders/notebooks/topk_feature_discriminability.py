"""Per-feature discriminability analysis for TopK SAE (Layer 11)."""
import ctypes
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")

TOPK = Path(__file__).parents[2] / "artifacts/activations/TOPK_tensors"


def to_numpy(t):
    t = t.contiguous().cpu().float()
    return np.frombuffer(
        (ctypes.c_float * t.numel()).from_address(t.data_ptr()), dtype=np.float32
    ).copy().reshape(t.shape)


v = to_numpy(torch.load(TOPK / "vulnerable_layer_11.pt", map_location="cpu", weights_only=True).float())
s = to_numpy(torch.load(TOPK / "safe_layer_11.pt", map_location="cpu", weights_only=True).float())
print(f"v={v.shape}, s={s.shape}")

f_v = (v > 0).mean(axis=0)
f_s = (s > 0).mean(axis=0)
delta_f = f_v - f_s
mean_v = v.mean(axis=0)
mean_s = s.mean(axis=0)
pooled_std = np.sqrt(((v.std(axis=0) ** 2) + (s.std(axis=0) ** 2)) / 2) + 1e-9
cohens_d = (mean_v - mean_s) / pooled_std

print(f"\nFiring rate stats across all {v.shape[1]} features:")
print(f"  mean f_v: {f_v.mean():.5f}  mean f_s: {f_s.mean():.5f}")
print(f"  max f_v: {f_v.max():.4f}  max f_s: {f_s.max():.4f}")
print(f"  max |delta_f|: {np.abs(delta_f).max():.5f}")
print(f"  |delta_f| > 1%:   {(np.abs(delta_f) > 0.01).sum()}")
print(f"  |delta_f| > 0.5%: {(np.abs(delta_f) > 0.005).sum()}")
print(f"  |delta_f| > 0.2%: {(np.abs(delta_f) > 0.002).sum()}")

top_by_df = np.argsort(-np.abs(delta_f))[:30]
print(f"\n── Top 30 features by |delta_f| ──")
print(f"{'Feat':>6}  {'f_v':>7}  {'f_s':>7}  {'delta_f':>9}  {'mean_v':>8}  {'mean_s':>8}  {'cohen_d':>8}")
for i in top_by_df:
    print(f"{i:>6}  {f_v[i]:.5f}  {f_s[i]:.5f}  {delta_f[i]:>+9.5f}  {mean_v[i]:>8.4f}  {mean_s[i]:>8.4f}  {cohens_d[i]:>+8.4f}")

top200 = np.argsort(-np.abs(delta_f))[:200]
print(f"\nComputing AUROCs for top 200 by |delta_f|...")
results = []
for i in top200:
    u_v, _ = mannwhitneyu(v[:, i], s[:, i], alternative="greater")
    auc = u_v / (len(v) * len(s))
    direction = "vuln" if auc >= 0.5 else "secure"
    auc = max(auc, 1 - auc)
    results.append((i, auc, delta_f[i], cohens_d[i], direction))

results.sort(key=lambda x: -x[1])
print(f"\n── Top 20 by AUROC ──")
print(f"{'Feat':>6}  {'AUROC':>6}  {'delta_f':>9}  {'cohen_d':>8}  {'f_v':>7}  {'f_s':>7}  {'dir':>6}")
for feat, auc, df, cd, direction in results[:20]:
    print(f"{feat:>6}  {auc:.4f}  {df:>+9.5f}  {cd:>+8.4f}  {f_v[feat]:.5f}  {f_s[feat]:.5f}  {direction:>6}")

print(f"\nMax single-feature AUROC: {results[0][1]:.4f}")
print(f"AUROC > 0.55: {sum(1 for _, a, *_ in results if a > 0.55)}")
print(f"AUROC > 0.60: {sum(1 for _, a, *_ in results if a > 0.60)}")
print(f"AUROC > 0.65: {sum(1 for _, a, *_ in results if a > 0.65)}")
