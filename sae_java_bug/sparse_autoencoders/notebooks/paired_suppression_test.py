"""
paired_suppression_test.py
==========================
Two analyses that strengthen the negative-space encoding claim beyond the
population-level 3.7× feature asymmetry already reported in the paper.

Analysis 1 — Within-pair paired suppression test
-------------------------------------------------
For each of the 2493 (vuln_i, secure_i) pairs, compute the fraction of
secure-enriched features (Δf < 0) where secure_i[k] > vuln_i[k].
If the 3.7× asymmetry is a genuine pair-level phenomenon, this fraction
should be reliably > 0.5 across most pairs.

Analysis 2 — Activation magnitude asymmetry
--------------------------------------------
For every feature k, plot mean_vuln[k] vs mean_secure[k].
Secure-enriched features (Δf < 0) should cluster above the diagonal;
vuln-enriched features below it.

Outputs
-------
  artifacts/activations/paired_suppression/
      paired_suppression_results.json   — summary statistics
      fig_paired_suppression.pdf        — two-panel figure
  Paper figures dir /
      fig_paired_suppression.pdf
"""

import json
import warnings
from pathlib import Path

import ctypes
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
ARTIFACTS    = SCRIPT_DIR.parent.parent / "artifacts" / "activations"
PAPER_FIGS   = SCRIPT_DIR.parent.parent.parent.parent / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations" / "figures"

SAE_FEAT_DIR = ARTIFACTS / "run_20260218_134529_vulnerable_code_qwen_coder_standard_16384_10M"
OUT_DIR      = ARTIFACTS / "paired_suppression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
def to_numpy(t: torch.Tensor) -> np.ndarray:
    """torch → numpy without relying on torch.Tensor.numpy() (NumPy 2.x compat)."""
    t = t.contiguous().cpu().float()
    arr = np.frombuffer(
        (ctypes.c_float * t.numel()).from_address(t.data_ptr()),
        dtype=np.float32
    ).copy()
    return arr.reshape(t.shape)


# ── Load activations ──────────────────────────────────────────────────────────
print("Loading SAE feature activations …")
vuln = torch.load(SAE_FEAT_DIR / "vulnerable_layer_11.pt",
                  map_location="cpu", weights_only=True).float()  # [2493, 16384]
safe = torch.load(SAE_FEAT_DIR / "safe_layer_11.pt",
                  map_location="cpu", weights_only=True).float()  # [2493, 16384]
print(f"  vuln: {tuple(vuln.shape)}  safe: {tuple(safe.shape)}")

N, D = vuln.shape

# ── Δf: identify secure-enriched features ────────────────────────────────────
f_vuln  = (vuln > 0).float().mean(dim=0)   # [D]
f_safe  = (safe > 0).float().mean(dim=0)   # [D]
delta_f = to_numpy(f_vuln - f_safe)         # negative = secure-enriched

sec_mask  = delta_f < 0   # [D] bool — 12,887 features
vuln_mask = delta_f > 0   # [D] bool — 3,445 features

n_sec  = sec_mask.sum()
n_vuln = vuln_mask.sum()
print(f"\nFeature asymmetry: {n_sec:,} secure-enriched  {n_vuln:,} vuln-enriched  ratio={n_sec/n_vuln:.2f}×")

# ── Analysis 1: Within-pair total activation on secure-enriched features ──────
print("\n── Analysis 1: Within-pair paired suppression ──────────────────────")
#
# Because SAE features are very sparse, most (vuln_i[k], safe_i[k]) pairs are
# (0, 0) — comparing 0 > 0 per feature is not meaningful.  Instead we compare
# the *total* activation that each sample accrues across ALL secure-enriched
# features:  total_safe_i = Σ_k safe[i,k]  vs  total_vuln_i = Σ_k vuln[i,k]
# for k in sec_mask.  This aggregates the sparse signal into a scalar per pair.

vuln_np = to_numpy(vuln)   # [N, D]
safe_np = to_numpy(safe)   # [N, D]

sec_idx = np.where(sec_mask)[0]   # indices of secure-enriched features

total_safe_sec = safe_np[:, sec_idx].sum(axis=1)   # [N]
total_vuln_sec = vuln_np[:, sec_idx].sum(axis=1)   # [N]

per_pair_win   = total_safe_sec > total_vuln_sec   # [N] bool
pct_wins       = per_pair_win.mean() * 100
mean_diff      = (total_safe_sec - total_vuln_sec).mean()

print(f"  n_secure_enriched features  : {n_sec:,}")
print(f"  % pairs: Σ(safe) > Σ(vuln) on secure-enriched features : {pct_wins:.1f}%")
print(f"  Mean pair Δ (safe − vuln total activation)             : {mean_diff:.4f}")

# Keep per_pair_frac as the normalised version for plotting
per_pair_frac = total_safe_sec / (total_safe_sec + total_vuln_sec + 1e-9)
mean_frac   = per_pair_frac.mean()
median_frac = np.median(per_pair_frac)
print(f"  Mean normalised share on secure side                   : {mean_frac:.3f}")

# ── Analysis 2: Activation magnitude asymmetry ───────────────────────────────
print("\n── Analysis 2: Activation magnitude asymmetry ──────────────────────")

mean_vuln_all = vuln_np.mean(axis=0)   # [D]
mean_safe_all = safe_np.mean(axis=0)   # [D]

above_diag_sec  = (mean_safe_all[sec_mask]  > mean_vuln_all[sec_mask]).mean()
above_diag_vuln = (mean_safe_all[vuln_mask] > mean_vuln_all[vuln_mask]).mean()

print(f"  Secure-enriched features above diagonal (mean_safe > mean_vuln): {above_diag_sec*100:.1f}%")
print(f"  Vuln-enriched  features above diagonal (mean_safe > mean_vuln): {above_diag_vuln*100:.1f}%")

# ── Save results ─────────────────────────────────────────────────────────────
results = {
    "n_secure_enriched": int(n_sec),
    "n_vuln_enriched":   int(n_vuln),
    "ratio":             round(float(n_sec / n_vuln), 2),
    "paired_suppression": {
        "pct_pairs_total_safe_exceeds_vuln": round(float(pct_wins), 1),
        "mean_pair_delta_activation": round(float(mean_diff), 4),
        "mean_normalised_share_secure": round(float(mean_frac), 4),
        "median_normalised_share_secure": round(float(median_frac), 4),
    },
    "magnitude_asymmetry": {
        "pct_sec_enriched_above_diagonal": round(float(above_diag_sec * 100), 1),
        "pct_vuln_enriched_above_diagonal": round(float(above_diag_vuln * 100), 1),
    },
}
with open(OUT_DIR / "paired_suppression_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {OUT_DIR / 'paired_suppression_results.json'}")

# ── Figure — magnitude scatter only (paper figure) ───────────────────────────
mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10})

fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))

rng = np.random.default_rng(42)
neutral_mask = ~sec_mask & ~vuln_mask
neutral_idx  = np.where(neutral_mask)[0]
sample_idx   = rng.choice(neutral_idx, size=min(3000, len(neutral_idx)), replace=False)

ax.scatter(mean_vuln_all[sample_idx], mean_safe_all[sample_idx],
           s=1, c="#aec7e8", alpha=0.3, rasterized=True, label="Neutral")
ax.scatter(mean_vuln_all[sec_mask],  mean_safe_all[sec_mask],
           s=2, c="#1f77b4", alpha=0.6, rasterized=True,
           label=f"Secure-enriched ({n_sec:,}, {above_diag_sec*100:.0f}% above diag.)")
ax.scatter(mean_vuln_all[vuln_mask], mean_safe_all[vuln_mask],
           s=2, c="#d62728", alpha=0.6, rasterized=True,
           label=f"Vuln-enriched ({n_vuln:,}, {above_diag_vuln*100:.0f}% above diag.)")

lim_max = max(mean_vuln_all.max(), mean_safe_all.max()) * 1.05
ax.plot([0, lim_max], [0, lim_max], "k--", lw=0.8, label="y = x")
ax.set_xlim(0, lim_max)
ax.set_ylim(0, lim_max)
ax.set_xlabel(r"Mean activation on vulnerable code")
ax.set_ylabel(r"Mean activation on secure code")
ax.set_title(f"Magnitude asymmetry — all {D:,} SAE features (Layer~11)")
ax.legend(fontsize=7, markerscale=3)

fig.tight_layout()

out_pdf = OUT_DIR / "fig_paired_suppression.pdf"
fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
print(f"Figure saved to {out_pdf}")

if PAPER_FIGS.exists():
    import shutil
    paper_out = PAPER_FIGS / "fig_paired_suppression.pdf"
    shutil.copy(out_pdf, paper_out)
    print(f"Copied to {paper_out}")

plt.show()
print("\nDone.")
