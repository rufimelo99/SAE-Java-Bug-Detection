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

SAE_RUNS = [
    {
        "label":   "Qwen-STD-SAE",
        "feat_dir": ARTIFACTS / "run_20260218_134529_vulnerable_code_qwen_coder_standard_16384_10M",
    },
    {
        "label":   "Qwen-TopK-SAE",
        "feat_dir": ARTIFACTS / "TOPK_tensors",
    },
]
OUT_DIR = ARTIFACTS / "paired_suppression"
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


def run_analysis(label, feat_dir):
    if not feat_dir.exists():
        print(f"[SKIP] {label}: directory not found ({feat_dir})")
        return None

    print(f"\n{'='*60}")
    print(f"Loading SAE feature activations — {label}")
    vuln = torch.load(feat_dir / "vulnerable_layer_11.pt",
                      map_location="cpu", weights_only=True).float()
    safe = torch.load(feat_dir / "safe_layer_11.pt",
                      map_location="cpu", weights_only=True).float()
    print(f"  vuln: {tuple(vuln.shape)}  safe: {tuple(safe.shape)}")

    N, D = vuln.shape

    # Δf: identify secure-enriched features
    f_vuln  = (vuln > 0).float().mean(dim=0)
    f_safe  = (safe > 0).float().mean(dim=0)
    delta_f = to_numpy(f_vuln - f_safe)

    sec_mask  = delta_f < 0
    vuln_mask = delta_f > 0

    n_sec  = sec_mask.sum()
    n_vuln = vuln_mask.sum()
    print(f"\nFeature asymmetry: {n_sec:,} secure-enriched  {n_vuln:,} vuln-enriched  ratio={n_sec/n_vuln:.2f}×")

    # Analysis 1: Within-pair paired suppression
    print("\n── Analysis 1: Within-pair paired suppression ──────────────────────")
    vuln_np = to_numpy(vuln)
    safe_np = to_numpy(safe)

    sec_idx        = np.where(sec_mask)[0]
    total_safe_sec = safe_np[:, sec_idx].sum(axis=1)
    total_vuln_sec = vuln_np[:, sec_idx].sum(axis=1)
    per_pair_win   = total_safe_sec > total_vuln_sec
    pct_wins       = per_pair_win.mean() * 100
    mean_diff      = (total_safe_sec - total_vuln_sec).mean()
    per_pair_frac  = total_safe_sec / (total_safe_sec + total_vuln_sec + 1e-9)
    mean_frac      = per_pair_frac.mean()
    median_frac    = np.median(per_pair_frac)

    print(f"  % pairs: Σ(safe) > Σ(vuln) on secure-enriched features : {pct_wins:.1f}%")
    print(f"  Mean pair Δ (safe − vuln total activation)             : {mean_diff:.4f}")
    print(f"  Mean normalised share on secure side                   : {mean_frac:.3f}")

    # Analysis 2: Activation magnitude asymmetry
    print("\n── Analysis 2: Activation magnitude asymmetry ──────────────────────")
    mean_vuln_all   = vuln_np.mean(axis=0)
    mean_safe_all   = safe_np.mean(axis=0)
    above_diag_sec  = (mean_safe_all[sec_mask]  > mean_vuln_all[sec_mask]).mean()
    above_diag_vuln = (mean_safe_all[vuln_mask] > mean_vuln_all[vuln_mask]).mean()

    print(f"  Secure-enriched above diagonal: {above_diag_sec*100:.1f}%")
    print(f"  Vuln-enriched  above diagonal: {above_diag_vuln*100:.1f}%")

    # Save JSON results
    results = {
        "label": label,
        "n_features": int(D),
        "n_samples": int(N),
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
    safe_label = label.replace(" ", "_").replace("-", "_")
    json_out = OUT_DIR / f"paired_suppression_results_{safe_label}.json"
    with open(json_out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved: {json_out}")

    return dict(
        label=label, D=D, N=N,
        sec_mask=sec_mask, vuln_mask=vuln_mask,
        mean_vuln_all=mean_vuln_all, mean_safe_all=mean_safe_all,
        n_sec=n_sec, n_vuln=n_vuln,
        above_diag_sec=above_diag_sec, above_diag_vuln=above_diag_vuln,
    )


# ── Run all SAEs ──────────────────────────────────────────────────────────────
all_results = []
for run in SAE_RUNS:
    res = run_analysis(run["label"], run["feat_dir"])
    if res is not None:
        all_results.append(res)

# ── Figure — one panel per SAE ────────────────────────────────────────────────
mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10})

n_panels = len(all_results)
if n_panels == 0:
    print("No results to plot.")
else:
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    for ax, res in zip(axes, all_results):
        neutral_mask = ~res["sec_mask"] & ~res["vuln_mask"]
        neutral_idx  = np.where(neutral_mask)[0]
        sample_idx   = rng.choice(neutral_idx, size=min(3000, len(neutral_idx)), replace=False)
        mv, ms = res["mean_vuln_all"], res["mean_safe_all"]

        ax.scatter(mv[sample_idx], ms[sample_idx],
                   s=1, c="#aec7e8", alpha=0.3, rasterized=True, label="Neutral")
        ax.scatter(mv[res["sec_mask"]],  ms[res["sec_mask"]],
                   s=2, c="#1f77b4", alpha=0.6, rasterized=True,
                   label=f"Secure-enriched ({res['n_sec']:,}, {res['above_diag_sec']*100:.0f}% above diag.)")
        ax.scatter(mv[res["vuln_mask"]], ms[res["vuln_mask"]],
                   s=2, c="#d62728", alpha=0.6, rasterized=True,
                   label=f"Vuln-enriched ({res['n_vuln']:,}, {res['above_diag_vuln']*100:.0f}% above diag.)")

        lim_max = max(mv.max(), ms.max()) * 1.05
        ax.plot([0, lim_max], [0, lim_max], "k--", lw=0.8, label="y = x")
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.set_xlabel("Mean activation on vulnerable code")
        ax.set_ylabel("Mean activation on secure code")
        ax.set_title(f"{res['label']}\nMagnitude asymmetry — {res['D']:,} SAE features (Layer 11)")
        ax.legend(fontsize=7, markerscale=3)

    fig.tight_layout()
    out_pdf = OUT_DIR / "fig_paired_suppression.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    print(f"\nFigure saved: {out_pdf}")

    if PAPER_FIGS.exists():
        import shutil
        paper_out = PAPER_FIGS / "fig_paired_suppression.pdf"
        shutil.copy(out_pdf, paper_out)
        print(f"Copied to {paper_out}")

    plt.show()

print("\nDone.")
