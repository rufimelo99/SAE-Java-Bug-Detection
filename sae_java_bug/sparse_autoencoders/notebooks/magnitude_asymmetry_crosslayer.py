"""
magnitude_asymmetry_crosslayer.py
==================================
Runs the activation magnitude asymmetry analysis (Analysis 2 from
paired_suppression_test.py) across all 8 standard SAE layers available
in artifacts/activations/TO_UPload/.

For each layer, streams the JSONL to build [N, 16384] tensors, then computes:
  - Δf per feature → identify secure-enriched (Δf < 0) vs vuln-enriched (Δf > 0)
  - % of secure-enriched features above the diagonal (mean_safe > mean_vuln)
  - % of vuln-enriched features above the diagonal (control)

Outputs
-------
  artifacts/activations/paired_suppression/
      magnitude_asymmetry_crosslayer.json
  Paper figures dir /
      fig_magnitude_asymmetry_crosslayer.pdf
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

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS  = SCRIPT_DIR.parent.parent / "artifacts" / "activations"
JSONL_DIR  = ARTIFACTS / "TO_UPload"
OUT_DIR    = ARTIFACTS / "paired_suppression"
PAPER_FIGS = (SCRIPT_DIR.parent.parent.parent.parent
              / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
              / "figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]

# ── Helper ────────────────────────────────────────────────────────────────────
def to_numpy(t: torch.Tensor) -> np.ndarray:
    t = t.contiguous().cpu().float()
    arr = np.frombuffer(
        (ctypes.c_float * t.numel()).from_address(t.data_ptr()),
        dtype=np.float32
    ).copy()
    return arr.reshape(t.shape)


def load_layer_jsonl(layer: int):
    """Stream JSONL for a layer → (vuln [N,16384], safe [N,16384]) tensors."""
    pattern = f"activations_layer_{layer}_sae_*.jsonl"
    matches = list(JSONL_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No JSONL found for layer {layer} in {JSONL_DIR}")
    path = matches[0]
    print(f"  Reading {path.name} …")

    vuln_rows, safe_rows = [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            vuln_rows.append(rec["vulnerable"])
            safe_rows.append(rec["secure"])

    vuln = torch.tensor(vuln_rows, dtype=torch.float32)
    safe = torch.tensor(safe_rows, dtype=torch.float32)
    return vuln, safe


# ── Main loop ─────────────────────────────────────────────────────────────────
results = {}
layer_data = {}   # store (sec_mask, vuln_mask, mean_vuln, mean_safe) per layer

for layer in LAYERS:
    print(f"\n── Layer {layer} ─────────────────────────────────────────────")
    vuln, safe = load_layer_jsonl(layer)
    N, D = vuln.shape
    print(f"  shape: {N} × {D}")

    vuln_np = to_numpy(vuln)
    safe_np = to_numpy(safe)

    # Δf
    f_vuln  = (vuln_np > 0).mean(axis=0)
    f_safe  = (safe_np > 0).mean(axis=0)
    delta_f = f_vuln - f_safe

    sec_mask  = delta_f < 0
    vuln_mask = delta_f > 0
    n_sec  = sec_mask.sum()
    n_vuln = vuln_mask.sum()
    ratio  = round(n_sec / max(n_vuln, 1), 2)
    print(f"  secure-enriched={n_sec:,}  vuln-enriched={n_vuln:,}  ratio={ratio}×")

    # Magnitude asymmetry
    mean_vuln = vuln_np.mean(axis=0)
    mean_safe = safe_np.mean(axis=0)

    above_sec  = (mean_safe[sec_mask]  > mean_vuln[sec_mask]).mean()
    above_vuln = (mean_safe[vuln_mask] > mean_vuln[vuln_mask]).mean()
    print(f"  secure-enriched above diagonal: {above_sec*100:.1f}%")
    print(f"  vuln-enriched   above diagonal: {above_vuln*100:.1f}%")

    results[layer] = {
        "n_secure_enriched": int(n_sec),
        "n_vuln_enriched":   int(n_vuln),
        "ratio":             ratio,
        "pct_sec_above_diagonal":  round(float(above_sec  * 100), 1),
        "pct_vuln_above_diagonal": round(float(above_vuln * 100), 1),
    }
    layer_data[layer] = (sec_mask, vuln_mask, mean_vuln, mean_safe)

# Save JSON
out_json = OUT_DIR / "magnitude_asymmetry_crosslayer.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_json}")

# ── Figure: 2×4 scatter grid ──────────────────────────────────────────────────
mpl.rcParams.update({"font.size": 8, "axes.titlesize": 9})

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
rng = np.random.default_rng(42)

for ax, layer in zip(axes.flat, LAYERS):
    sec_mask, vuln_mask, mean_vuln, mean_safe = layer_data[layer]
    res = results[layer]

    neutral_idx = np.where(~sec_mask & ~vuln_mask)[0]
    sample_idx  = rng.choice(neutral_idx, size=min(2000, len(neutral_idx)), replace=False)

    ax.scatter(mean_vuln[sample_idx], mean_safe[sample_idx],
               s=1, c="#aec7e8", alpha=0.3, rasterized=True)
    ax.scatter(mean_vuln[sec_mask],  mean_safe[sec_mask],
               s=1, c="#1f77b4", alpha=0.5, rasterized=True,
               label=f"Sec-enrich. ({res['pct_sec_above_diagonal']:.0f}%↑)")
    ax.scatter(mean_vuln[vuln_mask], mean_safe[vuln_mask],
               s=1, c="#d62728", alpha=0.5, rasterized=True,
               label=f"Vuln-enrich. ({res['pct_vuln_above_diagonal']:.0f}%↑)")

    lim = max(mean_vuln.max(), mean_safe.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.7)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title(f"L{layer}  (ratio {res['ratio']}×)")
    ax.set_xlabel("Mean act. — vuln", fontsize=7)
    ax.set_ylabel("Mean act. — secure", fontsize=7)
    ax.legend(fontsize=6, markerscale=3, loc="upper left")

fig.suptitle(
    "Activation magnitude asymmetry across all 8 standard SAE layers\n"
    "Blue (secure-enriched) should cluster above y = x diagonal",
    fontsize=10
)
fig.tight_layout()

out_pdf = OUT_DIR / "fig_magnitude_asymmetry_crosslayer.pdf"
fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
print(f"Figure saved to {out_pdf}")

if PAPER_FIGS.exists():
    import shutil
    shutil.copy(out_pdf, PAPER_FIGS / "fig_magnitude_asymmetry_crosslayer.pdf")
    print(f"Copied to paper figures dir")

plt.show()
print("\nDone.")
