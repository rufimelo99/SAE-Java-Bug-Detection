"""
feature_asymmetry_crosslayer.py
================================
Replicates the L11 feature-asymmetry analysis (12,887 vs 3,476) at all four
layers for which we have pre-stacked .pt activation files.

  - L11 : SAE feature activations [2493, 16384] from
          run_20260218_134529_vulnerable_code_qwen_coder_standard_16384_10M/
          (standard ReLU+L1 SAE)
  - L0, L14, L27 : TopK SAE feature activations [2493, 16384], encoded
          from raw residual [2493, 3584] using local TopK SAE weights at
          ~/Downloads/sae_DeltaSecommits_qwen-2.5-7b-instruct_tokenized_topk_blocks.{L}.hook_resid_post/

For each representation we compute, per feature/neuron k:
    Δf[k] = (fraction of vulnerable samples where k > 0)
           − (fraction of secure samples where k > 0)
and count how many dimensions have Δf < 0 (secure-enriched) vs Δf > 0.

Outputs
-------
  artifacts/activations/feature_asymmetry_crosslayer/
      delta_f_results.json          # per-layer counts
  Paper figures dir /
      fig_feature_asymmetry_crosslayer.pdf
"""

import json
import warnings
from pathlib import Path

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
RAW_DIR      = ARTIFACTS / "run_20260216_231950_cl_secure_code_qwen_coder_topk_16384_raw"
TOPK_SAE_DIR = Path.home() / "Downloads"  # local TopK SAE weights
OUT_DIR      = ARTIFACTS / "feature_asymmetry_crosslayer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Per-layer Δf computation ─────────────────────────────────────────────────
def to_numpy(t: torch.Tensor):
    """torch → numpy without relying on torch.Tensor.numpy() (NumPy 2.x compat)."""
    import ctypes, numpy as np
    t = t.contiguous().cpu().float()
    arr = np.frombuffer((ctypes.c_float * t.numel()).from_address(
        t.data_ptr()), dtype=np.float32).copy()
    return arr.reshape(t.shape)


def compute_delta_f(vuln: torch.Tensor, safe: torch.Tensor):
    """Δf[k] = fraction of vuln where k>0 − fraction of safe where k>0."""
    f_vuln  = (vuln > 0).float().mean(dim=0)
    f_safe  = (safe > 0).float().mean(dim=0)
    delta_f = f_vuln - f_safe
    return to_numpy(delta_f)


def load_safetensors(path: Path) -> dict:
    """Minimal safetensors reader using struct + torch.frombuffer (no numpy needed)."""
    import struct, json as _json
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = _json.loads(f.read(header_len))
        data_offset = 8 + header_len
        tensors = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            f.seek(data_offset + start)
            raw = bytearray(f.read(end - start))
            dtype_map = {"F32": torch.float32, "F16": torch.float16,
                         "BF16": torch.bfloat16}
            t_dtype = dtype_map.get(info["dtype"], torch.float32)
            t = torch.frombuffer(raw, dtype=t_dtype).reshape(info["shape"]).clone()
            tensors[name] = t
    return tensors


def encode_topk(raw: torch.Tensor, sae_dir: Path, k: int = 64) -> torch.Tensor:
    """Encode raw residual activations through a local TopK SAE.

    Forward pass (matching sae_lens TopK with apply_b_dec_to_input + layer_norm):
        x_adj   = raw - b_dec
        x_norm  = layer_norm(x_adj, [d_in])
        pre_act = x_norm @ W_enc + b_enc
        acts    = topk(pre_act, k)   — keep top-k, zero the rest
    """
    weights = load_safetensors(sae_dir / "sae_weights.safetensors")
    W_enc = weights["W_enc"].float()   # [d_in, d_sae]
    b_enc = weights["b_enc"].float()   # [d_sae]
    b_dec = weights["b_dec"].float()   # [d_in]

    x = raw.float()
    x_adj  = x - b_dec.unsqueeze(0)                              # [N, d_in]
    x_norm = torch.nn.functional.layer_norm(x_adj, [x_adj.shape[-1]])
    pre    = x_norm @ W_enc + b_enc.unsqueeze(0)                 # [N, d_sae]

    # TopK: zero out all but the top-k per sample
    topk_vals, topk_idx = torch.topk(pre, k, dim=-1)
    acts = torch.zeros_like(pre)
    acts.scatter_(1, topk_idx, topk_vals)
    return acts


def load_layer(layer: int):
    """Return (vuln, safe) tensors for the given layer in SAE feature space."""
    if layer == 11:
        # Pre-computed standard ReLU+L1 SAE feature activations [2493, 16384]
        v = torch.load(SAE_FEAT_DIR / "vulnerable_layer_11.pt",
                       map_location="cpu", weights_only=True)
        s = torch.load(SAE_FEAT_DIR / "safe_layer_11.pt",
                       map_location="cpu", weights_only=True)
    else:
        # Encode raw residual [2493, 3584] through local TopK SAE → [2493, 16384]
        sae_dir = (TOPK_SAE_DIR /
                   f"sae_DeltaSecommits_qwen-2.5-7b-instruct_tokenized_topk_"
                   f"blocks.{layer}.hook_resid_post")
        print(f"  Loading TopK SAE from {sae_dir.name} …")
        v_raw = torch.load(RAW_DIR / f"vulnerable_layer_{layer}.pt",
                           map_location="cpu", weights_only=True)
        s_raw = torch.load(RAW_DIR / f"safe_layer_{layer}.pt",
                           map_location="cpu", weights_only=True)
        v = encode_topk(v_raw, sae_dir)
        s = encode_topk(s_raw, sae_dir)
    return v, s


# ── Main ─────────────────────────────────────────────────────────────────────
# All layers → SAE features (16384 dims)
# L11: standard ReLU+L1 SAE; L0/L27: TopK SAE (local weights; L14 file truncated)
LAYERS    = [0,    11,    27]
REP_LABEL = ["TopK SAE\n(16384)", "Std SAE\n(16384)", "TopK SAE\n(16384)"]

results  = {}
delta_fs = {}

for layer in LAYERS:
    print(f"\n── Layer {layer} ──────────────────────────────────────")
    v_feats, s_feats = load_layer(layer)
    delta_f = compute_delta_f(v_feats, s_feats)

    n_total     = len(delta_f)
    n_secure    = int((delta_f < 0).sum())   # more on secure
    n_vuln      = int((delta_f > 0).sum())   # more on vuln
    n_equal     = n_total - n_secure - n_vuln
    ratio       = round(n_secure / max(n_vuln, 1), 2)

    print(f"  d_sae={n_total}  secure-enriched={n_secure}  vuln-enriched={n_vuln}  "
          f"equal={n_equal}  ratio={ratio}×")

    results[layer] = {
        "d_sae": n_total,
        "secure_enriched": n_secure,
        "vuln_enriched": n_vuln,
        "equal": n_equal,
        "ratio": ratio,
    }
    delta_fs[layer] = delta_f

# Save results
with open(OUT_DIR / "delta_f_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {OUT_DIR / 'delta_f_results.json'}")

# ── Figure ───────────────────────────────────────────────────────────────────
mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10})

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=False)

for ax, layer, rep in zip(axes, LAYERS, REP_LABEL):
    df  = delta_fs[layer]
    res = results[layer]

    bins     = np.linspace(df.min(), df.max(), 80)
    neg_mask = df < 0
    pos_mask = df > 0
    ax.hist(df[neg_mask], bins=bins, color="#1f77b4", alpha=0.75,
            label=f"Secure-enriched ({res['secure_enriched']:,})")
    ax.hist(df[pos_mask], bins=bins, color="#d62728", alpha=0.75,
            label=f"Vuln-enriched ({res['vuln_enriched']:,})")
    ax.axvline(0, color="k", lw=0.8, ls="--")

    ax.set_title(f"L{layer}  [{rep}]\n{res['ratio']}× asymmetry")
    ax.set_xlabel(r"$\Delta f = f_\mathrm{vuln} - f_\mathrm{secure}$")
    if ax is axes[0]:
        ax.set_ylabel("Feature / neuron count")
    ax.legend(fontsize=7, loc="upper right")

fig.suptitle(
    r"Feature activation asymmetry ($\Delta f$) across layers — "
    "SAE feature space at all four layers",
    fontsize=10, y=1.02
)
fig.tight_layout()

out_pdf = OUT_DIR / "fig_feature_asymmetry_crosslayer.pdf"
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Figure saved to {out_pdf}")

if PAPER_FIGS.exists():
    paper_out = PAPER_FIGS / "fig_feature_asymmetry_crosslayer.pdf"
    import shutil
    shutil.copy(out_pdf, paper_out)
    print(f"Copied to {paper_out}")

plt.show()
print("\nDone.")
