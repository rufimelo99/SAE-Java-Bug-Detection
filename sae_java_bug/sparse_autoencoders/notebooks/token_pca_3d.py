"""
token_pca_3d.py
===============
3-D PCA trajectory visualisation of per-token SAE feature activations,
supporting one or multiple SAE layers.

Strategy
--------
1. Load per_token_sae_l{layer}.jsonl  (output of collect_per_token_sae.py)
   for each requested layer.
2. Pool to fixed-length positional bins (default 20) per sample →
   [N × n_bins, D] matrices per class.
3. Fit PCA on the combined matrix for that layer → 3 PCs.
4. Plot secure (blue) and vulnerable (red) centroid trajectories as 3-D
   polylines through the n_bins positional centroids, plus low-opacity
   individual paths.

Outputs  (one set per layer, plus a multi-layer grid if >1 layer)
-------
  artifacts/activations/token_pca_3d/
      token_pca_3d_l{L}.pdf
      token_pca_3d_l{L}.html       (Plotly, optional)
      token_pca_3d_all_layers.pdf  (subplot grid, only if >1 layer)

Run examples
------------
  # single layer
  python token_pca_3d.py --layers 11

  # multiple layers
  python token_pca_3d.py --layers 0 3 7 11 15 19 23 27

  # custom JSONL pattern (must contain {layer})
  python token_pca_3d.py --layers 11 15 \
      --jsonl_pattern artifacts/activations/per_token_sae_l{layer}.jsonl

  # limit samples for speed, skip Plotly
  python token_pca_3d.py --layers 11 --n_samples 500 --no_plotly
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--layers", type=int, nargs="+", default=[11],
                    help="Which SAE layers to visualise (space-separated)")
parser.add_argument("--jsonl_pattern",
                    default="artifacts/activations/per_token_sae_l{layer}.jsonl",
                    help="Path pattern; {layer} is replaced with the layer number")
parser.add_argument("--n_bins",     type=int, default=20,
                    help="Positional bins (relative position 0→1)")
parser.add_argument("--n_samples",  type=int, default=0,
                    help="Max samples per layer (0 = all)")
parser.add_argument("--n_features", type=int, default=16384)
parser.add_argument("--min_tokens", type=int, default=10,
                    help="Skip samples shorter than this (tokens)")
parser.add_argument("--no_plotly",  action="store_true",
                    help="Skip interactive Plotly HTML output")
args = parser.parse_args()

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS  = SCRIPT_DIR.parent.parent / "artifacts" / "activations"
OUT_DIR    = ARTIFACTS / "token_pca_3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

D    = args.n_features
BINS = args.n_bins


# ── Helpers ────────────────────────────────────────────────────────────────────
def resolve_jsonl(layer: int) -> Path:
    raw = args.jsonl_pattern.format(layer=layer)
    p   = Path(raw)
    return p if p.is_absolute() else SCRIPT_DIR.parent.parent / p


def sparse_to_dense(tok: dict) -> np.ndarray:
    row = np.zeros(D, dtype=np.float32)
    if tok["indices"]:
        row[tok["indices"]] = tok["values"]
    return row


def bin_sequence(token_list: list) -> np.ndarray:
    """Average per-token sparse activations into BINS positional buckets → [BINS, D]."""
    T      = len(token_list)
    out    = np.zeros((BINS, D), dtype=np.float32)
    counts = np.zeros(BINS,     dtype=np.int32)
    for t, tok in enumerate(token_list):
        b = min(int(t / T * BINS), BINS - 1)
        out[b]    += sparse_to_dense(tok)
        counts[b] += 1
    nonzero = counts > 0
    out[nonzero] /= counts[nonzero, None]
    return out


def load_layer(layer: int):
    """Return sec_arr [N, BINS, D] and vul_arr [N, BINS, D] for one layer."""
    path = resolve_jsonl(layer)
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
    print(f"  Reading {path} …")

    sec_binned, vul_binned = [], []
    n_loaded = 0
    with open(path) as f:
        for line in f:
            rec   = json.loads(line)
            n_sec = rec.get("n_secure_tokens",     len(rec["secure"]))
            n_vul = rec.get("n_vulnerable_tokens", len(rec["vulnerable"]))
            if n_sec < args.min_tokens or n_vul < args.min_tokens:
                continue
            sec_binned.append(bin_sequence(rec["secure"]))
            vul_binned.append(bin_sequence(rec["vulnerable"]))
            n_loaded += 1
            if args.n_samples > 0 and n_loaded >= args.n_samples:
                break
            if n_loaded % 200 == 0:
                print(f"    {n_loaded} samples …")

    print(f"  Loaded {n_loaded} samples.")
    sec_arr = np.stack(sec_binned, axis=0)   # [N, BINS, D]
    vul_arr = np.stack(vul_binned, axis=0)
    return sec_arr, vul_arr


def fit_pca_and_project(sec_arr, vul_arr):
    N = len(sec_arr)
    combined = np.concatenate([
        sec_arr.reshape(N * BINS, D),
        vul_arr.reshape(N * BINS, D),
    ], axis=0)
    pca = PCA(n_components=3, random_state=42)
    pca.fit(combined)
    sec_pca = pca.transform(sec_arr.reshape(N * BINS, D)).reshape(N, BINS, 3)
    vul_pca = pca.transform(vul_arr.reshape(N * BINS, D)).reshape(N, BINS, 3)
    return pca, sec_pca, vul_pca


def draw_layer_3d(ax, sec_pca, vul_pca, pca, layer, N):
    """Draw centroid + individual trajectories onto a 3-D Axes."""
    sec_mean = sec_pca.mean(axis=0)   # [BINS, 3]
    vul_mean = vul_pca.mean(axis=0)

    rng        = np.random.default_rng(42)
    n_ind      = min(80, N)
    idx_sample = rng.choice(N, size=n_ind, replace=False)

    for i in idx_sample:
        ax.plot(sec_pca[i, :, 0], sec_pca[i, :, 1], sec_pca[i, :, 2],
                color="#1f77b4", alpha=0.05, lw=0.4)
        ax.plot(vul_pca[i, :, 0], vul_pca[i, :, 1], vul_pca[i, :, 2],
                color="#d62728", alpha=0.05, lw=0.4)

    ax.plot(sec_mean[:, 0], sec_mean[:, 1], sec_mean[:, 2],
            color="#1f77b4", lw=2.5, label="Secure")
    ax.plot(vul_mean[:, 0], vul_mean[:, 1], vul_mean[:, 2],
            color="#d62728", lw=2.5, label="Vulnerable")

    ax.scatter(*sec_mean[0],  s=60, c="navy",    marker="o", zorder=5)
    ax.scatter(*sec_mean[-1], s=60, c="navy",    marker="^", zorder=5)
    ax.scatter(*vul_mean[0],  s=60, c="darkred", marker="o", zorder=5)
    ax.scatter(*vul_mean[-1], s=60, c="darkred", marker="^", zorder=5)

    ev = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)", fontsize=7)
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)", fontsize=7)
    ax.set_zlabel(f"PC3 ({ev[2]:.1f}%)", fontsize=7)
    ax.set_title(f"L{layer}  (N={N})", fontsize=9)
    ax.legend(fontsize=6, loc="upper left")
    return sec_mean, vul_mean


def save_plotly(layer, sec_pca, vul_pca, sec_mean, vul_mean, pca, N):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  Plotly not installed — skipping HTML. (pip install plotly)")
        return

    pos_vals = np.linspace(0, 1, BINS)
    rng      = np.random.default_rng(42)
    idx_ind  = rng.choice(N, size=min(40, N), replace=False)

    fig_pl = go.Figure()
    for i in idx_ind:
        for arr, col in [(sec_pca, "royalblue"), (vul_pca, "firebrick")]:
            fig_pl.add_trace(go.Scatter3d(
                x=arr[i, :, 0], y=arr[i, :, 1], z=arr[i, :, 2],
                mode="lines", line=dict(color=col, width=1),
                opacity=0.1, showlegend=False,
            ))

    for arr, name, cs, show_cb in [
        (sec_mean, "Secure",     "Blues", True),
        (vul_mean, "Vulnerable", "Reds",  False),
    ]:
        cb = dict(title="Token pos", x=1.05) if show_cb else {}
        fig_pl.add_trace(go.Scatter3d(
            x=arr[:, 0], y=arr[:, 1], z=arr[:, 2],
            mode="lines+markers", name=name,
            line=dict(color=pos_vals, colorscale=cs, width=6),
            marker=dict(size=4, color=pos_vals, colorscale=cs,
                        showscale=show_cb, colorbar=cb),
        ))

    ev = pca.explained_variance_ratio_ * 100
    fig_pl.update_layout(
        title=f"3-D PCA token trajectory — SAE L{layer} (N={N})",
        scene=dict(
            xaxis_title=f"PC1 ({ev[0]:.1f}%)",
            yaxis_title=f"PC2 ({ev[1]:.1f}%)",
            zaxis_title=f"PC3 ({ev[2]:.1f}%)",
        ),
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    out_html = OUT_DIR / f"token_pca_3d_l{layer}.html"
    fig_pl.write_html(str(out_html))
    print(f"  Saved {out_html}")


# ── Main loop over layers ──────────────────────────────────────────────────────
mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10})

layer_results = {}   # layer → (sec_pca, vul_pca, pca, N)

for layer in args.layers:
    print(f"\n── Layer {layer} ────────────────────────────────────────────────")
    try:
        sec_arr, vul_arr = load_layer(layer)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        continue

    N = len(sec_arr)
    print("  Fitting PCA …")
    pca, sec_pca, vul_pca = fit_pca_and_project(sec_arr, vul_arr)
    ev = pca.explained_variance_ratio_ * 100
    print(f"  Explained variance: PC1={ev[0]:.1f}%  PC2={ev[1]:.1f}%  PC3={ev[2]:.1f}%")

    layer_results[layer] = (sec_pca, vul_pca, pca, N)

    # Per-layer single figure
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    sec_mean, vul_mean = draw_layer_3d(ax, sec_pca, vul_pca, pca, layer, N)

    sm = mpl.cm.ScalarMappable(cmap="coolwarm",
                                norm=mpl.colors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.45, pad=0.1, label="Relative token position")
    fig.suptitle(
        f"3-D PCA token trajectory — SAE L{layer}\n"
        f"Blue = secure  |  Red = vulnerable  |  N={N}, {BINS} bins",
        fontsize=10
    )
    fig.tight_layout()
    out_pdf = OUT_DIR / f"token_pca_3d_l{layer}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    print(f"  Saved {out_pdf}")
    plt.close(fig)

    if not args.no_plotly:
        save_plotly(layer, sec_pca, vul_pca, sec_mean, vul_mean, pca, N)

# ── Multi-layer grid (only if >1 layer loaded successfully) ───────────────────
done_layers = list(layer_results.keys())
if len(done_layers) > 1:
    print("\nBuilding multi-layer grid figure …")
    ncols = min(4, len(done_layers))
    nrows = (len(done_layers) + ncols - 1) // ncols
    fig   = plt.figure(figsize=(5 * ncols, 4.5 * nrows))

    for idx, layer in enumerate(done_layers):
        sec_pca, vul_pca, pca, N = layer_results[layer]
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        draw_layer_3d(ax, sec_pca, vul_pca, pca, layer, N)

    fig.suptitle(
        f"3-D PCA token trajectories across SAE layers — {BINS} positional bins\n"
        "Blue = secure  |  Red = vulnerable  |  circles = start, triangles = end",
        fontsize=11, y=1.01
    )
    fig.tight_layout()
    out_grid = OUT_DIR / "token_pca_3d_all_layers.pdf"
    fig.savefig(out_grid, bbox_inches="tight", dpi=150)
    print(f"Saved {out_grid}")
    plt.close(fig)

print("\nDone.")
