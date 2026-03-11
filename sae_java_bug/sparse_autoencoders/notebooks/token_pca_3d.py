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
   [N × n_bins, D] matrices per class/family.
3. Fit PCA on the combined matrix for that layer → 3 PCs.
4. Plot centroid trajectories as 3-D polylines through the n_bins positional
   centroids, plus low-opacity individual paths.

Modes
-----
  vuln_secure  (default) — blue=secure, red=vulnerable
  cwe_family             — one colour per CWE family (vulnerable side only)
  both                   — secure vs vulnerable AND by CWE family, side by side

Outputs  (one set per layer, plus a multi-layer grid if >1 layer)
-------
  artifacts/activations/token_pca_3d/
      token_pca_3d_l{L}.pdf
      token_pca_3d_l{L}.html       (Plotly, optional)
      token_pca_3d_all_layers.pdf  (subplot grid, only if >1 layer)

Run examples
------------
  # secure vs vulnerable (default)
  python token_pca_3d.py --layers 11

  # by CWE family
  python token_pca_3d.py --layers 0 3 7 11 15 19 23 27 --mode cwe_family

  # both panels side by side
  python token_pca_3d.py --layers 11 --mode both

  # custom JSONL pattern (must contain {layer})
  python token_pca_3d.py --layers 11 15 \
      --jsonl_pattern artifacts/activations/per_token_sae_l{layer}.jsonl
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

# ── CWE family definitions ─────────────────────────────────────────────────────
# Values are sets of CWE numbers (strings, without "CWE-" prefix).
# "Other" acts as catch-all for anything not listed above.
CWE_FAMILIES: dict[str, set | None] = {
    "Memory":    {"119", "125", "787", "476", "122", "121", "416", "415", "401"},
    "Injection": {"79", "89", "78", "77", "94", "434", "502"},
    "Other":     None,
}

FAMILY_COLORS = {
    "Memory":    "#d62728",   # red
    "Injection": "#1f77b4",   # blue
    "Other":     "#2ca02c",   # green
}

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--layers", type=int, nargs="+", default=[11],
                    help="Which SAE layers to visualise (space-separated)")
parser.add_argument("--mode", choices=["vuln_secure", "cwe_family", "both"],
                    default="vuln_secure",
                    help="Colour scheme: 'vuln_secure', 'cwe_family', or 'both'")
parser.add_argument("--jsonl_pattern",
                    default="per_token/per_token_sae_l{layer}.jsonl",
                    help="Path pattern inside artifacts/activations/; {layer} is replaced")
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
    return p if p.is_absolute() else ARTIFACTS / p


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


def cwe_to_family(cwe_str: str) -> str:
    """Map a raw CWE string like 'CWE-79' or '79' to a family name."""
    num = cwe_str.upper().replace("CWE-", "").strip()
    for family, members in CWE_FAMILIES.items():
        if members is None:
            continue
        if num in members:
            return family
    return "Other"


def load_layer(layer: int):
    """
    Return:
      sec_arr  [N, BINS, D]
      vul_arr  [N, BINS, D]
      families [N]  (string family label per sample)
    """
    path = resolve_jsonl(layer)
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
    print(f"  Reading {path} …")

    sec_binned, vul_binned, families = [], [], []
    n_loaded = 0
    n_skipped = 0
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue
            n_sec = rec.get("n_secure_tokens",     len(rec["secure"]))
            n_vul = rec.get("n_vulnerable_tokens", len(rec["vulnerable"]))
            if n_sec < args.min_tokens or n_vul < args.min_tokens:
                continue
            sec_binned.append(bin_sequence(rec["secure"]))
            vul_binned.append(bin_sequence(rec["vulnerable"]))
            families.append(cwe_to_family(rec.get("cwe", "")))
            n_loaded += 1
            if args.n_samples > 0 and n_loaded >= args.n_samples:
                break
            if n_loaded % 200 == 0:
                print(f"    {n_loaded} samples …")

    print(f"  Loaded {n_loaded} samples (skipped {n_skipped} malformed lines).")
    if n_loaded == 0:
        raise RuntimeError(
            f"No samples passed the min_tokens={args.min_tokens} filter for layer {layer}.\n"
            f"  File: {path}\n"
            "  If all records have n_tokens=1 this file contains pooled (not per-token) "
            "activations — run collect_per_token_sae.sh on the VM to generate real per-token data."
        )
    sec_arr = np.stack(sec_binned, axis=0)   # [N, BINS, D]
    vul_arr = np.stack(vul_binned, axis=0)
    return sec_arr, vul_arr, np.array(families)


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


# ── Drawing helpers ────────────────────────────────────────────────────────────
def _draw_trajectories(ax, arrays_meta: list[tuple], pca, layer, N, title_suffix=""):
    """
    arrays_meta: list of (arr [N_i, BINS, 3], label, color, linestyle)
    Draws faint individual paths + bold centroid for each group.
    """
    rng = np.random.default_rng(42)
    for arr, label, color, ls in arrays_meta:
        n = len(arr)
        if n == 0:
            continue
        n_ind = min(15, n)
        idx   = rng.choice(n, size=n_ind, replace=False)
        for i in idx:
            ax.plot(arr[i, :, 0], arr[i, :, 1], arr[i, :, 2],
                    color=color, alpha=0.015, lw=0.3, linestyle=ls)
        mean = arr.mean(axis=0)
        ax.plot(mean[:, 0], mean[:, 1], mean[:, 2],
                color=color, lw=3.0, linestyle=ls, label=f"{label} (n={n})",
                zorder=10)
        marker = "o" if ls == "-" else "s"   # circle=start solid, square=start dashed
        ax.scatter(*mean[0],  s=80, c=color, marker=marker, zorder=11, edgecolors="k", linewidths=0.5)
        ax.scatter(*mean[-1], s=80, c=color, marker="^",    zorder=11, edgecolors="k", linewidths=0.5)

    ev = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)", fontsize=7)
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)", fontsize=7)
    ax.set_zlabel(f"PC3 ({ev[2]:.1f}%)", fontsize=7)
    ax.set_title(f"L{layer}{title_suffix}", fontsize=9)
    ax.legend(fontsize=6, loc="upper left")


def draw_vuln_secure(ax, sec_pca, vul_pca, pca, layer, N):
    _draw_trajectories(ax, [
        (sec_pca, "Secure",     "#1f77b4", "-"),
        (vul_pca, "Vulnerable", "#d62728", "-"),
    ], pca, layer, N)


def draw_cwe_family(ax, sec_pca, vul_pca, families, pca, layer):
    """
    For each CWE family: solid line = vulnerable, dashed line = secure.
    Same colour per family so the vuln/secure gap is visible within each family.
    """
    groups = []
    for family, color in FAMILY_COLORS.items():
        mask = families == family
        n = mask.sum()
        if n == 0:
            continue
        groups.append((vul_pca[mask], f"{family} vuln", color, "-"))
        groups.append((sec_pca[mask], f"{family} secure", color, "--"))
    _draw_trajectories(ax, groups, pca, layer, len(vul_pca), title_suffix=" — CWE family")


# ── Plotly ─────────────────────────────────────────────────────────────────────
def save_plotly_vuln_secure(layer, sec_pca, vul_pca, pca, N):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  Plotly not installed — skipping HTML. (pip install plotly)")
        return

    pos_vals = np.linspace(0, 1, BINS)
    rng      = np.random.default_rng(42)
    idx_ind  = rng.choice(N, size=min(15, N), replace=False)

    fig_pl = go.Figure()
    for i in idx_ind:
        for arr, col in [(sec_pca, "royalblue"), (vul_pca, "firebrick")]:
            fig_pl.add_trace(go.Scatter3d(
                x=arr[i, :, 0], y=arr[i, :, 1], z=arr[i, :, 2],
                mode="lines", line=dict(color=col, width=1),
                opacity=0.03, showlegend=False,
            ))

    for arr, name, cs, show_cb in [
        (sec_pca.mean(0), "Secure",     "Blues", True),
        (vul_pca.mean(0), "Vulnerable", "Reds",  False),
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


def save_plotly_cwe_family(layer, sec_pca, vul_pca, families, pca):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  Plotly not installed — skipping HTML. (pip install plotly)")
        return

    pos_vals = np.linspace(0, 1, BINS)
    rng      = np.random.default_rng(42)

    PLOTLY_COLORS = {
        "Memory":    "firebrick",
        "Injection": "royalblue",
        "Other":     "green",
    }

    fig_pl = go.Figure()
    for family, color in PLOTLY_COLORS.items():
        mask = families == family
        if mask.sum() == 0:
            continue
        for arr, suffix, dash in [
            (vul_pca[mask], "vuln",   "solid"),
            (sec_pca[mask], "secure", "dash"),
        ]:
            n = len(arr)
            # faint individuals
            idx_ind = rng.choice(n, size=min(10, n), replace=False)
            for i in idx_ind:
                fig_pl.add_trace(go.Scatter3d(
                    x=arr[i, :, 0], y=arr[i, :, 1], z=arr[i, :, 2],
                    mode="lines", line=dict(color=color, width=1, dash=dash),
                    opacity=0.03, showlegend=False,
                ))
            # centroid
            mean = arr.mean(0)
            fig_pl.add_trace(go.Scatter3d(
                x=mean[:, 0], y=mean[:, 1], z=mean[:, 2],
                mode="lines+markers", name=f"{family} {suffix} (n={n})",
                line=dict(color=color, width=6, dash=dash),
                marker=dict(size=4, color=pos_vals,
                            colorscale="Viridis", showscale=False),
            ))

    ev = pca.explained_variance_ratio_ * 100
    fig_pl.update_layout(
        title=f"3-D PCA token trajectory by CWE family — SAE L{layer}",
        scene=dict(
            xaxis_title=f"PC1 ({ev[0]:.1f}%)",
            yaxis_title=f"PC2 ({ev[1]:.1f}%)",
            zaxis_title=f"PC3 ({ev[2]:.1f}%)",
        ),
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    out_html = OUT_DIR / f"token_pca_3d_cwe_l{layer}.html"
    fig_pl.write_html(str(out_html))
    print(f"  Saved {out_html}")


# ── Main loop over layers ──────────────────────────────────────────────────────
mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10})

layer_results = {}   # layer → (sec_pca, vul_pca, families, pca, N)

for layer in args.layers:
    print(f"\n── Layer {layer} ────────────────────────────────────────────────")
    try:
        sec_arr, vul_arr, families = load_layer(layer)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        continue

    N = len(sec_arr)
    # Print family breakdown
    unique, counts = np.unique(families, return_counts=True)
    print(f"  Family breakdown: { {k: int(v) for k, v in zip(unique, counts)} }")

    print("  Fitting PCA …")
    pca, sec_pca, vul_pca = fit_pca_and_project(sec_arr, vul_arr)
    ev = pca.explained_variance_ratio_ * 100
    print(f"  Explained variance: PC1={ev[0]:.1f}%  PC2={ev[1]:.1f}%  PC3={ev[2]:.1f}%")

    layer_results[layer] = (sec_pca, vul_pca, families, pca, N)

    # ── Per-layer figure ──────────────────────────────────────────────────────
    if args.mode == "both":
        fig = plt.figure(figsize=(16, 7))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        draw_vuln_secure(ax1, sec_pca, vul_pca, pca, layer, N)
        draw_cwe_family(ax2, sec_pca, vul_pca, families, pca, layer)
    elif args.mode == "cwe_family":
        fig = plt.figure(figsize=(9, 7))
        ax  = fig.add_subplot(111, projection="3d")
        draw_cwe_family(ax, sec_pca, vul_pca, families, pca, layer)
    else:  # vuln_secure
        fig = plt.figure(figsize=(9, 7))
        ax  = fig.add_subplot(111, projection="3d")
        draw_vuln_secure(ax, sec_pca, vul_pca, pca, layer, N)

    mode_tag = f"_{args.mode}" if args.mode != "vuln_secure" else ""
    fig.suptitle(
        f"3-D PCA token trajectory — SAE L{layer}\n"
        "circles = start  |  triangles = end  |  "
        f"N={N}, {BINS} bins",
        fontsize=10
    )
    fig.tight_layout()
    out_pdf = OUT_DIR / f"token_pca_3d_l{layer}{mode_tag}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    print(f"  Saved {out_pdf}")
    plt.close(fig)

    if not args.no_plotly:
        if args.mode in ("vuln_secure", "both"):
            save_plotly_vuln_secure(layer, sec_pca, vul_pca, pca, N)
        if args.mode in ("cwe_family", "both"):
            save_plotly_cwe_family(layer, sec_pca, vul_pca, families, pca)


# ── Multi-layer grid (only if >1 layer loaded successfully) ───────────────────
done_layers = list(layer_results.keys())
if len(done_layers) > 1:
    print("\nBuilding multi-layer grid figure …")
    n_panels = 2 if args.mode == "both" else 1
    ncols    = min(4, len(done_layers)) * n_panels
    nrows    = (len(done_layers) + min(4, len(done_layers)) - 1) // min(4, len(done_layers))
    fig      = plt.figure(figsize=(5 * ncols, 4.5 * nrows))

    panel = 1
    for layer in done_layers:
        sec_pca, vul_pca, families, pca, N = layer_results[layer]
        if args.mode == "both":
            ax1 = fig.add_subplot(nrows, ncols, panel,     projection="3d")
            ax2 = fig.add_subplot(nrows, ncols, panel + 1, projection="3d")
            draw_vuln_secure(ax1, sec_pca, vul_pca, pca, layer, N)
            draw_cwe_family(ax2, sec_pca, vul_pca, families, pca, layer)
            panel += 2
        elif args.mode == "cwe_family":
            ax = fig.add_subplot(nrows, ncols, panel, projection="3d")
            draw_cwe_family(ax, sec_pca, vul_pca, families, pca, layer)
            panel += 1
        else:
            ax = fig.add_subplot(nrows, ncols, panel, projection="3d")
            draw_vuln_secure(ax, sec_pca, vul_pca, pca, layer, N)
            panel += 1

    mode_tag = f"_{args.mode}" if args.mode != "vuln_secure" else ""
    fig.suptitle(
        f"3-D PCA token trajectories across SAE layers — {BINS} positional bins\n"
        "circles = start, triangles = end",
        fontsize=11, y=1.01
    )
    fig.tight_layout()
    out_grid = OUT_DIR / f"token_pca_3d_all_layers{mode_tag}.pdf"
    fig.savefig(out_grid, bbox_inches="tight", dpi=150)
    print(f"Saved {out_grid}")
    plt.close(fig)

print("\nDone.")
