"""
Cross-layer vulnerability direction analysis.

For each layer L, computes the mean vulnerability direction:
    d^L = normalize(mean(v^L) - mean(s^L))

then measures:

1. Direction stability — cosine similarity between d^L and d^{L'} for all layer pairs.
   If the vulnerability direction is consistent across depth, the heatmap is uniformly high.
   If it rotates, each layer encodes vulnerability along a different axis.

2. Paired delta norm — ||δ_i^L||₂ = ||v_i^L - s_i^L||₂ per pair per layer.
   Does the model progressively push paired representations apart, or do they stay close?

3. Alignment score — δ_i^L · d^L per pair per layer.
   Do individual pairs agree on the mean vulnerability direction?
   Fraction > 0 should equal the linear probe AUROC rank (internal consistency check).

All analyses run globally AND within-C (language-controlled, n=1618).

Usage:
    conda run -n sae python cross_layer_direction_probe.py
    conda run -n sae python cross_layer_direction_probe.py --run_dir PATH

Requires:
    artifacts/activations/mean_pool/<ts>/safe_layer_{L}.pt + vulnerable_layer_{L}.pt + meta.json
"""

import argparse
import ctypes
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
ARTIFACTS = Path(__file__).parents[2] / "artifacts" / "activations"
PAPER_FIGS = (
    Path(__file__).parents[4]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]
SEED = 42

C_EXTS = {"c", "cc", "cpp", "h"}

# ── CWE Families (from paper) ──────────────────────────────────────────────────
CWE_FAMILIES = {
    "Memory Safety": [
        "CWE-119",
        "CWE-120",
        "CWE-125",
        "CWE-787",
        "CWE-416",
        "CWE-415",
        "CWE-401",
        "CWE-476",
    ],
    "Injection": ["CWE-79", "CWE-89", "CWE-78", "CWE-94"],
    "Resource Management": ["CWE-400", "CWE-399", "CWE-362"],
    "Information Disclosure": ["CWE-200"],
}

# ── Style ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ── Tensor loading (NumPy 2.x workaround) ─────────────────────────────────────
def _t2np(t: "torch.Tensor") -> np.ndarray:
    """Fast tensor → numpy via ctypes (avoids Python-level bytes() iteration)."""
    t = t.float().contiguous()
    buf = (ctypes.c_float * t.numel()).from_address(t.data_ptr())
    return np.ctypeslib.as_array(buf).reshape(t.shape).copy()


def find_latest_mean_pool_run() -> Path:
    runs = sorted((ARTIFACTS / "mean_pool").glob("*/meta.json"))
    if not runs:
        raise FileNotFoundError(f"No mean_pool runs under {ARTIFACTS}/mean_pool/")
    return runs[-1].parent


def load_layer(run_dir: Path, layer: int) -> tuple[np.ndarray, np.ndarray]:
    safe = _t2np(torch.load(run_dir / f"safe_layer_{layer}.pt", weights_only=True))
    vuln = _t2np(
        torch.load(run_dir / f"vulnerable_layer_{layer}.pt", weights_only=True)
    )
    return safe, vuln


# ── Direction analysis ─────────────────────────────────────────────────────────
def vuln_direction(safe: np.ndarray, vuln: np.ndarray) -> np.ndarray:
    """Unit vector pointing from mean-secure to mean-vulnerable."""
    d = vuln.mean(0) - safe.mean(0)
    norm = np.linalg.norm(d)
    return d / norm if norm > 1e-10 else d


def cosine_matrix(unit_vecs: list[np.ndarray]) -> np.ndarray:
    """[L, d] → [L, L] cosine similarity (unit vecs → just dot products)."""
    D = np.stack(unit_vecs)
    return D @ D.T


# ── Figures ───────────────────────────────────────────────────────────────────
def _layer_labels():
    return [f"L{l}" for l in LAYERS]


def fig_cosine_heatmap(cos_global: np.ndarray, cos_c: np.ndarray, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    ll = _layer_labels()

    for ax, mat, title in zip(axes, [cos_global, cos_c], ["Global", "Within-C"]):
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(len(LAYERS)))
        ax.set_yticks(range(len(LAYERS)))
        ax.set_xticklabels(ll, rotation=45)
        ax.set_yticklabels(ll)
        ax.set_title(title, fontsize=9)
        for i in range(len(LAYERS)):
            for j in range(len(LAYERS)):
                v = mat[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if abs(v) > 0.65 else "black",
                )

    plt.colorbar(im, ax=axes, shrink=0.80, pad=0.02, label="Cosine similarity")
    fig.suptitle(
        r"Vulnerability direction cosine similarity across layers"
        "\n"
        r"$\mathbf{d}^L = \mathrm{normalize}(\bar{v}^L - \bar{s}^L)$",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def fig_delta_norm(
    norms_per_layer: list[np.ndarray], c_mask: np.ndarray, out_path: Path
):
    ll = _layer_labels()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)

    for ax, mask, title, color in zip(
        axes,
        [np.ones(len(norms_per_layer[0]), dtype=bool), c_mask],
        ["Global (n=2493)", f"Within-C (n={c_mask.sum()})"],
        ["#4878cf", "#4878cf"],
    ):
        data = [norms[mask] for norms in norms_per_layer]
        means = [d.mean() for d in data]

        parts = ax.violinplot(
            data,
            positions=range(len(LAYERS)),
            showmedians=True,
            showextrema=False,
            widths=0.7,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.45)
        parts["cmedians"].set_color("#1a3a6c")
        parts["cmedians"].set_linewidth(1.5)

        ax.plot(
            range(len(LAYERS)),
            means,
            "o--",
            color="#1a3a6c",
            ms=4,
            lw=1.2,
            label="Mean",
        )

        ax.set_xticks(range(len(LAYERS)))
        ax.set_xticklabels(ll, rotation=45)
        ax.set_xlabel("Layer")
        ax.set_title(title)

    axes[0].set_ylabel(r"$\|\delta_i^L\|_2 = \|v_i^L - s_i^L\|_2$")
    fig.suptitle("Paired representation distance across layers", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def fig_alignment(
    align_per_layer: list[np.ndarray], c_mask: np.ndarray, out_path: Path
):
    ll = _layer_labels()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)

    for ax, mask, title, color in zip(
        axes,
        [np.ones(len(align_per_layer[0]), dtype=bool), c_mask],
        ["Global (n=2493)", f"Within-C (n={c_mask.sum()})"],
        ["#e07b39", "#e07b39"],
    ):
        data = [al[mask] for al in align_per_layer]
        means = [d.mean() for d in data]
        frac_pos = [np.mean(d > 0) for d in data]

        parts = ax.violinplot(
            data,
            positions=range(len(LAYERS)),
            showmedians=True,
            showextrema=False,
            widths=0.7,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.45)
        parts["cmedians"].set_color("#8b3a00")
        parts["cmedians"].set_linewidth(1.5)

        ax.plot(
            range(len(LAYERS)),
            means,
            "s--",
            color="#8b3a00",
            ms=4,
            lw=1.2,
            label="Mean",
        )
        ax.axhline(0, color="grey", lw=0.8, ls=":", alpha=0.6)

        # Annotate fraction with positive alignment above the violin
        y_top = max(np.percentile(d, 97) for d in data) * 1.08
        for xi, fp in enumerate(frac_pos):
            ax.text(
                xi,
                y_top,
                f"{fp:.0%}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#1a3a6c",
            )

        ax.set_xticks(range(len(LAYERS)))
        ax.set_xticklabels(ll, rotation=45)
        ax.set_xlabel("Layer")
        ax.set_title(f"{title}\n(% positive alignment shown)")

    axes[0].set_ylabel(r"$\delta_i^L \cdot \mathbf{d}^L$")
    fig.suptitle(
        r"Per-pair alignment to mean vulnerability direction $\mathbf{d}^L$"
        "\n(positive = vulnerable side is further in the vulnerability direction)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Combined summary figure ───────────────────────────────────────────────────
def fig_direction_summary(
    norms_per_layer: list[np.ndarray],
    align_per_layer: list[np.ndarray],
    c_mask: np.ndarray,
    out_path: Path,
):
    """
    Two-panel summary figure (C-only):
      A (left)  — Mean paired distance ||δ||₂ across layers
      B (right) — Fraction of pairs with positive alignment across layers
    """
    n_layers = len(LAYERS)
    ll = _layer_labels()
    xs = list(range(n_layers))

    fig, (ax_norm, ax_aln) = plt.subplots(
        1, 2, figsize=(9, 3.8), gridspec_kw={"wspace": 0.38}
    )

    # ── Panel A: mean ||δ||₂ trajectory ──────────────────────────────────────
    means_c = [n[c_mask].mean() for n in norms_per_layer]
    stds_c = [n[c_mask].std() for n in norms_per_layer]

    ax_norm.plot(xs, means_c, "o-", color="#4878cf", lw=1.5, ms=5)
    ax_norm.fill_between(
        xs,
        [m - s for m, s in zip(means_c, stds_c)],
        [m + s for m, s in zip(means_c, stds_c)],
        color="#4878cf",
        alpha=0.15,
    )
    ax_norm.axvline(n_layers - 1, color="#d62728", lw=0.8, ls=":", alpha=0.7)

    ax_norm.set_xticks(xs)
    ax_norm.set_xticklabels(ll, rotation=45)
    ax_norm.set_xlabel("Layer")
    ax_norm.set_ylabel(r"Mean $\|\delta_i^L\|_2$")
    ax_norm.set_title("(Left) Paired representation distance\n", fontsize=8)

    # ── Panel B: fraction positive alignment ──────────────────────────────────
    frac_c = [np.mean(a[c_mask] > 0) for a in align_per_layer]

    ax_aln.plot(xs, frac_c, "o-", color="#4878cf", lw=1.5, ms=5)
    ax_aln.axhline(0.5, color="grey", lw=0.8, ls=":", alpha=0.5, label="Chance (50%)")
    ax_aln.axvline(n_layers - 1, color="#d62728", lw=0.8, ls=":", alpha=0.7)
    ax_aln.axvspan(1, n_layers - 2, color="#4878cf", alpha=0.06)

    ax_aln.set_xticks(xs)
    ax_aln.set_xticklabels(ll, rotation=45)
    ax_aln.set_xlabel("Layer")
    ax_aln.set_ylabel(r"Fraction pairs: $\delta_i^L \cdot \mathbf{d}^L > 0$")
    ax_aln.set_title(
        "(Right) Per-pair alignment to vulnerability direction\n", fontsize=8
    )
    ax_aln.set_ylim(0.45, 0.95)
    ax_aln.legend(fontsize=7, loc="lower left")

    fig.suptitle(
        "Direction geometry across layers (C, n="
        + str(c_mask.sum())
        + ", mean-token pooling)",
        fontsize=9,
        y=1.02,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Per-family direction geometry ──────────────────────────────────────────────
def fig_direction_per_family(
    align_per_layer: list[np.ndarray],
    c_mask: np.ndarray,
    meta: list,
    out_path: Path,
):
    """
    Per-CWE-family alignment breakdown across layers.
    """
    n_layers = len(LAYERS)
    ll = _layer_labels()
    xs = list(range(n_layers))

    # Build family masks
    family_masks = {}
    for family_name, cwe_list in CWE_FAMILIES.items():
        family_mask = np.array(
            [c_mask[i] and meta[i].get("cwe") in cwe_list for i in range(len(meta))]
        )
        if family_mask.sum() > 0:  # Only include families with samples
            family_masks[family_name] = family_mask

    # Color palette
    colors = {
        "Memory Safety": "#4878cf",
        "Injection": "#e07b39",
        "Resource Management": "#6cc644",
        "Information Disclosure": "#c71585",
    }

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for family_name, family_mask in sorted(family_masks.items()):
        frac = [np.mean(a[family_mask] > 0) for a in align_per_layer]
        n_samples = family_mask.sum()
        ax.plot(
            xs,
            frac,
            "o-",
            color=colors.get(family_name, "#999999"),
            lw=1.5,
            ms=5,
            label=f"{family_name} (n={n_samples})",
        )

    ax.axhline(0.5, color="grey", lw=0.8, ls=":", alpha=0.5, label="Chance (50%)")
    ax.axvline(n_layers - 1, color="#d62728", lw=0.8, ls=":", alpha=0.7)
    ax.axvspan(1, n_layers - 2, color="#4878cf", alpha=0.04)

    ax.set_xticks(xs)
    ax.set_xticklabels(ll, rotation=45)
    ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel(r"Fraction pairs: $\delta_i^L \cdot \mathbf{d}^L > 0$", fontsize=9)
    ax.set_ylim(0.45, 0.95)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.95)

    ax.set_title(
        "Per-pair alignment across vulnerability families (C only, mean-token pooling)",
        fontsize=9,
        pad=10,
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Combined comprehensive figure ──────────────────────────────────────────────
def fig_direction_comprehensive(
    norms_per_layer: list[np.ndarray],
    align_per_layer: list[np.ndarray],
    c_mask: np.ndarray,
    meta: list,
    out_path: Path,
):
    """
    Three-panel comprehensive direction geometry figure:
      A (left)   — Mean paired distance ||δ||₂ across layers
      B (middle) — Global alignment across layers
      C (right)  — Per-family alignment breakdown
    """
    n_layers = len(LAYERS)
    ll = _layer_labels()
    xs = list(range(n_layers))

    fig = plt.figure(figsize=(16, 3.8))
    gs = fig.add_gridspec(1, 3, wspace=0.32, width_ratios=[1, 1, 1.1])
    ax_norm = fig.add_subplot(gs[0, 0])
    ax_aln_global = fig.add_subplot(gs[0, 1])
    ax_aln_family = fig.add_subplot(gs[0, 2])

    # ── Panel A: Mean paired distance ──────────────────────────────────────────
    means_c = [n[c_mask].mean() for n in norms_per_layer]
    stds_c = [n[c_mask].std() for n in norms_per_layer]

    ax_norm.plot(xs, means_c, "o-", color="#4878cf", lw=1.5, ms=5)
    ax_norm.fill_between(
        xs,
        [m - s for m, s in zip(means_c, stds_c)],
        [m + s for m, s in zip(means_c, stds_c)],
        color="#4878cf",
        alpha=0.15,
    )
    ax_norm.axvline(n_layers - 1, color="#d62728", lw=0.8, ls=":", alpha=0.7)

    ax_norm.set_xticks(xs)
    ax_norm.set_xticklabels(ll, rotation=45)
    ax_norm.set_xlabel("Layer")
    ax_norm.set_ylabel(r"Mean $\|\delta_i^L\|_2$")
    ax_norm.set_title("(A) Paired representation distance", fontsize=8)

    # ── Panel B: Global alignment ──────────────────────────────────────────────
    frac_c = [np.mean(a[c_mask] > 0) for a in align_per_layer]

    ax_aln_global.plot(xs, frac_c, "o-", color="#4878cf", lw=1.5, ms=5)
    ax_aln_global.axhline(
        0.5, color="grey", lw=0.8, ls=":", alpha=0.5, label="Chance (50%)"
    )
    ax_aln_global.axvline(n_layers - 1, color="#d62728", lw=0.8, ls=":", alpha=0.7)
    ax_aln_global.axvspan(1, n_layers - 2, color="#4878cf", alpha=0.06)

    ax_aln_global.set_xticks(xs)
    ax_aln_global.set_xticklabels(ll, rotation=45)
    ax_aln_global.set_xlabel("Layer")
    ax_aln_global.set_ylabel(r"Fraction pairs: $\delta_i^L \cdot \mathbf{d}^L > 0$")
    ax_aln_global.set_title("(B) Global per-pair alignment", fontsize=8)
    ax_aln_global.set_ylim(0.45, 0.95)
    ax_aln_global.legend(fontsize=7, loc="lower left")

    # ── Panel C: Per-family alignment ──────────────────────────────────────────
    family_masks = {}
    for family_name, cwe_list in CWE_FAMILIES.items():
        family_mask = np.array(
            [c_mask[i] and meta[i].get("cwe") in cwe_list for i in range(len(meta))]
        )
        if family_mask.sum() > 0:  # Only include families with samples
            family_masks[family_name] = family_mask

    colors = {
        "Memory Safety": "#4878cf",
        "Injection": "#e07b39",
        "Resource Management": "#6cc644",
        "Information Disclosure": "#c71585",
    }

    for family_name, family_mask in sorted(family_masks.items()):
        frac = [np.mean(a[family_mask] > 0) for a in align_per_layer]
        ax_aln_family.plot(
            xs,
            frac,
            "o-",
            color=colors.get(family_name, "#999999"),
            lw=1.5,
            ms=5,
            label=family_name,
        )

    ax_aln_family.axhline(
        0.5, color="grey", lw=0.8, ls=":", alpha=0.5, label="Chance (50%)"
    )
    ax_aln_family.axvline(n_layers - 1, color="#d62728", lw=0.8, ls=":", alpha=0.7)
    ax_aln_family.axvspan(1, n_layers - 2, color="#4878cf", alpha=0.04)

    ax_aln_family.set_xticks(xs)
    ax_aln_family.set_xticklabels(ll, rotation=45)
    ax_aln_family.set_xlabel("Layer")
    ax_aln_family.set_ylabel(r"Fraction pairs: $\delta_i^L \cdot \mathbf{d}^L > 0$")
    ax_aln_family.set_title("(C) Per-family alignment", fontsize=8)
    ax_aln_family.set_ylim(0.45, 0.95)
    ax_aln_family.legend(fontsize=7, loc="lower left", framealpha=0.95)

    fig.suptitle(
        "Direction geometry across layers (C, n=" + str(c_mask.sum()) + ")",
        fontsize=9,
        y=1.02,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Split figures for double-column paper layout ──────────────────────────────
def fig_direction_distance_only(
    norms_per_layer: list[np.ndarray],
    c_mask: np.ndarray,
    out_path: Path,
):
    """
    Single-panel figure (Panel A):
      Mean paired distance ||δ||₂ across layers
    Optimized for single-column layout in papers.
    """
    n_layers = len(LAYERS)
    ll = _layer_labels()
    xs = list(range(n_layers))

    fig, ax = plt.subplots(figsize=(4, 3.2))

    # Panel A: Mean paired distance
    means_c = [n[c_mask].mean() for n in norms_per_layer]
    stds_c = [n[c_mask].std() for n in norms_per_layer]

    ax.plot(xs, means_c, "o-", color="#4878cf", lw=1.5, ms=5)
    ax.fill_between(
        xs,
        [m - s for m, s in zip(means_c, stds_c)],
        [m + s for m, s in zip(means_c, stds_c)],
        color="#4878cf",
        alpha=0.15,
    )
    ax.axvline(n_layers - 1, color="#d62728", lw=0.8, ls=":", alpha=0.7)

    ax.set_xticks(xs)
    ax.set_xticklabels(ll, rotation=45)
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"Mean $\|\delta_i^L\|_2$")
    ax.set_title("Paired representation distance", fontsize=9, fontweight="bold")

    fig.suptitle(
        "Direction geometry: paired distance (C, n=" + str(c_mask.sum()) + ")",
        fontsize=8,
        y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def fig_direction_alignment_combined(
    align_per_layer: list[np.ndarray],
    c_mask: np.ndarray,
    meta: list,
    out_path: Path,
):
    """
    Single-panel figure combining global and per-family alignment:
      All bug-type families plus global alignment on one plot
    Same axes (layer vs alignment %) for direct comparison.
    """
    n_layers = len(LAYERS)
    ll = _layer_labels()
    xs = list(range(n_layers))

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    # Global alignment (bold line) with 90% binomial CI
    n_c = int(c_mask.sum())
    frac_c = [np.mean(a[c_mask] > 0) for a in align_per_layer]
    ci_lo_c = [max(0.0, p - 1.645 * np.sqrt(max(p * (1 - p) / n_c, 0))) for p in frac_c]
    ci_hi_c = [min(1.0, p + 1.645 * np.sqrt(max(p * (1 - p) / n_c, 0))) for p in frac_c]
    ax.plot(xs, frac_c, "o-", color="black", lw=2.5, ms=6, label="Global", zorder=10)
    ax.fill_between(xs, ci_lo_c, ci_hi_c, alpha=0.12, color="black", linewidth=0, zorder=9)

    # Per-family alignment
    family_masks = {}
    for family_name, cwe_list in CWE_FAMILIES.items():
        family_mask = np.array(
            [c_mask[i] and meta[i].get("cwe") in cwe_list for i in range(len(meta))]
        )
        if family_mask.sum() > 0:
            family_masks[family_name] = family_mask

    colors = {
        "Memory Safety": "#4878cf",
        "Injection": "#e07b39",
        "Resource Management": "#6cc644",
        "Information Disclosure": "#c71585",
    }

    for family_name, family_mask in sorted(family_masks.items()):
        n_fam = int(family_mask.sum())
        frac = [np.mean(a[family_mask] > 0) for a in align_per_layer]
        ci_lo_f = [max(0.0, p - 1.645 * np.sqrt(max(p * (1 - p) / n_fam, 0))) for p in frac]
        ci_hi_f = [min(1.0, p + 1.645 * np.sqrt(max(p * (1 - p) / n_fam, 0))) for p in frac]
        c = colors.get(family_name, "#999999")
        ax.plot(
            xs, frac, "o-", color=c, lw=1.5, ms=5,
            label=family_name, alpha=0.75,
        )
        ax.fill_between(xs, ci_lo_f, ci_hi_f, alpha=0.10, color=c, linewidth=0)

    ax.axhline(0.5, color="grey", lw=0.8, ls=":", alpha=0.5, label="Chance (50%)")
    ax.axvline(n_layers - 1, color="#d62728", lw=0.8, ls=":", alpha=0.7)
    ax.axvspan(1, n_layers - 2, color="lightgrey", alpha=0.03)

    ax.set_xticks(xs)
    ax.set_xticklabels(ll, rotation=45)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel(r"Fraction pairs: $\delta_i^L \cdot \mathbf{d}^L > 0$", fontsize=10)
    ax.set_title(
        "Vulnerability direction alignment: global and per-family",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_ylim(0.45, 0.95)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(
        "Direction geometry: alignment across bug types (C, n="
        + str(c_mask.sum())
        + ")",
        fontsize=8,
        y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_mean_pool_run()
    print(f"Using mean-pool run: {run_dir}")

    with (run_dir / "meta.json").open() as f:
        meta = json.load(f)
    extensions = [r["file_extension"] for r in meta]
    c_mask = np.array([ext in C_EXTS for ext in extensions])
    print(f"  n={len(meta)}, C={c_mask.sum()}, other={len(meta)-c_mask.sum()}")

    dirs_global: list[np.ndarray] = []
    dirs_c: list[np.ndarray] = []
    norms_per_layer: list[np.ndarray] = []
    align_per_layer: list[np.ndarray] = []

    print(
        f"\n{'Layer':>6}  {'||d_glob||':>10}  {'frac+_glob':>10}  "
        f"{'frac+_C':>9}  {'mean_norm':>10}  {'mean_norm_C':>12}"
    )
    print("-" * 70)

    for layer in LAYERS:
        if not (run_dir / f"safe_layer_{layer}.pt").exists():
            print(f"  [SKIP] L{layer}")
            continue

        safe, vuln = load_layer(run_dir, layer)

        # ── Vulnerability directions
        d_glob = vuln_direction(safe, vuln)
        d_c = vuln_direction(safe[c_mask], vuln[c_mask])
        dirs_global.append(d_glob)
        dirs_c.append(d_c)

        # ── Paired delta
        delta = vuln - safe  # [N, 3584]
        norms = np.linalg.norm(delta, axis=1)  # [N]
        norms_per_layer.append(norms)

        # ── Alignment to GLOBAL direction (so global and C use same reference)
        align = delta @ d_glob  # [N]
        align_per_layer.append(align)

        frac_pos_glob = np.mean(align > 0)
        frac_pos_c = np.mean(align[c_mask] > 0)
        mean_norm = norms.mean()
        mean_norm_c = norms[c_mask].mean()

        print(
            f"  L{layer:>2}   {np.linalg.norm(d_glob):>10.4f}  "
            f"{frac_pos_glob:>10.3f}  {frac_pos_c:>9.3f}  "
            f"{mean_norm:>10.3f}  {mean_norm_c:>12.3f}"
        )

    # ── Cosine similarity matrices
    cos_global = cosine_matrix(dirs_global)
    cos_c = cosine_matrix(dirs_c)

    print("\nCosine similarity matrix — GLOBAL directions:")
    print(np.round(cos_global, 3))
    print("\nCosine similarity matrix — WITHIN-C directions:")
    print(np.round(cos_c, 3))

    # ── How aligned are global vs within-C directions?
    print("\nGlobal vs. Within-C direction cosine similarity per layer:")
    for i, layer in enumerate(LAYERS):
        c = float(dirs_global[i] @ dirs_c[i])
        print(f"  L{layer}: {c:.4f}")

    # ── Figures
    # Generate split figures for double-column paper layout
    fig_direction_distance_only(
        norms_per_layer,
        c_mask,
        PAPER_FIGS / "fig_direction_paired_distance.pdf",
    )
    fig_direction_alignment_combined(
        align_per_layer,
        c_mask,
        meta,
        PAPER_FIGS / "fig_direction_alignment.pdf",
    )

    # Also keep the comprehensive 3-panel version for reference
    fig_direction_comprehensive(
        norms_per_layer,
        align_per_layer,
        c_mask,
        meta,
        PAPER_FIGS / "fig_direction_summary.pdf",
    )
    fig_cosine_heatmap(cos_global, cos_c, PAPER_FIGS / "fig_direction_cosine_sim.pdf")
    fig_delta_norm(
        norms_per_layer, c_mask, PAPER_FIGS / "fig_delta_norm_trajectory.pdf"
    )
    fig_alignment(align_per_layer, c_mask, PAPER_FIGS / "fig_alignment_trajectory.pdf")


if __name__ == "__main__":
    main()
