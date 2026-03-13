"""
Generate publication-quality ablation figures for the NeurIPS paper.

Figures (saved to the paper repo's figures/ directory):
  1. fig_cwe_language_confound.pdf  — CWE-family × file-extension co-occurrence
  2. fig_delta_auc_heatmap.pdf      — ΔAUC (after−before residualisation) across all
                                       8 layers × 2 representations (raw, SAE)
  3. fig_ablation_l11.pdf           — Raw vs SAE at L11: before/after bar chart

Run from anywhere:
  python generate_ablation_figures.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

HERE       = Path(__file__).parent
ARTIFACTS  = Path(__file__).parents[2] / "artifacts" / "activations"
PAPER_FIGS = (
    Path(__file__).parents[4]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)

# ── NeurIPS-compatible style ──────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":    "serif",
    "font.size":      9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi":     150,
    "pdf.fonttype":   42,
    "ps.fonttype":    42,
})

FAMILY_COLORS = {
    "Memory Safety":   "#4878cf",
    "Injection":       "#e07b39",
    "Info Disclosure": "#6acc65",
    "Access Control":  "#d65f5f",
    "Resource Mgmt":   "#b47cc7",
}
FAMILIES = list(FAMILY_COLORS.keys())
LAYERS   = [0, 3, 7, 11, 15, 19, 23, 27]

# ── ΔAUC = AUC_residualised − AUC_raw (from notebook run) ────────────────────
# Negative  → language was helping  → confound
# Positive  → language was hurting  → suppression
DELTA_AUC = {
    "raw": {
        0:  {"Memory Safety": -0.068, "Injection": -0.096, "Info Disclosure": -0.015, "Access Control":  0.068, "Resource Mgmt":  0.013},
        3:  {"Memory Safety": -0.066, "Injection": -0.120, "Info Disclosure": -0.024, "Access Control":  0.038, "Resource Mgmt":  0.004},
        7:  {"Memory Safety": -0.050, "Injection": -0.105, "Info Disclosure": -0.004, "Access Control":  0.041, "Resource Mgmt":  0.006},
        11: {"Memory Safety": -0.055, "Injection": -0.142, "Info Disclosure": -0.013, "Access Control":  0.005, "Resource Mgmt":  0.013},
        15: {"Memory Safety": -0.036, "Injection": -0.124, "Info Disclosure": -0.009, "Access Control":  0.035, "Resource Mgmt": -0.004},
        19: {"Memory Safety": -0.031, "Injection": -0.153, "Info Disclosure": -0.032, "Access Control": -0.005, "Resource Mgmt": -0.021},
        23: {"Memory Safety": -0.064, "Injection": -0.166, "Info Disclosure": -0.007, "Access Control": -0.013, "Resource Mgmt": -0.035},
        27: {"Memory Safety": -0.072, "Injection": -0.173, "Info Disclosure":  0.000, "Access Control": -0.025, "Resource Mgmt": -0.021},
    },
    "sae": {
        0:  {"Memory Safety":  0.223, "Injection":  0.299, "Info Disclosure":  0.098, "Access Control":  0.134, "Resource Mgmt":  0.063},
        3:  {"Memory Safety":  0.118, "Injection":  0.169, "Info Disclosure":  0.083, "Access Control":  0.085, "Resource Mgmt":  0.092},
        7:  {"Memory Safety":  0.012, "Injection":  0.078, "Info Disclosure": -0.051, "Access Control":  0.077, "Resource Mgmt":  0.032},
        11: {"Memory Safety": -0.052, "Injection": -0.124, "Info Disclosure": -0.025, "Access Control":  0.033, "Resource Mgmt":  0.005},
        15: {"Memory Safety": -0.049, "Injection": -0.150, "Info Disclosure": -0.012, "Access Control":  0.008, "Resource Mgmt": -0.006},
        19: {"Memory Safety": -0.056, "Injection": -0.137, "Info Disclosure": -0.032, "Access Control": -0.035, "Resource Mgmt": -0.005},
        23: {"Memory Safety": -0.043, "Injection": -0.121, "Info Disclosure": -0.023, "Access Control":  0.017, "Resource Mgmt": -0.019},
        27: {"Memory Safety": -0.065, "Injection": -0.148, "Info Disclosure": -0.025, "Access Control": -0.033, "Resource Mgmt": -0.034},
    },
}

# ── CWE × file-extension raw counts ──────────────────────────────────────────
EXTENSIONS = ["c", "cc", "cpp", "java", "js", "php", "py", "rb"]
CWE_EXT_COUNTS = {
    "Memory Safety":   [941, 122,  46,  3,  2,   4,  0,  0],
    "Injection":       [  7,   1,   1, 10, 82, 373, 23, 14],
    "Info Disclosure": [ 97,   0,   3,  4,  6,  24,  1,  3],
    "Access Control":  [ 50,   1,   4, 13, 17,  48, 17,  4],
    "Resource Mgmt":   [237,  52,  18, 12, 40,  24, 14,  2],
}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — CWE × Language Confound (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def fig_confound():
    counts = pd.DataFrame(CWE_EXT_COUNTS, index=EXTENSIONS).T
    pct    = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4))
    kw = dict(linewidths=0.4, annot=True, cbar=False)
    sns.heatmap(counts, fmt="d",   cmap="YlOrRd", ax=axes[0], **kw, annot_kws={"size": 7})
    sns.heatmap(pct,    fmt=".0f", cmap="YlOrRd", ax=axes[1], **kw, annot_kws={"size": 7})

    axes[0].set_title("Sample counts", fontweight="bold")
    axes[1].set_title(r"$P(\mathrm{ext}\mid\mathrm{CWE\ family})$  [%]", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("File extension")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle("CWE family strongly correlates with programming language",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    out = PAPER_FIGS / "fig_cwe_language_confound.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — ΔAUC Heatmap: Raw vs SAE across all layers
# ─────────────────────────────────────────────────────────────────────────────

def fig_delta_auc_heatmap():
    """
    Two side-by-side heatmaps: Raw (left) and SAE (right).
    Rows = CWE families, columns = layers 0–27.
    Diverging colormap: red = confound (ΔAUC < 0), blue = suppression (ΔAUC > 0).
    """
    def _build_df(kind):
        return pd.DataFrame(
            {f"L{l}": [DELTA_AUC[kind][l][f] for f in FAMILIES] for l in LAYERS},
            index=FAMILIES,
        )

    df_raw = _build_df("raw")
    df_sae = _build_df("sae")

    # Shared colour scale: symmetric around 0
    vmax = max(df_raw.abs().values.max(), df_sae.abs().values.max())
    vmax = round(vmax + 0.01, 2)

    # Reserve space for colorbar on the right
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.5))
    fig.subplots_adjust(left=0.18, right=0.86, wspace=0.06)

    kw = dict(
        vmin=-vmax, vmax=vmax,
        cmap="RdBu",
        linewidths=0.4,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        cbar=False,
    )

    # Left panel: show y-axis labels (family names)
    sns.heatmap(df_raw, ax=axes[0], yticklabels=True,  **kw)
    # Right panel: suppress y-axis labels (shared axis)
    sns.heatmap(df_sae, ax=axes[1], yticklabels=False, **kw)

    axes[0].set_title("Raw residual stream", fontweight="bold")
    axes[1].set_title("SAE features",        fontweight="bold")

    for ax in axes:
        ax.set_xlabel("Layer")
        ax.set_ylabel("")
        ax.tick_params(axis="y", rotation=0)
        ax.tick_params(axis="x", rotation=0)

    # Dashed vertical line at the boundary between L11 and L15
    # (L11 = index 3 in LAYERS; right edge of that column = x=4)
    l11_right = LAYERS.index(11) + 1
    for ax in axes:
        ax.axvline(l11_right, color="black", linewidth=1.2, linestyle="--")

    # Explicit colorbar axes — no space stealing
    cax = fig.add_axes([0.88, 0.15, 0.018, 0.68])
    sm  = plt.cm.ScalarMappable(cmap="RdBu",
                                 norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(r"$\Delta$AUC", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        r"$\Delta$AUC after file-type residualisation"
        "  (blue = language suppresses CWE signal;"
        "  red = language helps)",
        fontsize=8, x=0.50, y=1.02,
    )
    out = PAPER_FIGS / "fig_delta_auc_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Raw vs SAE at L11: ΔAUC comparison
# ─────────────────────────────────────────────────────────────────────────────

def fig_ablation_l11():
    """
    Grouped bar chart comparing ΔAUC at L11 for raw vs SAE,
    for Memory Safety and Injection (the two most separable families).
    Highlights that SAE L11 ≈ raw L11 (no disentanglement at training layer).
    """
    main_families = ["Memory Safety", "Injection"]
    delta_raw_l11 = [DELTA_AUC["raw"][11][f] for f in main_families]
    delta_sae_l11 = [DELTA_AUC["sae"][11][f] for f in main_families]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    x = np.arange(len(main_families))
    w = 0.32
    colors = [FAMILY_COLORS[f] for f in main_families]

    ax.bar(x - w/2, delta_raw_l11, w,
           color=colors, label="Raw residual stream (L11)", zorder=3)
    ax.bar(x + w/2, delta_sae_l11, w,
           color=colors, alpha=0.5, hatch="///",
           label="SAE features (L11)", zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(main_families)
    ax.set_ylabel(r"$\Delta$AUC (resid. $-$ raw)")
    ax.set_title(
        "SAE L11 confound $\\approx$ raw L11 confound\n"
        "(SAE provides no disentanglement at training layer)",
        fontweight="bold", fontsize=8,
    )
    ax.legend(loc="lower right", framealpha=0.9, fontsize=7)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.35)
    ax.set_ylim(-0.22, 0.05)

    # Annotate difference
    for i, (r, s) in enumerate(zip(delta_raw_l11, delta_sae_l11)):
        diff = abs(r - s)
        ax.annotate(f"$\\Delta={diff:.3f}$",
                    xy=(i, min(r, s) - 0.015),
                    ha="center", fontsize=7, color="dimgray")

    fig.tight_layout()
    out = PAPER_FIGS / "fig_ablation_l11.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Pairwise Family ΔAUC matrices (Raw L11 | SAE L0 | SAE L11)
# ─────────────────────────────────────────────────────────────────────────────

def fig_pairwise_auc():
    """
    Three-panel figure: pairwise ΔAUC = AUC_residualised − AUC_raw for all
    5×5 family pairs at three representative layers.
    Blue = language suppresses CWE signal (positive); red = language confound (negative).
    """
    FAM_SHORT = ["Mem. Safety", "Injection", "Info Disc.", "Acc. Control", "Res. Mgmt"]

    # Hardcoded from notebook output
    PAIRWISE_DELTA = {
        "Raw L11": np.array([
            [np.nan, -0.099,  0.061,  0.077,  0.057],
            [-0.099, np.nan, -0.064, -0.066, -0.094],
            [ 0.061, -0.064, np.nan,  0.017,  0.043],
            [ 0.077, -0.066,  0.017, np.nan,  0.093],
            [ 0.057, -0.094,  0.043,  0.093, np.nan],
        ]),
        "SAE L0": np.array([
            [np.nan,  0.381,  0.175,  0.234,  0.116],
            [ 0.381, np.nan,  0.282,  0.174,  0.296],
            [ 0.175,  0.282, np.nan,  0.090,  0.094],
            [ 0.234,  0.174,  0.090, np.nan,  0.175],
            [ 0.116,  0.296,  0.094,  0.175, np.nan],
        ]),
        "SAE L11": np.array([
            [np.nan, -0.092,  0.053,  0.075,  0.010],
            [-0.092, np.nan, -0.019, -0.059, -0.073],
            [ 0.053, -0.019, np.nan,  0.007,  0.023],
            [ 0.075, -0.059,  0.007, np.nan,  0.067],
            [ 0.010, -0.073,  0.023,  0.067, np.nan],
        ]),
    }

    vmax = 0.40  # symmetric colorscale

    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.5))
    fig.subplots_adjust(left=0.15, right=0.88, wspace=0.08)

    for idx, (title, mat) in enumerate(PAIRWISE_DELTA.items()):
        ax = axes[idx]
        df = pd.DataFrame(mat, index=FAM_SHORT, columns=FAM_SHORT)
        nan_mask = np.isnan(mat)

        sns.heatmap(
            df, ax=ax,
            mask=nan_mask,
            vmin=-vmax, vmax=vmax,
            cmap="RdBu",
            annot=True, fmt=".2f", annot_kws={"size": 6},
            linewidths=0.4,
            cbar=False,
            yticklabels=(idx == 0),
            square=True,
        )
        # Grey diagonal cells
        for i in range(len(FAM_SHORT)):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color="#cccccc", lw=0))

        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=30, labelsize=6)
        ax.tick_params(axis="y", rotation=0,  labelsize=6)

    # Shared colorbar
    cax = fig.add_axes([0.90, 0.12, 0.018, 0.70])
    sm = plt.cm.ScalarMappable(
        cmap="RdBu", norm=plt.Normalize(vmin=-vmax, vmax=vmax)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(r"$\Delta$AUC", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        r"Pairwise $\Delta$AUC (resid. $-$ raw) across all CWE-family pairs"
        "  —  blue = language suppresses,  red = language confound",
        fontsize=7.5, y=1.03,
    )

    out = PAPER_FIGS / "fig_pairwise_delta_auc.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Paper figures dir : {PAPER_FIGS}")
    print(f"Activations dir   : {ARTIFACTS}")
    print()
    print("Generating ablation figures...")
    fig_confound()
    fig_delta_auc_heatmap()
    fig_ablation_l11()
    fig_pairwise_auc()
    print("Done.")
