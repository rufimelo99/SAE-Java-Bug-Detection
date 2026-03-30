"""
Plot fig_vuln_secure_by_layer from cached within_lang_baseline_results.json.

Run:
  python plot_vuln_secure_by_layer.py

Skips any TopK entries in the cache.
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ARTIFACTS  = Path(__file__).parents[2] / "artifacts" / "activations"
CACHE_JSON = ARTIFACTS / "within_lang_baseline_results.json"
PAPER_FIGS = (
    Path(__file__).parents[4]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

# Runs to include (order matters for panel layout)
INCLUDE_RUNS = ["Qwen-Raw", "Qwen-STD-SAE"]
RUN_TITLES   = {"Qwen-Raw": "Raw residual", "Qwen-STD-SAE": "SAE features"}

# ── Style ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":     "serif",
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi":      150,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

LINE_STYLES = {
    "global":   {"color": "#333333", "marker": "o", "label": "All files"},
    "c_only":   {"color": "#4878cf", "marker": "^", "label": "Within C"},
    "php_only": {"color": "#d62728", "marker": "s", "label": "Within PHP"},
    "js_only":  {"color": "#2ca02c", "marker": "D", "label": "Within JS"},
}

LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]


def make_figure(cache):
    runs = [r for r in INCLUDE_RUNS if r in cache]
    n_panels = len(runs)
    if n_panels == 0:
        raise ValueError("No matching runs found in cache.")

    fig, axes = plt.subplots(1, n_panels, figsize=(3.8 * n_panels, 3.0), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, run_name in zip(axes, runs):
        run_data = cache[run_name]
        layers   = sorted(int(l) for l in run_data if int(l) in LAYERS)
        xs       = list(range(len(layers)))
        layer_lbls = [f"L{l}" for l in layers]

        for key, style in LINE_STYLES.items():
            aucs = [run_data[str(l)][key]["roc_auc"] if str(l) in run_data else np.nan for l in layers]
            los  = [run_data[str(l)][key]["ci_lo"]   if str(l) in run_data else np.nan for l in layers]
            his  = [run_data[str(l)][key]["ci_hi"]   if str(l) in run_data else np.nan for l in layers]
            ax.plot(xs, aucs, style["marker"] + "-",
                    color=style["color"], label=style["label"], lw=1.5, ms=5)
            ax.fill_between(xs, los, his, color=style["color"], alpha=0.15)

        ax.axhline(0.5, color="grey", lw=0.7, ls=":", alpha=0.5, label="Chance")

        if 11 in layers:
            x11 = layers.index(11)
            ax.axvline(x11, color="black", lw=1.0, ls="--", alpha=0.4)
            ax.text(x11 + 0.1, 0.51, "L11", fontsize=7, color="black", alpha=0.6)

        ax.set_xticks(xs)
        ax.set_xticklabels(layer_lbls, rotation=45)
        ax.set_xlabel("Layer")
        ax.set_ylabel("ROC-AUC  (vuln vs. secure)")
        ax.set_title(RUN_TITLES.get(run_name, run_name), fontweight="bold")
        ax.set_ylim(0.00, 0.75)
        ax.legend(fontsize=7)

    fig.suptitle(
        "Vulnerable vs. Secure probe: global, within-C, within-PHP, within-JS\n"
        "(shaded band = 95% bootstrap CI)",
        fontsize=9,
    )
    fig.tight_layout()
    out = PAPER_FIGS / "fig_vuln_secure_by_layer.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    if not CACHE_JSON.exists():
        raise FileNotFoundError(
            f"Cache not found: {CACHE_JSON}\n"
            "Run within_language_baseline.py first to generate it."
        )
    cache = json.load(open(CACHE_JSON))
    make_figure(cache)
