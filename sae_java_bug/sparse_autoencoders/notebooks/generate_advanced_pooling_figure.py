"""
Regenerate fig_advanced_pooling_comparison.pdf from a saved probe_results.json.

Usage:
    conda run -n sae python generate_advanced_pooling_figure.py
    conda run -n sae python generate_advanced_pooling_figure.py --results PATH/probe_results.json

Automatically finds the most recent advanced_pool run if no path is given.
Excludes L27 (fp16 overflow → constant probe output).
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE       = Path(__file__).parent
ARTIFACTS  = Path(__file__).parents[2] / "artifacts" / "activations"
PAPER_FIGS = (
    Path(__file__).parents[4]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

ALL_LAYERS  = [0, 3, 7, 11, 15, 19, 23, 27]
PLOT_LAYERS = [0, 3, 7, 11, 15, 19, 23]   # L27 excluded (fp16 overflow)

mpl.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 150,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

COLORS = {
    "last_token":      "#555555",
    "mean_token":      "#d65f5f",
    "attn_weighted":   "#3a82c4",
    "diff_restricted": "#2ca02c",
}
LABELS = {
    "last_token":      "Last-token (baseline)",
    "mean_token":      "Mean-token",
    "attn_weighted":   "Attention-weighted",
    "diff_restricted": "Diff-restricted",
}
MARKERS = {
    "last_token": "o", "mean_token": "s",
    "attn_weighted": "^", "diff_restricted": "D",
}


def find_latest_results() -> Path:
    runs = sorted((ARTIFACTS / "advanced_pool").glob("*/probe_results.json"))
    if not runs:
        raise FileNotFoundError(
            f"No probe_results.json found under {ARTIFACTS}/advanced_pool/"
        )
    return runs[-1]


def load_results(json_path: Path) -> dict:
    with json_path.open() as f:
        payload = json.load(f)
    # Convert string keys back to int
    return {
        method: {int(k): v for k, v in res.items()}
        for method, res in payload["probe_results"].items()
    }


def plot(all_results: dict, plot_layers: list[int], out_path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    # Whether diff_restricted ≡ mean_token (degenerate case)
    diff_eq_mean = False
    if "diff_restricted" in all_results and "mean_token" in all_results:
        diff_eq_mean = all(
            abs(all_results["diff_restricted"].get(l, {}).get("roc_auc", -1)
                - all_results["mean_token"].get(l, {}).get("roc_auc", -2)) < 1e-9
            for l in plot_layers
        )

    keys_to_plot = ["last_token", "mean_token", "attn_weighted"]
    if not diff_eq_mean and "diff_restricted" in all_results:
        keys_to_plot.append("diff_restricted")

    for key in keys_to_plot:
        if key not in all_results:
            continue
        res = all_results[key]
        layers_present = [l for l in plot_layers if l in res]
        xs   = [plot_layers.index(l) for l in layers_present]
        aucs = [res[l]["roc_auc"] for l in layers_present]
        los  = [res[l]["ci_lo"]   for l in layers_present]
        his  = [res[l]["ci_hi"]   for l in layers_present]

        ax.plot(xs, aucs, f"{MARKERS[key]}-",
                color=COLORS[key], label=LABELS[key], lw=1.5, ms=5)
        ax.fill_between(xs, los, his, color=COLORS[key], alpha=0.15)

    # Annotate below-chance attention region
    if "attn_weighted" in all_results:
        below = [l for l in plot_layers
                 if all_results["attn_weighted"].get(l, {}).get("ci_hi", 1.0) < 0.5]
        if below:
            x0 = plot_layers.index(below[0])  - 0.4
            x1 = plot_layers.index(below[-1]) + 0.4
            ax.axvspan(x0, x1, color=COLORS["attn_weighted"], alpha=0.07, zorder=0)
            ax.text(
                (x0 + x1) / 2, 0.415, "below\nchance",
                ha="center", va="top", fontsize=6.5,
                color=COLORS["attn_weighted"], alpha=0.8,
            )

    if diff_eq_mean:
        ax.text(
            0.02, 0.02,
            "Diff-restricted ≡ Mean-token\n(full-file diff coverage 100%)",
            transform=ax.transAxes, fontsize=6.5,
            color=COLORS["diff_restricted"], alpha=0.8,
            va="bottom",
        )

    ax.axhline(0.5, color="grey", lw=0.7, ls=":", alpha=0.5, label="Chance (0.5)")
    if 11 in plot_layers:
        x11 = plot_layers.index(11)
        ax.axvline(x11, color="black", lw=0.8, ls="--", alpha=0.3)
        ax.text(x11 + 0.1, 0.395, "L11", fontsize=7, color="black", alpha=0.5)

    ax.set_xticks(range(len(plot_layers)))
    ax.set_xticklabels([f"L{l}" for l in plot_layers], rotation=45)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC  (vuln vs.\\ secure)")
    ax.set_title(
        "Pooling strategy comparison — raw residual stream\n"
        "(shaded = 95\\% bootstrap CI; L27 excluded, fp16 overflow)",
        fontsize=8,
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0.38, 0.68)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_table(all_results: dict, plot_layers: list[int]):
    methods = [m for m in ["last_token", "mean_token", "attn_weighted", "diff_restricted"]
               if m in all_results]
    print("\n" + "=" * 95)
    print("AUROC by pooling strategy (raw residual, L27 excluded)")
    print("=" * 95)
    header = f"{'Layer':>6}  " + "  ".join(f"{m:^27}" for m in methods)
    print(header)
    print("-" * 95)
    for l in plot_layers:
        row = f"  L{l:>2}   "
        for m in methods:
            d = all_results[m].get(l)
            if d:
                row += f"{d['roc_auc']:.3f} [{d['ci_lo']:.3f}–{d['ci_hi']:.3f}]   "
            else:
                row += f"{'—':^27}   "
        # Flag below-chance attention layers
        attn = all_results.get("attn_weighted", {}).get(l, {})
        if attn.get("ci_hi", 1.0) < 0.50:
            row += " ◀ attn below chance"
        print(row)
    print("=" * 95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=None,
                        help="Path to probe_results.json (default: latest run)")
    args = parser.parse_args()

    json_path = args.results or find_latest_results()
    print(f"Loading results from {json_path}")
    all_results = load_results(json_path)

    print_table(all_results, PLOT_LAYERS)

    out = PAPER_FIGS / "fig_advanced_pooling_comparison.pdf"
    plot(all_results, PLOT_LAYERS, out)
    # Also save next to the JSON
    plot(all_results, PLOT_LAYERS, json_path.parent / "fig_advanced_pooling_comparison.pdf")


if __name__ == "__main__":
    main()
