"""
Nonlinear probe figure — reads from JSONL produced by nonlinear_probe_compute.py.

Run:
  python nonlinear_probe_plot.py
  python nonlinear_probe_plot.py --results path/to/nonlinear_probe_results.jsonl
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ARTIFACTS  = Path(__file__).parents[2] / "artifacts" / "activations"
DEFAULT_IN = ARTIFACTS / "nonlinear_probe" / "nonlinear_probe_results.jsonl"
PAPER_FIGS = (
    Path(__file__).parents[4]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":     "serif",
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi":      150,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

PROBE_STYLES = {
    "LogReg":  {"color": "#333333", "marker": "o", "ls": "-",  "lw": 1.8},
    "MLP":     {"color": "#d62728", "marker": "s", "ls": "--", "lw": 1.6},
    "RBF-SVM": {"color": "#1f77b4", "marker": "^", "ls": ":",  "lw": 1.6},
}


def load_results(jsonl_path):
    """Return {label -> {probe -> {layer -> (auroc, ci_lo, ci_hi)}}}."""
    results = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            label = r["label"]
            probe = r["probe"]
            layer = r["layer"]
            results.setdefault(label, {}).setdefault(probe, {})[layer] = (
                r["auroc"], r["ci_lo"], r["ci_hi"]
            )
    return results


def make_figure(results, out_path):
    n_panels = len(results)
    if n_panels == 0:
        print("No results to plot.")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 3.2), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (label, probe_results) in zip(axes, results.items()):
        all_layers = sorted({l for pd in probe_results.values() for l in pd})
        xs         = list(range(len(all_layers)))
        layer_lbls = [f"L{l}" for l in all_layers]

        for probe_name, style in PROBE_STYLES.items():
            if probe_name not in probe_results:
                continue
            pd = probe_results[probe_name]
            aucs = [pd[l][0] if l in pd else np.nan for l in all_layers]
            los  = [pd[l][1] if l in pd else np.nan for l in all_layers]
            his  = [pd[l][2] if l in pd else np.nan for l in all_layers]
            ax.plot(xs, aucs, style["marker"] + style["ls"],
                    color=style["color"], lw=style["lw"], ms=5, label=probe_name)
            ax.fill_between(xs, los, his, color=style["color"], alpha=0.12)

        ax.axhline(0.5, color="grey", lw=0.7, ls=":", alpha=0.5, label="Chance")

        if 11 in all_layers:
            x11 = all_layers.index(11)
            ax.axvline(x11, color="black", lw=1.0, ls="--", alpha=0.35)
            ax.text(x11 + 0.1, 0.505, "L11", fontsize=7, color="black", alpha=0.55)

        ax.set_xticks(xs)
        ax.set_xticklabels(layer_lbls, rotation=45)
        ax.set_xlabel("Layer")
        ax.set_ylabel("ROC-AUC  (vuln vs. secure)")
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(0.40, 0.70)
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Nonlinear probe comparison: LogReg vs MLP vs RBF-SVM\n"
        "(last-token pooled; PCA-50; shaded = 95% bootstrap CI)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_IN,
                        help="Path to nonlinear_probe_results.jsonl")
    args = parser.parse_args()

    if not args.results.exists():
        raise FileNotFoundError(
            f"Results not found: {args.results}\n"
            "Run nonlinear_probe_compute.py first."
        )

    results = load_results(args.results)
    out = PAPER_FIGS / "fig_nonlinear_probes.pdf"
    make_figure(results, out)
