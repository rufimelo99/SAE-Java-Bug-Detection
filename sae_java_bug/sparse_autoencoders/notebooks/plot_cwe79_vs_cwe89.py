"""
Plot CWE-79 (XSS) vs CWE-89 (SQLi) within-PHP probe AUROC across layers.

Usage:
    conda run -n sae python plot_cwe79_vs_cwe89.py
    conda run -n sae python plot_cwe79_vs_cwe89.py --results PATH/cwe79_vs_cwe89_results.json
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ARTIFACTS  = Path(__file__).parents[2] / "artifacts" / "activations"
PAPER_FIGS = (
    Path(__file__).parents[4]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]

mpl.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 150,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

COLOR = "#8c4b8c"   # purple — distinct from existing red/blue/grey palette


def find_latest_results() -> Path:
    runs = sorted((ARTIFACTS / "mean_pool").glob("*/cwe79_vs_cwe89_results.json"))
    if not runs:
        raise FileNotFoundError(
            f"No cwe79_vs_cwe89_results.json found under {ARTIFACTS}/mean_pool/"
        )
    return runs[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    json_path = args.results or find_latest_results()
    with json_path.open() as f:
        data = json.load(f)

    n79 = data["n_cwe79_php"]
    n89 = data["n_cwe89_php"]
    res = {int(k): v for k, v in data["results_per_layer"].items()}

    layers_present = [l for l in LAYERS if l in res]
    xs   = list(range(len(layers_present)))
    aucs = [res[l]["roc_auc"] for l in layers_present]
    los  = [res[l]["ci_lo"]   for l in layers_present]
    his  = [res[l]["ci_hi"]   for l in layers_present]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    ax.plot(xs, aucs, "o-", color=COLOR, lw=1.5, ms=5,
            label=f"CWE-79 vs CWE-89 within PHP\n($n_{{79}}={n79}$, $n_{{89}}={n89}$)")
    ax.fill_between(xs, los, his, color=COLOR, alpha=0.15)

    ax.axhline(0.5, color="grey", lw=0.7, ls=":", alpha=0.5, label="Chance (0.5)")

    if 11 in layers_present:
        x11 = layers_present.index(11)
        ax.axvline(x11, color="black", lw=0.8, ls="--", alpha=0.3)
        ax.text(x11 + 0.1, 0.03, "L11", fontsize=7, color="black", alpha=0.5)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{l}" for l in layers_present], rotation=45)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC  (CWE-79 vs.\\ CWE-89)")
    ax.set_title(
        "XSS vs.\\ SQL injection within PHP\n"
        "(shaded = 95\\% bootstrap CI; vulnerable-side mean-token)",
        fontsize=8,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0.0, 1.02)

    fig.tight_layout()

    out_paper = PAPER_FIGS / "fig_cwe79_vs_cwe89_php.pdf"
    out_local = json_path.parent / "fig_cwe79_vs_cwe89_php.pdf"
    fig.savefig(out_paper, bbox_inches="tight")
    fig.savefig(out_local, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_paper}")
    print(f"Saved: {out_local}")


if __name__ == "__main__":
    main()
