"""
feature_ablation_plot.py
========================
Reads feature_ablation_results.jsonl and generates fig_feature_ablation.pdf.

Run
---
    conda run -n sae python feature_ablation_plot.py

Input
-----
    artifacts/activations/feature_ablation/feature_ablation_results.jsonl

Output
------
    <paper_figs>/fig_feature_ablation.pdf
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
IN_JSONL   = Path(__file__).parents[2] / "artifacts" / "activations" / \
             "feature_ablation" / "feature_ablation_results.jsonl"
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

COLORS = {
    "secure_enriched": "#4878cf",
    "vuln_enriched":   "#e07b39",
}
LABELS = {
    "secure_enriched": "Ablate secure-enriched features",
    "vuln_enriched":   "Ablate vuln-enriched features (control)",
}


def load(path: Path) -> dict[str, list]:
    data: dict[str, list] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            c = r["condition"]
            data.setdefault(c, []).append(r)
    for c in data:
        data[c].sort(key=lambda x: x["n_ablated"])
    return data


def main():
    if not IN_JSONL.exists():
        raise FileNotFoundError(
            f"Results not found: {IN_JSONL}\n"
            "Run feature_ablation_compute.py first."
        )

    data = load(IN_JSONL)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for condition, rows in data.items():
        xs      = [r["n_ablated"]  for r in rows]
        ys      = [r["eff_auc"]    for r in rows]
        lo_err  = [r["eff_auc"] - r["eff_ci_lo"] for r in rows]
        hi_err  = [r["eff_ci_hi"] - r["eff_auc"] for r in rows]

        color = COLORS.get(condition, "grey")
        label = LABELS.get(condition, condition)

        ax.plot(xs, ys, "o-", color=color, lw=1.5, ms=4, label=label, zorder=3)
        ax.fill_between(
            xs,
            [y - l for y, l in zip(ys, lo_err)],
            [y + h for y, h in zip(ys, hi_err)],
            color=color, alpha=0.15, zorder=2,
        )

    ax.axhline(0.5, color="grey", lw=0.8, ls=":", alpha=0.7, label="Chance (0.5)", zorder=1)

    # Annotate baseline (N=0)
    for condition, rows in data.items():
        baseline = next(r for r in rows if r["n_ablated"] == 0)
        ax.annotate(
            f"{baseline['eff_auc']:.3f}",
            xy=(0, baseline["eff_auc"]),
            xytext=(120, baseline["eff_auc"] + 0.005),
            fontsize=7,
            color=COLORS.get(condition, "grey"),
        )

    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("Number of features zeroed out")
    ax.set_ylabel("Effective AUROC  (1 − raw AUROC)")
    ax.set_title(
        "Feature ablation: zeroing secure-enriched SAE features\n"
        "collapses mean-token separation toward chance",
        fontsize=9,
    )
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0.44, 0.68)

    # x-axis ticks matching ablation sizes
    ticks = [0, 10, 100, 1000, 10000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])

    fig.tight_layout()
    out = PAPER_FIGS / "fig_feature_ablation.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
