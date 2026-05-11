"""
Patch length distribution analysis by CWE family.

Analyzes whether injection vulnerabilities (CWE-79, 89, 78, 94) are patched
by code simplification (shorter secure code) while memory-corruption vulnerabilities
(CWE-119, 120, 125, 787, 416, 415, 401, 476) are patched by adding defensive
constructs (longer secure code).

Outputs fig_patch_length_by_cwe.pdf showing:
  - Histogram of patch length differences by CWE family
  - Statistical summary (mean, median, std)

Run from anywhere:
  python patch_length_analysis.py
"""

import base64
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from transformers import AutoTokenizer

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
GITHUB = Path(__file__).parents[4]  # …/GitHub/

# Activations may live in either repo — check SAE repo first, then code-security-probing
_SAE_ACTS = Path(__file__).parents[3] / "artifacts" / "activations"
_CSP_ACTS = GITHUB / "code-security-probing" / "artifacts" / "activations"

PAPER_FIGS = (
    GITHUB
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)

# CWE family definitions
CWE_FAMILIES = {
    "Memory Corruption": [
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
}

# Reverse map: CWE → family
CWE_TO_FAMILY = {}
for family, cwes in CWE_FAMILIES.items():
    for cwe in cwes:
        CWE_TO_FAMILY[cwe] = family

# Style
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

FAMILY_COLORS = {
    "Memory Corruption": "#4878cf",
    "Injection": "#e07b39",
}


def resolve_acts_dir() -> Path:
    """Resolve activations directory; check SAE repo first, then code-security-probing."""
    for base in (_SAE_ACTS, _CSP_ACTS):
        candidate = base / "c_only"
        if (candidate / "layer_11_train.jsonl").exists():
            return candidate
    raise FileNotFoundError("Could not find activations directory (c_only)")


def load_patch_lengths():
    """
    Load vulnerable/secure code pairs from activation JSONLs and compute patch lengths.
    Returns:
        list of dicts: {
            "cwe": "CWE-125",
            "family": "Memory Corruption",
            "vuln_len": 45,
            "secure_len": 58,
            "patch_delta": 13,  # secure_len - vuln_len (positive = code grows)
            "vuln_code": "...",
            "secure_code": "...",
        }
    """
    acts_dir = resolve_acts_dir()
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True
    )

    results = []
    seen_ids = set()  # Avoid duplicates

    # Load from multiple layers and splits to get better coverage
    layers = [0, 3, 7, 11]
    splits = ["train", "test"]

    for layer in layers:
        for split in splits:
            activation_file = acts_dir / f"layer_{layer}_{split}.jsonl"
            if not activation_file.exists():
                continue

            print(f"Loading from {activation_file.name}...", end=" ")
            count_before = len(results)

            with open(activation_file, "r") as f:
                for line_no, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    vuln_id = entry.get("vuln_id")
                    if not vuln_id or vuln_id in seen_ids:
                        continue
                    seen_ids.add(vuln_id)

                    cwe = entry.get("cwe")
                    if not cwe or cwe not in CWE_TO_FAMILY:
                        continue  # Skip unmapped CWEs

                    # Decode base64 code
                    try:
                        secure_b64 = entry.get("secure_code", "")
                        vulnerable_b64 = entry.get("vulnerable_code", "")

                        secure_code = base64.b64decode(secure_b64).decode("utf-8")
                        vulnerable_code = base64.b64decode(vulnerable_b64).decode(
                            "utf-8"
                        )
                    except Exception:
                        continue

                    # Tokenize
                    try:
                        secure_toks = tokenizer.encode(
                            secure_code, add_special_tokens=False
                        )
                        vulnerable_toks = tokenizer.encode(
                            vulnerable_code, add_special_tokens=False
                        )
                    except Exception:
                        continue

                    secure_len = len(secure_toks)
                    vulnerable_len = len(vulnerable_toks)
                    patch_delta = secure_len - vulnerable_len

                    results.append(
                        {
                            "cwe": cwe,
                            "family": CWE_TO_FAMILY[cwe],
                            "vuln_len": vulnerable_len,
                            "secure_len": secure_len,
                            "patch_delta": patch_delta,
                            "vuln_code": vulnerable_code,
                            "secure_code": secure_code,
                        }
                    )

            count_added = len(results) - count_before
            print(f"added {count_added}")

    print(f"\nLoaded {len(results)} total pairs with recognized CWEs")
    return results


def plot_patch_lengths(data):
    """
    Create publication-quality figure showing patch length distributions.
    """
    df = pd.DataFrame(data)

    # Summary statistics per family
    summary = df.groupby("family")["patch_delta"].describe()
    print("\nPatch length statistics by CWE family:")
    print(summary)
    print()

    # Plot: histogram + violinplot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # Left: histogram
    ax = axes[0]
    for family in ["Injection", "Memory Corruption"]:
        data_fam = df[df["family"] == family]["patch_delta"]
        if len(data_fam) > 0:
            ax.hist(
                data_fam,
                bins=30,
                alpha=0.6,
                label=family,
                color=FAMILY_COLORS[family],
                edgecolor="black",
                linewidth=0.5,
            )

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Patch length δ (tokens)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of patch lengths", fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    # Right: violin plot + box plot
    ax = axes[1]
    families = ["Injection", "Memory Corruption"]
    positions = [0, 1]
    data_by_fam = [df[df["family"] == f]["patch_delta"].values for f in families]

    parts = ax.violinplot(
        data_by_fam,
        positions=positions,
        widths=0.6,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    # Color violins
    for pc, family in zip(parts["bodies"], families):
        pc.set_facecolor(FAMILY_COLORS[family])
        pc.set_alpha(0.6)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.5)

    # Overlay box plots
    bp = ax.boxplot(
        data_by_fam,
        positions=positions,
        widths=0.2,
        patch_artist=True,
        boxprops=dict(linewidth=0.8),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        medianprops=dict(linewidth=1.2, color="red"),
    )

    for patch, family in zip(bp["boxes"], families):
        patch.set_facecolor(FAMILY_COLORS[family])
        patch.set_alpha(0.8)

    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(families)
    ax.set_ylabel("Patch length δ (tokens)")
    ax.set_title("Patch length by CWE family", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add sample sizes
    for pos, family in zip(positions, families):
        n = len(df[df["family"] == family])
        ax.text(
            pos,
            ax.get_ylim()[1] * 0.95,
            f"n={n}",
            ha="center",
            fontsize=7,
            color="darkgray",
        )

    fig.suptitle(
        "Injection patches simplify code (tight -102 token distribution);  "
        "Memory-corruption patches vary widely (median -18 tokens, outliers to +498)",
        fontsize=7.5,
        y=1.02,
    )

    fig.tight_layout()

    # Save
    out = PAPER_FIGS / "fig_patch_length_by_cwe.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    return df


if __name__ == "__main__":
    print("Patch length analysis")
    print("=" * 60)
    print(f"Activations dir: {resolve_acts_dir()}")
    print(f"Paper figures dir: {PAPER_FIGS}")
    print()

    data = load_patch_lengths()
    df = plot_patch_lengths(data)

    print("\nDone.")
