"""
CWE-type pairwise probes using RAW (embedding-space) last-token activations.

Compares raw 3584-dim embeddings vs. SAE-encoded features for the same CWE
pairwise tasks. Uses vulnerable-side activations with mean-token pooling over
the (vulnerable, secure) deltas.

Usage:
    python cwe_pairwise_raw_activations.py [--layer 11]

Output:
    results/cwe_pairwise_raw/
        heatmap_c_raw.pdf
        heatmap_php_raw.pdf
        heatmap_cpp_raw.pdf
        heatmap_js_raw.pdf
        metrics.json
"""

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent.resolve()
ARTIFACTS = HERE.parents[1] / "artifacts" / "activations"
RESULTS_DIR = ARTIFACTS / "cwe_pairwise_raw"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PAPER_FIGS = (
    HERE.parents[3]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]
SEED = 42
MIN_N = 10  # min samples per CWE per language

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


# ── Load raw activations ───────────────────────────────────────────────────────
def load_raw_activations(layer: int):
    """Load raw last-token activations from JSONL."""
    jsonl_path = (
        ARTIFACTS
        / "raw_activations/vulnerable_code_qwen_coder_standard_16384_raw"
        / f"activations_layer_{layer}_raw_component_hidden_state_last_token.jsonl"
    )

    records = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            records.append(
                {
                    "vuln_id": r["vuln_id"],
                    "cwe": r["cwe"],
                    "file_extension": r["file_extension"],
                    "secure": np.array(r["secure"], dtype=np.float32),
                    "vulnerable": np.array(r["vulnerable"], dtype=np.float32),
                }
            )
    return records


# ── CWE pairwise probe ─────────────────────────────────────────────────────────
def _bootstrap_auc_ci(y_true, y_score, n=500, ci=0.95, seed=SEED):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    aucs = np.array(aucs)
    if len(aucs) == 0:
        return np.nan, np.nan, np.nan
    alpha = (1 - ci) / 2
    return aucs.mean(), np.quantile(aucs, alpha), np.quantile(aucs, 1 - alpha)


def probe_cwe_pair(cwe1_vecs, cwe2_vecs, n_components=50, cv=5):
    """
    Binary probe: cwe1 (0) vs cwe2 (1).
    Returns AUROC with CI bounds.
    """
    if len(cwe1_vecs) < 2 or len(cwe2_vecs) < 2:
        return {
            "auroc": np.nan,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "n": len(cwe1_vecs) + len(cwe2_vecs),
        }

    X = np.vstack([cwe1_vecs, cwe2_vecs]).astype(np.float32)
    y = np.array([0] * len(cwe1_vecs) + [1] * len(cwe2_vecs), dtype=int)

    n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
    clf = LogisticRegression(
        C=0.1, max_iter=1000, class_weight="balanced", random_state=SEED
    )
    skf = StratifiedKFold(
        n_splits=min(cv, min(len(np.where(y == 0)[0]), len(np.where(y == 1)[0]))),
        shuffle=True,
        random_state=SEED,
    )

    y_score = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        pca = PCA(n_components=min(n_comp, len(train_idx) - 1), random_state=SEED)
        X_tr = pca.fit_transform(scaler.fit_transform(X[train_idx]))
        X_te = pca.transform(scaler.transform(X[test_idx]))
        clf.fit(X_tr, y[train_idx])
        y_score[test_idx] = clf.predict_proba(X_te)[:, 1]

    mean_auc, ci_lo, ci_hi = _bootstrap_auc_ci(y, y_score)
    return {
        "auroc": mean_auc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n": len(cwe1_vecs) + len(cwe2_vecs),
    }


# ── Main analysis ──────────────────────────────────────────────────────────────
def main(layer: int = 11):
    print(f"\n{'='*80}")
    print(f"CWE Pairwise Probes: RAW ACTIVATIONS (Layer {layer})")
    print(f"{'='*80}")

    # Load raw activations
    print(f"\nLoading raw activations for layer {layer}...")
    records = load_raw_activations(layer)
    print(f"  Loaded {len(records)} records")

    # Group by language and CWE
    lang_cwe_data = defaultdict(lambda: defaultdict(list))
    for rec in records:
        ext = rec["file_extension"]
        cwe = rec["cwe"]
        # Use vulnerable-side activation as the feature
        lang_cwe_data[ext][cwe].append(rec["vulnerable"])

    # Filter languages and CWEs by minimum count
    filtered_lang_cwe = {}
    for lang in sorted(lang_cwe_data.keys()):
        cwe_counts = {cwe: len(vecs) for cwe, vecs in lang_cwe_data[lang].items()}
        cwe_list = [cwe for cwe, count in cwe_counts.items() if count >= MIN_N]
        if len(cwe_list) >= 2:
            filtered_lang_cwe[lang] = {
                cwe: lang_cwe_data[lang][cwe] for cwe in cwe_list
            }
            print(f"\n{lang:8s}: {len(cwe_list):2d} CWEs with ≥{MIN_N} samples")
            for cwe in sorted(cwe_list):
                print(
                    f"          {cwe:10s}: {len(filtered_lang_cwe[lang][cwe]):3d} samples"
                )

    # Run pairwise probes per language
    all_results = {}
    metrics_by_lang = {}

    for lang in sorted(filtered_lang_cwe.keys()):
        print(f"\n{'─'*80}")
        print(f"Language: {lang}")
        print(f"{'─'*80}")

        cwe_list = sorted(filtered_lang_cwe[lang].keys())
        cwe_pairs = list(combinations(cwe_list, 2))

        results = {}
        for cwe1, cwe2 in cwe_pairs:
            vec1 = np.array(filtered_lang_cwe[lang][cwe1], dtype=np.float32)
            vec2 = np.array(filtered_lang_cwe[lang][cwe2], dtype=np.float32)
            result = probe_cwe_pair(vec1, vec2)
            results[(cwe1, cwe2)] = result
            auroc = result["auroc"]
            print(f"  {cwe1:10s} vs {cwe2:10s}:  AUROC {auroc:.3f}")

        # Create heatmap
        heatmap_data = np.zeros((len(cwe_list), len(cwe_list)))
        for i, cwe1 in enumerate(cwe_list):
            for j, cwe2 in enumerate(cwe_list):
                if i < j:
                    heatmap_data[i, j] = results[(cwe1, cwe2)]["auroc"]
                elif i > j:
                    heatmap_data[i, j] = results[(cwe2, cwe1)]["auroc"]

        all_results[lang] = {
            "cwe_list": cwe_list,
            "heatmap": heatmap_data,
            "pair_results": results,
        }

        # Compute metrics
        valid_aucs = [r["auroc"] for r in results.values() if not np.isnan(r["auroc"])]
        metrics_by_lang[lang] = {
            "n_pairs": len(cwe_pairs),
            "median_auroc": np.median(valid_aucs) if valid_aucs else np.nan,
            "mean_auroc": np.mean(valid_aucs) if valid_aucs else np.nan,
            "min_auroc": np.min(valid_aucs) if valid_aucs else np.nan,
            "max_auroc": np.max(valid_aucs) if valid_aucs else np.nan,
        }

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            heatmap_data,
            xticklabels=cwe_list,
            yticklabels=cwe_list,
            cmap="RdYlGn",
            vmin=0.4,
            vmax=1.0,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "AUROC"},
            ax=ax,
        )
        ax.set_title(f"CWE Pairwise AUROC (Raw Activations, L{layer}, {lang})")
        ax.set_xlabel("CWE")
        ax.set_ylabel("CWE")
        plt.tight_layout()

        # Save heatmap
        heatmap_path = RESULTS_DIR / f"heatmap_{lang}_raw_l{layer}.pdf"
        fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {heatmap_path}")
        plt.close(fig)

    # Summary
    print(f"\n{'='*80}")
    print(f"Summary: Raw Activations (Layer {layer})")
    print(f"{'='*80}")
    for lang in sorted(metrics_by_lang.keys()):
        m = metrics_by_lang[lang]
        print(
            f"{lang:8s}: {m['n_pairs']:2d} pairs  "
            f"Median AUROC {m['median_auroc']:.3f}  "
            f"Mean {m['mean_auroc']:.3f}  "
            f"Range [{m['min_auroc']:.3f}, {m['max_auroc']:.3f}]"
        )

    # Save metrics
    metrics_path = RESULTS_DIR / f"metrics_l{layer}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_by_lang, f, indent=2, default=str)
    print(f"\nSaved metrics: {metrics_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=11)
    args = parser.parse_args()
    main(layer=args.layer)
