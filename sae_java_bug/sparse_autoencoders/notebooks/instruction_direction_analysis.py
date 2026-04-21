"""
Direction Analysis from Saved Instruction Activations

Analyzes direction properties from pre-computed activations:
1. When the vulnerability direction emerges (emergence layer)
2. Cross-layer consistency (direction stability)
3. Per-pair alignment (do pairs agree on the direction?)
4. Binary vulnerability probing (AUROC comparison)

Input:
    Activations saved by instruction_activations_extract.py

Usage:
    # Analyze the latest run:
    conda run -n sae python instruction_direction_analysis.py

    # Analyze a specific run:
    conda run -n sae python instruction_direction_analysis.py --run_ts 20260421_120000

Outputs:
    artifacts/activations/instruction_comparison/<run_ts>/
        comparison_metrics.json
    figures/fig_instruction_direction_comparison.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
ARTIFACTS = Path(__file__).parents[2] / "artifacts" / "activations"
PAPER_FIGS = (
    Path(__file__).parents[4]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]
SEED = 42

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# Activation loading
# ─────────────────────────────────────────────────────────────────────────────


def find_latest_instruction_run() -> Path:
    """Find the latest instruction_comparison run directory."""
    runs = sorted((ARTIFACTS / "instruction_comparison").glob("*/meta.json"))
    if not runs:
        raise FileNotFoundError(
            f"No instruction_comparison runs found under {ARTIFACTS}/instruction_comparison/"
        )
    return runs[-1].parent


def load_activations(run_dir: Path, condition: str) -> dict[int, np.ndarray]:
    """
    Load activations for a condition (baseline or instruction).
    condition: 'baseline' or 'instruction'
    Returns {layer: [safe_acts, vuln_acts]}
    """
    if condition not in ["baseline", "instruction"]:
        raise ValueError(f"Invalid condition: {condition}")

    result = {}
    for layer in LAYERS:
        safe_path = run_dir / f"{condition}_safe_layer_{layer}.pt"
        vuln_path = run_dir / f"{condition}_vulnerable_layer_{layer}.pt"

        if not safe_path.exists() or not vuln_path.exists():
            raise FileNotFoundError(
                f"Missing activations for layer {layer} in {run_dir}"
            )

        safe = torch.load(safe_path, weights_only=True).numpy()
        vuln = torch.load(vuln_path, weights_only=True).numpy()
        result[layer] = (safe, vuln)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Direction analysis
# ─────────────────────────────────────────────────────────────────────────────


def vuln_direction(safe: np.ndarray, vuln: np.ndarray) -> np.ndarray:
    """Unit vector pointing from mean-secure to mean-vulnerable."""
    d = vuln.mean(0) - safe.mean(0)
    norm = np.linalg.norm(d)
    return d / norm if norm > 1e-10 else d


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)


def per_pair_alignment(
    safe: np.ndarray, vuln: np.ndarray, direction: np.ndarray
) -> float:
    """
    Fraction of pairs where (vuln - safe) · direction > 0.
    This should correlate with probe AUROC.
    """
    deltas = vuln - safe
    dots = np.dot(deltas, direction)
    return float((dots > 0).mean())


def direction_emergence_layer(
    safe_dict: dict, vuln_dict: dict, threshold: float = 0.5
) -> int:
    """
    Find the earliest layer where direction alignment exceeds threshold.
    Returns the layer index, or -1 if no layer meets threshold.
    """
    for layer in LAYERS:
        safe, vuln = safe_dict[layer], vuln_dict[layer]
        direction = vuln_direction(safe, vuln)
        alignment = per_pair_alignment(safe, vuln, direction)
        if alignment >= threshold:
            return layer
    return -1


# ─────────────────────────────────────────────────────────────────────────────
# Probe utilities
# ─────────────────────────────────────────────────────────────────────────────


def bootstrap_auc_ci(y_true, y_score, n_bootstrap=500, ci=0.95, seed=SEED):
    """Compute AUROC with 95% bootstrap CI."""
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_score[idx]))
    scores = np.array(scores)
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return float(scores.mean()), float(lo), float(hi)


def probe_activations(safe: np.ndarray, vuln: np.ndarray, seed=SEED):
    """
    Logistic regression probe: vulnerable vs. secure.
    Return AUROC with 95% CI.
    """
    X = np.vstack([safe, vuln])
    y = np.array([0] * len(safe) + [1] * len(vuln))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA-50 for dimensionality reduction
    pca = PCA(n_components=50, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)

    # Cross-validated probe
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    y_pred = np.zeros_like(y, dtype=float)

    for train_idx, test_idx in cv.split(X_pca, y):
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_pca[train_idx], y[train_idx])
        y_pred[test_idx] = clf.predict_proba(X_pca[test_idx])[:, 1]

    auroc, lo, hi = bootstrap_auc_ci(y, y_pred)
    return auroc, lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────


def main(run_dir: Path = None):
    """Run direction analysis on saved activations."""
    if run_dir is None:
        print("[*] Finding latest instruction_comparison run...")
        run_dir = find_latest_instruction_run()

    print(f"[*] Analyzing run: {run_dir}")

    # Load metadata
    meta_path = run_dir / "meta.json"
    with meta_path.open() as f:
        meta = json.load(f)

    print(f"    Timestamp: {meta['run_ts']}")
    print(f"    Baseline pairs: {meta['n_baseline_pairs']}")
    print(f"    Instruction pairs: {meta['n_instruction_pairs']}")

    # Load activations
    print("[*] Loading activations...")
    baseline_acts = load_activations(run_dir, "baseline")
    instruction_acts = load_activations(run_dir, "instruction")

    # Compute metrics
    print("[*] Computing direction metrics...")
    metrics = {
        "baseline": {},
        "instruction": {},
        "comparison": {},
    }

    # Baseline direction analysis
    baseline_directions = {}
    for layer in LAYERS:
        safe, vuln = baseline_acts[layer]
        direction = vuln_direction(safe, vuln)
        baseline_directions[layer] = direction
        alignment = per_pair_alignment(safe, vuln, direction)
        auroc, lo, hi = probe_activations(safe, vuln)

        metrics["baseline"][layer] = {
            "alignment_score": float(alignment),
            "auroc": float(auroc),
            "auroc_ci": [float(lo), float(hi)],
        }

    # Instruction direction analysis
    instruction_directions = {}
    for layer in LAYERS:
        safe, vuln = instruction_acts[layer]
        direction = vuln_direction(safe, vuln)
        instruction_directions[layer] = direction
        alignment = per_pair_alignment(safe, vuln, direction)
        auroc, lo, hi = probe_activations(safe, vuln)

        metrics["instruction"][layer] = {
            "alignment_score": float(alignment),
            "auroc": float(auroc),
            "auroc_ci": [float(lo), float(hi)],
        }

    # Cross-condition direction stability
    print("[*] Computing cross-condition direction stability...")
    for layer in LAYERS:
        cos_sim = cosine_similarity(
            baseline_directions[layer], instruction_directions[layer]
        )
        metrics["comparison"][layer] = {
            "direction_cosine_similarity": float(cos_sim),
            "baseline_alignment": float(metrics["baseline"][layer]["alignment_score"]),
            "instruction_alignment": float(
                metrics["instruction"][layer]["alignment_score"]
            ),
            "alignment_delta": float(
                metrics["instruction"][layer]["alignment_score"]
                - metrics["baseline"][layer]["alignment_score"]
            ),
            "baseline_auroc": float(metrics["baseline"][layer]["auroc"]),
            "instruction_auroc": float(metrics["instruction"][layer]["auroc"]),
            "auroc_delta": float(
                metrics["instruction"][layer]["auroc"]
                - metrics["baseline"][layer]["auroc"]
            ),
        }

    # Direction emergence
    baseline_safe = {l: baseline_acts[l][0] for l in LAYERS}
    baseline_vuln = {l: baseline_acts[l][1] for l in LAYERS}
    instruction_safe = {l: instruction_acts[l][0] for l in LAYERS}
    instruction_vuln = {l: instruction_acts[l][1] for l in LAYERS}

    baseline_emergence = direction_emergence_layer(
        baseline_safe, baseline_vuln, threshold=0.5
    )
    instruction_emergence = direction_emergence_layer(
        instruction_safe, instruction_vuln, threshold=0.5
    )

    metrics["emergence"] = {
        "baseline_emergence_layer": baseline_emergence,
        "instruction_emergence_layer": instruction_emergence,
    }

    # Save metrics
    metrics_path = run_dir / "comparison_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[+] Metrics saved to {metrics_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("DIRECTION EMERGENCE")
    print("=" * 70)
    print(f"Baseline emergence layer (threshold 0.5): {baseline_emergence}")
    print(f"Instruction emergence layer (threshold 0.5):   {instruction_emergence}")

    print("\n" + "=" * 70)
    print("CROSS-LAYER ALIGNMENT & AUROC COMPARISON")
    print("=" * 70)
    print(
        f"{'Layer':<8} {'Baseline Align':<16} {'Instr Align':<14} {'Align Δ':<12} {'Dir Cos-Sim':<12}"
    )
    print("-" * 70)
    for layer in LAYERS:
        baseline_align = metrics["baseline"][layer]["alignment_score"]
        instruction_align = metrics["instruction"][layer]["alignment_score"]
        align_delta = metrics["comparison"][layer]["alignment_delta"]
        cos_sim = metrics["comparison"][layer]["direction_cosine_similarity"]
        print(
            f"{layer:<8} {baseline_align:.4f}{'':<11} {instruction_align:.4f}{'':<9} {align_delta:+.4f}{'':<7} {cos_sim:.4f}"
        )

    print("\n" + "=" * 70)
    print("BINARY VULNERABILITY PROBING (AUROC)")
    print("=" * 70)
    print(f"{'Layer':<8} {'Baseline AUROC':<16} {'Instr AUROC':<14} {'AUROC Δ':<12}")
    print("-" * 70)
    for layer in LAYERS:
        baseline_auroc = metrics["baseline"][layer]["auroc"]
        instruction_auroc = metrics["instruction"][layer]["auroc"]
        auroc_delta = metrics["comparison"][layer]["auroc_delta"]
        print(
            f"{layer:<8} {baseline_auroc:.4f}{'':<11} {instruction_auroc:.4f}{'':<9} {auroc_delta:+.4f}"
        )

    # Generate comparison figure
    print("\n[*] Generating comparison figure...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Alignment score comparison
    ax = axes[0, 0]
    baseline_aligns = [metrics["baseline"][l]["alignment_score"] for l in LAYERS]
    instruction_aligns = [metrics["instruction"][l]["alignment_score"] for l in LAYERS]
    ax.plot(LAYERS, baseline_aligns, "o-", label="Baseline", linewidth=2, markersize=6)
    ax.plot(
        LAYERS,
        instruction_aligns,
        "s-",
        label="With Instruction",
        linewidth=2,
        markersize=6,
    )
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Per-Pair Alignment Score")
    ax.set_title("Direction Alignment Across Layers")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: AUROC comparison
    ax = axes[0, 1]
    baseline_aurocs = [metrics["baseline"][l]["auroc"] for l in LAYERS]
    instruction_aurocs = [metrics["instruction"][l]["auroc"] for l in LAYERS]
    ax.plot(LAYERS, baseline_aurocs, "o-", label="Baseline", linewidth=2, markersize=6)
    ax.plot(
        LAYERS,
        instruction_aurocs,
        "s-",
        label="With Instruction",
        linewidth=2,
        markersize=6,
    )
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC (Vulnerable vs. Secure)")
    ax.set_title("Binary Vulnerability Probing")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 0.7])

    # Plot 3: Direction cosine similarity
    ax = axes[1, 0]
    cos_sims = [metrics["comparison"][l]["direction_cosine_similarity"] for l in LAYERS]
    ax.plot(LAYERS, cos_sims, "o-", color="purple", linewidth=2, markersize=6)
    ax.axhline(
        0.9, color="gray", linestyle="--", alpha=0.5, label="High consistency (0.9)"
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity (Baseline vs. Instruction Direction)")
    ax.set_title("Direction Stability Across Conditions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.7, 1.05])

    # Plot 4: Alignment delta
    ax = axes[1, 1]
    align_deltas = [metrics["comparison"][l]["alignment_delta"] for l in LAYERS]
    colors = ["green" if d > 0 else "red" for d in align_deltas]
    ax.bar(range(len(LAYERS)), align_deltas, color=colors, alpha=0.7)
    ax.axhline(0, color="black", linestyle="-", linewidth=1)
    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels(LAYERS)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Alignment Score Delta (Instruction - Baseline)")
    ax.set_title("Effect of Security Instruction on Direction Alignment")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig_path = PAPER_FIGS / "fig_instruction_direction_comparison.pdf"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"[+] Figure saved to {fig_path}")

    print("\n[+] Analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze saved instruction activation comparisons"
    )
    parser.add_argument(
        "--run_ts",
        type=str,
        default=None,
        help="Run timestamp (e.g., 20260421_120000). If not provided, uses latest.",
    )
    args = parser.parse_args()

    if args.run_ts:
        run_dir = ARTIFACTS / "instruction_comparison" / args.run_ts
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
    else:
        run_dir = None

    main(run_dir)
