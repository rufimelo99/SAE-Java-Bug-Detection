"""
Paired sample visualization with vulnerability direction overlay.

Creates a scatter plot showing:
- Paired vulnerable/secure samples connected by lines
- Vulnerability direction prominently overlaid
- Clear per-pair alignment validation

Usage:
    conda run -n sae python visualize_paired_direction.py --layer 15 --n-pairs 30
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from adjustText import adjust_text
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
ARTIFACTS = Path(__file__).parents[2] / "artifacts" / "activations"
PAPER_FIGS = (
    Path(__file__).parents[4]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

RAW_RUN_DIR = (
    ARTIFACTS / "raw_activations" / "vulnerable_code_qwen_coder_standard_16384_raw"
)

SOURCE_JSONL = (
    RAW_RUN_DIR / "activations_layer_0_raw_component_hidden_state_last_token.jsonl"
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]
SEED = 42

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


def load_activations(layer):
    """Load pre-computed activations."""
    safe_pt = RAW_RUN_DIR / f"safe_layer_{layer}.pt"
    vuln_pt = RAW_RUN_DIR / f"vulnerable_layer_{layer}.pt"

    if safe_pt.exists() and vuln_pt.exists():
        import ctypes as _ct

        def _t2np_local(p):
            t = torch.load(p, weights_only=True).float().contiguous()
            buf = (_ct.c_float * t.numel()).from_address(t.data_ptr())
            return np.ctypeslib.as_array(buf).reshape(t.shape).copy()

        return _t2np_local(safe_pt), _t2np_local(vuln_pt)
    else:
        # Try JSONL fallback
        matches = list(RAW_RUN_DIR.glob(f"activations_layer_{layer}_*.jsonl"))
        if not matches:
            return None, None

        safe_rows, vuln_rows = [], []
        with matches[0].open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                safe_rows.append(r["secure"])
                vuln_rows.append(r["vulnerable"])
        return (
            np.array(safe_rows, dtype=np.float32),
            np.array(vuln_rows, dtype=np.float32),
        )


def load_metadata(jsonl_path=SOURCE_JSONL):
    """Load CVE IDs and CWEs from source JSONL."""
    metadata = []
    if not jsonl_path.exists():
        return metadata
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            metadata.append(
                {
                    "vuln_id": r.get("vuln_id", "unknown"),
                    "cwe": r.get("cwe", ""),
                }
            )
    return metadata


def compute_vulnerability_direction(safe_mat, vuln_mat):
    """Compute d = normalize(mean(vuln) - mean(safe))."""
    mean_vuln = vuln_mat.mean(axis=0)
    mean_safe = safe_mat.mean(axis=0)
    direction = mean_vuln - mean_safe
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    return direction


def visualize_paired_direction(layer, n_pairs=30, out_path=None):
    """
    Create scatter plot showing paired samples and vulnerability direction.

    - Safe samples (blue), vulnerable samples (red)
    - Lines connect pairs
    - Vulnerability direction shown as bold arrow
    """
    print(f"Loading activations for layer {layer}...")
    safe_mat, vuln_mat = load_activations(layer)
    if safe_mat is None:
        print(f"  No data for layer {layer}")
        return

    n = min(n_pairs, len(safe_mat))
    safe_mat = safe_mat[:n]
    vuln_mat = vuln_mat[:n]

    print(f"  Loaded {n} pairs, selecting cleanest examples...")

    # Compute vulnerability direction
    direction = compute_vulnerability_direction(safe_mat, vuln_mat)

    # Stack for PCA (reduce to 2D)
    X_all = np.vstack([safe_mat, vuln_mat])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    pca = PCA(n_components=2, random_state=SEED)
    X_2d = pca.fit_transform(X_scaled)

    # Project direction onto PCA space
    direction_scaled = scaler.transform(direction.reshape(1, -1))[0]
    direction_2d = pca.transform(direction_scaled.reshape(1, -1))[0]
    direction_2d = direction_2d / (np.linalg.norm(direction_2d) + 1e-8)

    safe_2d = X_2d[:n]
    vuln_2d = X_2d[n : 2 * n]

    # Select cleanest pairs: those where pair vector aligns with direction in 2D
    X_all_scaled = scaler.transform(X_all)
    projections = X_all_scaled @ direction_scaled

    # For each pair i, compute alignment in 2D space
    # pair_vector = vuln_2d - safe_2d, alignment = cos(angle) with direction_2d
    pair_scores = []
    for i in range(n):
        pair_vec_2d = vuln_2d[i] - safe_2d[i]
        pair_norm = np.linalg.norm(pair_vec_2d) + 1e-8
        # Cosine similarity between pair vector and direction
        alignment_2d = np.dot(pair_vec_2d, direction_2d) / pair_norm
        pair_scores.append((alignment_2d, i))

    # Sort by alignment (descending) and take top n_clean
    n_clean = 5  # Keep original function at 5 for backwards compatibility
    pair_scores.sort(reverse=True)
    clean_indices = [idx for _, idx in pair_scores[:n_clean]]
    clean_indices.sort()

    safe_2d_clean = safe_2d[clean_indices]
    vuln_2d_clean = vuln_2d[clean_indices]

    # Load metadata for annotations
    metadata = load_metadata()

    # Create figure
    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    # Plot paired connections
    for i, idx in enumerate(clean_indices):
        ax.plot(
            [safe_2d_clean[i, 0], vuln_2d_clean[i, 0]],
            [safe_2d_clean[i, 1], vuln_2d_clean[i, 1]],
            color="#888888",
            alpha=0.5,
            lw=1.3,
            zorder=1,
        )

    # Plot samples (larger, with borders)
    ax.scatter(
        safe_2d_clean[:, 0],
        safe_2d_clean[:, 1],
        c="#2E86AB",
        s=140,
        alpha=0.9,
        label="Secure",
        zorder=3,
        edgecolors="black",
        linewidth=0.7,
    )
    ax.scatter(
        vuln_2d_clean[:, 0],
        vuln_2d_clean[:, 1],
        c="#A23B72",
        s=140,
        alpha=0.9,
        label="Vulnerable",
        zorder=3,
        edgecolors="black",
        linewidth=0.7,
    )

    # Add vulnerability ID/CWE annotations near vulnerable samples
    if metadata:
        for i, idx in enumerate(clean_indices):
            if idx < len(metadata):
                meta = metadata[idx]
                vuln_id = meta.get("vuln_id", "unknown")
                cwe = meta.get("cwe", "")
                label_text = f"{vuln_id}\n{cwe}"
                ax.text(
                    vuln_2d_clean[i, 0] + 1.0,
                    vuln_2d_clean[i, 1],
                    label_text,
                    fontsize=6.5,
                    color="#333333",
                    alpha=0.85,
                    verticalalignment="center",
                    bbox=dict(
                        boxstyle="round,pad=0.35",
                        facecolor="lightyellow",
                        alpha=0.75,
                        edgecolor="none",
                    ),
                    zorder=4,
                )

    # Don't plot arrow—will be shown in histogram companion

    # Compute per-pair alignment across ALL pairs in 2D space (for reference)
    pair_alignment_all = 0
    for i in range(n):
        pair_vec_2d = vuln_2d[i] - safe_2d[i]
        alignment_2d = np.dot(pair_vec_2d, direction_2d)
        if alignment_2d > 0:  # if pair points in same general direction as arrow
            pair_alignment_all += 1
    alignment_pct_all = 100.0 * pair_alignment_all / n

    # Tight axis scaling
    x_margin = (safe_2d_clean[:, 0].max() - safe_2d_clean[:, 0].min()) * 0.15
    y_margin = (safe_2d_clean[:, 1].max() - safe_2d_clean[:, 1].min()) * 0.15
    x_margin = max(
        x_margin, (vuln_2d_clean[:, 0].max() - vuln_2d_clean[:, 0].min()) * 0.15
    )
    y_margin = max(
        y_margin, (vuln_2d_clean[:, 1].max() - vuln_2d_clean[:, 1].min()) * 0.15
    )

    ax.set_xlim(
        min(safe_2d_clean[:, 0].min(), vuln_2d_clean[:, 0].min()) - x_margin,
        max(safe_2d_clean[:, 0].max(), vuln_2d_clean[:, 0].max()) + x_margin,
    )
    ax.set_ylim(
        min(safe_2d_clean[:, 1].min(), vuln_2d_clean[:, 1].min()) - y_margin,
        max(safe_2d_clean[:, 1].max(), vuln_2d_clean[:, 1].max()) + y_margin,
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        f"Layer {layer}: Paired Examples",
        fontsize=11,
        pad=12,
        weight="normal",
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")

    fig.tight_layout()
    if out_path is None:
        out_path = PAPER_FIGS / f"fig_paired_direction_L{layer}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(
        f"  Full dataset alignment: {alignment_pct_all:.0f}% ({pair_alignment_all}/{n} pairs)\n"
    )


def visualize_paired_with_alignment_histogram(layer, n_pairs=100, out_path=None):
    """
    Create side-by-side visualization:
    - LEFT: Scatter plot of paired examples
    - RIGHT: Histogram showing per-pair alignment distribution
    """
    print(f"Loading activations for layer {layer}...")
    safe_mat, vuln_mat = load_activations(layer)
    if safe_mat is None:
        print(f"  No data for layer {layer}")
        return

    n = min(n_pairs, len(safe_mat))
    safe_mat = safe_mat[:n]
    vuln_mat = vuln_mat[:n]

    # Compute vulnerability direction and alignments
    direction = compute_vulnerability_direction(safe_mat, vuln_mat)
    X_all = np.vstack([safe_mat, vuln_mat])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    pca = PCA(n_components=2, random_state=SEED)
    X_2d = pca.fit_transform(X_scaled)

    direction_scaled = scaler.transform(direction.reshape(1, -1))[0]
    direction_2d = pca.transform(direction_scaled.reshape(1, -1))[0]
    direction_2d = direction_2d / (np.linalg.norm(direction_2d) + 1e-8)

    safe_2d = X_2d[:n]
    vuln_2d = X_2d[n : 2 * n]

    metadata = load_metadata()

    # Select cleanest pairs for LEFT plot
    pair_vectors_2d = vuln_2d - safe_2d
    pair_scores = []
    for i in range(n):
        pair_vec_2d = pair_vectors_2d[i]
        pair_norm = np.linalg.norm(pair_vec_2d) + 1e-8
        alignment_2d = np.dot(pair_vec_2d, direction_2d) / pair_norm
        pair_scores.append((alignment_2d, i))

    n_clean = 10  # Show 10 pairs
    pair_scores.sort(reverse=True)
    clean_indices = [idx for _, idx in pair_scores[:n_clean]]
    clean_indices.sort()

    safe_2d_clean = safe_2d[clean_indices]
    vuln_2d_clean = vuln_2d[clean_indices]

    # Compute alignment for ALL pairs in original space (for RIGHT histogram)
    # This matches the per-pair alignment reported in the paper
    all_alignments = []
    for i in range(n):
        # Pair vector in original high-dimensional space
        pair_vec_orig = vuln_mat[i] - safe_mat[i]
        # Alignment with vulnerability direction in original space
        alignment_orig = np.dot(pair_vec_orig, direction)
        all_alignments.append(alignment_orig)
    all_alignments = np.array(all_alignments)

    # Create side-by-side figure (increased size for 50 pairs)
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.1, 0.9]}
    )

    # LEFT PLOT: Scatter without arrow
    for i, idx in enumerate(clean_indices):
        ax_left.plot(
            [safe_2d_clean[i, 0], vuln_2d_clean[i, 0]],
            [safe_2d_clean[i, 1], vuln_2d_clean[i, 1]],
            color="#888888",
            alpha=0.5,
            lw=1.3,
            zorder=1,
        )

    ax_left.scatter(
        safe_2d_clean[:, 0],
        safe_2d_clean[:, 1],
        c="#2E86AB",
        s=140,
        alpha=0.9,
        label="Secure",
        zorder=3,
        edgecolors="black",
        linewidth=0.7,
    )
    ax_left.scatter(
        vuln_2d_clean[:, 0],
        vuln_2d_clean[:, 1],
        c="#A23B72",
        s=140,
        alpha=0.9,
        label="Vulnerable",
        zorder=3,
        edgecolors="black",
        linewidth=0.7,
    )

    # Add CVE/CWE annotations — adjustText handles non-overlap automatically
    texts = []
    if metadata:
        for i, idx in enumerate(clean_indices):
            if idx < len(metadata):
                meta = metadata[idx]
                vuln_id = meta.get("vuln_id", "unknown")
                cwe = meta.get("cwe", "")
                label_text = f"{vuln_id}\n{cwe}"
                t = ax_left.text(
                    vuln_2d_clean[i, 0],
                    vuln_2d_clean[i, 1],
                    label_text,
                    fontsize=6,
                    color="#333333",
                    alpha=0.90,
                    ha="center",
                    va="bottom",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="lightyellow",
                        alpha=0.80,
                        edgecolor="#cccccc",
                        linewidth=0.5,
                    ),
                    zorder=4,
                )
                texts.append((t, vuln_2d_clean[i, 0], vuln_2d_clean[i, 1]))

    if texts:
        text_objs = [t for t, _, _ in texts]
        xs = np.array([x for _, x, _ in texts])
        ys = np.array([y for _, _, y in texts])
        adjust_text(
            text_objs,
            x=xs,
            y=ys,
            ax=ax_left,
            expand=(1.6, 2.0),
            force_text=(0.8, 1.5),
            force_points=(0.4, 0.8),
            only_move={"text": "y", "points": "y"},
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.6),
        )

    x_margin = (safe_2d_clean[:, 0].max() - safe_2d_clean[:, 0].min()) * 0.15
    y_margin = (safe_2d_clean[:, 1].max() - safe_2d_clean[:, 1].min()) * 0.15
    x_margin = max(
        x_margin, (vuln_2d_clean[:, 0].max() - vuln_2d_clean[:, 0].min()) * 0.15
    )
    y_margin = max(
        y_margin, (vuln_2d_clean[:, 1].max() - vuln_2d_clean[:, 1].min()) * 0.15
    )

    ax_left.set_xlim(
        min(safe_2d_clean[:, 0].min(), vuln_2d_clean[:, 0].min()) - x_margin,
        max(safe_2d_clean[:, 0].max(), vuln_2d_clean[:, 0].max()) + x_margin,
    )
    ax_left.set_ylim(
        min(safe_2d_clean[:, 1].min(), vuln_2d_clean[:, 1].min()) - y_margin,
        max(safe_2d_clean[:, 1].max(), vuln_2d_clean[:, 1].max()) + y_margin,
    )

    ax_left.set_xlabel("PC1")
    ax_left.set_ylabel("PC2")
    ax_left.set_title(f"Layer {layer}: Paired Examples", fontsize=11, pad=10)
    ax_left.legend(
        loc="lower center",
        fontsize=8,
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
    )
    ax_left.grid(False)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.spines["left"].set_color("#CCCCCC")
    ax_left.spines["bottom"].set_color("#CCCCCC")

    # RIGHT PLOT: Histogram of alignment scores
    alignment_pct_positive = 100.0 * np.sum(all_alignments > 0) / len(all_alignments)

    ax_right.hist(
        all_alignments,
        bins=20,
        color="#555555",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.7,
    )
    ax_right.axvline(
        0, color="#D62828", lw=2.5, linestyle="--", alpha=0.8, label="Zero alignment"
    )
    ax_right.set_xlabel("Per-pair alignment score", fontsize=10)
    ax_right.set_ylabel("Frequency", fontsize=10)
    ax_right.set_title(
        f"Alignment distribution\n({alignment_pct_positive:.0f}% > 0)",
        fontsize=10,
        pad=10,
    )
    ax_right.grid(False)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.spines["left"].set_color("#CCCCCC")
    ax_right.spines["bottom"].set_color("#CCCCCC")
    ax_right.legend(loc="best", fontsize=8)

    fig.tight_layout()
    if out_path is None:
        out_path = PAPER_FIGS / f"fig_paired_direction_L{layer}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(
        f"  Alignment: {alignment_pct_positive:.0f}% positive ({np.sum(all_alignments > 0)}/{n} pairs)\n"
    )


def visualize_paired_with_alignment_histogram_3d(layer, n_pairs=100, out_path=None):
    """
    Create side-by-side visualization with 3D:
    - LEFT: 3D scatter plot of paired examples (PC1, PC2, PC3)
    - RIGHT: Histogram showing per-pair alignment distribution
    """
    print(f"Loading activations for layer {layer}...")
    safe_mat, vuln_mat = load_activations(layer)
    if safe_mat is None:
        print(f"  No data for layer {layer}")
        return

    n = min(n_pairs, len(safe_mat))
    safe_mat = safe_mat[:n]
    vuln_mat = vuln_mat[:n]

    # Compute vulnerability direction and alignments
    direction = compute_vulnerability_direction(safe_mat, vuln_mat)
    X_all = np.vstack([safe_mat, vuln_mat])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    pca = PCA(n_components=3, random_state=SEED)
    X_3d = pca.fit_transform(X_scaled)

    direction_scaled = scaler.transform(direction.reshape(1, -1))[0]
    direction_3d = pca.transform(direction_scaled.reshape(1, -1))[0]
    direction_3d = direction_3d / (np.linalg.norm(direction_3d) + 1e-8)

    safe_3d = X_3d[:n]
    vuln_3d = X_3d[n : 2 * n]

    metadata = load_metadata()

    # Select cleanest pairs for LEFT plot
    pair_vectors_3d = vuln_3d - safe_3d
    pair_scores = []
    for i in range(n):
        pair_vec_3d = pair_vectors_3d[i]
        pair_norm = np.linalg.norm(pair_vec_3d) + 1e-8
        alignment_3d = np.dot(pair_vec_3d, direction_3d) / pair_norm
        pair_scores.append((alignment_3d, i))

    n_clean = 10  # Show 10 pairs
    pair_scores.sort(reverse=True)
    clean_indices = [idx for _, idx in pair_scores[:n_clean]]
    clean_indices.sort()

    safe_3d_clean = safe_3d[clean_indices]
    vuln_3d_clean = vuln_3d[clean_indices]

    # Compute alignment for ALL pairs in original space (for RIGHT histogram)
    all_alignments = []
    for i in range(n):
        pair_vec_orig = vuln_mat[i] - safe_mat[i]
        alignment_orig = np.dot(pair_vec_orig, direction)
        all_alignments.append(alignment_orig)
    all_alignments = np.array(all_alignments)

    # Create side-by-side figure with 3D on left
    fig = plt.figure(figsize=(14, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9])
    ax_left = fig.add_subplot(gs[0], projection="3d")
    ax_right = fig.add_subplot(gs[1])

    # LEFT PLOT: 3D Scatter
    for i, idx in enumerate(clean_indices):
        ax_left.plot(
            [safe_3d_clean[i, 0], vuln_3d_clean[i, 0]],
            [safe_3d_clean[i, 1], vuln_3d_clean[i, 1]],
            [safe_3d_clean[i, 2], vuln_3d_clean[i, 2]],
            color="#888888",
            alpha=0.5,
            lw=1.3,
            zorder=1,
        )

    ax_left.scatter(
        safe_3d_clean[:, 0],
        safe_3d_clean[:, 1],
        safe_3d_clean[:, 2],
        c="#2E86AB",
        s=140,
        alpha=0.9,
        label="Secure",
        zorder=3,
        edgecolors="black",
        linewidth=0.7,
    )
    ax_left.scatter(
        vuln_3d_clean[:, 0],
        vuln_3d_clean[:, 1],
        vuln_3d_clean[:, 2],
        c="#A23B72",
        s=140,
        alpha=0.9,
        label="Vulnerable",
        zorder=3,
        edgecolors="black",
        linewidth=0.7,
    )

    # Add CWE annotations only to top 5 pairs to reduce clutter
    if metadata:
        for i, idx in enumerate(clean_indices[:5]):  # Label only first 5 pairs
            meta = metadata[idx]
            cwe = meta.get("cwe", "")
            if cwe:
                # Simple offset: move slightly right and up
                ax_left.text(
                    vuln_3d_clean[i, 0] + 3,
                    vuln_3d_clean[i, 1] + 1.5,
                    vuln_3d_clean[i, 2],
                    cwe,
                    fontsize=7.5,
                    color="#333333",
                    alpha=0.95,
                    weight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.35",
                        facecolor="lightyellow",
                        alpha=0.85,
                        edgecolor="#999999",
                        linewidth=0.6,
                    ),
                    zorder=5,
                )

    ax_left.set_xlabel("PC1", fontsize=9)
    ax_left.set_ylabel("PC2", fontsize=9)
    ax_left.set_zlabel("PC3", fontsize=9)
    ax_left.set_title(f"Layer {layer}: Paired Examples (3D)", fontsize=11, pad=10)
    ax_left.legend(loc="upper left", fontsize=8)
    ax_left.grid(False)

    # RIGHT PLOT: Histogram of alignment scores
    alignment_pct_positive = 100.0 * np.sum(all_alignments > 0) / len(all_alignments)

    ax_right.hist(
        all_alignments,
        bins=20,
        color="#555555",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.7,
    )
    ax_right.axvline(
        0, color="#D62828", lw=2.5, linestyle="--", alpha=0.8, label="Zero alignment"
    )
    ax_right.set_xlabel("Per-pair alignment score", fontsize=10)
    ax_right.set_ylabel("Frequency", fontsize=10)
    ax_right.set_title(
        f"Alignment distribution\n({alignment_pct_positive:.0f}% > 0)",
        fontsize=10,
        pad=10,
    )
    ax_right.grid(False)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.spines["left"].set_color("#CCCCCC")
    ax_right.spines["bottom"].set_color("#CCCCCC")
    ax_right.legend(loc="best", fontsize=8)

    fig.tight_layout()
    if out_path is None:
        out_path = PAPER_FIGS / f"fig_paired_direction_L{layer}_3d.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(
        f"  Alignment: {alignment_pct_positive:.0f}% positive ({np.sum(all_alignments > 0)}/{n} pairs)\n"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=15, help="Layer to visualize")
    parser.add_argument(
        "--n-pairs", type=int, default=30, help="Number of pairs to show"
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, help="Multiple layers (overrides --layer)"
    )
    parser.add_argument(
        "--3d",
        action="store_true",
        help="Use 3D visualization instead of 2D",
    )
    args = parser.parse_args()

    layers_to_plot = args.layers if args.layers else [args.layer]

    print("\n[Paired Direction Visualization]\n")
    viz_func = (
        visualize_paired_with_alignment_histogram_3d
        if args.__dict__.get("3d")
        else visualize_paired_with_alignment_histogram
    )
    for layer in layers_to_plot:
        viz_func(layer, n_pairs=args.n_pairs)


if __name__ == "__main__":
    main()
