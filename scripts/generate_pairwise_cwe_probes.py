#!/usr/bin/env python3
"""
Generate pairwise CWE-type probe AUROC heatmaps for all models.

This creates heatmaps showing binary classification AUROC between each pair
of specific CWE types (e.g., CWE-119 vs CWE-120) across all layers.

Usage:
    python scripts/generate_pairwise_cwe_probes.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Style matching paper
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

_SCRIPTS_DIR = Path(__file__).parent
_PROJECT_DIR = _SCRIPTS_DIR.parent

ACTIVATIONS_DIR = _PROJECT_DIR / "sae_java_bug" / "artifacts" / "multi_model_probing"
METADATA_FILE = (
    _PROJECT_DIR
    / "sae_java_bug"
    / "artifacts"
    / "activations"
    / "advanced_pool"
    / "20260308_214306"
    / "meta.json"
)
OUTPUT_DIR = (
    _PROJECT_DIR.parent
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
RESULTS_DIR = _PROJECT_DIR / "results" / "raw_data"

# Top CWE types to include
# CWE types to include — covers DeltaSecommits, SVEN, and PreciseBugs
CWE_TYPES = {
    # DeltaSecommits / SVEN
    "CWE-119", "CWE-120", "CWE-125", "CWE-787",
    "CWE-78",  "CWE-89",  "CWE-22",  "CWE-401",
    # PreciseBugs
    "CWE-190", "CWE-122", "CWE-476", "CWE-369",
    "CWE-457", "CWE-121", "CWE-416",
}


# Maps dataset prefix → raw JSONL with CWE labels
_DATASET_METADATA: Dict[str, Path] = {
    "deltasecommits": METADATA_FILE,  # existing meta.json
    "sven": _PROJECT_DIR / "sae_java_bug" / "artifacts" / "data" / "sven_raw" / "sven_c_pairs.jsonl",
    "precisebugs": _PROJECT_DIR / "sae_java_bug" / "artifacts" / "data" / "precisebugs_raw" / "precisebugs_c_pairs.jsonl",
}


def discover_models() -> List[str]:
    """Auto-discover models from available NPZ files (all datasets included)."""
    return [
        npz.stem.replace("activations_", "")
        for npz in sorted(ACTIVATIONS_DIR.glob("activations_*.npz"))
    ]


def _dataset_of(model_full: str) -> str:
    """Return the dataset prefix for a model name, or 'deltasecommits' if none."""
    for ds in _DATASET_METADATA:
        if model_full.startswith(ds + "_"):
            return ds
    return "deltasecommits"


def load_metadata_for(model_full: str) -> np.ndarray:
    """Load CWE labels from the appropriate metadata file for this model/dataset."""
    ds = _dataset_of(model_full)
    path = _DATASET_METADATA.get(ds, METADATA_FILE)
    if not path.exists():
        logger.warning(f"Metadata file not found for dataset '{ds}': {path}")
        return np.array([])
    with open(path) as f:
        content = f.read().strip()
    if content.startswith("["):
        records = json.loads(content)
    else:
        records = [json.loads(line) for line in content.splitlines() if line.strip()]
    def _normalize(cwe: str) -> str:
        # Normalize CWE-022 → CWE-22, CWE-078 → CWE-78, etc.
        if cwe.startswith("CWE-"):
            return "CWE-" + str(int(cwe[4:]))
        return cwe

    return np.array([_normalize(r.get("cwe", "unknown")) for r in records])


def get_peak_layer(activations: Dict[str, np.ndarray]) -> int:
    """Return the highest available layer index from the NPZ keys."""
    layers = sorted(
        int(k.split("_")[1])
        for k in activations
        if k.startswith("layer_") and k.endswith("_vuln_mean")
    )
    if not layers:
        return 15
    # Use the 75th percentile layer as "peak" (avoid very last which may be noisy)
    idx = max(0, int(len(layers) * 0.75) - 1)
    return layers[idx]


def probe_pairwise(
    X: np.ndarray, y: np.ndarray, n_components: int = 50, cv: int = 5
) -> float:
    """
    Binary classification probe: y=1 for CWE1, y=0 for CWE2.
    Returns AUROC.
    """
    if len(np.unique(y)) < 2:
        return 0.5

    n_splits = min(cv, int(y.sum()), int((y == 0).sum()))
    if n_splits < 2:
        return 0.5

    n_comp = min(n_components, X.shape[1], X.shape[0] - 1)

    clf = LogisticRegression(
        C=0.1, max_iter=1000, class_weight="balanced", random_state=42
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_score = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        pca = PCA(n_components=min(n_comp, len(tr) - 1), random_state=42)

        Xtr = pca.fit_transform(scaler.fit_transform(X[tr]))
        Xte = pca.transform(scaler.transform(X[te]))

        clf.fit(Xtr, y[tr])
        y_score[te] = clf.predict_proba(Xte)[:, 1]

    try:
        auroc = roc_auc_score(y, y_score)
    except Exception:
        auroc = 0.5

    return auroc


def save_results(
    model_full: str,
    available_cwes: List[str],
    layer_matrices: Dict[int, np.ndarray],
    peak_layer: int,
):
    """Save per-layer AUROC matrices to JSON for later figure regeneration."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_full,
        "peak_layer": peak_layer,
        "cwe_types": available_cwes,
        "layers": {
            str(layer): {"auroc_matrix": mat.tolist()}
            for layer, mat in sorted(layer_matrices.items())
        },
    }
    out = RESULTS_DIR / f"{model_full}_cwe_pairwise_probe.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"✓ Data saved: {out}")


def load_results(model_full: str) -> Dict:
    """Load previously saved AUROC matrix from JSON."""
    path = RESULTS_DIR / f"{model_full}_cwe_pairwise_probe.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def plot_heatmap(
    model_full: str, available_cwes: List[str], matrix: np.ndarray, peak_layer: int
):
    """Render and save the heatmap PDF from a matrix."""
    # Normalise to [0, 1] for display
    if matrix.max() > 1.5:
        matrix_norm = matrix / 100.0
    else:
        matrix_norm = matrix.copy()

    n = len(available_cwes)
    cwe_short = [c.split("-")[1] for c in available_cwes]

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    im = ax.imshow(matrix_norm, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="equal")

    # Grey diagonal for self-pairs
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color="#cccccc", lw=0))

    # Cell annotations
    for i in range(n):
        for j in range(n):
            color = "black"
            ax.text(j, i, f"{matrix_norm[i, j]:.2f}",
                    ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cwe_short, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cwe_short, fontsize=9)
    ax.set_xlabel("CWE Type (target)", fontsize=10)
    ax.set_ylabel("CWE Type (source)", fontsize=10)
    ax.set_title(
        f"Pairwise CWE probe across layers — peak AUROC\n"
        f"(vulnerable-side mean-token; green = separable, yellow = chance)",
        fontsize=10, pad=12,
    )

    plt.tight_layout()

    output_file = OUTPUT_DIR / f"fig_cwe_pairwise_probe_{model_full.split('-')[0]}.pdf"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"✓ Figure saved: {output_file}")
    plt.close()


def _probe_matrix(
    X_all: np.ndarray, cwes: np.ndarray, available_cwes: List[str]
) -> np.ndarray:
    """Compute the n_cwes × n_cwes AUROC matrix for one layer's activations."""
    n_cwes = len(available_cwes)
    matrix = np.zeros((n_cwes, n_cwes))
    for i, cwe1 in enumerate(available_cwes):
        for j, cwe2 in enumerate(available_cwes):
            if i == j:
                matrix[i, j] = 1.0
                continue
            mask1 = cwes == cwe1
            mask2 = cwes == cwe2
            if mask1.sum() == 0 or mask2.sum() == 0:
                matrix[i, j] = 0.5
                continue
            y = np.concatenate([np.ones(mask1.sum()), np.zeros(mask2.sum())])
            X_subset = np.vstack([X_all[mask1], X_all[mask2]])
            matrix[i, j] = probe_pairwise(X_subset, y)
    return matrix


def run_probing(model_full: str):
    """Compute AUROC matrix for every layer, save all to JSON, plot peak layer."""
    logger.info(f"\nGenerating pairwise CWE probes for {model_full}...")
    cwes = load_metadata_for(model_full)
    if len(cwes) == 0:
        logger.warning(f"No metadata for {model_full} — skipping")
        return

    npz_file = ACTIVATIONS_DIR / f"activations_{model_full}.npz"
    if not npz_file.exists():
        logger.warning(f"Activations not found: {npz_file}")
        return

    data = np.load(npz_file)
    activations = {key: data[key] for key in data.files}

    if len(cwes) != next(iter(activations.values())).shape[0]:
        logger.warning(
            f"CWE label count ({len(cwes)}) != activation rows "
            f"({next(iter(activations.values())).shape[0]}) for {model_full}"
        )
        return

    all_layers = sorted(
        int(k.split("_")[1])
        for k in activations
        if k.startswith("layer_") and k.endswith("_vuln_mean")
    )
    if not all_layers:
        logger.warning(f"No layer keys found in {npz_file.name}")
        return

    available_cwes = sorted([c for c in np.unique(cwes) if c in CWE_TYPES])
    if len(available_cwes) < 2:
        logger.warning(
            f"Not enough CWE types for {model_full} (found: {available_cwes})"
        )
        return

    peak_layer = get_peak_layer(activations)
    logger.info(
        f"Available CWEs: {available_cwes}, layers: {all_layers}, peak: {peak_layer}"
    )

    layer_matrices: Dict[int, np.ndarray] = {}
    for layer in all_layers:
        key = f"layer_{layer}_vuln_mean"
        logger.info(f"  Layer {layer}...")
        layer_matrices[layer] = _probe_matrix(activations[key], cwes, available_cwes)

    save_results(model_full, available_cwes, layer_matrices, peak_layer)
    plot_heatmap(model_full, available_cwes, layer_matrices[peak_layer], peak_layer)


def figures_from_saved(model_full: str, layer: int = None):
    """Regenerate figure from saved JSON without re-running probing."""
    results = load_results(model_full)
    if not results:
        logger.warning(
            f"No saved data for {model_full} — run without --figures-only first"
        )
        return
    # Handle both old format (layers dict) and new format (single auroc_matrix)
    if "auroc_matrix" in results:
        # New format: single AUROC matrix at peak layer
        matrix = np.array(results["auroc_matrix"])
        plot_heatmap(model_full, results["cwe_types"], matrix, results["peak_layer"])
    elif "layers" in results:
        # Old format: layer-by-layer data
        plot_layer = layer if layer is not None else results["peak_layer"]
        layer_data = results["layers"].get(str(plot_layer))
        if layer_data is None:
            available = sorted(results["layers"].keys(), key=int)
            logger.warning(
                f"Layer {plot_layer} not found for {model_full}. Available: {available}"
            )
            return
        matrix = np.array(layer_data["auroc_matrix"])
        plot_heatmap(model_full, results["cwe_types"], matrix, plot_layer)
    else:
        logger.error(f"No auroc_matrix or layers found in {model_full} results")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate figures from saved JSON without re-running probing",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have a saved JSON result",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer to plot (default: peak layer stored in JSON)",
    )
    args = parser.parse_args()

    if args.figures_only:
        logger.info("Regenerating figures from saved data...")
        saved = sorted(RESULTS_DIR.glob("*_cwe_pairwise_probe.json"))
        if not saved:
            logger.error(f"No saved results found in {RESULTS_DIR}")
            return
        for path in saved:
            model_full = path.stem.replace("_cwe_pairwise_probe", "")
            try:
                figures_from_saved(model_full, layer=args.layer)
            except Exception as e:
                logger.error(f"Error plotting {model_full}: {e}")
        logger.info("\n✓ Figures regenerated!")
        return

    logger.info("Generating pairwise CWE probe AUROC heatmaps...")
    models = discover_models()
    if not models:
        logger.warning(f"No NPZ files found in {ACTIVATIONS_DIR}")
        return

    logger.info(f"Models found: {models}")

    for model_full in models:
        if args.skip_existing:
            json_path = RESULTS_DIR / f"{model_full}_cwe_pairwise_probe.json"
            if json_path.exists():
                logger.info(f"  Skipping {model_full} (already computed, regenerating figure)")
                figures_from_saved(model_full, layer=args.layer)
                continue
        try:
            run_probing(model_full)
        except Exception as e:
            logger.error(f"Error processing {model_full}: {e}")

    logger.info("\n✓ Pairwise CWE probe generation complete!")


if __name__ == "__main__":
    main()
