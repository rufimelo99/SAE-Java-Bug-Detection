#!/usr/bin/env python3
"""
Probe whether model activations can distinguish security bugs (known CWE) from
non-security bugs (unknown CWE) in the PreciseBugs dataset.

Reuses existing NPZ activations — no re-computation needed.

Usage:
    python scripts/probe_security_vs_nonsecurity.py
    python scripts/probe_security_vs_nonsecurity.py --figures-only
"""

import json
import logging
from pathlib import Path

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
    / "data"
    / "precisebugs_raw"
    / "precisebugs_c_pairs.jsonl"
)
OUTPUT_DIR = (
    _PROJECT_DIR.parent
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
RESULTS_DIR = _PROJECT_DIR / "results" / "raw_data"

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

MODEL_LABELS = {
    "precisebugs_codellama-7b":   "CodeLlama-7B",
    "precisebugs_codellama-13b":  "CodeLlama-13B",
    "precisebugs_deepseek-6.7b":  "DeepSeek-6.7B",
    "precisebugs_qwen-7b":        "Qwen-7B",
    "precisebugs_qwen-coder-7b":  "Qwen-Coder-7B",
    "precisebugs_qwen-14b":       "Qwen-14B",
    "precisebugs_starcoder2-7b":  "StarCoder2-7B",
    "precisebugs_starcoder2-15b": "StarCoder2-15B",
}


def load_labels() -> np.ndarray:
    """Load binary labels: 1=security bug (known CWE), 0=non-security (unknown)."""
    with open(METADATA_FILE) as f:
        records = [json.loads(l) for l in f if l.strip()]

    def _normalize(cwe: str) -> str:
        if cwe.startswith("CWE-"):
            return "CWE-" + str(int(cwe[4:]))
        return cwe

    cwes = np.array([_normalize(r.get("cwe", "unknown")) for r in records])
    labels = (cwes != "unknown").astype(int)
    n_sec = labels.sum()
    n_nonsec = (labels == 0).sum()
    logger.info(f"Labels: {n_sec} security bugs, {n_nonsec} non-security bugs")
    return labels, cwes


def probe_binary(X: np.ndarray, y: np.ndarray, n_components: int = 50, cv: int = 5) -> float:
    """Stratified k-fold LogReg AUROC for binary y."""
    if len(np.unique(y)) < 2:
        return 0.5

    n_splits = min(cv, int(y.sum()), int((y == 0).sum()))
    if n_splits < 2:
        return 0.5

    n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
    clf = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=42)
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
        return roc_auc_score(y, y_score)
    except Exception:
        return 0.5


def run_model(model_full: str, labels: np.ndarray) -> dict:
    npz_path = ACTIVATIONS_DIR / f"activations_{model_full}.npz"
    if not npz_path.exists():
        logger.warning(f"NPZ not found: {npz_path}")
        return {}

    data = np.load(npz_path)
    layers = sorted(
        int(k.split("_")[1])
        for k in data.files
        if k.startswith("layer_") and k.endswith("_vuln_mean")
    )

    if not layers:
        logger.warning(f"No layer keys in {npz_path.name}")
        return {}

    if data[f"layer_{layers[0]}_vuln_mean"].shape[0] != len(labels):
        logger.warning(
            f"Row count mismatch: {data[f'layer_{layers[0]}_vuln_mean'].shape[0]} "
            f"activations vs {len(labels)} labels for {model_full}"
        )
        return {}

    logger.info(f"Probing {model_full} across {len(layers)} layers...")
    aurocs = {}
    for layer in layers:
        X = data[f"layer_{layer}_vuln_mean"]
        auroc = probe_binary(X, labels)
        aurocs[layer] = auroc
        logger.info(f"  Layer {layer}: AUROC={auroc:.4f}")

    return aurocs


def save_results(results: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "precisebugs_security_vs_nonsecurity_probe.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {out}")


def load_results() -> dict:
    path = RESULTS_DIR / "precisebugs_security_vs_nonsecurity_probe.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def plot_results(results: dict):
    fig, ax = plt.subplots(figsize=(8, 5))

    models = sorted(results.keys())
    for i, model in enumerate(models):
        aurocs = results[model]
        if not aurocs:
            continue
        layers = sorted(int(l) for l in aurocs)
        vals = [aurocs[str(l)] for l in layers]
        label = MODEL_LABELS.get(model, model)
        color = _PALETTE[i % len(_PALETTE)]
        ax.plot(layers, vals, marker="o", label=label, color=color, linewidth=2, markersize=5)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Chance (0.50)")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("AUROC (security vs non-security)", fontsize=12)
    ax.set_title("Security Bug vs Non-Security Bug Discrimination\n(PreciseBugs, LogReg on vuln. activations)", fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.set_ylim([0.4, 0.85])
    ax.grid(True, alpha=0.25)

    out = OUTPUT_DIR / "fig_security_vs_nonsecurity_probe.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {out}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-only", action="store_true",
                        help="Regenerate figure from saved JSON without re-running probes")
    args = parser.parse_args()

    if args.figures_only:
        results = load_results()
        if not results:
            logger.error("No saved results found — run without --figures-only first")
            return
        plot_results(results)
        return

    labels, _ = load_labels()

    models = sorted(
        npz.stem.replace("activations_", "")
        for npz in ACTIVATIONS_DIR.glob("activations_precisebugs_*.npz")
    )
    logger.info(f"Models: {models}")

    results = {}
    for model in models:
        aurocs = run_model(model, labels)
        if aurocs:
            results[model] = {str(l): v for l, v in aurocs.items()}

    save_results(results)
    plot_results(results)
    logger.info("Done.")


if __name__ == "__main__":
    main()
