"""
Nonlinear probe computation — saves results to JSONL.

Run once to compute; use nonlinear_probe_plot.py to regenerate the figure.

Run:
  conda run -n sae python nonlinear_probe_compute.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ARTIFACTS  = Path(__file__).parents[2] / "artifacts" / "activations"
OUT_DIR    = ARTIFACTS / "nonlinear_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSONL  = OUT_DIR / "nonlinear_probe_results.jsonl"

SEED = 42
np.random.seed(SEED)

LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]

RUNS_REGISTRY = [
    {
        "run_dir": ARTIFACTS / "raw_activations/vulnerable_code_qwen_coder_standard_16384_raw",
        "label":   "Raw residual",
    },
    {
        "run_dir": ARTIFACTS / "TO_UPLOAD",
        "label":   "SAE features (STD)",
    },
]

N_PCA       = 50
N_CV_FOLDS  = 5
N_BOOTSTRAP = 500
CI_LEVEL    = 0.95


# ── Data loading ──────────────────────────────────────────────────────────────

def _discover_layers(run_dir):
    layers = set()
    for f in run_dir.glob("activations_layer_*_*.jsonl"):
        try:
            layers.add(int(f.name.split("_")[2]))
        except (IndexError, ValueError):
            pass
    return sorted(layers)


def _load_layer(run_dir, layer):
    safe_pt = run_dir / f"safe_layer_{layer}.pt"
    vuln_pt = run_dir / f"vulnerable_layer_{layer}.pt"
    if safe_pt.exists() and vuln_pt.exists():
        safe = torch.load(safe_pt, weights_only=True).numpy().astype(np.float32)
        vuln = torch.load(vuln_pt, weights_only=True).numpy().astype(np.float32)
        return safe, vuln
    matches = list(run_dir.glob(f"activations_layer_{layer}_*.jsonl"))
    if not matches:
        return None
    secure_rows, vuln_rows = [], []
    with matches[0].open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            secure_rows.append(r["secure"])
            vuln_rows.append(r["vulnerable"])
    return np.array(secure_rows, dtype=np.float32), np.array(vuln_rows, dtype=np.float32)


def load_data():
    data = {}
    for entry in RUNS_REGISTRY:
        run_dir = entry["run_dir"]
        label   = entry["label"]
        if not run_dir.exists():
            print(f"[SKIP] {label}: {run_dir} not found")
            continue
        available = _discover_layers(run_dir)
        layers    = [l for l in LAYERS if l in available]
        if not layers:
            print(f"[SKIP] {label}: no layers found in {run_dir}")
            continue
        data[label] = {}
        for layer in tqdm(layers, desc=f"Loading {label}", leave=False):
            result = _load_layer(run_dir, layer)
            if result is not None:
                data[label][layer] = result
        print(f"Loaded {label}: layers {list(data[label].keys())}")
    return data


# ── Probe helpers ─────────────────────────────────────────────────────────────

def _bootstrap_auc_ci(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    rng  = np.random.default_rng(SEED)
    n    = len(y_true)
    aucs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    aucs  = np.array(aucs)
    alpha = (1 - CI_LEVEL) / 2
    return float(aucs.mean()), float(np.quantile(aucs, alpha)), float(np.quantile(aucs, 1 - alpha))


PROBE_NAMES = ["LogReg", "MLP", "RBF-SVM"]

def _make_probe(name):
    if name == "LogReg":
        return LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=SEED)
    if name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(256, 128), activation="relu", alpha=1e-3,
            max_iter=500, early_stopping=True, validation_fraction=0.1,
            random_state=SEED, verbose=False,
        )
    if name == "RBF-SVM":
        return SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True,
            class_weight="balanced", random_state=SEED, max_iter=2000,
        )
    raise ValueError(f"Unknown probe: {name}")


def run_probe_cv(safe_mat, vuln_mat, probe_name):
    n = len(safe_mat)
    X = np.vstack([safe_mat, vuln_mat])
    y = np.array([0] * n + [1] * n, dtype=int)
    n_comp  = min(N_PCA, X.shape[1], X.shape[0] - 1)
    skf     = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
    y_score = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        pca    = PCA(n_components=n_comp, random_state=SEED)
        X_tr   = pca.fit_transform(scaler.fit_transform(X[train_idx]))
        X_te   = pca.transform(scaler.transform(X[test_idx]))
        p = _make_probe(probe_name)
        p.fit(X_tr, y[train_idx])
        y_score[test_idx] = p.predict_proba(X_te)[:, 1]
    return _bootstrap_auc_ci(y, y_score)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Output: {OUT_JSONL}\n")
    data = load_data()

    records = []
    for label, layer_dict in data.items():
        for layer in tqdm(sorted(layer_dict.keys()), desc=label):
            safe_mat, vuln_mat = layer_dict[layer]
            for probe_name in PROBE_NAMES:
                mean, lo, hi = run_probe_cv(safe_mat, vuln_mat, probe_name)
                rec = {"label": label, "probe": probe_name, "layer": layer,
                       "auroc": mean, "ci_lo": lo, "ci_hi": hi}
                records.append(rec)
                print(f"  {label}  L{layer:2d}  {probe_name:8s}  AUROC={mean:.3f} [{lo:.3f}–{hi:.3f}]")

    with OUT_JSONL.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nSaved {len(records)} records to {OUT_JSONL}")
