#!/usr/bin/env python3
"""
Unpaired vulnerable-vs-secure binary probing across models (reviewer response).

This is the *categorical* counterpart to the paired direction-geometry analysis
in rebuttal_data/. It pools all secure and vulnerable samples together, labels them
0/1, and asks a linear probe to draw a single global threshold — the experiment
whose failure (AUROC ~0.65) motivates the "relational not categorical" thesis.

For each model, pooling strategy (mean / last), and layer it reports AUROC with
a 95% bootstrap CI under two CV schemes:
    - standard StratifiedKFold        (samples treated independently)
    - pair-stratified KFold           (both members of a pair kept in one fold)

Sources:
    npz: activations_{dataset}_{model}.npz with keys layer_{L}_{vuln,secure}_{mean,last}

Usage:
    python scripts/unpaired_binary_probe.py --model deltasecommits_codellama-7b
    python scripts/unpaired_binary_probe.py --model all
    python scripts/unpaired_binary_probe.py --model all --probe mlp
    python scripts/unpaired_binary_probe.py --model all --probe rbf-svm
    python scripts/unpaired_binary_probe.py --model all --length-resid

Output:
    rebuttal_data/raw_data/{model}_binary_probe.json
    rebuttal_data/raw_data/{model}_binary_probe_mlp.json
    rebuttal_data/raw_data/{model}_binary_probe_rbf-svm.json
    rebuttal_data/raw_data/{model}_binary_probe_length_resid.json
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "sae_java_bug" / "artifacts"
MULTI_MODEL_DIR = ARTIFACTS / "multi_model_probing"
RESULTS_DIR = REPO / "rebuttal_data" / "raw_data"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Shared dataset metadata (same ordering across all DeltaSecommits activation caches)
METADATA_FILE = (
    ARTIFACTS
    / "activations"
    / "TO_UPLOAD"
    / "activations_layer_11_sae_blocks.11.hook_resid_post_component_hook_resid_post.hook_sae_acts_post.jsonl"
)

C_EXTS = {"c"}  # gives n=1368 pairs from DeltaSecommits

SEED = 42

DELTASECOMMITS_MODELS = [
    "deltasecommits_codellama-7b",
    "deltasecommits_codellama-13b",
    "deltasecommits_deepseek-6.7b",
    "deltasecommits_qwen-7b",
    "deltasecommits_qwen-coder-7b",
    "deltasecommits_qwen-14b",
    "deltasecommits_starcoder2-7b",
    "deltasecommits_starcoder2-15b",
]

PROBE_LABEL = {
    "logreg": "LogReg(C=0.1, balanced) on StandardScaler+PCA(50)",
    "mlp": "MLP(64,) on StandardScaler+PCA(50)",
    "rbf-svm": "RBF-SVM(C=1.0, balanced, decision_function) on StandardScaler+PCA(50)",
}


# ── Classifier factory ─────────────────────────────────────────────────────────

def make_clf(probe_type: str):
    if probe_type == "logreg":
        return LogisticRegression(
            C=0.1, max_iter=1000, class_weight="balanced", random_state=SEED
        )
    if probe_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64,), max_iter=500, random_state=SEED
        )
    if probe_type == "rbf-svm":
        # probability=False: avoids Platt-scaling internal CV that ignores pair structure
        # and inverts pair-stratified AUROC. Use decision_function for scoring instead.
        return SVC(
            kernel="rbf", C=1.0, probability=False,
            class_weight="balanced", random_state=SEED
        )
    raise ValueError(f"Unknown probe type: {probe_type}")


# ── Pair-stratified CV splitter ────────────────────────────────────────────────

class PairStratifiedKFold:
    """KFold that keeps both members of each pair in the same fold.

    Expects X layout: [secure_0..secure_{n-1} | vuln_0..vuln_{n-1}]
    so that pair i = (row i, row n+i).
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        n = len(X) // 2
        pair_idx = np.arange(n)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(pair_idx)
        kf = KFold(n_splits=self.n_splits)
        for train_pairs, test_pairs in kf.split(pair_idx):
            tp = pair_idx[train_pairs]
            vp = pair_idx[test_pairs]
            train_idx = np.concatenate([tp, tp + n])
            test_idx = np.concatenate([vp, vp + n])
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# ── Bootstrap CI ───────────────────────────────────────────────────────────────

def bootstrap_auroc(y_true, y_score, n_boot: int = 1000, ci: float = 0.90, seed: int = SEED):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    lo = float(np.percentile(aucs, (1 - ci) / 2 * 100))
    hi = float(np.percentile(aucs, (1 + ci) / 2 * 100))
    return lo, hi


# ── Probing ────────────────────────────────────────────────────────────────────

def probe(secure_mat, vuln_mat, splitter, probe_type: str = "logreg"):
    """Run CV probe; return (mean_auroc, ci_lo, ci_hi)."""
    X = np.vstack([secure_mat, vuln_mat]).astype(np.float32)
    y = np.array([0] * len(secure_mat) + [1] * len(vuln_mat))

    all_scores = np.zeros(len(y))
    for train_idx, test_idx in splitter.split(X, y):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te = X[test_idx]

        scaler = StandardScaler()
        pca = PCA(n_components=50, random_state=SEED)
        X_tr = pca.fit_transform(scaler.fit_transform(X_tr))
        X_te = pca.transform(scaler.transform(X_te))

        clf = make_clf(probe_type)
        clf.fit(X_tr, y_tr)

        if probe_type == "rbf-svm":
            scores = clf.decision_function(X_te)
        else:
            scores = clf.predict_proba(X_te)[:, 1]
        all_scores[test_idx] = scores

    auroc = float(roc_auc_score(y, all_scores))
    ci_lo, ci_hi = bootstrap_auroc(y, all_scores)
    return auroc, ci_lo, ci_hi


# ── Metadata helpers ───────────────────────────────────────────────────────────

def load_extensions_from_jsonl(path: Path):
    exts = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            exts.append(r.get("file_extension", "").lstrip(".").lower())
    return exts


def load_lengths_from_jsonl(path: Path):
    len_s, len_v = [], []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for arr, key in [(len_s, "secure_code"), (len_v, "vulnerable_code")]:
                raw = r.get(key, "")
                try:
                    text = base64.b64decode(raw).decode("utf-8", errors="replace")
                except Exception:
                    text = raw if isinstance(raw, str) else ""
                arr.append(len(text))
    return np.array(len_s, dtype=np.float32), np.array(len_v, dtype=np.float32)


def residualise_length(secure, vuln, len_s, len_v):
    """Ridge-regress log(char_length) out of activations."""
    log_s = np.log1p(len_s).reshape(-1, 1)
    log_v = np.log1p(len_v).reshape(-1, 1)
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(np.vstack([log_s, log_v]), np.vstack([secure, vuln]))
    secure_r = (secure - ridge.predict(log_s)).astype(np.float32)
    vuln_r = (vuln - ridge.predict(log_v)).astype(np.float32)
    return secure_r, vuln_r


# ── Per-model runner ───────────────────────────────────────────────────────────

def run_model(model: str, c_only: bool = True, probe_type: str = "logreg", length_resid: bool = False):
    npz_path = MULTI_MODEL_DIR / f"activations_{model}.npz"
    if not npz_path.exists():
        print(f"  [skip] {npz_path} not found")
        return

    print(f"\n=== {model}  probe={probe_type}  length_resid={length_resid} ===")
    npz = np.load(npz_path, allow_pickle=True)

    # Detect layers from npz keys
    layers = sorted(
        {int(k.split("_")[1]) for k in npz.keys() if k.startswith("layer_")},
        key=int,
    )
    print(f"  Layers: {layers}")

    # Detect sample count from npz
    first_key = next(k for k in npz.keys() if k.startswith("layer_"))
    n_npz = npz[first_key].shape[0]

    # Load metadata for C-only mask
    meta_matched = False
    if METADATA_FILE.exists():
        exts = load_extensions_from_jsonl(METADATA_FILE)
        if len(exts) == n_npz:
            meta_matched = True
        else:
            print(f"  [warn] metadata length {len(exts)} != npz length {n_npz}; skipping C filter")

    if c_only and meta_matched:
        exts = load_extensions_from_jsonl(METADATA_FILE)
        mask = np.array([e in C_EXTS for e in exts])
        n_c = int(mask.sum())
        print(f"  C-only: {n_c}/{n_npz} pairs")
    else:
        mask = np.ones(n_npz, dtype=bool)
        n_c = n_npz

    # Optionally load lengths for residualisation
    len_s_full = len_v_full = None
    if length_resid and meta_matched:
        len_s_full, len_v_full = load_lengths_from_jsonl(METADATA_FILE)

    std_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pst_splitter = PairStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results_by_pooling = {}
    for pool in ("mean", "last"):
        layer_results = {}
        for L in layers:
            sk = f"layer_{L}_secure_{pool}"
            vk = f"layer_{L}_vuln_{pool}"
            if sk not in npz or vk not in npz:
                continue
            secure = npz[sk][mask].astype(np.float32)
            vuln = npz[vk][mask].astype(np.float32)

            if length_resid and len_s_full is not None:
                secure, vuln = residualise_length(
                    secure, vuln, len_s_full[mask], len_v_full[mask]
                )

            std_auc, std_lo, std_hi = probe(secure, vuln, std_splitter, probe_type)
            pst_auc, pst_lo, pst_hi = probe(secure, vuln, pst_splitter, probe_type)

            layer_results[str(L)] = {
                "standard_cv": round(std_auc, 4),
                "standard_cv_ci90": [round(std_lo, 4), round(std_hi, 4)],
                "pair_stratified_cv": round(pst_auc, 4),
                "pair_stratified_cv_ci90": [round(pst_lo, 4), round(pst_hi, 4)],
            }
            print(
                f"  L{L:2d} [{pool}]  std={std_auc:.4f} [{std_lo:.4f},{std_hi:.4f}]"
                f"  pst={pst_auc:.4f} [{pst_lo:.4f},{pst_hi:.4f}]"
            )

        if layer_results:
            best_std = max(layer_results, key=lambda l: layer_results[l]["standard_cv"])
            best_pst = max(layer_results, key=lambda l: layer_results[l]["pair_stratified_cv"])
            results_by_pooling[pool] = {
                "layers": layer_results,
                "peak_standard_cv": best_std,
                "peak_pair_stratified_cv": best_pst,
            }

    suffix = f"_{probe_type}" if probe_type != "logreg" else ""
    if length_resid:
        suffix += "_length_resid"
    out_path = RESULTS_DIR / f"{model}_binary_probe{suffix}.json"

    dataset = "DeltaSecommits" if "deltasecommits" in model else \
              "PreciseBugs" if "precisebugs" in model else \
              "SVEN" if "sven" in model else "Unknown"

    out = {
        "model": model,
        "probe": PROBE_LABEL[probe_type],
        "length_residualised": length_resid,
        "dataset": dataset,
        "c_only": c_only,
        "n_pairs": int(n_c),
        "poolings": results_by_pooling,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  -> saved {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="all",
        help="Model name (e.g. deltasecommits_codellama-7b) or 'all'",
    )
    ap.add_argument(
        "--probe",
        default="logreg",
        choices=["logreg", "mlp", "rbf-svm"],
        help="Classifier type (default: logreg)",
    )
    ap.add_argument(
        "--length-resid",
        action="store_true",
        help="Residualise log(char_length) from activations before probing",
    )
    ap.add_argument(
        "--no-c-only",
        action="store_true",
        help="Use all languages (default: C-only)",
    )
    args = ap.parse_args()

    c_only = not args.no_c_only

    if args.model == "all":
        models = DELTASECOMMITS_MODELS
    else:
        models = [args.model]

    for model in models:
        run_model(
            model,
            c_only=c_only,
            probe_type=args.probe,
            length_resid=args.length_resid,
        )


if __name__ == "__main__":
    main()
