"""
Language Persistence After Residualisation — verification script.

For each layer, measures how predictable the file extension is from the
SAE delta vectors (vuln - secure) before and after Ridge file-type residualisation.

Output:
  artifacts/activations/lang_persistence_results.jsonl  — one record per layer x rep
"""

import json
import glob
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

REPO    = Path(__file__).parent.parent.parent.parent
ACTS_DIR = REPO / "sae_java_bug/artifacts/activations/TO_UPLOAD"
OUT      = REPO / "sae_java_bug/artifacts/activations/lang_persistence_results.jsonl"

LAYERS  = [0, 3, 7, 11, 15, 19, 23, 27]
MIN_EXT = 20
N_PCA   = 50
SEED    = 42


def load_sae_layer(layer: int) -> list[dict]:
    pat   = str(ACTS_DIR / f"activations_layer_{layer}_sae_*.jsonl")
    files = glob.glob(pat)
    assert files, f"No SAE file for layer {layer}"
    records = []
    with open(files[0]) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def lang_auroc(delta: np.ndarray, exts: np.ndarray, n_pca: int = N_PCA) -> float:
    """Macro-averaged one-vs-rest language AUROC (z-score + PCA-50, 5-fold CV)."""
    unique, counts = np.unique(exts, return_counts=True)
    valid_exts     = set(unique[counts >= MIN_EXT])
    mask           = np.array([e in valid_exts for e in exts])
    X, y_raw       = delta[mask], exts[mask]

    classes = sorted(valid_exts)
    y       = np.array([classes.index(e) for e in y_raw])

    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_y, oof_p = [], []

    for tr, te in cv.split(X, y):
        sc  = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        n   = min(n_pca, Xtr.shape[1], Xtr.shape[0] - 1)
        pca = PCA(n_components=n, random_state=SEED)
        Xtr = pca.fit_transform(Xtr)
        Xte = pca.transform(Xte)
        clf = LogisticRegression(max_iter=2000, random_state=SEED)
        clf.fit(Xtr, y[tr])
        oof_y.append(y[te])
        oof_p.append(clf.predict_proba(Xte))

    y_all = np.concatenate(oof_y)
    p_all = np.concatenate(oof_p, axis=0)

    aurocs = []
    for i in range(len(classes)):
        yb = (y_all == i).astype(int)
        if yb.sum() < 5:
            continue
        aurocs.append(roc_auc_score(yb, p_all[:, i]))
    return float(np.mean(aurocs))


def residualise(delta: np.ndarray, exts: np.ndarray) -> np.ndarray:
    """Ridge regression from one-hot file-extension to delta; return residuals."""
    classes = sorted(set(exts))
    onehot  = np.array([[1.0 if e == c else 0.0 for c in classes] for e in exts])
    ridge   = Ridge(alpha=1.0)
    ridge.fit(onehot, delta)
    return delta - ridge.predict(onehot)


def main() -> None:
    print(f"{'Layer':<8} {'Rep':<8} {'Before':>8} {'After':>8} {'Change':>8}")
    print("-" * 46)

    records: list[dict] = []

    for layer in LAYERS:
        try:
            rows    = load_sae_layer(layer)
            vuln    = np.array([r["vulnerable"]                          for r in rows], dtype=np.float32)
            secure  = np.array([r["secure"]                              for r in rows], dtype=np.float32)
            exts    = np.array([r["file_extension"].lstrip(".").lower()  for r in rows])
            delta   = vuln - secure

            before = lang_auroc(delta, exts)
            resid  = residualise(delta, exts)
            after  = lang_auroc(resid, exts)

            rec = {
                "layer":          layer,
                "representation": "SAE",
                "auroc_before":   round(before, 4),
                "auroc_after":    round(after,  4),
                "delta":          round(after - before, 4),
            }
            records.append(rec)
            print(f"L{layer:<7} {'SAE':<8} {before:>8.3f} {after:>8.3f} {after - before:>+8.3f}")

        except Exception as e:
            print(f"L{layer:<7} SAE  error: {e}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(records)} records → {OUT}")


if __name__ == "__main__":
    main()
