"""
feature_ablation_compute.py
===========================
Feature-level causal test of the negative-space hypothesis.

Hypothesis: the mean-token separation signal in SAE feature space is driven by
secure-enriched features (those with Δf < 0).  Zeroing them out should reduce
the effective separation toward chance; zeroing vuln-enriched features should
not.

Method
------
1. Load pre-computed mean-token SAE feature vectors (shape [N, 16384]) from the
   most recent mean_pool_sae run — no GPU needed.
2. Load secure-enriched feature rankings from delta_f_results.json (features
   sorted by Δf, most negative = most secure-enriched first).
3. For each ablation budget N in ABLATION_SIZES:
     a. Zero out the top-N secure-enriched features from both safe and vuln
        matrices (zero-ablation = set those columns to 0).
     b. Re-run the standard probe (z-score → PCA-50 → logistic regression,
        5-fold CV, 500-bootstrap 95% CI).
4. Repeat for vuln-enriched features as a negative control.
5. Save all results to JSONL.

Interpretation
--------------
The unablated probe gives AUROC ≈ 0.395 (below chance, inverted direction).
1 - AUROC ≈ 0.605 is the effective separation in the secure > vulnerable
direction predicted by the negative-space hypothesis.  As N increases and
secure-enriched features are zeroed, (1 - AUROC) should fall toward 0.5.
Ablating vuln-enriched features should have little effect.

Run
---
    conda run -n sae python feature_ablation_compute.py

Output
------
    artifacts/activations/feature_ablation/feature_ablation_results.jsonl
"""

import ctypes
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
ARTIFACTS  = Path(__file__).parents[2] / "artifacts" / "activations"
OUT_DIR    = Path(__file__).parents[2] / "artifacts" / "activations" / "feature_ablation"
OUT_JSONL  = OUT_DIR / "feature_ablation_results.jsonl"

DELTA_F_JSON   = ARTIFACTS / "feature_asymmetry_crosslayer" / "delta_f_results.json"
MEAN_SAE_ROOT  = ARTIFACTS / "mean_pool_sae"

SAE_LAYER      = 11
SEED           = 42
N_COMPONENTS   = 50
CV_FOLDS       = 5
N_BOOTSTRAP    = 500

ABLATION_SIZES = [0, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 12588]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _t2np(path: Path) -> np.ndarray:
    t = torch.load(path, weights_only=True, map_location="cpu").float().contiguous()
    buf = (ctypes.c_float * t.numel()).from_address(t.data_ptr())
    return np.ctypeslib.as_array(buf).reshape(t.shape).copy()


def find_latest_mean_sae_run() -> Path:
    candidates = sorted(
        (d for d in MEAN_SAE_ROOT.iterdir()
         if (d / f"safe_mean_sae_layer_{SAE_LAYER}.pt").exists()),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No completed mean_pool_sae run found under {MEAN_SAE_ROOT}\n"
            "Run mean_pool_sae_probe.py first."
        )
    return candidates[0]


def bootstrap_auc_ci(y_true, y_score, n=N_BOOTSTRAP, seed=SEED):
    from sklearn.metrics import roc_auc_score
    rng  = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    aucs = np.array(aucs)
    return float(np.mean(aucs)), float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))


def probe(safe_mat: np.ndarray, vuln_mat: np.ndarray) -> dict:
    X = np.vstack([safe_mat, vuln_mat]).astype(np.float32)
    y = np.array([0] * len(safe_mat) + [1] * len(vuln_mat), dtype=int)
    n_comp = min(N_COMPONENTS, X.shape[1], X.shape[0] - 1)

    clf = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced",
                             random_state=SEED)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    y_score = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y):
        sc  = StandardScaler()
        pca = PCA(n_components=min(n_comp, len(tr) - 1), random_state=SEED)
        Xtr = pca.fit_transform(sc.fit_transform(X[tr]))
        Xte = pca.transform(sc.transform(X[te]))
        clf.fit(Xtr, y[tr])
        y_score[te] = clf.predict_proba(Xte)[:, 1]

    auc, lo, hi = bootstrap_auc_ci(y, y_score)
    return {"roc_auc": auc, "ci_lo": lo, "ci_hi": hi, "n": len(safe_mat)}


def ablate(safe_mat: np.ndarray, vuln_mat: np.ndarray,
           feature_ids: list[int], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Zero out the first n features from feature_ids in both matrices."""
    if n == 0:
        return safe_mat, vuln_mat
    mask = feature_ids[:n]
    s = safe_mat.copy()
    v = vuln_mat.copy()
    s[:, mask] = 0.0
    v[:, mask] = 0.0
    return s, v


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load mean-token SAE matrices ─────────────────────────────────────────
    run_dir = find_latest_mean_sae_run()
    print(f"Loading mean-token SAE tensors from: {run_dir}")
    safe_mat = _t2np(run_dir / f"safe_mean_sae_layer_{SAE_LAYER}.pt")   # [N, 16384]
    vuln_mat = _t2np(run_dir / f"vulnerable_mean_sae_layer_{SAE_LAYER}.pt")
    N = len(safe_mat)
    print(f"  Loaded: {N} samples × {safe_mat.shape[1]} features")

    # ── Compute per-feature Δf from the mean-token SAE matrices ─────────────
    # We rank features by their mean-token activation-frequency shift directly
    # from the matrices we're about to ablate.  This is more appropriate than
    # using last-token Δf rankings for a mean-token ablation experiment.
    print("\nComputing per-feature Δf from mean-token SAE matrices …")
    freq_sec    = (safe_mat > 0).mean(axis=0)      # [d_sae]
    freq_vul    = (vuln_mat > 0).mean(axis=0)
    delta_f_vec = freq_vul - freq_sec              # negative = secure-enriched

    print(f"  Secure-enriched (Δf<0): {(delta_f_vec < 0).sum()}")
    print(f"  Vuln-enriched  (Δf>0): {(delta_f_vec > 0).sum()}")

    # Sorted indices: most secure-enriched first (most negative Δf)
    sec_ranked  = np.argsort(delta_f_vec).tolist()          # ascending Δf
    vuln_ranked = np.argsort(-delta_f_vec).tolist()         # descending Δf

    # ── Run ablation sweep ────────────────────────────────────────────────────
    results = []
    total   = len(ABLATION_SIZES) * 2
    done    = 0

    for condition, ranked_ids in [("secure_enriched", sec_ranked),
                                   ("vuln_enriched",   vuln_ranked)]:
        for n_ablate in ABLATION_SIZES:
            s_abl, v_abl = ablate(safe_mat, vuln_mat, ranked_ids, n_ablate)
            res           = probe(s_abl, v_abl)
            done         += 1

            row = {
                "condition":   condition,
                "n_ablated":   n_ablate,
                "roc_auc":     res["roc_auc"],
                "ci_lo":       res["ci_lo"],
                "ci_hi":       res["ci_hi"],
                "eff_auc":     1.0 - res["roc_auc"],   # effective separation (inverted direction)
                "eff_ci_lo":   1.0 - res["ci_hi"],
                "eff_ci_hi":   1.0 - res["ci_lo"],
                "n":           res["n"],
            }
            results.append(row)
            print(f"  [{done:>2}/{total}] {condition:20s}  N={n_ablate:>6}  "
                  f"AUROC={res['roc_auc']:.3f}  eff={row['eff_auc']:.3f} "
                  f"[{row['eff_ci_lo']:.3f}–{row['eff_ci_hi']:.3f}]")

    # ── Save ─────────────────────────────────────────────────────────────────
    with OUT_JSONL.open("w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    print(f"\nSaved {len(results)} records → {OUT_JSONL}")


if __name__ == "__main__":
    main()
