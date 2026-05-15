"""
Multi-model vulnerability probing experiment.

Replicates the Qwen2.5-7B-Instruct probing pipeline on CodeLlama-7B-Instruct
and StarCoder2-7B, to test whether the key findings (near-chance last-token
probing, above-chance mean-token probing, vulnerability direction geometry)
generalise across model families.

Experiments:
  1. Binary probe (vulnerable vs secure):
       - last-token and mean-token pooling at 8 layers
       - 5-fold stratified CV, 95% bootstrap CIs
  2. Within-language breakdown (C, PHP, JS, C++, Python)
       - mean-token pooling only, all layers
  3. Vulnerability direction geometry:
       - mean paired distance ||delta||_2 per layer
       - per-pair alignment: % pairs where vuln is further along the mean direction

Output:
  results/activations_{model_name}.npz  — cached activations (skipped if exists)
  results/probing_{model_name}.jsonl    — one JSON record per experiment x layer
  results/probing_all_models.jsonl      — combined file

Install:
  pip install torch transformers scikit-learn numpy tqdm

Run:
  python -m sae_java_bug.evaluation.multi_model_probing
  python -m sae_java_bug.evaluation.multi_model_probing --model codellama-7b
  python -m sae_java_bug.evaluation.multi_model_probing --model starcoder2-7b
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ── Repository root ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent.parent   # SAE-Java-Bug-Detection/

# ── Configuration ──────────────────────────────────────────────────────────────

MODELS: dict[str, dict] = {
    "qwen-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "layers": None,  # None = all layers
    },
    "qwen-coder-7b": {
        "hf_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "layers": None,
    },
    "codellama-7b": {
        "hf_id": "codellama/CodeLlama-7b-Instruct-hf",
        "layers": None,
    },
    "codellama-13b": {
        "hf_id": "codellama/CodeLlama-13b-Instruct-hf",
        "layers": None,
    },
    "starcoder2-7b": {
        "hf_id": "bigcode/starcoder2-7b",
        "layers": None,
    },
    "starcoder2-15b": {
        "hf_id": "bigcode/starcoder2-15b",
        "layers": None,
    },
    "deepseekcoder-7b": {
        "hf_id": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "layers": None,
    },
    "llama3-8b": {
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "layers": None,
    },
}

# Paired code data: layer-11 activations JSONL has base64 secure/vulnerable code
# for all 2,493 pairs — we use only the code fields, not the Qwen activations.
DATA_JSONL = (
    REPO_ROOT
    / "sae_java_bug/artifacts/activations/TO_UPLOAD"
    / "activations_layer_11_sae_blocks.11.hook_resid_post_component_hook_resid_post"
      ".hook_sae_acts_post.jsonl"
)

# ── Probing hyperparameters ────────────────────────────────────────────────────
MAX_TOKENS   = 2048
N_PCA        = 50
N_FOLDS      = 5
N_BOOTSTRAP  = 500
RNG_SEED     = 42

WITHIN_LANGS: dict[str, str] = {
    "c":   "C",
    "php": "PHP",
    "js":  "JavaScript",
    "cpp": "C++",
    "py":  "Python",
}
MIN_WITHIN_LANG = 50   # skip language strata smaller than this

DEFAULT_ACTIVATIONS_DIR = REPO_ROOT / "sae_java_bug" / "artifacts" / "multi_model_probing"


# ── Data loading ───────────────────────────────────────────────────────────────

def _b64decode(s: str) -> str:
    try:
        return base64.b64decode(s).decode("utf-8", errors="replace")
    except Exception:
        return s


def load_pairs() -> list[dict]:
    if not DATA_JSONL.exists():
        print(
            f"ERROR: data file not found:\n  {DATA_JSONL}\n"
            "Make sure you are running from the SAE-Java-Bug-Detection repo.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading pairs from {DATA_JSONL} …")
    pairs: list[dict] = []
    with open(DATA_JSONL) as f:
        for line in f:
            r = json.loads(line)
            vuln_code = _b64decode(r.get("vulnerable_code", ""))
            sec_code  = _b64decode(r.get("secure_code", ""))
            if not vuln_code.strip() or not sec_code.strip():
                continue
            pairs.append({
                "vuln_id":         r.get("vuln_id", ""),
                "cwe":             r.get("cwe", ""),
                "file_extension":  r.get("file_extension", "").lstrip(".").lower(),
                "vulnerable_code": vuln_code,
                "secure_code":     sec_code,
            })
    print(f"  {len(pairs):,} valid pairs loaded.")
    return pairs


# ── Activation extraction ──────────────────────────────────────────────────────

def extract_activations(
    pairs:      list[dict],
    model_cfg:  dict,
    cache_path: Path,
    device:     str,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Returns {layer: {"vuln_last", "vuln_mean", "secure_last", "secure_mean"}}
    where each value is an (N, D) float32 array.

    Activations are cached to cache_path (.npz); subsequent calls load from disk.
    """
    if cache_path.exists():
        print(f"  Loading cached activations from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        layers = sorted({
            int(k.split("_")[1])
            for k in data.files
            if k.startswith("layer_") and k.endswith("_vuln_last")
        })
        return {
            L: {
                "vuln_last":   data[f"layer_{L}_vuln_last"],
                "vuln_mean":   data[f"layer_{L}_vuln_mean"],
                "secure_last": data[f"layer_{L}_secure_last"],
                "secure_mean": data[f"layer_{L}_secure_mean"],
            }
            for L in layers
        }

    hf_id  = model_cfg["hf_id"]

    print(f"  Loading tokenizer: {hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model: {hf_id}")
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device).eval()
    torch.set_grad_enabled(False)

    n_layers = model.config.num_hidden_layers
    layers = model_cfg["layers"] if model_cfg["layers"] is not None else list(range(n_layers))
    print(f"  Model has {n_layers} layers. Probing: {len(layers)} layers (0–{layers[-1]})")

    # Accumulators: layer → list[ndarray | None]
    bufs: dict[str, dict[int, list]] = {
        key: {L: [] for L in layers}
        for key in ("vuln_last", "vuln_mean", "secure_last", "secure_mean")
    }

    skipped = 0
    for pair in tqdm(pairs, desc="  Extracting", unit="pair"):
        for code_field, last_key, mean_key in [
            ("vulnerable_code", "vuln_last",   "vuln_mean"),
            ("secure_code",     "secure_last", "secure_mean"),
        ]:
            code = pair[code_field]
            ids  = tokenizer(
                code,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_TOKENS,
            ).to(device)

            if ids["input_ids"].shape[1] == 0:
                skipped += 1
                for L in layers:
                    bufs[last_key][L].append(None)
                    bufs[mean_key][L].append(None)
                continue

            out = model(**ids, output_hidden_states=True)
            # hidden_states[0] = embedding; hidden_states[L+1] = after block L
            for L in layers:
                h = out.hidden_states[L + 1]          # (1, seq_len, D)
                bufs[last_key][L].append(h[0, -1, :].cpu().float().numpy())
                bufs[mean_key][L].append(h[0].mean(0).cpu().float().numpy())

    if skipped:
        print(f"  Warning: {skipped} empty token sequences skipped.")

    # Build valid mask (both sides must be non-None)
    valid = np.array([
        bufs["vuln_last"][layers[0]][i] is not None
        and bufs["secure_last"][layers[0]][i] is not None
        for i in range(len(pairs))
    ])
    print(f"  Valid pairs after extraction: {valid.sum():,} / {len(pairs):,}")

    result:    dict[int, dict[str, np.ndarray]] = {}
    save_dict: dict[str, np.ndarray]            = {"valid_mask": valid}

    for L in layers:
        for key in ("vuln_last", "vuln_mean", "secure_last", "secure_mean"):
            arr = np.stack([bufs[key][L][i] for i in range(len(pairs)) if valid[i]])
            result.setdefault(L, {})[key] = arr
            save_dict[f"layer_{L}_{key}"] = arr

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **save_dict)
    print(f"  Activations cached → {cache_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ── Probing utilities ──────────────────────────────────────────────────────────

def bootstrap_ci(
    y_true:  np.ndarray,
    y_score: np.ndarray,
    n:       int = N_BOOTSTRAP,
    rng:     np.random.Generator = None,
) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(roc_auc_score(yt, ys))
    arr = np.array(scores)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def run_probe(
    X_vuln:   np.ndarray,
    X_secure: np.ndarray,
    n_boot:   int = N_BOOTSTRAP,
) -> dict:
    """
    5-fold stratified CV with z-score normalisation + PCA(50) + logistic regression.
    Returns auroc_mean, auroc_ci_lower, auroc_ci_upper, auroc_folds, n_pairs.
    """
    N = len(X_vuln)
    X = np.concatenate([X_vuln, X_secure], axis=0)
    y = np.array([1] * N + [0] * N)

    cv   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
    rng  = np.random.default_rng(RNG_SEED)
    fold_aurocs: list[float] = []
    oof_y, oof_p = [], []

    for tr, te in cv.split(X, y):
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X[tr])
        X_te   = scaler.transform(X[te])

        n_comp = min(N_PCA, X_tr.shape[1], X_tr.shape[0] - 1)
        pca    = PCA(n_components=n_comp, random_state=RNG_SEED)
        X_tr   = pca.fit_transform(X_tr)
        X_te   = pca.transform(X_te)

        clf = LogisticRegression(max_iter=1000, random_state=RNG_SEED)
        clf.fit(X_tr, y[tr])

        proba = clf.predict_proba(X_te)[:, 1]
        fold_aurocs.append(roc_auc_score(y[te], proba))
        oof_y.append(y[te])
        oof_p.append(proba)

    ci_lo, ci_hi = bootstrap_ci(
        np.concatenate(oof_y), np.concatenate(oof_p), n=n_boot, rng=rng
    )
    return {
        "n_pairs":        N,
        "auroc_mean":     float(np.mean(fold_aurocs)),
        "auroc_ci_lower": ci_lo,
        "auroc_ci_upper": ci_hi,
        "auroc_folds":    [round(s, 4) for s in fold_aurocs],
    }


def direction_geometry(
    vuln_mean:   np.ndarray,
    secure_mean: np.ndarray,
) -> dict:
    """
    Paired vulnerability direction statistics (mean-token representations).
    - mean_paired_distance: mean ||v_i - s_i||_2
    - pct_aligned: fraction of pairs where delta_i points along the mean direction
    - mean_cosine_to_mean_dir: mean cosine similarity of delta_i to the mean direction
    """
    deltas    = vuln_mean - secure_mean                       # (N, D)
    norms     = np.linalg.norm(deltas, axis=1)                # (N,)
    mean_dir  = deltas.mean(axis=0)
    mean_dir /= np.linalg.norm(mean_dir) + 1e-12
    proj      = deltas @ mean_dir                             # (N,)
    cos_sims  = (deltas / (norms[:, None] + 1e-12)) @ mean_dir
    return {
        "mean_paired_distance":    float(norms.mean()),
        "std_paired_distance":     float(norms.std()),
        "pct_aligned":             float((proj > 0).mean()),
        "mean_cosine_to_mean_dir": float(cos_sims.mean()),
    }


# ── Main experiment runner ─────────────────────────────────────────────────────

def run_experiments(
    model_name: str,
    model_cfg:  dict,
    pairs:      list[dict],
    device:     str,
    output_dir: Path,
) -> list[dict]:
    print(f"\n{'='*60}\n  Model: {model_name}  ({model_cfg['hf_id']})\n{'='*60}")

    cache_path  = output_dir / f"activations_{model_name}.npz"
    acts        = extract_activations(pairs, model_cfg, cache_path, device)
    layers      = model_cfg["layers"]

    # Reload valid_mask to filter pairs metadata for within-language
    valid_mask  = np.load(cache_path)["valid_mask"]
    extensions  = np.array([p["file_extension"] for p, v in zip(pairs, valid_mask) if v])

    records: list[dict] = []

    # ── 1. Global binary probe ───────────────────────────────────────────────
    print("\n  [1/3] Global binary probe …")
    for L in tqdm(layers, desc="    layers"):
        for pooling, vk, sk in [
            ("last_token", "vuln_last",  "secure_last"),
            ("mean_token", "vuln_mean",  "secure_mean"),
        ]:
            res = run_probe(acts[L][vk], acts[L][sk])
            records.append({
                "model":          model_name,
                "experiment":     "global_probe",
                "pooling":        pooling,
                "layer":          L,
                "language":       "all",
                **res,
            })

    # ── 2. Within-language breakdown (mean-token) ────────────────────────────
    print("\n  [2/3] Within-language probe (mean-token) …")
    for ext, lang_name in WITHIN_LANGS.items():
        mask = extensions == ext
        n    = int(mask.sum())
        if n < MIN_WITHIN_LANG:
            print(f"    Skipping {lang_name}: only {n} pairs (< {MIN_WITHIN_LANG})")
            continue
        for L in tqdm(layers, desc=f"    {lang_name} (n={n})", leave=False):
            res = run_probe(
                acts[L]["vuln_mean"][mask],
                acts[L]["secure_mean"][mask],
            )
            records.append({
                "model":          model_name,
                "experiment":     "within_language",
                "pooling":        "mean_token",
                "layer":          L,
                "language":       lang_name,
                "file_extension": ext,
                **res,
            })

    # ── 3. Vulnerability direction geometry ──────────────────────────────────
    print("\n  [3/3] Direction geometry …")
    for L in layers:
        geo = direction_geometry(acts[L]["vuln_mean"], acts[L]["secure_mean"])
        records.append({
            "model":      model_name,
            "experiment": "direction_geometry",
            "layer":      L,
            **geo,
        })

    return records


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-model vulnerability probing")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()) + ["all"],
        default="all",
        help="Which model(s) to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_ACTIVATIONS_DIR),
        help="Directory for cached NPZ activations and probing results",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = (
        "cuda"  if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    pairs       = load_pairs()
    model_names = list(MODELS.keys()) if args.model == "all" else [args.model]

    all_records: list[dict] = []
    for name in model_names:
        records = run_experiments(name, MODELS[name], pairs, device, output_dir)
        all_records.extend(records)

        out = output_dir / f"probing_{name}.jsonl"
        with open(out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"\n  {len(records)} records → {out}")

    combined = output_dir / "probing_all_models.jsonl"
    with open(combined, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"\nAll results → {combined}  ({len(all_records)} records total)")


if __name__ == "__main__":
    main()
