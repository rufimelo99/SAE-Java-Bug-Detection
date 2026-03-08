"""
Advanced pooling strategies probe — extends mean-token pooling.

Compares four strategies at each target layer:
  1. Last-token       — from pre-computed .pt / JSONL (baseline)
  2. Mean-token       — uniform mean over all token positions
  3. Attention-weighted — token j weighted by mean(over heads) last-token→j attention
  4. Diff-restricted  — mean over tokens that fall on lines changed by the security patch

Research question: can a smarter pooling strategy recover more of the distributed
vulnerability signal, and does restricting to the patch site concentrate it?

Usage:
    conda run -n sae python advanced_pooling_probe.py
    conda run -n sae python advanced_pooling_probe.py --layers 11 15
    conda run -n sae python advanced_pooling_probe.py --no_attn   # skip attn-weighted
    conda run -n sae python advanced_pooling_probe.py --no_diff   # skip diff-restricted
    conda run -n sae python advanced_pooling_probe.py --n_sample 200  # quick test

Outputs:
    artifacts/activations/advanced_pool/<ts>/
        mean_safe_layer_{L}.pt            [N, 3584]
        mean_vuln_layer_{L}.pt            [N, 3584]
        attn_safe_layer_{L}.pt            [N, 3584]   (if --no_attn not set)
        attn_vuln_layer_{L}.pt            [N, 3584]
        diff_safe_layer_{L}.pt            [N, 3584]   (if --no_diff not set)
        diff_vuln_layer_{L}.pt            [N, 3584]
        meta.json
        probe_results.json
    paper figures / fig_advanced_pooling_comparison.pdf
"""

import argparse
import base64
import bisect
import difflib
import json
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
ARTIFACTS  = Path(__file__).parents[2] / "artifacts" / "activations"
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

MODEL_ID   = "Qwen/Qwen2.5-7B-Instruct"
ALL_LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]
MAX_TOKENS = 2048
SEED       = 42

# ── Style ──────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 150,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_source_records(jsonl_path: Path, n_sample: int = -1):
    records = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            try:
                secure_code  = base64.b64decode(r["secure_code"]).decode("utf-8", errors="replace")
                vuln_code    = base64.b64decode(r["vulnerable_code"]).decode("utf-8", errors="replace")
            except Exception:
                # r.get(..., "") only uses default for missing keys, not for JSON null values;
                # "or ''" handles null (None) values correctly.
                secure_code  = r.get("secure_code") or ""
                vuln_code    = r.get("vulnerable_code") or ""
            records.append({
                "vuln_id":        r["vuln_id"],
                "cwe":            r["cwe"],
                "file_extension": r["file_extension"],
                "secure_code":    secure_code,
                "vulnerable_code": vuln_code,
            })
            if n_sample > 0 and len(records) >= n_sample:
                break
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Diff utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_changed_line_sets(secure_code: str, vuln_code: str):
    """
    Returns (secure_changed_lines, vuln_changed_lines) — sets of 0-indexed
    line numbers changed/added in each version relative to the other.
    Uses SequenceMatcher on line lists (fast, handles moves well).
    """
    s_lines = secure_code.splitlines()
    v_lines = vuln_code.splitlines()

    sm = difflib.SequenceMatcher(None, v_lines, s_lines, autojunk=False)
    secure_changed: set[int] = set()
    vuln_changed:   set[int] = set()

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "insert":   # lines j1:j2 added in secure
            secure_changed.update(range(j1, j2))
        elif tag == "delete":  # lines i1:i2 removed from vuln
            vuln_changed.update(range(i1, i2))
        elif tag == "replace":
            secure_changed.update(range(j1, j2))
            vuln_changed.update(range(i1, i2))
        # "equal" — skip

    return secure_changed, vuln_changed


def build_token_diff_mask(
    code: str,
    changed_lines: set,
    offset_mapping: torch.Tensor,
) -> torch.Tensor:
    """
    Returns a bool tensor [seq_len] where True = token overlaps a changed line.
    Uses character offsets (return_offsets_mapping=True) to map tokens → lines.
    Falls back to all-True if offset_mapping is unavailable or all-zero.
    """
    # Build cumulative char offsets per line
    lines = code.split("\n")
    line_starts: list[int] = []
    pos = 0
    for ln in lines:
        line_starts.append(pos)
        pos += len(ln) + 1   # +1 for the '\n'
    line_starts.append(pos)  # sentinel

    def char_to_line(char_idx: int) -> int:
        idx = bisect.bisect_right(line_starts, char_idx) - 1
        return max(0, min(idx, len(lines) - 1))

    offsets = offset_mapping.tolist()  # list of (start, end) pairs
    mask = []
    for start, end in offsets:
        # (0, 0) marks special tokens — exclude them from diff mask (treat as unchanged)
        if start == 0 and end == 0:
            mask.append(False)
        else:
            line_num = char_to_line(start)
            mask.append(line_num in changed_lines)

    return torch.tensor(mask, dtype=torch.bool)


# ─────────────────────────────────────────────────────────────────────────────
# Forward pass + pooling extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_pooled_vectors(
    secure_code: str,
    vuln_code:   str,
    tokenizer,
    model,
    target_layers: list[int],
    device,
    do_attn: bool = True,
    do_diff: bool = True,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    """
    Single-pair extraction. Returns:
      {
        "mean":  {"secure": {layer: np.array}, "vuln": {...}},
        "attn":  {"secure": {layer: np.array}, "vuln": {...}},   # if do_attn
        "diff":  {"secure": {layer: np.array}, "vuln": {...}},   # if do_diff
        "diff_frac_secure": float,   # fraction of tokens in changed lines (secure)
        "diff_frac_vuln":   float,
        "diff_n_changed_lines": (int, int),
      }
    """
    # Guard against null/empty code strings (JSON null records in dataset)
    secure_code = secure_code or ""
    vuln_code   = vuln_code   or ""

    # ── Diff masks ────────────────────────────────────────────────────────────
    s_changed, v_changed = get_changed_line_sets(secure_code, vuln_code) if do_diff else (set(), set())

    result: dict = {
        "mean": {"secure": {}, "vuln": {}},
        "diff_frac_secure": 0.0,
        "diff_frac_vuln":   0.0,
        "diff_n_changed_lines": (len(s_changed), len(v_changed)),
    }
    if do_attn:
        result["attn"] = {"secure": {}, "vuln": {}}
    if do_diff:
        result["diff"] = {"secure": {}, "vuln": {}}

    sides = [
        ("secure", secure_code, s_changed),
        ("vuln",   vuln_code,   v_changed),
    ]

    for side_name, code, changed_lines in sides:
        # Tokenise with offset mapping for diff masking
        encoding = tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
            return_offsets_mapping=True,
        )
        # offset_mapping may be absent or None if the slow tokenizer is used
        _om = encoding.pop("offset_mapping", None)
        offset_mapping = _om[0] if _om is not None else None   # [seq_len, 2] or None

        # Diff mask
        if do_diff and changed_lines and offset_mapping is not None:
            diff_mask = build_token_diff_mask(code, changed_lines, offset_mapping)
            # Record fraction
            if side_name == "secure":
                result["diff_frac_secure"] = diff_mask.float().mean().item()
            else:
                result["diff_frac_vuln"] = diff_mask.float().mean().item()
        else:
            diff_mask = None

        inputs = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                output_attentions=do_attn,
            )

        for layer_idx in target_layers:
            # hidden_states[0] = embeddings; hidden_states[i+1] = after layer i
            h = outputs.hidden_states[layer_idx + 1][0].float()  # [seq_len, d_model]

            # ── Mean-token pooling ───────────────────────────────────────────
            mean_vec = h.mean(dim=0).cpu().numpy()
            result["mean"][side_name][layer_idx] = mean_vec

            # ── Attention-weighted pooling ───────────────────────────────────
            if do_attn:
                # outputs.attentions may be None (Flash Attention 2 / SDPA on CUDA
                # does not return attention weights).  In that case fall back to mean-token.
                attn_layer = (
                    outputs.attentions[layer_idx]
                    if outputs.attentions is not None
                    else None
                )
                if attn_layer is not None:
                    # attentions[layer_idx]: [1, n_heads, seq_len, seq_len]
                    # Use float32 for attention weights to avoid fp16 overflow (NaN)
                    attn      = attn_layer[0].float()           # [n_heads, seq_len, seq_len]
                    last_attn = attn[:, -1, :]                  # [n_heads, seq_len]
                    weights   = last_attn.mean(dim=0)           # [seq_len]
                    # Replace any NaN weights (degenerate sequences) with uniform
                    if torch.isnan(weights).any():
                        weights = torch.ones_like(weights)
                    weights   = weights / (weights.sum() + 1e-8)
                    attn_vec  = (h * weights.unsqueeze(-1)).sum(dim=0).cpu().numpy()
                else:
                    # Flash/SDPA backend — attention weights unavailable; use mean-token
                    attn_vec = mean_vec
                result["attn"][side_name][layer_idx] = attn_vec

            # ── Diff-restricted pooling ──────────────────────────────────────
            if do_diff:
                if diff_mask is not None and diff_mask.any():
                    seq_len = h.shape[0]
                    m = diff_mask[:seq_len].to(device)  # guard against truncation
                    diff_vec = h[m].mean(dim=0).cpu().numpy()
                else:
                    # No changed lines (e.g. identical files or truncated diff)
                    diff_vec = h.mean(dim=0).cpu().numpy()   # fallback = mean-token
                result["diff"][side_name][layer_idx] = diff_vec

        # Free GPU memory explicitly
        del outputs
        if device == "cuda":
            torch.cuda.empty_cache()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Probing utilities
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_auc_ci(y_true, y_score, n_bootstrap=500, ci=0.95, seed=SEED):
    from sklearn.metrics import roc_auc_score
    rng  = np.random.default_rng(seed)
    n    = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    aucs = np.array(aucs)
    alpha = (1 - ci) / 2
    return float(aucs.mean()), float(np.quantile(aucs, alpha)), float(np.quantile(aucs, 1 - alpha))


def probe_vuln_secure(safe_mat, vuln_mat, n_components=50, cv=5, n_bootstrap=500):
    n  = len(safe_mat)
    X  = np.vstack([safe_mat, vuln_mat]).astype(np.float32)
    y  = np.array([0] * n + [1] * n, dtype=int)

    # Replace NaN/Inf that can arise from fp16 overflow or zero-filled fallback rows.
    # np.nan_to_num converts NaN→0, +inf→large finite, -inf→large negative.
    n_bad = int(np.sum(~np.isfinite(X)))
    if n_bad > 0:
        print(f"    [probe] replacing {n_bad} non-finite values (NaN/Inf) with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    pca    = PCA(n_components=min(n_components, X.shape[1], X.shape[0] - 1), random_state=SEED)
    X_pca  = pca.fit_transform(scaler.fit_transform(X))

    clf    = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=SEED)
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)

    y_score = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in skf.split(X_pca, y):
        clf.fit(X_pca[train_idx], y[train_idx])
        y_score[test_idx] = clf.predict_proba(X_pca[test_idx])[:, 1]

    mean_auc, ci_lo, ci_hi = _bootstrap_auc_ci(y, y_score, n_bootstrap=n_bootstrap)
    return {"roc_auc": mean_auc, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": n}


# ─────────────────────────────────────────────────────────────────────────────
# Load pre-computed last-token results
# ─────────────────────────────────────────────────────────────────────────────

def load_last_token_results(target_layers: list[int]):
    results = {}
    for layer in target_layers:
        safe_pt = RAW_RUN_DIR / f"safe_layer_{layer}.pt"
        vuln_pt = RAW_RUN_DIR / f"vulnerable_layer_{layer}.pt"
        if safe_pt.exists() and vuln_pt.exists():
            safe_mat = torch.load(safe_pt, weights_only=True).numpy().astype(np.float32)
            vuln_mat = torch.load(vuln_pt, weights_only=True).numpy().astype(np.float32)
        else:
            matches = list(RAW_RUN_DIR.glob(f"activations_layer_{layer}_*.jsonl"))
            if not matches:
                print(f"  [SKIP] last-token L{layer}: no data found")
                continue
            safe_rows, vuln_rows = [], []
            with matches[0].open() as f:
                for line in f:
                    r = json.loads(line.strip())
                    safe_rows.append(r["secure"])
                    vuln_rows.append(r["vulnerable"])
            safe_mat = np.array(safe_rows, dtype=np.float32)
            vuln_mat = np.array(vuln_rows, dtype=np.float32)

        results[layer] = probe_vuln_secure(safe_mat, vuln_mat)
        print(f"  Last-token L{layer}: AUROC {results[layer]['roc_auc']:.3f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(all_results: dict, target_layers: list[int], out_path: Path):
    """
    Line plot: AUROC vs layer for each pooling strategy.
    all_results keys: "last_token", "mean_token", "attn_weighted", "diff_restricted"
    """
    COLORS = {
        "last_token":      "#555555",
        "mean_token":      "#d65f5f",
        "attn_weighted":   "#3a82c4",
        "diff_restricted": "#2ca02c",
    }
    LABELS = {
        "last_token":      "Last-token (baseline)",
        "mean_token":      "Mean-token",
        "attn_weighted":   "Attention-weighted",
        "diff_restricted": "Diff-restricted",
    }
    MARKERS = {
        "last_token": "o", "mean_token": "s",
        "attn_weighted": "^", "diff_restricted": "D",
    }

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for key in ["last_token", "mean_token", "attn_weighted", "diff_restricted"]:
        if key not in all_results:
            continue
        res   = all_results[key]
        xs    = [target_layers.index(l) for l in target_layers if l in res]
        aucs  = [res[l]["roc_auc"] for l in target_layers if l in res]
        los   = [res[l]["ci_lo"]   for l in target_layers if l in res]
        his   = [res[l]["ci_hi"]   for l in target_layers if l in res]

        ax.plot(xs, aucs, f"{MARKERS[key]}-",
                color=COLORS[key], label=LABELS[key], lw=1.5, ms=5)
        ax.fill_between(xs, los, his, color=COLORS[key], alpha=0.15)

    ax.axhline(0.5, color="grey", lw=0.7, ls=":", alpha=0.5, label="Chance (0.5)")
    if 11 in target_layers:
        x11 = target_layers.index(11)
        ax.axvline(x11, color="black", lw=0.8, ls="--", alpha=0.3)
        ax.text(x11 + 0.1, ax.get_ylim()[0] + 0.01, "L11",
                fontsize=7, color="black", alpha=0.5)

    ax.set_xticks(range(len(target_layers)))
    ax.set_xticklabels([f"L{l}" for l in target_layers], rotation=45)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC  (vuln vs.\\ secure)")
    ax.set_title(
        "Pooling strategy comparison — raw residual stream\n"
        "(shaded = 95\\% bootstrap CI)",
        fontsize=8,
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_diff_stats(meta_rows: list, diff_fracs_s: list, diff_fracs_v: list,
                    n_changed_lines_s: list, n_changed_lines_v: list, out_path: Path):
    """Diagnostic: histogram of diff-restricted token fractions."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    for ax, fracs, label in zip(axes,
                                [diff_fracs_s, diff_fracs_v],
                                ["Secure (fix lines)", "Vulnerable (removed lines)"]):
        ax.hist(fracs, bins=30, color="#2ca02c", alpha=0.7, edgecolor="white")
        ax.axvline(np.mean(fracs), color="red", lw=1.5, ls="--",
                   label=f"Mean = {np.mean(fracs):.2f}")
        ax.set_xlabel("Fraction of tokens in changed lines")
        ax.set_ylabel("Count")
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=8)

    # Fraction of zero-diff samples
    n_zero_s = sum(1 for f in diff_fracs_s if f == 0)
    n_zero_v = sum(1 for f in diff_fracs_v if f == 0)
    fig.suptitle(
        f"Diff-restricted token coverage\n"
        f"(zero-diff fallback: secure={n_zero_s}, vuln={n_zero_v})",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers",    nargs="+", type=int, default=ALL_LAYERS,
                        help="Which layers to probe (default: all 8)")
    parser.add_argument("--no_attn",   action="store_true",
                        help="Skip attention-weighted pooling")
    parser.add_argument("--no_diff",   action="store_true",
                        help="Skip diff-restricted pooling")
    parser.add_argument("--n_sample",  type=int, default=-1,
                        help="Subsample N records (-1 = all)")
    args = parser.parse_args()

    target_layers = sorted(args.layers)
    do_attn       = not args.no_attn
    do_diff       = not args.no_diff

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print(f"Layers: {target_layers}")
    print(f"Attention-weighted: {do_attn}")
    print(f"Diff-restricted:    {do_diff}")
    print(f"Source JSONL: {SOURCE_JSONL}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\nLoading source records...")
    records = load_source_records(SOURCE_JSONL, n_sample=args.n_sample)
    N = len(records)
    print(f"  {N} samples")

    # ── 2. Load model ─────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Use eager attention when attention weights are needed (Flash Attention 2 and
    # SDPA return None for attention tensors, so we force the eager backend when
    # do_attn=True).  On MPS, "eager" is always used anyway.
    attn_impl_kwargs = (
        {"attn_implementation": "eager"} if do_attn and device == "cuda" else {}
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, **attn_impl_kwargs
    ).to(device)
    if do_attn and device == "cuda":
        print("  Using eager attention implementation (required for output_attentions=True)")
    model.eval()
    print("  Model ready.")

    # ── 3. Extract activations ────────────────────────────────────────────────
    ts      = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ARTIFACTS / "advanced_pool" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Buffers: pooling_type → side → layer → [N, d_model] list
    mean_bufs = {s: {l: [] for l in target_layers} for s in ("secure", "vuln")}
    attn_bufs = {s: {l: [] for l in target_layers} for s in ("secure", "vuln")} if do_attn else None
    diff_bufs = {s: {l: [] for l in target_layers} for s in ("secure", "vuln")} if do_diff else None

    diff_fracs_s, diff_fracs_v   = [], []
    n_changed_s,  n_changed_v    = [], []
    meta_rows = []

    print(f"\nExtracting pooled activations ({N} samples)...")
    t0 = time.time()

    for i, rec in enumerate(records):
        if (i + 1) % 50 == 0:
            el  = time.time() - t0
            eta = el / (i + 1) * (N - i - 1)
            print(f"  [{i+1:>4}/{N}]  elapsed {el:.0f}s  ETA {eta:.0f}s")

        try:
            r = extract_pooled_vectors(
                rec["secure_code"],
                rec["vulnerable_code"],
                tokenizer, model, target_layers, device,
                do_attn=do_attn, do_diff=do_diff,
            )
        except Exception as e:
            print(f"  [WARN] sample {i} ({rec['vuln_id']}): {e}")
            d_model = model.config.hidden_size
            for l in target_layers:
                z = np.zeros(d_model, dtype=np.float32)
                mean_bufs["secure"][l].append(z)
                mean_bufs["vuln"][l].append(z)
                if do_attn:
                    attn_bufs["secure"][l].append(z)
                    attn_bufs["vuln"][l].append(z)
                if do_diff:
                    diff_bufs["secure"][l].append(z)
                    diff_bufs["vuln"][l].append(z)
            diff_fracs_s.append(0.0)
            diff_fracs_v.append(0.0)
            n_changed_s.append(0)
            n_changed_v.append(0)
            meta_rows.append({k: rec[k] for k in ("vuln_id", "cwe", "file_extension")})
            continue

        for side in ("secure", "vuln"):
            for l in target_layers:
                mean_bufs[side][l].append(r["mean"][side][l])
                if do_attn:
                    attn_bufs[side][l].append(r["attn"][side][l])
                if do_diff:
                    diff_bufs[side][l].append(r["diff"][side][l])

        diff_fracs_s.append(r["diff_frac_secure"])
        diff_fracs_v.append(r["diff_frac_vuln"])
        nc = r["diff_n_changed_lines"]
        n_changed_s.append(nc[0])
        n_changed_v.append(nc[1])
        meta_rows.append({k: rec[k] for k in ("vuln_id", "cwe", "file_extension")})

    # ── 4. Save tensors ───────────────────────────────────────────────────────
    print(f"\nSaving tensors to {out_dir} ...")
    for l in target_layers:
        for side, bufs in [("safe", mean_bufs["secure"]), ("vuln", mean_bufs["vuln"])]:
            torch.save(torch.tensor(np.array(bufs[l]), dtype=torch.float32),
                       out_dir / f"mean_{side}_layer_{l}.pt")
        if do_attn:
            for side, bufs in [("safe", attn_bufs["secure"]), ("vuln", attn_bufs["vuln"])]:
                torch.save(torch.tensor(np.array(bufs[l]), dtype=torch.float32),
                           out_dir / f"attn_{side}_layer_{l}.pt")
        if do_diff:
            for side, bufs in [("safe", diff_bufs["secure"]), ("vuln", diff_bufs["vuln"])]:
                torch.save(torch.tensor(np.array(bufs[l]), dtype=torch.float32),
                           out_dir / f"diff_{side}_layer_{l}.pt")

    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta_rows, f)

    # Diff coverage stats
    if do_diff:
        print(f"\n  Diff coverage (fraction of tokens in changed lines):")
        print(f"    Secure : mean={np.mean(diff_fracs_s):.3f}  "
              f"median={np.median(diff_fracs_s):.3f}  "
              f"zero-diff={sum(1 for x in diff_fracs_s if x == 0)}/{N}")
        print(f"    Vuln   : mean={np.mean(diff_fracs_v):.3f}  "
              f"median={np.median(diff_fracs_v):.3f}  "
              f"zero-diff={sum(1 for x in diff_fracs_v if x == 0)}/{N}")
        print(f"  Changed lines: secure mean={np.mean(n_changed_s):.1f}, "
              f"vuln mean={np.mean(n_changed_v):.1f}")

    # ── 5. Run probes ─────────────────────────────────────────────────────────
    print("\nRunning probes ...")
    all_results: dict = {}

    print("\n  [mean-token]")
    all_results["mean_token"] = {}
    for l in target_layers:
        sm = np.array(mean_bufs["secure"][l], dtype=np.float32)
        vm = np.array(mean_bufs["vuln"][l],   dtype=np.float32)
        res = probe_vuln_secure(sm, vm)
        all_results["mean_token"][l] = res
        print(f"    L{l:>2}: {res['roc_auc']:.3f} [{res['ci_lo']:.3f}–{res['ci_hi']:.3f}]")

    if do_attn:
        print("\n  [attention-weighted]")
        all_results["attn_weighted"] = {}
        for l in target_layers:
            sm = np.array(attn_bufs["secure"][l], dtype=np.float32)
            vm = np.array(attn_bufs["vuln"][l],   dtype=np.float32)
            res = probe_vuln_secure(sm, vm)
            all_results["attn_weighted"][l] = res
            print(f"    L{l:>2}: {res['roc_auc']:.3f} [{res['ci_lo']:.3f}–{res['ci_hi']:.3f}]")

    if do_diff:
        print("\n  [diff-restricted]")
        all_results["diff_restricted"] = {}
        for l in target_layers:
            sm = np.array(diff_bufs["secure"][l], dtype=np.float32)
            vm = np.array(diff_bufs["vuln"][l],   dtype=np.float32)
            res = probe_vuln_secure(sm, vm)
            all_results["diff_restricted"][l] = res
            print(f"    L{l:>2}: {res['roc_auc']:.3f} [{res['ci_lo']:.3f}–{res['ci_hi']:.3f}]")

    print("\n  [last-token (from pre-computed .pt / JSONL)]")
    all_results["last_token"] = load_last_token_results(target_layers)

    # ── 6. Print summary table ────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("AUROC comparison across pooling strategies (raw residual stream)")
    print("=" * 90)
    methods = ["last_token", "mean_token"] + (["attn_weighted"] if do_attn else []) + (["diff_restricted"] if do_diff else [])
    header  = f"{'Layer':>6}  " + "  ".join(f"{'  '.join(m.split('_')):^25}" for m in methods)
    print(header)
    print("-" * 90)
    for l in target_layers:
        row = f"  L{l:>2}   "
        for m in methods:
            d = all_results.get(m, {}).get(l)
            row += (f"{d['roc_auc']:.3f} [{d['ci_lo']:.3f}–{d['ci_hi']:.3f}]   " if d else f"{'—':^25}   ")
        print(row)
    print("=" * 90)

    # Peak layer across mean-token
    if all_results.get("mean_token"):
        best_l = max(all_results["mean_token"], key=lambda l: all_results["mean_token"][l]["roc_auc"])
        print(f"\n  Peak mean-token: L{best_l} = {all_results['mean_token'][best_l]['roc_auc']:.3f}")
    if do_attn and all_results.get("attn_weighted"):
        best_l = max(all_results["attn_weighted"], key=lambda l: all_results["attn_weighted"][l]["roc_auc"])
        print(f"  Peak attn-weighted: L{best_l} = {all_results['attn_weighted'][best_l]['roc_auc']:.3f}")
    if do_diff and all_results.get("diff_restricted"):
        best_l = max(all_results["diff_restricted"], key=lambda l: all_results["diff_restricted"][l]["roc_auc"])
        print(f"  Peak diff-restricted: L{best_l} = {all_results['diff_restricted'][best_l]['roc_auc']:.3f}")

    # ── 7. Save figures ───────────────────────────────────────────────────────
    fig_path = PAPER_FIGS / "fig_advanced_pooling_comparison.pdf"
    plot_comparison(all_results, target_layers, fig_path)
    # Also save locally
    plot_comparison(all_results, target_layers, out_dir / "fig_advanced_pooling_comparison.pdf")

    if do_diff:
        plot_diff_stats(
            meta_rows, diff_fracs_s, diff_fracs_v, n_changed_s, n_changed_v,
            out_dir / "fig_diff_coverage.pdf",
        )

    # ── 8. Save results JSON ──────────────────────────────────────────────────
    payload = {
        "run_ts": ts, "layers": target_layers,
        "do_attn": do_attn, "do_diff": do_diff, "n_sample": N,
        "diff_coverage": {
            "mean_frac_secure":   float(np.mean(diff_fracs_s)),
            "mean_frac_vuln":     float(np.mean(diff_fracs_v)),
            "median_frac_secure": float(np.median(diff_fracs_s)),
            "median_frac_vuln":   float(np.median(diff_fracs_v)),
            "n_zero_diff_secure": int(sum(1 for x in diff_fracs_s if x == 0)),
            "n_zero_diff_vuln":   int(sum(1 for x in diff_fracs_v if x == 0)),
        } if do_diff else {},
        "probe_results": {
            method: {str(l): res for l, res in results.items()}
            for method, results in all_results.items()
        },
    }
    with (out_dir / "probe_results.json").open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {out_dir}/probe_results.json")
    print("Done.")


if __name__ == "__main__":
    main()
