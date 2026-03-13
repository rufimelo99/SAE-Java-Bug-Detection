"""
Length-controlled re-analysis of vuln-vs-secure probe.

Uses actual tokenizer token counts (Qwen2.5-7B-Instruct tokenizer) rather than
character lengths as a proxy.

Two controls:
  1. Length-residualized: regress log(tok_len) out of both S and V matrices via Ridge,
     then probe the residuals.  If AUROC doesn't change, length is not driving anything.
  2. Length-stratified: bin pairs by len_ratio = tok_len_vuln / tok_len_secure into
     quartiles and probe within each bin.  Consistent near-chance across bins rules out
     a length-driven effect.

Output:
  fig_length_controlled.pdf   — left: AUROC vs layer (raw / length-residualized)
                                 right: within-quartile bar chart at L11
Run:
  conda run -n sae python length_controlled_probe.py
"""

import base64
import json
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

HERE       = Path(__file__).parent
ARTIFACTS  = Path(__file__).parents[2] / "artifacts" / "activations"
PAPER_FIGS = (
    Path(__file__).parents[4]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

MODEL_ID  = "Qwen/Qwen2.5-7B-Instruct"
try:
    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"Tokenizer loaded: {MODEL_ID}")
except Exception as _e:
    print(f"[WARN] Could not load tokenizer ({_e}); falling back to character length")
    _TOKENIZER = None

mpl.rcParams.update({
    "font.family":     "serif",
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi":      150,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]

RUNS_REGISTRY = [
    {
        "run_dir": ARTIFACTS / "raw_activations/vulnerable_code_qwen_coder_standard_16384_raw",
        "label":   "Qwen-Raw",
    },
    {
        "run_dir": ARTIFACTS / "TO_UPLOAD",
        "label":   "Qwen-STD-SAE",
    },
    {
        "run_dir": ARTIFACTS / "TOPK",
        "label":   "Qwen-TopK-SAE",
    },
]


# ── Data loading ──────────────────────────────────────────────────────────────

def _tok_len(b64_str) -> int:
    """Decode base64 source code and return tokenizer token count.
    Falls back to character count if tokenizer is unavailable."""
    try:
        code = base64.b64decode(b64_str).decode("utf-8", errors="replace")
    except Exception:
        code = b64_str if isinstance(b64_str, str) else ""
    if _TOKENIZER is not None:
        return len(_TOKENIZER.encode(code, add_special_tokens=False))
    return len(code)


def load_jsonl(jsonl_path):
    """Load activations + lengths from one JSONL file.
    Returns (safe_mat, vuln_mat, meta_df).
    """
    secure_rows, vuln_rows, meta_rows = [], [], []
    with jsonl_path.open() as f:
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
            meta_rows.append({
                "vuln_id":        r["vuln_id"],
                "cwe":            r["cwe"],
                "file_extension": r["file_extension"],
                "tok_len_secure": _tok_len(r.get("secure_code", "")),
                "tok_len_vuln":   _tok_len(r.get("vulnerable_code", "")),
            })
    safe_mat = np.array(secure_rows, dtype=np.float32)
    vuln_mat = np.array(vuln_rows,   dtype=np.float32)
    meta     = pd.DataFrame(meta_rows)
    return safe_mat, vuln_mat, meta


def load_meta_only(jsonl_path):
    """Lightweight: only load metadata + lengths, no activation vectors."""
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append({
                "vuln_id":         r["vuln_id"],
                "cwe":             r["cwe"],
                "file_extension":  r["file_extension"],
                "tok_len_secure": _tok_len(r.get("secure_code", "")),
                "tok_len_vuln":   _tok_len(r.get("vulnerable_code", "")),
            })
    return pd.DataFrame(rows)


def discover_layers(run_dir):
    layers = set()
    for f in run_dir.glob("activations_layer_*_*.jsonl"):
        try:
            layers.add(int(f.name.split("_")[2]))
        except (IndexError, ValueError):
            pass
    return sorted(layers)


def load_layer(run_dir, layer):
    """Load (safe_mat, vuln_mat, meta) for one layer.
    Prefers pre-stacked .pt tensors for activations; always reads JSONL for lengths.
    """
    meta_matches = list(run_dir.glob(f"activations_layer_{layer}_*.jsonl"))
    if not meta_matches:
        return None

    safe_pt = run_dir / f"safe_layer_{layer}.pt"
    vuln_pt = run_dir / f"vulnerable_layer_{layer}.pt"
    if safe_pt.exists() and vuln_pt.exists():
        safe_mat = torch.load(safe_pt, weights_only=True).numpy().astype(np.float32)
        vuln_mat = torch.load(vuln_pt, weights_only=True).numpy().astype(np.float32)
        meta     = load_meta_only(meta_matches[0])
    else:
        safe_mat, vuln_mat, meta = load_jsonl(meta_matches[0])

    return safe_mat, vuln_mat, meta


def load_all_layers():
    layers_data = {}
    for entry in tqdm(RUNS_REGISTRY, desc="Loading runs"):
        run_dir, run_label = entry["run_dir"], entry["label"]
        if not run_dir.exists():
            print(f"[SKIP] {run_label}: directory not found ({run_dir})")
            continue
        available = discover_layers(run_dir)
        for layer in tqdm([l for l in LAYERS if l in available], desc=f"  {run_label}", leave=False):
            result = load_layer(run_dir, layer)
            if result is None:
                print(f"  [SKIP] {run_label} L{layer}: no JSONL")
                continue
            safe_mat, vuln_mat, meta = result
            layers_data[f"{run_label} L{layer}"] = (safe_mat, vuln_mat, meta)
    print(f"\nLoaded {len(layers_data)} layer entries")
    return layers_data


# ── Analysis helpers ──────────────────────────────────────────────────────────

def _bootstrap_auc_ci(y_true, y_score, n_bootstrap=500, ci=0.95):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(SEED)
    n   = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    aucs  = np.array(aucs)
    alpha = (1 - ci) / 2
    return float(aucs.mean()), float(np.quantile(aucs, alpha)), float(np.quantile(aucs, 1 - alpha))


def run_probe(S, V, n_components=50, cv=5, n_bootstrap=500):
    """Stack [S, V], PCA-reduce inside each CV fold, logistic-regression OOF scores, bootstrap CI.
    Scaler and PCA are fit inside each fold to avoid data leakage."""
    n  = len(S)
    X  = np.vstack([S, V])
    y  = np.array([0] * n + [1] * n, dtype=int)

    n_comp  = min(n_components, X.shape[1], X.shape[0] - 1)
    clf     = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=SEED)
    skf     = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    y_score = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        pca    = PCA(n_components=n_comp, random_state=SEED)
        X_tr   = pca.fit_transform(scaler.fit_transform(X[train_idx]))
        X_te   = pca.transform(scaler.transform(X[test_idx]))
        clf.fit(X_tr, y[train_idx])
        y_score[test_idx] = clf.predict_proba(X_te)[:, 1]

    mean_auc, ci_lo, ci_hi = _bootstrap_auc_ci(y, y_score, n_bootstrap=n_bootstrap)
    return {"roc_auc": mean_auc, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": n}


def residualise_length(S, V, meta):
    """
    Regress log(tok_len) out of both activation matrices via Ridge.
    Each side is residualised with its own length covariate.
    Returns (S_residualised, V_residualised).
    """
    log_s = np.log1p(meta["tok_len_secure"].values).reshape(-1, 1).astype(np.float32)
    log_v = np.log1p(meta["tok_len_vuln"].values).reshape(-1, 1).astype(np.float32)

    # Fit one Ridge on all 2n rows so both sides use the same regression
    X_all = np.vstack([S, V])
    L_all = np.vstack([log_s, log_v])
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(L_all, X_all)

    S_r = (S - ridge.predict(log_s)).astype(np.float32)
    V_r = (V - ridge.predict(log_v)).astype(np.float32)
    return S_r, V_r


def stratified_by_length_ratio(S, V, meta, n_quartiles=4, n_components=30, n_bootstrap=500):
    """
    Bin pairs by len_ratio = tok_len_vuln / tok_len_secure into n_quartiles.
    Probe within each bin.  Returns list of result dicts (None if bin too small).
    """
    ratio  = (meta["tok_len_vuln"].values + 1.0) / (meta["tok_len_secure"].values + 1.0)
    edges  = np.quantile(ratio, np.linspace(0, 1, n_quartiles + 1))
    results = []
    for q in range(n_quartiles):
        lo_e, hi_e = edges[q], edges[q + 1]
        mask = (ratio >= lo_e) & (ratio < hi_e) if q < n_quartiles - 1 else (ratio >= lo_e)
        n_q  = mask.sum()
        if n_q < 30:
            results.append(None)
            continue
        nc = min(n_components, n_q - 1, S.shape[1])
        res = run_probe(S[mask], V[mask], n_components=nc, n_bootstrap=n_bootstrap)
        res["quartile"]      = q + 1
        res["len_ratio_lo"]  = float(lo_e)
        res["len_ratio_hi"]  = float(hi_e)
        results.append(res)
    return results


# ── Figure ─────────────────────────────────────────────────────────────────────

def fig_length_controlled(layers_data):
    """
    For each run: two-panel figure.
      Left:  AUROC vs layer — raw probe vs length-residualized probe.
      Right: Within-length-ratio-quartile AUROC at L11.
    """
    run_results = {}

    for label, (safe_mat, vuln_mat, meta) in tqdm(layers_data.items(), desc="Probing layers"):
        run_name, layer_str = label.rsplit(" L", 1)
        layer = int(layer_str)
        if run_name not in run_results:
            run_results[run_name] = {
                "layers":        [],
                "raw":           [],
                "len_resid":     [],
                "quartiles_l11": None,
            }

        raw_res            = run_probe(safe_mat, vuln_mat)
        S_r, V_r           = residualise_length(safe_mat, vuln_mat, meta)
        len_resid_res      = run_probe(S_r, V_r)

        run_results[run_name]["layers"].append(layer)
        run_results[run_name]["raw"].append(raw_res)
        run_results[run_name]["len_resid"].append(len_resid_res)

        if layer == 11:
            run_results[run_name]["quartiles_l11"] = stratified_by_length_ratio(
                safe_mat, vuln_mat, meta
            )

    # Sort entries by layer
    for rd in run_results.values():
        order = np.argsort(rd["layers"])
        for key in ("layers", "raw", "len_resid"):
            rd[key] = [rd[key][i] for i in order]

    n_runs = len(run_results)
    if n_runs == 0:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(n_runs, 2, figsize=(8.0, 3.2 * n_runs))
    if n_runs == 1:
        axes = [axes]

    for row_axes, (run_name, rd) in zip(axes, run_results.items()):
        ax_left, ax_right = row_axes
        layers     = rd["layers"]
        layer_lbls = [f"L{l}" for l in layers]
        xs         = list(range(len(layers)))

        def _plot_line(ax, results, color, label, marker):
            aucs = [r["roc_auc"] if r else np.nan for r in results]
            los  = [r["ci_lo"]   if r else np.nan for r in results]
            his  = [r["ci_hi"]   if r else np.nan for r in results]
            ax.plot(xs, aucs, f"{marker}-", color=color, label=label, lw=1.5, ms=5)
            ax.fill_between(xs, los, his, color=color, alpha=0.15)

        _plot_line(ax_left, rd["raw"],       "#333333", "Raw",                 "o")
        _plot_line(ax_left, rd["len_resid"], "#e07b39", "Length-residualized", "s")

        ax_left.axhline(0.5, color="grey", lw=0.7, ls=":", alpha=0.5, label="Chance")
        if 11 in layers:
            x11 = layers.index(11)
            ax_left.axvline(x11, color="black", lw=1.0, ls="--", alpha=0.4)
            ax_left.text(x11 + 0.12, 0.505, "L11", fontsize=7, color="black", alpha=0.6)
        ax_left.set_xticks(xs)
        ax_left.set_xticklabels(layer_lbls, rotation=45)
        ax_left.set_xlabel("Layer")
        ax_left.set_ylabel("ROC-AUC  (vuln vs. secure)")
        ax_left.set_title(f"{run_name}: raw vs. length-residualized", fontweight="bold")
        ax_left.set_ylim(0.38, 0.65)
        ax_left.legend(fontsize=7, loc="upper right")

        # Right panel: quartile bars at L11
        qr = rd.get("quartiles_l11") or []
        q_valid = [q for q in qr if q is not None]
        if q_valid:
            x_bars = [q["quartile"] for q in q_valid]
            aucs   = [q["roc_auc"] for q in q_valid]
            errs   = np.array([
                [q["roc_auc"] - q["ci_lo"], q["ci_hi"] - q["roc_auc"]]
                for q in q_valid
            ]).T
            labels = [
                f"Q{q['quartile']}\n[{q['len_ratio_lo']:.2f}–{q['len_ratio_hi']:.2f}]\n(n={q['n']})"
                for q in q_valid
            ]
            ax_right.bar(
                x_bars, aucs,
                yerr=errs, color="#4878cf", alpha=0.7,
                capsize=4, error_kw={"lw": 1.2},
            )
            ax_right.axhline(0.5, color="grey", lw=0.7, ls=":", alpha=0.5, label="Chance")
            ax_right.set_xticks(x_bars)
            ax_right.set_xticklabels(labels, fontsize=7)
            ax_right.set_xlabel("Length-ratio quartile  (tok_len_vuln / tok_len_secure)  at L11")
            ax_right.set_ylabel("ROC-AUC")
            ax_right.set_title(f"{run_name}: stratified by length ratio (L11)", fontweight="bold")
            ax_right.set_ylim(0.35, 0.65)
            ax_right.legend(fontsize=7)
        else:
            ax_right.text(0.5, 0.5, "No L11 data", ha="center", va="center", transform=ax_right.transAxes)
            ax_right.set_visible(False)

    fig.suptitle(
        "Length-controlled re-analysis  —  vulnerable vs. secure AUROC\n"
        "(shaded band = 95% bootstrap CI)",
        fontsize=9,
    )
    fig.tight_layout()
    out = PAPER_FIGS / "fig_length_controlled.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")

    # Print numerical summary
    print("\n" + "=" * 72)
    print("Length-controlled re-analysis — AUROC summary")
    print("=" * 72)
    for run_name, rd in run_results.items():
        print(f"\n{run_name}")
        print(f"  {'Layer':>5}  {'Raw AUC':>24}  {'Length-residualized':>24}")
        print("  " + "-" * 58)
        def _fmt(r):
            return f"{r['roc_auc']:.3f} [{r['ci_lo']:.3f}–{r['ci_hi']:.3f}]" if r else "        —       "
        for layer, raw, lr in zip(rd["layers"], rd["raw"], rd["len_resid"]):
            print(f"  {layer:>5}  {_fmt(raw):>24}  {_fmt(lr):>24}")

        print(f"\n  Within length-ratio quartiles at L11:")
        qr = rd.get("quartiles_l11") or []
        for q in qr:
            if q:
                print(
                    f"    Q{q['quartile']} [{q['len_ratio_lo']:.2f}–{q['len_ratio_hi']:.2f}]:"
                    f"  AUROC={q['roc_auc']:.3f} [{q['ci_lo']:.3f}–{q['ci_hi']:.3f}]"
                    f"  n={q['n']}"
                )
    print()


# ── Length distribution summary ───────────────────────────────────────────────

def print_length_stats(layers_data):
    """Print token-length summary from one layer's metadata (representative)."""
    label = next(iter(layers_data))
    _, _, meta = layers_data[label]
    ratio = (meta["tok_len_vuln"] + 1) / (meta["tok_len_secure"] + 1)
    print("\nToken-length summary (from first loaded layer):")
    print(f"  Secure  — mean={meta['tok_len_secure'].mean():.0f}  "
          f"median={meta['tok_len_secure'].median():.0f}  "
          f"std={meta['tok_len_secure'].std():.0f}")
    print(f"  Vuln    — mean={meta['tok_len_vuln'].mean():.0f}  "
          f"median={meta['tok_len_vuln'].median():.0f}  "
          f"std={meta['tok_len_vuln'].std():.0f}")
    print(f"  Ratio (vuln/secure) — mean={ratio.mean():.3f}  "
          f"median={ratio.median():.3f}  "
          f"% vuln > secure: {(meta['tok_len_vuln'] > meta['tok_len_secure']).mean()*100:.1f}%")
    print(f"  Quartile edges: {np.quantile(ratio, [0, 0.25, 0.5, 0.75, 1.0]).round(3)}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Paper figures dir : {PAPER_FIGS}")
    print(f"Activations dir   : {ARTIFACTS}")
    print()

    layers_data = load_all_layers()
    print_length_stats(layers_data)

    fig_length_controlled(layers_data)
    print("Done.")
