"""
Within-language baseline analysis — reviewer Gap 3 response.

Addresses the reviewer's request for within-language probing as a stronger control
than global residualisation. Three figures:

  fig_within_lang_by_layer.pdf  — Raw / Residualised / Within-language AUC across layers
                                   (C→Memory Safety, PHP→Injection). Key narrative figure.
  fig_within_vs_resid.pdf       — Scatter: within-language AUC vs residualised AUC.
                                   Near-diagonal → Ridge residualisation was sufficient.
  fig_within_c_pairwise.pdf     — Pairwise CWE-family AUC within C samples across layers.
                                   Shows genuine CWE structure when language is constant.

Run from anywhere:
  conda run -n sae python within_language_baseline.py
"""

import json
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

HERE       = Path(__file__).parent
ARTIFACTS  = Path(__file__).parents[2] / "artifacts" / "activations"
PAPER_FIGS = (
    Path(__file__).parents[4]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

SEED  = 42
np.random.seed(SEED)

# ── NeurIPS-compatible style ──────────────────────────────────────────────────
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

# ── CWE family mapping ────────────────────────────────────────────────────────
CWE_FAMILIES = {
    "Memory Safety":   ["CWE-119","CWE-125","CWE-787","CWE-476","CWE-120",
                        "CWE-416","CWE-401","CWE-190","CWE-122","CWE-121","CWE-415"],
    "Injection":       ["CWE-89","CWE-79","CWE-78","CWE-94","CWE-77"],
    "Info Disclosure": ["CWE-200","CWE-209"],
    "Access Control":  ["CWE-264","CWE-287","CWE-22","CWE-284","CWE-285"],
    "Resource Mgmt":   ["CWE-400","CWE-399","CWE-369","CWE-362","CWE-20","CWE-770"],
}
CWE_TO_FAMILY = {c: f for f, cs in CWE_FAMILIES.items() for c in cs}
MAIN_FAMILIES = list(CWE_FAMILIES.keys())

FAMILY_COLORS = {
    "Memory Safety":   "#4878cf",
    "Injection":       "#e07b39",
    "Info Disclosure": "#6acc65",
    "Access Control":  "#d65f5f",
    "Resource Mgmt":   "#b47cc7",
}

LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]

# Dominant (language, family) pairs — where within-language probing is feasible
LANG_FAMILY_PAIRS = [
    ("c",   "Memory Safety"),
    ("php", "Injection"),
    ("js",  "Injection"),
    ("c",   "Access Control"),
    ("c",   "Resource Mgmt"),
]

RUNS_REGISTRY = [
    {
        "run_dir":     ARTIFACTS / "raw_activations/vulnerable_code_qwen_coder_standard_16384_raw",
        "label":       "Qwen-Raw",
        "kind":        "raw",
        "min_samples": 100,
    },
    {
        "run_dir":     ARTIFACTS / "TO_UPLOAD",
        "label":       "Qwen-STD-SAE",
        "kind":        "sae",
        "min_samples": 100,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl_meta(jsonl_path):
    records = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records.append({
                "vuln_id":        r["vuln_id"],
                "cwe":            r["cwe"],
                "file_extension": r["file_extension"],
            })
    return pd.DataFrame(records)


def load_jsonl_activations(jsonl_path):
    secure_rows, vuln_rows = [], []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            secure_rows.append(r["secure"])
            vuln_rows.append(r["vulnerable"])
    return np.array(secure_rows, dtype=np.float32), np.array(vuln_rows, dtype=np.float32)


def discover_layers(run_dir):
    layers = set()
    for f in run_dir.glob("activations_layer_*_*.jsonl"):
        try:
            layers.add(int(f.name.split("_")[2]))
        except (IndexError, ValueError):
            pass
    return sorted(layers)


def load_layer(run_dir, layer):
    """Load (safe_mat, vuln_mat, meta) for one layer. Prefers .pt tensors, falls back to JSONL."""
    safe_pt = run_dir / f"safe_layer_{layer}.pt"
    vuln_pt = run_dir / f"vulnerable_layer_{layer}.pt"
    if safe_pt.exists() and vuln_pt.exists():
        safe_mat = torch.load(safe_pt, weights_only=True).numpy().astype(np.float32)
        vuln_mat = torch.load(vuln_pt, weights_only=True).numpy().astype(np.float32)
    else:
        matches = list(run_dir.glob(f"activations_layer_{layer}_*.jsonl"))
        if not matches:
            return None
        safe_mat, vuln_mat = load_jsonl_activations(matches[0])

    meta_matches = list(run_dir.glob(f"activations_layer_{layer}_*.jsonl"))
    if not meta_matches:
        return None
    return safe_mat, vuln_mat, load_jsonl_meta(meta_matches[0])


def load_all_layers():
    """Return layers_data: {label -> (delta, safe_mat, vuln_mat, meta, dim_label)}"""
    layers_data = {}
    for entry in tqdm(RUNS_REGISTRY, desc="Loading runs"):
        run_dir   = entry["run_dir"]
        run_label = entry["label"]
        kind      = entry["kind"]
        min_n     = entry.get("min_samples", 100)

        if not run_dir.exists():
            print(f"[SKIP] {run_label}: directory not found ({run_dir})")
            continue

        available    = discover_layers(run_dir)
        layers_to_load = [l for l in LAYERS if l in available]

        for layer in tqdm(layers_to_load, desc=f"  {run_label}", leave=False):
            result = load_layer(run_dir, layer)
            if result is None:
                print(f"  [SKIP] {run_label} L{layer}: no data")
                continue
            safe_mat, vuln_mat, meta = result
            if len(meta) < min_n:
                print(f"  [SKIP] {run_label} L{layer}: {len(meta)} < {min_n}")
                continue
            meta["family"] = meta["cwe"].map(CWE_TO_FAMILY).fillna("Other")
            delta = vuln_mat - safe_mat
            layers_data[f"{run_label} L{layer}"] = (delta, safe_mat, vuln_mat, meta, f"{kind} {delta.shape[1]}-dim")

    print(f"\nLoaded {len(layers_data)} layer entries:")
    for lbl, (d, s, v, m, dim) in layers_data.items():
        print(f"  {lbl:40s}  delta={d.shape}  n={len(m)}")
    return layers_data


# ─────────────────────────────────────────────────────────────────────────────
# Analysis functions
# ─────────────────────────────────────────────────────────────────────────────

def probe_cwe_families(delta, meta, families, n_components=50, cv=5):
    scaler = StandardScaler()
    pca    = PCA(n_components=min(n_components, delta.shape[1], delta.shape[0]-1), random_state=SEED)
    X      = pca.fit_transform(scaler.fit_transform(delta))
    rows   = []
    for fam in families:
        y = (meta["family"].values == fam).astype(int)
        if y.sum() < 10 or (1-y).sum() < 10:
            continue
        clf = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=SEED)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
        auc = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc").mean()
        rows.append({"family": fam, "n_pos": int(y.sum()), "roc_auc": auc})
    return pd.DataFrame(rows).set_index("family")


def residualise_filetype(delta, meta):
    ext_ohe = pd.get_dummies(meta["file_extension"], drop_first=False).astype(float).values
    ridge   = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(ext_ohe, delta)
    return (delta - ridge.predict(ext_ohe)).astype(np.float32)


def probe_within_language(delta, meta, ext, families, n_components=30, cv=5):
    """One-vs-rest CWE probes restricted to samples with file_extension == ext."""
    mask = (meta["file_extension"] == ext).values
    if mask.sum() < 30:
        return None
    d_sub = delta[mask]
    m_sub = meta[mask].reset_index(drop=True)
    scaler = StandardScaler()
    pca    = PCA(n_components=min(n_components, d_sub.shape[1], d_sub.shape[0]-1), random_state=SEED)
    X      = pca.fit_transform(scaler.fit_transform(d_sub))
    rows   = []
    for fam in families:
        y = (m_sub["family"].values == fam).astype(int)
        if y.sum() < 5 or (1-y).sum() < 5:
            continue
        clf = LogisticRegression(C=0.1, max_iter=500, class_weight="balanced", random_state=SEED)
        skf = StratifiedKFold(n_splits=min(cv, int(y.sum())), shuffle=True, random_state=SEED)
        try:
            auc = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc").mean()
            rows.append({"family": fam, "n_pos": int(y.sum()), "n_total": int(mask.sum()), "roc_auc": auc})
        except Exception:
            pass
    if not rows:
        return None
    return pd.DataFrame(rows).set_index("family")


def pairwise_family_probe(delta, meta, families, n_components=50, cv=5):
    """
    NxN symmetric DataFrame: entry (i,j) = binary logistic-regression ROC-AUC
    for family_i vs family_j. Diagonal and unreachable pairs are NaN.
    """
    scaler = StandardScaler()
    pca    = PCA(n_components=min(n_components, delta.shape[1], delta.shape[0]-1), random_state=SEED)
    X      = pca.fit_transform(scaler.fit_transform(delta))
    n      = len(families)
    mat    = np.full((n, n), np.nan)
    for i, fam_i in enumerate(families):
        for j, fam_j in enumerate(families):
            if i >= j:
                continue
            mask = (meta["family"].values == fam_i) | (meta["family"].values == fam_j)
            if mask.sum() < 20:
                continue
            X_sub = X[mask]
            y     = (meta["family"].values[mask] == fam_i).astype(int)
            if y.sum() < 5 or (1-y).sum() < 5:
                continue
            clf = LogisticRegression(C=0.1, max_iter=500, class_weight="balanced", random_state=SEED)
            k   = min(cv, int(y.sum()), int((1-y).sum()))
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
            try:
                auc       = cross_val_score(clf, X_sub, y, cv=skf, scoring="roc_auc").mean()
                mat[i, j] = auc
                mat[j, i] = auc
            except Exception:
                pass
    return pd.DataFrame(mat, index=families, columns=families)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Within-language AUC across layers
# ─────────────────────────────────────────────────────────────────────────────

def fig_within_lang_by_layer(layers_data):
    """
    Two panels: C → Memory Safety, PHP → Injection.
    Three lines each: Raw AUC / Residualised AUC / Within-language AUC.
    Dashed vertical at L11 (SAE training layer).
    """
    targets = [("c", "Memory Safety"), ("php", "Injection")]

    # Collect results
    # results[run_kind][lang_fam][layer] = {raw, resid, within}
    run_kinds = {}
    for label, (delta, _s, _v, meta, _) in layers_data.items():
        run_name, layer_str = label.rsplit(" L", 1)
        layer = int(layer_str)
        resid = residualise_filetype(delta, meta)

        raw_probe   = probe_cwe_families(delta, meta, MAIN_FAMILIES)
        resid_probe = probe_cwe_families(resid, meta, MAIN_FAMILIES)

        if run_name not in run_kinds:
            run_kinds[run_name] = {}

        for ext, fam in targets:
            key = (ext, fam)
            if key not in run_kinds[run_name]:
                run_kinds[run_name][key] = {}

            within_probe = probe_within_language(delta, meta, ext, MAIN_FAMILIES)

            run_kinds[run_name][key][layer] = {
                "raw":    raw_probe["roc_auc"].get(fam, np.nan) if fam in raw_probe.index else np.nan,
                "resid":  resid_probe["roc_auc"].get(fam, np.nan) if fam in resid_probe.index else np.nan,
                "within": within_probe["roc_auc"].get(fam, np.nan) if (within_probe is not None and fam in within_probe.index) else np.nan,
            }

    for run_name, kf_data in run_kinds.items():
        fig, axes = plt.subplots(1, len(targets), figsize=(6.75, 2.8), sharey=False)
        if len(targets) == 1:
            axes = [axes]

        for ax, (ext, fam) in zip(axes, targets):
            d = kf_data.get((ext, fam), {})
            sorted_layers = sorted(d.keys())
            raw_aucs    = [d[l]["raw"]    for l in sorted_layers]
            resid_aucs  = [d[l]["resid"]  for l in sorted_layers]
            within_aucs = [d[l]["within"] for l in sorted_layers]

            layer_labels = [f"L{l}" for l in sorted_layers]

            ax.plot(layer_labels, raw_aucs,    "o-",  color="#333333", label="Raw",            lw=1.5)
            ax.plot(layer_labels, resid_aucs,  "s--", color="#d65f5f", label="Residualised",   lw=1.5)
            ax.plot(layer_labels, within_aucs, "^:",  color="#4878cf", label=f"Within-{ext}",  lw=1.5)

            # Dashed vertical at L11
            if 11 in sorted_layers:
                x11 = sorted_layers.index(11)
                ax.axvline(x11, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
                ax.text(x11 + 0.1, ax.get_ylim()[0] + 0.01, "L11", fontsize=7, color="black", alpha=0.6)

            ax.axhline(0.5, color="grey", linewidth=0.6, linestyle=":", alpha=0.4)
            ax.set_title(f"{fam}  (within-{ext})", fontweight="bold")
            ax.set_xlabel("Layer")
            ax.set_ylabel("ROC-AUC")
            ax.legend(fontsize=7, loc="upper left")
            ax.tick_params(axis="x", rotation=45)
            ax.set_ylim(0.45, 1.0)

        fig.suptitle(
            f"{run_name}: Raw vs Residualised vs Within-language AUC across layers",
            fontsize=9, y=1.02,
        )
        fig.tight_layout()
        safe_name = run_name.replace(" ", "_")
        out = PAPER_FIGS / f"fig_within_lang_by_layer_{safe_name}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Scatter: within-language AUC vs residualised AUC
# ─────────────────────────────────────────────────────────────────────────────

def fig_within_vs_resid(layers_data):
    """
    Scatter plot: x = residualised AUC, y = within-language AUC.
    Each point = one (layer, lang, family) combination.
    Near-diagonal → Ridge residualisation was sufficient.
    One panel per run (Raw / SAE).
    """
    run_points = {}

    for label, (delta, _s, _v, meta, _) in layers_data.items():
        run_name, layer_str = label.rsplit(" L", 1)
        layer = int(layer_str)
        resid = residualise_filetype(delta, meta)

        resid_probe = probe_cwe_families(resid, meta, MAIN_FAMILIES)

        if run_name not in run_points:
            run_points[run_name] = []

        for ext, fam in LANG_FAMILY_PAIRS:
            within = probe_within_language(delta, meta, ext, MAIN_FAMILIES)
            if within is None or fam not in within.index:
                continue
            if fam not in resid_probe.index:
                continue
            run_points[run_name].append({
                "layer":  layer,
                "ext":    ext,
                "family": fam,
                "resid":  resid_probe.loc[fam, "roc_auc"],
                "within": within.loc[fam, "roc_auc"],
            })

    n_runs = len(run_points)
    if n_runs == 0:
        print("fig_within_vs_resid: no data to plot")
        return

    fig, axes = plt.subplots(1, n_runs, figsize=(3.5 * n_runs, 3.2))
    if n_runs == 1:
        axes = [axes]

    for ax, (run_name, pts) in zip(axes, run_points.items()):
        if not pts:
            ax.set_title(run_name)
            continue
        df = pd.DataFrame(pts)
        for (ext, fam), grp in df.groupby(["ext", "family"]):
            color = FAMILY_COLORS.get(fam, "grey")
            ax.scatter(grp["resid"], grp["within"], label=f"{ext}/{fam}",
                       color=color, s=30, alpha=0.8, zorder=3)

        # Diagonal reference
        lo = min(df["resid"].min(), df["within"].min()) - 0.02
        hi = max(df["resid"].max(), df["within"].max()) + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y = x")

        ax.set_title(run_name, fontweight="bold")
        ax.set_xlabel("Residualised AUC")
        ax.set_ylabel("Within-language AUC")
        ax.legend(fontsize=6, loc="lower right", ncol=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")

    fig.suptitle(
        "Within-language AUC vs Residualised AUC\n"
        "(near diagonal = Ridge residualisation captured language effect)",
        fontsize=8, y=1.04,
    )
    fig.tight_layout()
    out = PAPER_FIGS / "fig_within_vs_resid.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Within-C pairwise AUC heatmaps across layers
# ─────────────────────────────────────────────────────────────────────────────

def fig_within_c_pairwise(layers_data):
    """
    For C samples only, pairwise family AUC at each layer.
    One heatmap per layer, grouped by run.
    Shows genuine CWE structure when language is held constant.
    """
    # Families present in meaningful quantity in C
    C_FAMILIES = ["Memory Safety", "Access Control", "Resource Mgmt", "Info Disclosure"]

    for run_name in {label.rsplit(" L", 1)[0] for label in layers_data}:
        run_layers = {
            int(lbl.rsplit(" L", 1)[1]): layers_data[lbl]
            for lbl in layers_data if lbl.startswith(run_name + " L")
        }
        sorted_layers = sorted(run_layers.keys())
        n = len(sorted_layers)
        if n == 0:
            continue

        ncols = (n + 1) // 2  # 4 for 8 layers
        nrows = 2
        fig, axes2d = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.8 * nrows))
        axes_flat = axes2d.flatten()
        # hide any spare cells
        for spare in axes_flat[n:]:
            spare.axis("off")

        vmin, vmax = 0.5, 1.0

        for idx, (ax, layer) in enumerate(zip(axes_flat, sorted_layers)):
            delta, _s, _v, meta, _ = run_layers[layer]
            mask = meta["file_extension"].values == "c"
            if mask.sum() < 30:
                ax.set_title(f"L{layer}\n(n<30)")
                ax.axis("off")
                continue

            d_c   = delta[mask]
            m_c   = meta[mask].reset_index(drop=True)
            valid = [f for f in C_FAMILIES if (m_c["family"] == f).sum() >= 5]
            if len(valid) < 2:
                ax.set_title(f"L{layer}\n(<2 fam)")
                ax.axis("off")
                continue

            pw = pairwise_family_probe(d_c, m_c, valid)
            short = [f.replace("Memory Safety", "Mem.S").replace("Access Control","Acc.C")
                      .replace("Resource Mgmt","Res.M").replace("Info Disclosure","Info.D")
                     for f in valid]
            pw.index   = short
            pw.columns = short

            sns.heatmap(
                pw, ax=ax,
                mask=np.isnan(pw.values),
                vmin=vmin, vmax=vmax,
                cmap="RdYlGn",
                annot=True, fmt=".2f", annot_kws={"size": 6},
                linewidths=0.4,
                cbar=False,
                square=True,
                yticklabels=(idx % ncols == 0),
            )
            # Grey diagonal
            for i in range(len(valid)):
                ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color="#cccccc", lw=0))

            l11_marker = " ←L11" if layer == 11 else ""
            ax.set_title(f"L{layer}{l11_marker}", fontweight="bold", fontsize=8)
            ax.tick_params(axis="x", rotation=30, labelsize=6)
            ax.tick_params(axis="y", rotation=0,  labelsize=6)

        # Shared colorbar
        cax = fig.add_axes([0.92, 0.1, 0.015, 0.8])
        sm  = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label("ROC-AUC", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        fig.suptitle(
            f"{run_name}: Pairwise CWE-family AUC within C files only\n"
            "(no language confound — genuine vulnerability-type structure)",
            fontsize=8, y=1.02,
        )
        fig.tight_layout(rect=[0, 0, 0.91, 1.0])
        safe_name = run_name.replace(" ", "_")
        out = PAPER_FIGS / f"fig_within_c_pairwise_{safe_name}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Vuln vs. secure probe — within language (critical missing experiment)
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_auc_ci(y_true, y_score, n_bootstrap=1000, ci=0.95, rng=None):
    """Return (mean_auc, lo, hi) via bootstrap resampling."""
    from sklearn.metrics import roc_auc_score
    if rng is None:
        rng = np.random.default_rng(SEED)
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    aucs = np.array(aucs)
    alpha = (1 - ci) / 2
    return aucs.mean(), np.quantile(aucs, alpha), np.quantile(aucs, 1 - alpha)


def probe_vuln_secure(safe_mat, vuln_mat, meta, ext=None, n_components=50, cv=5, n_bootstrap=500):
    """
    Binary probe: vulnerable (1) vs secure (0).
    Stacks the two sides of each paired sample and trains a logistic regression.
    Optionally restricted to file_extension == ext.
    Returns dict with roc_auc, ci_lo, ci_hi, n.
    """
    from sklearn.metrics import roc_auc_score

    if ext is not None:
        mask = (meta["file_extension"] == ext).values
    else:
        mask = np.ones(len(meta), dtype=bool)

    if mask.sum() < 30:
        return None

    S = safe_mat[mask].astype(np.float32)
    V = vuln_mat[mask].astype(np.float32)
    n = mask.sum()

    X = np.vstack([S, V])
    y = np.array([0] * n + [1] * n, dtype=int)

    scaler = StandardScaler()
    pca    = PCA(n_components=min(n_components, X.shape[1], X.shape[0] - 1), random_state=SEED)
    X_pca  = pca.fit_transform(scaler.fit_transform(X))

    clf = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=SEED)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)

    # Collect out-of-fold scores for bootstrap
    y_score = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in skf.split(X_pca, y):
        clf.fit(X_pca[train_idx], y[train_idx])
        y_score[test_idx] = clf.predict_proba(X_pca[test_idx])[:, 1]

    mean_auc, ci_lo, ci_hi = _bootstrap_auc_ci(y, y_score, n_bootstrap=n_bootstrap)
    return {"roc_auc": mean_auc, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": int(n)}


def fig_vuln_secure_by_layer(layers_data):
    """
    Key figure: vulnerable vs. secure binary probe AUROC across layers.
    Two lines per panel: global (all files) and within-C.
    One panel per run (Raw / SAE).
    95% bootstrap CI shown as shaded band.
    Dashed vertical at L11.
    """
    run_results = {}

    for label, (delta, safe_mat, vuln_mat, meta, _) in layers_data.items():
        run_name, layer_str = label.rsplit(" L", 1)
        layer = int(layer_str)

        if run_name not in run_results:
            run_results[run_name] = {"layers": [], "global": [], "c_only": [], "php_only": [], "js_only": []}

        global_res = probe_vuln_secure(safe_mat, vuln_mat, meta, ext=None)
        c_res      = probe_vuln_secure(safe_mat, vuln_mat, meta, ext="c")
        php_res    = probe_vuln_secure(safe_mat, vuln_mat, meta, ext="php")
        js_res     = probe_vuln_secure(safe_mat, vuln_mat, meta, ext="js")

        run_results[run_name]["layers"].append(layer)
        run_results[run_name]["global"].append(global_res)
        run_results[run_name]["c_only"].append(c_res)
        run_results[run_name]["php_only"].append(php_res)
        run_results[run_name]["js_only"].append(js_res)

    # Sort by layer
    for rn in run_results:
        order = np.argsort(run_results[rn]["layers"])
        for key in ("layers", "global", "c_only", "php_only", "js_only"):
            run_results[rn][key] = [run_results[rn][key][i] for i in order]

    n_runs = len(run_results)
    if n_runs == 0:
        print("fig_vuln_secure_by_layer: no data")
        return

    fig, axes = plt.subplots(1, n_runs, figsize=(3.8 * n_runs, 3.0), sharey=True)
    if n_runs == 1:
        axes = [axes]

    for ax, (run_name, rd) in zip(axes, run_results.items()):
        layers      = rd["layers"]
        layer_lbls  = [f"L{l}" for l in layers]

        def _plot_line(results, color, label, marker):
            aucs = [r["roc_auc"] if r else np.nan for r in results]
            los  = [r["ci_lo"]   if r else np.nan for r in results]
            his  = [r["ci_hi"]   if r else np.nan for r in results]
            xs   = list(range(len(layers)))
            ax.plot(xs, aucs, marker + "-", color=color, label=label, lw=1.5, ms=5)
            ax.fill_between(xs, los, his, color=color, alpha=0.15)

        _plot_line(rd["global"],   "#333333", "All files",      "o")
        _plot_line(rd["c_only"],   "#4878cf", "Within C",       "^")
        _plot_line(rd["php_only"], "#d62728", "Within PHP",     "s")
        _plot_line(rd["js_only"],  "#2ca02c", "Within JS",      "D")

        ax.axhline(0.5, color="grey", lw=0.7, ls=":", alpha=0.5, label="Chance")
        if 11 in layers:
            x11 = layers.index(11)
            ax.axvline(x11, color="black", lw=1.0, ls="--", alpha=0.4)
            ax.text(x11 + 0.1, 0.51, "L11", fontsize=7, color="black", alpha=0.6)

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layer_lbls, rotation=45)
        ax.set_xlabel("Layer")
        ax.set_ylabel("ROC-AUC  (vuln vs. secure)")
        ax.set_title(run_name, fontweight="bold")
        ax.set_ylim(0.40, 0.65)
        ax.legend(fontsize=7)

    fig.suptitle(
        "Vulnerable vs. Secure probe: global, within-C, within-PHP, within-JS\n"
        "(shaded band = 95% bootstrap CI)",
        fontsize=9,
    )
    fig.tight_layout()
    out = PAPER_FIGS / "fig_vuln_secure_by_layer.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Print numerical summary
    print("\nVuln vs. Secure AUROC (global / within-C / within-PHP / within-JS) with 95% CI:")
    for run_name, rd in run_results.items():
        print(f"\n{run_name}")
        print(f"  {'Layer':>5}  {'Global':>24}  {'C-only':>24}  {'PHP-only':>24}  {'JS-only':>24}")
        print("  " + "-" * 105)
        def _fmt(r):
            return f"{r['roc_auc']:.3f} [{r['ci_lo']:.3f}-{r['ci_hi']:.3f}]" if r else "       —      "
        for layer, g, c, p, j in zip(rd["layers"], rd["global"], rd["c_only"], rd["php_only"], rd["js_only"]):
            print(f"  {layer:>5}  {_fmt(g):>24}  {_fmt(c):>24}  {_fmt(p):>24}  {_fmt(j):>24}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(layers_data):
    rows = []
    for label, (delta, _s, _v, meta, _) in layers_data.items():
        run_name, layer_str = label.rsplit(" L", 1)
        layer = int(layer_str)
        resid = residualise_filetype(delta, meta)
        raw_p   = probe_cwe_families(delta, meta, MAIN_FAMILIES)
        resid_p = probe_cwe_families(resid, meta, MAIN_FAMILIES)
        for ext, fam in LANG_FAMILY_PAIRS:
            wp = probe_within_language(delta, meta, ext, MAIN_FAMILIES)
            rows.append({
                "Run":    run_name,
                "Layer":  layer,
                "Ext":    ext,
                "Family": fam,
                "Raw":    round(raw_p.loc[fam, "roc_auc"], 3) if fam in raw_p.index else float("nan"),
                "Resid":  round(resid_p.loc[fam, "roc_auc"], 3) if fam in resid_p.index else float("nan"),
                "Within": round(wp.loc[fam, "roc_auc"], 3) if (wp is not None and fam in wp.index) else float("nan"),
            })
    df = pd.DataFrame(rows)
    df["Δ(Within−Resid)"] = (df["Within"] - df["Resid"]).round(3)
    print("\n" + "="*80)
    print("Within-language baseline summary")
    print("="*80)
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Paper figures dir : {PAPER_FIGS}")
    print(f"Activations dir   : {ARTIFACTS}")
    print()

    layers_data = load_all_layers()

    print("\nGenerating figures...")
    fig_vuln_secure_by_layer(layers_data)
    fig_within_lang_by_layer(layers_data)
    fig_within_vs_resid(layers_data)
    fig_within_c_pairwise(layers_data)

    print_summary(layers_data)
    print("\nDone.")
