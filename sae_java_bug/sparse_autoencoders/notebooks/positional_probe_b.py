"""
positional_probe_b.py
=====================
Option B analysis: does discriminative signal survive excluding the first position bin?

Loads the positional_profiles_raw.jsonl checkpoint (no GPU needed).
For each sample we have binned SAE activations [20 bins × 10 features] for
both secure and vulnerable code.

We run three probes using only bins 1–19 (dropping bin 0):
  1. Per-feature scalar probe  — mean over bins 1-19 for each feature individually
  2. All-feature combined probe — mean over bins 1-19 for all 10 features (10-dim)
  3. Full-profile probe        — all 19 bins × 10 features (190-dim, L2-regularised)

We also compute, for each feature:
  - frac_pos: fraction of bins 1-19 where secure > vulnerable (diffuseness)
  - sign test p-value

Outputs
-------
  out_dir/
    fig_perfeature_auroc_bins1_19.pdf   — per-feature AUROC bar chart
    fig_f1185_profile.pdf               — F1185 profile across all bins, secure vs vuln
    positional_probe_b_results.json     — all numbers
"""

import json
import scipy.stats as stats
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ── Config ─────────────────────────────────────────────────────────────────────

CKPT = Path("sae_java_bug/artifacts/token_viz/figures/positional_profiles_raw.jsonl")
OUT  = Path("sae_java_bug/artifacts/token_viz/figures")
PAPER_FIGS = Path(
    "../../../On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations/figures"
)

TOP_FEATURES_META = [
    (80,    "Cond. branching & resource dispatch"),
    (4135,  "Function entry & parameter setup"),
    (1185,  "Security config & input sanitisation"),
    (6401,  "Short snippets, incomplete security"),
    (13920, "Control flow past validation"),
    (2860,  "Input parsing, inadequate validation"),
    (14439, "End-of-file & file termination"),
    (15720, "Config/protocol & HTTP headers"),
    (15765, "Raw memory ops & pointer manip."),
    (9193,  "Resource cleanup & error-handling"),
]
FEATURE_IDS = [f[0] for f in TOP_FEATURES_META]
N_FEATURES  = len(FEATURE_IDS)
N_BINS      = 20
BINS_1_19   = slice(1, 20)   # exclude bin 0
BIN_CENTERS = np.linspace(0, 1, N_BINS, endpoint=False) + 0.5 / N_BINS

SEED = 42

mpl.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "figure.dpi": 150, "pdf.fonttype": 42,
})

# ── Load checkpoint ────────────────────────────────────────────────────────────

def load_checkpoint(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} samples from {path.name}")
    return rows


# ── Probe utilities ────────────────────────────────────────────────────────────

def bootstrap_auc(y_true, y_score, n=1000, seed=SEED):
    rng  = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    aucs = np.array(aucs)
    return aucs.mean(), np.quantile(aucs, 0.025), np.quantile(aucs, 0.975)


def cv_probe(X, y, C=0.1, cv=5):
    """Cross-validated logistic regression, returns per-sample scores."""
    scaler = StandardScaler()
    clf    = LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=SEED)
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    scores = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf.fit(Xtr, y[tr])
        scores[te] = clf.predict_proba(Xte)[:, 1]
    return scores


# ── Sign test ─────────────────────────────────────────────────────────────────

def sign_test(diff_bins_1_19):
    """
    Two-sided sign test: are differences (secure - vuln) across bins 1-19
    consistently positive?
    H0: median difference = 0 (i.e. p_positive = 0.5)
    Returns (n_pos, n_total, p_value).
    """
    diffs = np.array(diff_bins_1_19)
    diffs = diffs[diffs != 0]     # exclude exact zeros (rare)
    n_pos = int((diffs > 0).sum())
    n     = len(diffs)
    # One-sided: P(X >= n_pos | n, 0.5) — testing if secure fires MORE
    p_one = stats.binomtest(n_pos, n, 0.5, alternative="greater").pvalue
    return n_pos, n, p_one


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    rows = load_checkpoint(CKPT)
    N    = len(rows)

    # Stack arrays: shape [N, 20, 10]
    v_all = np.array([r["v_binned"] for r in rows], dtype=np.float32)  # [N, 20, 10]
    s_all = np.array([r["s_binned"] for r in rows], dtype=np.float32)

    # Bins 1-19 only
    v_body = v_all[:, 1:, :]   # [N, 19, 10]
    s_body = s_all[:, 1:, :]

    # Mean over bins 1-19 per feature per sample → [N, 10]
    v_mean = v_body.mean(axis=1)
    s_mean = s_body.mean(axis=1)

    # Aggregate diff per feature across all samples and bins
    # diff[n, b, f] = secure - vulnerable for sample n, bin b, feature f
    diff_body = s_body - v_body   # [N, 19, 10]
    # Per-bin mean diff: [19, 10]
    diff_mean = diff_body.mean(axis=0)

    print("\n" + "=" * 72)
    print(f"{'Feat':>6}  {'frac_pos':>8}  {'n_pos/19':>9}  {'p (sign)':>10}  {'label'}")
    print("-" * 72)
    sign_results = []
    for fi, (fid, label) in enumerate(TOP_FEATURES_META):
        d = diff_mean[:, fi]           # [19] diff per bin (bins 1-19)
        n_pos, n, pval = sign_test(d)
        frac = n_pos / n
        sign_results.append({"feature_idx": fid, "frac_pos": frac,
                              "n_pos": n_pos, "n_bins": n, "p_sign": pval, "label": label})
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        print(f"  F{fid:>5}  {frac:>8.2f}  {n_pos:>4}/{n:<4}  {pval:>10.4f}  {label} {sig}")
    print("=" * 72)

    # ── Per-feature scalar probe (bins 1-19 mean) ─────────────────────────────
    print("\nPer-feature scalar probe (mean over bins 1-19):")
    print("-" * 72)
    per_feat_results = []
    for fi, (fid, label) in enumerate(TOP_FEATURES_META):
        # X: [2N, 1], y: [2N]
        X = np.concatenate([s_mean[:, fi:fi+1], v_mean[:, fi:fi+1]], axis=0)
        y = np.array([0]*N + [1]*N)
        scores = cv_probe(X, y)
        auc, lo, hi = bootstrap_auc(y, scores)
        per_feat_results.append({"feature_idx": fid, "auc": auc, "ci_lo": lo, "ci_hi": hi, "label": label})
        print(f"  F{fid:>5}  AUROC {auc:.3f} [{lo:.3f}–{hi:.3f}]  {label}")

    # ── Combined 10-feature probe (bins 1-19 mean) ────────────────────────────
    X_comb = np.concatenate([s_mean, v_mean], axis=0)   # [2N, 10]
    y_comb = np.array([0]*N + [1]*N)
    scores_comb = cv_probe(X_comb, y_comb)
    auc_comb, lo_comb, hi_comb = bootstrap_auc(y_comb, scores_comb)
    print(f"\nAll-10-feature combined (bins 1-19 mean):")
    print(f"  AUROC {auc_comb:.3f} [{lo_comb:.3f}–{hi_comb:.3f}]")

    # ── Full-profile probe (19 bins × 10 features = 190-dim) ─────────────────
    v_flat = v_body.reshape(N, -1)   # [N, 190]
    s_flat = s_body.reshape(N, -1)
    X_full = np.concatenate([s_flat, v_flat], axis=0)
    y_full = np.array([0]*N + [1]*N)
    scores_full = cv_probe(X_full, y_full, C=0.01)   # stronger regularisation
    auc_full, lo_full, hi_full = bootstrap_auc(y_full, scores_full)
    print(f"\nFull-profile probe (19 bins × 10 features):")
    print(f"  AUROC {auc_full:.3f} [{lo_full:.3f}–{hi_full:.3f}]")

    # ── F1185 with bin 0 vs bins 1-19 comparison ─────────────────────────────
    fi_1185 = FEATURE_IDS.index(1185)
    # Bin 0 only
    X_b0 = np.concatenate([s_all[:, 0:1, fi_1185], v_all[:, 0:1, fi_1185]], axis=0)
    y_b0 = np.array([0]*N + [1]*N)
    scores_b0 = cv_probe(X_b0, y_b0)
    auc_b0, lo_b0, hi_b0 = bootstrap_auc(y_b0, scores_b0)
    # Bins 1-19 only
    X_b1 = np.concatenate([s_mean[:, fi_1185:fi_1185+1], v_mean[:, fi_1185:fi_1185+1]], axis=0)
    scores_b1 = cv_probe(X_b1, y_b0)
    auc_b1, lo_b1, hi_b1 = bootstrap_auc(y_b0, scores_b1)
    print(f"\nF1185 bin-0-only probe:    AUROC {auc_b0:.3f} [{lo_b0:.3f}–{hi_b0:.3f}]")
    print(f"F1185 bins-1-19 probe:     AUROC {auc_b1:.3f} [{lo_b1:.3f}–{hi_b1:.3f}]")

    # ── Save results ──────────────────────────────────────────────────────────
    payload = {
        "n_samples": N,
        "sign_test": sign_results,
        "per_feature_probe_bins1_19": per_feat_results,
        "combined_probe_bins1_19": {"auc": auc_comb, "ci_lo": lo_comb, "ci_hi": hi_comb},
        "full_profile_probe_bins1_19": {"auc": auc_full, "ci_lo": lo_full, "ci_hi": hi_full},
        "f1185_bin0_probe": {"auc": auc_b0, "ci_lo": lo_b0, "ci_hi": hi_b0},
        "f1185_bins1_19_probe": {"auc": auc_b1, "ci_lo": lo_b1, "ci_hi": hi_b1},
    }
    out_json = OUT / "positional_probe_b_results.json"
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved: {out_json}")

    # ── Figure 1: per-feature AUROC bar chart (bins 1-19) ────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    xs     = np.arange(N_FEATURES)
    aucs   = [r["auc"] for r in per_feat_results]
    los    = [r["ci_lo"] for r in per_feat_results]
    his    = [r["ci_hi"] for r in per_feat_results]
    colors = ["#2166ac" if a > 0.5 else "#d6604d" for a in aucs]
    ax.bar(xs, aucs, color=colors, alpha=0.75,
           yerr=[np.array(aucs)-np.array(los), np.array(his)-np.array(aucs)],
           capsize=3, error_kw={"lw": 0.8})
    ax.axhline(0.5, color="#333333", lw=0.9, ls="--", label="Chance (0.5)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"F{fid}" for fid in FEATURE_IDS], rotation=45, ha="right")
    ax.set_ylabel("AUROC  (secure vs. vulnerable)")
    ax.set_ylim(0.44, 0.58)
    ax.set_title(
        f"Per-feature binary probe using ONLY bins 1–19 (first 5% of tokens excluded), n={N}\n"
        "Blue = above chance. Error bars = 95% bootstrap CI.",
        fontsize=8,
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    p1 = OUT / "fig_perfeature_auroc_bins1_19.pdf"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p1}")

    # ── Figure 2: F1185 full positional profile ───────────────────────────────
    fi = fi_1185
    # Per-sample mean across samples, per bin
    s_prof = s_all[:, :, fi].mean(axis=0)   # [20]
    v_prof = v_all[:, :, fi].mean(axis=0)   # [20]
    s_se   = s_all[:, :, fi].std(axis=0) / np.sqrt(N)
    v_se   = v_all[:, :, fi].std(axis=0) / np.sqrt(N)
    diff   = s_prof - v_prof

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 5.0), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # Top: secure vs vulnerable profiles
    ax1.plot(BIN_CENTERS, s_prof, color="#2166ac", lw=1.5, label="Secure")
    ax1.fill_between(BIN_CENTERS, s_prof - s_se, s_prof + s_se, color="#2166ac", alpha=0.18)
    ax1.plot(BIN_CENTERS, v_prof, color="#d6604d", lw=1.5, label="Vulnerable")
    ax1.fill_between(BIN_CENTERS, v_prof - v_se, v_prof + v_se, color="#d6604d", alpha=0.18)
    ax1.axvline(BIN_CENTERS[0] + 0.5/N_BINS, color="#888888", lw=0.7, ls=":",
                label="Bin 0 boundary")
    ax1.set_ylabel("Mean activation")
    ax1.set_title(
        f"Feature 1185 ('Security config & input sanitisation')\n"
        f"Positional activation profile (Layer 11, n={N}). Shaded = ±1 SE.",
        fontsize=8,
    )
    ax1.legend(fontsize=7)

    # Bottom: diff = secure - vulnerable, with horizontal zero line
    bar_colors = ["#2166ac" if d > 0 else "#d6604d" for d in diff]
    ax2.bar(BIN_CENTERS, diff, width=0.045, color=bar_colors, alpha=0.8)
    ax2.axhline(0, color="#333333", lw=0.8)
    ax2.axvline(BIN_CENTERS[0] + 0.5/N_BINS, color="#888888", lw=0.7, ls=":")
    ax2.set_ylabel("Δ (secure − vuln)")
    ax2.set_xlabel("Normalised token position")
    ax2.set_xticks(np.linspace(0, 1, 11))
    ax2.set_xticklabels([f"{x:.1f}" for x in np.linspace(0, 1, 11)])

    # Annotate sign test result
    sr = next(r for r in sign_results if r["feature_idx"] == 1185)
    ax2.text(0.98, 0.92,
             f"Bins 1–19: {sr['n_pos']}/{sr['n_bins']} positive\n"
             f"Sign test p = {sr['p_sign']:.4f}",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=7, bbox=dict(facecolor="white", edgecolor="#cccccc", pad=3))

    fig.tight_layout()
    p2 = OUT / "fig_f1185_positional_profile.pdf"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p2}")

    # ── Figure 3: diffuseness score vs AUROC scatter ──────────────────────────
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for fi2, ((fid, label), sr, pr) in enumerate(
        zip(TOP_FEATURES_META, sign_results, per_feat_results)
    ):
        color = "#2166ac" if pr["auc"] > 0.5 else "#d6604d"
        ax.scatter(sr["frac_pos"], pr["auc"], color=color, s=50, zorder=3)
        ax.annotate(f"F{fid}", (sr["frac_pos"], pr["auc"]),
                    textcoords="offset points", xytext=(4, 2), fontsize=6)
    ax.axhline(0.5, color="#888888", lw=0.8, ls="--")
    ax.axvline(0.5, color="#888888", lw=0.8, ls=":")
    ax.set_xlabel("Diffuseness: fraction of bins 1–19 with secure > vulnerable")
    ax.set_ylabel("AUROC (bins 1–19 scalar probe)")
    ax.set_title(
        "Diffuseness vs. discriminability (bins 1–19)\n"
        "Features in upper-right support diffuse negative-space encoding.",
        fontsize=8,
    )
    fig.tight_layout()
    p3 = OUT / "fig_diffuseness_vs_auroc.pdf"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p3}")

    print("\nDone.")


if __name__ == "__main__":
    main()
