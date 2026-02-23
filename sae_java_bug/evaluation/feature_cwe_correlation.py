"""
Feature-CWE Correlation Analysis

Computes correlations between SAE feature activation deltas (vulnerable - secure)
and CWE class membership indicators, identifying which features are most
discriminative for each vulnerability type.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch  # only used for .pt fallback in load_activations
from scipy.stats import t as t_dist


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _resolve_run_path(run_id: str) -> Path:
    """
    Resolve *run_id* to an absolute directory path.

    Accepted forms:
    - bare run name:  "run_20260128_184854"
    - relative path:  "artifacts/activations/run_20260128_184854"
    - absolute path:  "/some/path/run_20260128_184854"
    """
    p = Path(run_id)
    if p.is_absolute() and p.exists():
        return p

    # Try the path as-is relative to cwd first (handles notebook-relative paths)
    if p.exists():
        return p.resolve()

    # Strip any leading "artifacts/activations/" prefix the caller may have included
    parts = p.parts
    for i, part in enumerate(parts):
        if part == "activations":
            run_name = Path(*parts[i + 1:])
            break
    else:
        run_name = p

    # Canonical location: <repo>/sae_java_bug/artifacts/activations/<run_name>
    canonical = Path(__file__).parent.parent / "artifacts" / "activations" / run_name
    if canonical.exists():
        return canonical

    raise FileNotFoundError(
        f"Could not resolve run_id '{run_id}'. "
        f"Tried: {p.resolve()}, {canonical}"
    )


def load_activations_from_jsonl(
    run_id: str,
    layer: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load secure/vulnerable activations and CWE labels directly from the JSONL file.

    This avoids any torch↔NumPy ABI issues and guarantees that the arrays and
    labels are always aligned (they come from the same record).

    *run_id* may be a bare run name (e.g. ``"run_20260128_184854"``), a path
    relative to the repo root, or an absolute path.

    Returns
    -------
    secure : ndarray, shape (n_samples, n_features)
    vulnerable : ndarray, shape (n_samples, n_features)
    cwe_labels : list of str, length n_samples
    """
    base = _resolve_run_path(run_id)
    jsonl_files = list(base.glob(f"activations_layer_{layer}_*.jsonl"))
    if not jsonl_files:
        available = sorted(
            int(p.name.split("_")[2])
            for p in base.glob("activations_layer_*_*.jsonl")
        )
        raise FileNotFoundError(
            f"No JSONL file found for layer {layer} in {base}. "
            f"Available layers: {available}"
        )

    secure_rows: list[list[float]] = []
    vuln_rows: list[list[float]] = []
    cwe_labels: list[str] = []

    with open(jsonl_files[0]) as f:
        for line in f:
            try:
                record = json.loads(line)
                secure_rows.append(record["secure"])
                vuln_rows.append(record["vulnerable"])
                cwe_labels.append(record.get("cwe", "unknown"))
            except (json.JSONDecodeError, KeyError):
                pass  # skip corrupt or incomplete lines

    return (
        np.array(secure_rows, dtype=np.float32),
        np.array(vuln_rows, dtype=np.float32),
        cwe_labels,
    )


def load_activations(run_id: str, layer: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Load secure and vulnerable activation tensors as numpy arrays.

    Tries to load from the .pt tensor files first (fast); falls back to the
    JSONL if torch cannot convert tensors to numpy (NumPy ABI mismatch).
    """
    base = _resolve_run_path(run_id)
    secure_pt = base / f"safe_layer_{layer}.pt"
    vuln_pt = base / f"vulnerable_layer_{layer}.pt"

    if secure_pt.exists() and vuln_pt.exists():
        try:
            sec = torch.load(secure_pt, weights_only=True).numpy()
            vuln = torch.load(vuln_pt, weights_only=True).numpy()
            return sec, vuln
        except (RuntimeError, Exception):
            pass  # fall through to JSONL loading

    sec, vuln, _ = load_activations_from_jsonl(run_id, layer)
    return sec, vuln


def load_cwe_labels(run_id: str, layer: int = 0) -> list[str]:
    """Load CWE labels from the activation JSONL file, in tensor order."""
    _, _, labels = load_activations_from_jsonl(run_id, layer)
    return labels


# ---------------------------------------------------------------------------
# Core correlation computation
# ---------------------------------------------------------------------------

def _pearson_r_vectorized(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute Pearson r between each column of X (n × d) and a vector y (n,).

    Returns an array of shape (d,) with one correlation per feature.
    """
    n = len(y)
    X_c = X - X.mean(axis=0)
    y_c = y - y.mean()

    num = y_c @ X_c                                   # (d,)
    denom = np.sqrt((X_c ** 2).sum(axis=0)) * np.sqrt((y_c ** 2).sum())
    return num / (denom + 1e-12)


def _pearson_pvalues(r: np.ndarray, n: int) -> np.ndarray:
    """Two-sided p-values for Pearson r given sample size n (t-distribution)."""
    t_stat = r * np.sqrt(n - 2) / np.sqrt(np.maximum(1 - r ** 2, 1e-12))
    return 2 * t_dist.sf(np.abs(t_stat), df=n - 2)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FeatureCWECorrelations:
    """
    Container for feature × CWE correlation results.

    Attributes
    ----------
    correlation_matrix : ndarray, shape (n_cwes, n_features)
        Pearson r between feature activation delta (vuln − secure) and a
        binary CWE-membership indicator for every (CWE, feature) pair.
    pvalue_matrix : ndarray, shape (n_cwes, n_features)
        Two-sided p-values (t-distribution approximation).
    cwe_names : list[str]
        Ordered list of CWE identifiers (rows of the matrices).
    cwe_counts : dict[str, int]
        Number of samples per CWE.
    n_samples : int
        Total number of samples.
    n_features : int
        Number of SAE features.
    """
    correlation_matrix: np.ndarray
    pvalue_matrix: np.ndarray
    cwe_names: list[str]
    cwe_counts: dict[str, int]
    n_samples: int
    n_features: int

    # ------------------------------------------------------------------
    # Accessor helpers
    # ------------------------------------------------------------------

    def correlations_for_cwe(self, cwe_id: str) -> np.ndarray:
        """Return the correlation vector (n_features,) for a single CWE."""
        idx = self.cwe_names.index(cwe_id)
        return self.correlation_matrix[idx]

    def top_features(
        self,
        cwe_id: str,
        k: int = 20,
        direction: str = "positive",
    ) -> list[tuple[int, float, float]]:
        """
        Return the top-k features most correlated with *cwe_id*.

        Parameters
        ----------
        cwe_id : str
            CWE identifier, e.g. "CWE-79".
        k : int
            Number of features to return.
        direction : {"positive", "negative", "absolute"}
            Whether to rank by positive, negative, or absolute correlation.

        Returns
        -------
        list of (feature_idx, correlation, p_value) tuples, ordered by
        correlation magnitude.
        """
        idx = self.cwe_names.index(cwe_id)
        corr = self.correlation_matrix[idx]
        pval = self.pvalue_matrix[idx]

        if direction == "positive":
            order = np.argsort(-corr)
        elif direction == "negative":
            order = np.argsort(corr)
        else:  # absolute
            order = np.argsort(-np.abs(corr))

        top = order[:k]
        return [(int(f), float(corr[f]), float(pval[f])) for f in top]

    def top_features_all_cwes(
        self,
        k: int = 10,
        direction: str = "absolute",
    ) -> dict[str, list[tuple[int, float, float]]]:
        """Return top-k features for every CWE."""
        return {
            cwe: self.top_features(cwe, k=k, direction=direction)
            for cwe in self.cwe_names
        }

    def cwe_similarity_matrix(self) -> np.ndarray:
        """
        Compute pairwise cosine similarity between CWE correlation profiles.

        Returns an (n_cwes, n_cwes) matrix where entry [i, j] is how similar
        the feature-correlation fingerprints of CWE i and CWE j are.
        A high value means both CWEs activate and suppress the same features.
        """
        C = self.correlation_matrix  # (n_cwes, n_features)
        norms = np.linalg.norm(C, axis=1, keepdims=True)
        C_norm = C / (norms + 1e-12)
        return C_norm @ C_norm.T  # (n_cwes, n_cwes)

    def feature_enrichment(
        self,
        secure: np.ndarray,
        vulnerable: np.ndarray,
        cwe_labels: list[str],
        threshold: float = 1e-6,
    ) -> np.ndarray:
        """
        Compute feature enrichment (fold-change in firing rate) per CWE.

        For each (CWE, feature) pair, the enrichment score is:
            P(delta > threshold | CWE=c) / P(delta > threshold | CWE≠c)

        where delta = vulnerable - secure.  Values > 1 mean the feature fires
        *more* on vulnerable code specifically in that CWE class.  Log2 of this
        ratio gives an intuitive "bits" scale centred at 0.

        Returns
        -------
        log2_enrichment : ndarray, shape (n_cwes, n_features)
            log2(fold-change), clipped to [-6, 6].
        """
        delta = (vulnerable - secure).astype(np.float32)
        fired = (delta > threshold).astype(np.float32)  # (n, d)
        labels_arr = np.array(cwe_labels)

        n_cwes = len(self.cwe_names)
        enrichment = np.zeros((n_cwes, self.n_features), dtype=np.float32)

        for i, cwe in enumerate(self.cwe_names):
            mask = labels_arr == cwe
            p_in = fired[mask].mean(axis=0)          # (d,)  firing rate inside CWE
            p_out = fired[~mask].mean(axis=0)        # (d,)  firing rate outside CWE
            log2fc = np.log2((p_in + 1e-4) / (p_out + 1e-4))
            enrichment[i] = np.clip(log2fc, -6, 6)

        return enrichment

    # ------------------------------------------------------------------
    # Feature selectivity
    # ------------------------------------------------------------------

    def feature_selectivity(self, enrichment: np.ndarray) -> np.ndarray:
        """
        Compute selectivity score for every (CWE, feature) pair.

        For each feature f and CWE c:
            selectivity[c, f] = enrichment[c, f] − max_{j ≠ c}(enrichment[j, f])

        Interpretation
        --------------
        > 0  Feature is more enriched in CWE c than in any other CWE.
               The margin indicates how exclusively it signals this class.
        ≈ 0  General vulnerability signal — fires similarly across CWEs.
        < 0  Feature is more enriched in a different CWE.

        Parameters
        ----------
        enrichment : ndarray, shape (n_cwes, n_features)
            Output of :meth:`feature_enrichment`.

        Returns
        -------
        selectivity : ndarray, shape (n_cwes, n_features)
        """
        n_cwes = enrichment.shape[0]
        selectivity = np.empty_like(enrichment)
        for i in range(n_cwes):
            rest = np.delete(enrichment, i, axis=0)      # (n_cwes-1, d)
            selectivity[i] = enrichment[i] - rest.max(axis=0)
        return selectivity

    def top_selective_features(
        self,
        cwe_id: str,
        enrichment: np.ndarray,
        k: int = 20,
    ) -> list[tuple[int, float, float]]:
        """
        Return the top-k features most *exclusively* enriched in *cwe_id*.

        Returns
        -------
        list of (feature_idx, selectivity_score, enrichment_in_cwe) tuples,
        sorted by selectivity descending.
        """
        sel = self.feature_selectivity(enrichment)
        idx = self.cwe_names.index(cwe_id)
        order = np.argsort(-sel[idx])[:k]
        return [(int(f), float(sel[idx, f]), float(enrichment[idx, f])) for f in order]

    def print_selectivity_summary(
        self,
        enrichment: np.ndarray,
        k: int = 10,
        min_samples: int = 5,
        min_selectivity: float = 0.0,
    ) -> None:
        """
        Print top-k most selective features per CWE.

        Only features with selectivity > *min_selectivity* are shown, to
        focus on genuinely CWE-exclusive features.
        """
        filtered = [c for c in self.cwe_names if self.cwe_counts[c] >= min_samples]
        print(f"Feature Selectivity  (n_samples={self.n_samples}, n_features={self.n_features})")
        print("  selectivity = enrichment[CWE] − max enrichment in any other CWE")
        print("=" * 72)
        for cwe in filtered:
            top = self.top_selective_features(cwe, enrichment, k=k)
            top = [(f, s, e) for f, s, e in top if s > min_selectivity]
            if not top:
                continue
            print(f"\n{cwe}  (n={self.cwe_counts[cwe]})")
            print(f"  {'Feature':>8}  {'Selectivity':>12}  {'Enrichment (log2FC)':>20}")
            print(f"  {'-'*8}  {'-'*12}  {'-'*20}")
            for feat_idx, sel, enr in top:
                print(f"  {feat_idx:>8}  {sel:>12.3f}  {enr:>20.3f}")
        print("=" * 72)

    def print_summary(
        self,
        k: int = 10,
        min_samples: int = 5,
        alpha: float = 0.05,
    ) -> None:
        """Print a text summary of the top features per CWE."""
        filtered = [c for c in self.cwe_names if self.cwe_counts[c] >= min_samples]
        print(f"Feature–CWE Correlations  (n_samples={self.n_samples}, n_features={self.n_features})")
        print("=" * 70)
        for cwe in filtered:
            n = self.cwe_counts[cwe]
            top = self.top_features(cwe, k=k, direction="absolute")
            sig = [(f, r, p) for f, r, p in top if p < alpha]
            print(f"\n{cwe}  (n={n})")
            print(f"  {'Feature':>8}  {'Corr':>8}  {'p-value':>10}  {'sig':>4}")
            print(f"  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*4}")
            for feat_idx, r, p in top:
                marker = "*" * sum([p < 0.05, p < 0.01, p < 0.001])
                print(f"  {feat_idx:>8}  {r:>8.4f}  {p:>10.2e}  {marker:<4}")
        print("=" * 70)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def compute_feature_cwe_correlations(
    secure: np.ndarray,
    vulnerable: np.ndarray,
    cwe_labels: list[str],
    min_samples: int = 5,
) -> FeatureCWECorrelations:
    """
    Compute Pearson correlations between SAE feature activation deltas and
    CWE class indicators.

    Parameters
    ----------
    secure : ndarray, shape (n_samples, n_features)
        SAE activations for secure code.
    vulnerable : ndarray, shape (n_samples, n_features)
        SAE activations for vulnerable code.
    cwe_labels : list[str]
        CWE label for each sample (length = n_samples).
    min_samples : int
        CWEs with fewer samples are still included in the matrix but flagged.

    Returns
    -------
    FeatureCWECorrelations
    """
    assert secure.shape == vulnerable.shape, "secure and vulnerable must have the same shape"
    assert len(cwe_labels) == len(secure), "cwe_labels length must equal number of samples"

    delta = (vulnerable - secure).astype(np.float32)      # (n, d)
    n_samples, n_features = delta.shape

    cwe_counts = dict(Counter(cwe_labels))
    cwe_names = sorted(cwe_counts.keys())
    n_cwes = len(cwe_names)

    corr_matrix = np.zeros((n_cwes, n_features), dtype=np.float32)
    pval_matrix = np.ones((n_cwes, n_features), dtype=np.float32)

    labels_arr = np.array(cwe_labels)

    for i, cwe in enumerate(cwe_names):
        n_cwe = cwe_counts[cwe]
        if n_cwe < 2:
            continue                # can't compute correlation with a single sample
        y = (labels_arr == cwe).astype(np.float32)
        r = _pearson_r_vectorized(delta, y)
        corr_matrix[i] = r
        pval_matrix[i] = _pearson_pvalues(r, n_samples)

    return FeatureCWECorrelations(
        correlation_matrix=corr_matrix,
        pvalue_matrix=pval_matrix,
        cwe_names=cwe_names,
        cwe_counts=cwe_counts,
        n_samples=n_samples,
        n_features=n_features,
    )


def compute_from_run(
    run_id: str,
    layer: int = 0,
    min_samples: int = 5,
) -> FeatureCWECorrelations:
    """
    Convenience wrapper: load data and compute correlations for a given run.

    Parameters
    ----------
    run_id : str
        Directory name under artifacts/activations/, e.g.
        "run_20260128_184854".
    layer : int
        Transformer layer index.
    min_samples : int
        Minimum CWE samples to include.

    Returns
    -------
    FeatureCWECorrelations
    """
    print(f"Loading activations for {run_id} layer {layer}...")
    secure, vuln, cwe_labels = load_activations_from_jsonl(run_id, layer)
    print(f"  secure: {secure.shape}, vulnerable: {vuln.shape}")
    print(f"  {len(cwe_labels)} labels, {len(set(cwe_labels))} unique CWEs")

    print("Computing correlations...")
    result = compute_feature_cwe_correlations(secure, vuln, cwe_labels, min_samples)
    print(f"  Done. Correlation matrix: {result.correlation_matrix.shape}")
    return result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_top_features_heatmap(
    result: FeatureCWECorrelations,
    top_k: int = 30,
    min_samples: int = 5,
    alpha: float = 0.05,
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (16, 8),
) -> None:
    """
    Plot a heatmap of the top-k most correlated features (union across CWEs).

    Rows = CWE classes, columns = features (sorted by max absolute correlation).
    Significant cells (p < alpha) are marked with a dot.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import rcParams

    rcParams.update({"font.family": "serif", "font.size": 10})

    # Filter CWEs with enough samples
    cwes = [c for c in result.cwe_names if result.cwe_counts[c] >= min_samples]
    cwe_idx = [result.cwe_names.index(c) for c in cwes]

    corr_sub = result.correlation_matrix[cwe_idx]   # (n_filtered_cwes, n_features)
    pval_sub = result.pvalue_matrix[cwe_idx]

    # Pick top-k features by max absolute correlation across CWEs
    max_abs = np.abs(corr_sub).max(axis=0)           # (n_features,)
    top_feat_idx = np.argsort(-max_abs)[:top_k]
    top_feat_idx_sorted = top_feat_idx[np.argsort(top_feat_idx)]  # keep order

    heat = corr_sub[:, top_feat_idx_sorted]
    sig = pval_sub[:, top_feat_idx_sorted] < alpha

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    vmax = np.abs(heat).max()
    im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    # Mark significant cells
    for row in range(heat.shape[0]):
        for col in range(heat.shape[1]):
            if sig[row, col]:
                ax.plot(col, row, ".", color="black", markersize=3)

    # Axes
    ax.set_yticks(range(len(cwes)))
    ax.set_yticklabels([f"{c} (n={result.cwe_counts[c]})" for c in cwes], fontsize=9)
    ax.set_xticks(range(len(top_feat_idx_sorted)))
    ax.set_xticklabels(top_feat_idx_sorted, rotation=90, fontsize=7)
    ax.set_xlabel("Feature index")
    ax.set_ylabel("CWE class")
    ax.set_title(
        f"Feature–CWE Correlation (Pearson r, top {top_k} features)\n"
        f"Dots = significant at p < {alpha}"
    )

    plt.colorbar(im, ax=ax, label="Pearson r (delta = vuln − secure)", shrink=0.8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


def plot_top_features_per_cwe(
    result: FeatureCWECorrelations,
    cwe_id: str,
    k: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """Bar chart of the top-k features for a specific CWE."""
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({"font.family": "serif", "font.size": 11})

    top = result.top_features(cwe_id, k=k, direction="absolute")
    feature_ids = [str(f) for f, _, _ in top]
    corrs = [r for _, r, _ in top]
    pvals = [p for _, _, p in top]

    colors = ["#2ecc71" if r > 0 else "#e74c3c" for r in corrs]
    alpha_markers = ["*" * sum([p < 0.05, p < 0.01, p < 0.001]) for p in pvals]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    bars = ax.bar(range(len(corrs)), corrs, color=colors, edgecolor="white")

    for i, (bar, marker) in enumerate(zip(bars, alpha_markers)):
        if marker:
            y = bar.get_height() + (0.002 if bar.get_height() >= 0 else -0.005)
            ax.text(i, y, marker, ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(feature_ids)))
    ax.set_xticklabels(feature_ids, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Pearson r (vuln − secure vs CWE indicator)")
    ax.set_title(
        f"Top {k} features for {cwe_id} (n={result.cwe_counts[cwe_id]})\n"
        "Green = fires more on vuln, Red = fires more on secure"
    )
    ax.axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def plot_cwe_similarity(
    result: FeatureCWECorrelations,
    min_samples: int = 5,
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """
    Plot a heatmap of the pairwise cosine similarity between CWE
    feature-correlation profiles.

    Similar CWEs (same geometry in feature space) will cluster together.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({"font.family": "serif", "font.size": 10})

    # Filter to CWEs with enough samples
    keep = [c for c in result.cwe_names if result.cwe_counts[c] >= min_samples]
    idx = [result.cwe_names.index(c) for c in keep]

    sim = result.cwe_similarity_matrix()[np.ix_(idx, idx)]

    # Cluster rows/columns by hierarchical clustering
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
        from scipy.spatial.distance import squareform

        dist = np.clip(1 - sim, 0, 2)
        np.fill_diagonal(dist, 0)
        condensed = squareform(dist)
        Z = linkage(condensed, method="average")
        order = leaves_list(Z)
    except Exception:
        order = list(range(len(keep)))

    sim_ord = sim[np.ix_(order, order)]
    labels_ord = [f"{keep[i]} (n={result.cwe_counts[keep[i]]})" for i in order]

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    im = ax.imshow(sim_ord, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Cosine similarity of feature-correlation profiles", shrink=0.8)

    ax.set_xticks(range(len(labels_ord)))
    ax.set_yticks(range(len(labels_ord)))
    ax.set_xticklabels(labels_ord, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels_ord, fontsize=9)
    ax.set_title("CWE–CWE similarity (shared feature-delta patterns)")

    # Annotate cells
    for i in range(len(labels_ord)):
        for j in range(len(labels_ord)):
            ax.text(j, i, f"{sim_ord[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if abs(sim_ord[i, j]) < 0.7 else "white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def plot_enrichment_heatmap(
    result: FeatureCWECorrelations,
    secure: np.ndarray,
    vulnerable: np.ndarray,
    cwe_labels: list[str],
    top_k: int = 30,
    min_samples: int = 5,
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (16, 8),
) -> None:
    """
    Plot a log2 fold-change enrichment heatmap (firing rate inside CWE vs outside).

    Rows = CWE classes, columns = top-k most enriched features (union across CWEs).
    Positive (red) = feature fires more on vulnerable code in this CWE.
    Negative (blue) = feature is suppressed on vulnerable code in this CWE.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({"font.family": "serif", "font.size": 10})

    enrichment = result.feature_enrichment(secure, vulnerable, cwe_labels)

    cwes = [c for c in result.cwe_names if result.cwe_counts[c] >= min_samples]
    cwe_idx = [result.cwe_names.index(c) for c in cwes]
    enr_sub = enrichment[cwe_idx]                   # (n_filtered, n_features)

    # Pick top-k features by max absolute enrichment
    max_abs = np.abs(enr_sub).max(axis=0)
    top_feat_idx = np.argsort(-max_abs)[:top_k]
    top_feat_idx_sorted = np.sort(top_feat_idx)

    heat = enr_sub[:, top_feat_idx_sorted]

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    vmax = max(np.abs(heat).max(), 1.0)
    im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_yticks(range(len(cwes)))
    ax.set_yticklabels(
        [f"{c} (n={result.cwe_counts[c]})" for c in cwes], fontsize=9
    )
    ax.set_xticks(range(len(top_feat_idx_sorted)))
    ax.set_xticklabels(top_feat_idx_sorted, rotation=90, fontsize=7)
    ax.set_xlabel("Feature index")
    ax.set_ylabel("CWE class")
    ax.set_title(
        f"Feature enrichment (log2 firing-rate fold-change, top {top_k} features)\n"
        "Red = fires more on vulnerable in this CWE, Blue = suppressed"
    )
    plt.colorbar(im, ax=ax, label="log2(fold-change)", shrink=0.8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def plot_selectivity_heatmap(
    result: FeatureCWECorrelations,
    enrichment: np.ndarray,
    top_k: int = 30,
    min_samples: int = 5,
    min_selectivity: float = 0.1,
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (16, 8),
) -> None:
    """
    Heatmap of the top-k most *selectively* enriched features per CWE.

    Only features with selectivity > *min_selectivity* are eligible.
    Unlike the enrichment heatmap (which shows globally top features),
    each column here is chosen because it exclusively signals one CWE.

    Rows = CWE classes, columns = features ranked by max selectivity across CWEs.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({"font.family": "serif", "font.size": 10})

    sel = result.feature_selectivity(enrichment)              # (n_cwes, n_features)

    cwes = [c for c in result.cwe_names if result.cwe_counts[c] >= min_samples]
    cwe_idx = [result.cwe_names.index(c) for c in cwes]
    sel_sub = sel[cwe_idx]
    enr_sub = enrichment[cwe_idx]

    # For each CWE pick the top-k selective features; take the union
    candidate_sets = []
    per_cwe_k = max(1, top_k // len(cwes) + 2)
    for i in range(len(cwes)):
        row = sel_sub[i]
        top_i = np.argsort(-row)[:per_cwe_k]
        top_i = top_i[row[top_i] > min_selectivity]
        candidate_sets.extend(top_i.tolist())

    # Deduplicate and cap at top_k by max selectivity
    candidates = list(dict.fromkeys(candidate_sets))
    if len(candidates) > top_k:
        max_sel = sel_sub[:, candidates].max(axis=0)
        order = np.argsort(-max_sel)[:top_k]
        candidates = [candidates[i] for i in order]

    if not candidates:
        print("No features with selectivity > min_selectivity. Try lowering the threshold.")
        return

    top_feat_idx = np.array(sorted(candidates))
    heat = enr_sub[:, top_feat_idx]
    sel_heat = sel_sub[:, top_feat_idx]

    fig, axes = plt.subplots(
        1, 2, figsize=figsize, dpi=150,
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05},
    )

    # Left: enrichment (log2 FC) coloured
    vmax = max(np.abs(heat).max(), 1.0)
    im = axes[0].imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_yticks(range(len(cwes)))
    axes[0].set_yticklabels(
        [f"{c} (n={result.cwe_counts[c]})" for c in cwes], fontsize=9
    )
    axes[0].set_xticks(range(len(top_feat_idx)))
    axes[0].set_xticklabels(top_feat_idx, rotation=90, fontsize=7)
    axes[0].set_xlabel("Feature index  (selected by selectivity)")
    axes[0].set_title(
        f"Selective features: log2 fold-change  (top {len(top_feat_idx)} features)\n"
        "Each column is enriched in at most one CWE"
    )
    plt.colorbar(im, ax=axes[0], label="log2(fold-change)", shrink=0.8)

    # Right: selectivity margin
    im2 = axes[1].imshow(sel_heat, aspect="auto", cmap="Greens",
                         vmin=0, vmax=sel_heat.max())
    axes[1].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_xlabel("Margin")
    axes[1].set_title("Selectivity\n(margin)")
    plt.colorbar(im2, ax=axes[1], label="selectivity margin", shrink=0.8)

    plt.suptitle(
        "Feature Selectivity  —  features that uniquely signal one CWE type",
        y=1.01, fontsize=11,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Feature clustering
# ---------------------------------------------------------------------------

@dataclass
class FeatureClusterResult:
    """
    Result of clustering SAE features by their CWE enrichment profiles.

    Each cluster groups features that respond similarly across CWE types —
    i.e. they all fire more (or less) on the same vulnerability classes.

    Attributes
    ----------
    cluster_labels : ndarray, shape (n_features,)
        Cluster assignment for every feature.  -1 = filtered out (low signal).
    centroids : ndarray, shape (n_clusters, n_cwes)
        Mean enrichment profile for each cluster.
    cwe_names : list[str]
        CWE names (column names of *centroids*).
    dominant_cwes : dict[int, list[str]]
        For each cluster, the CWE(s) whose centroid enrichment is highest
        (above the mean + 0.5 std across CWEs).
    cluster_sizes : dict[int, int]
        Number of features per cluster.
    feature_to_cluster : dict[int, int]
        Feature index → cluster id (only for active features).
    n_clusters : int
    n_active_features : int
        Features with enough signal to be clustered.
    """
    cluster_labels: np.ndarray
    centroids: np.ndarray
    cwe_names: list[str]
    dominant_cwes: dict[int, list[str]]
    cluster_sizes: dict[int, int]
    feature_to_cluster: dict[int, int]
    n_clusters: int
    n_active_features: int

    def features_in_cluster(self, cluster_id: int) -> list[int]:
        """Return the feature indices assigned to *cluster_id*."""
        return [f for f, c in self.feature_to_cluster.items() if c == cluster_id]

    def cluster_activation_scores(self, delta: np.ndarray) -> np.ndarray:
        """
        Compute per-cluster activation scores for each sample.

        For each sample, the score for cluster k is the mean of
        ``delta[:, features_in_cluster(k)]``.  This gives an
        (n_samples, n_clusters) matrix usable as a low-dimensional
        vulnerability-type representation.

        Parameters
        ----------
        delta : ndarray, shape (n_samples, n_features)
            Feature activation delta (vulnerable − secure).
        """
        scores = np.zeros((delta.shape[0], self.n_clusters), dtype=np.float32)
        for k in range(self.n_clusters):
            feats = self.features_in_cluster(k)
            if feats:
                scores[:, k] = delta[:, feats].mean(axis=1)
        return scores

    def print_summary(self) -> None:
        """Print a summary of each cluster and its dominant CWEs."""
        print(f"Feature Clusters  ({self.n_clusters} clusters, "
              f"{self.n_active_features} active features)")
        print("=" * 65)
        # Sort by size descending
        for k in sorted(self.cluster_sizes, key=lambda x: -self.cluster_sizes[x]):
            size = self.cluster_sizes[k]
            cwes = self.dominant_cwes.get(k, ["(no dominant CWE)"])
            # Show top-3 centroid entries
            centroid = self.centroids[k]
            top3_idx = np.argsort(-centroid)[:3]
            top3 = [(self.cwe_names[i], f"{centroid[i]:.2f}") for i in top3_idx]
            print(f"\n  Cluster {k:>2}  ({size:>5} features)  dominant: {', '.join(cwes)}")
            print(f"           top centroid: " +
                  "  ".join(f"{c}={v}" for c, v in top3))
        print("=" * 65)


def compute_feature_clusters(
    result: FeatureCWECorrelations,
    enrichment: np.ndarray,
    n_clusters: int = 20,
    min_enrichment: float = 0.3,
    random_state: int = 42,
) -> FeatureClusterResult:
    """
    Cluster SAE features by their CWE enrichment profiles.

    Each feature is represented as an (n_cwes,)-dimensional vector of its
    enrichment scores.  K-means groups features that respond to the same
    set of CWEs.

    Parameters
    ----------
    result : FeatureCWECorrelations
        Correlation result (provides CWE names and metadata).
    enrichment : ndarray, shape (n_cwes, n_features)
        Output of ``result.feature_enrichment()``.
    n_clusters : int
        Number of k-means clusters.
    min_enrichment : float
        Features whose max absolute enrichment is below this threshold are
        labelled −1 (background / no signal) and excluded from clustering.
    random_state : int

    Returns
    -------
    FeatureClusterResult
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize, StandardScaler

    n_cwes, n_features = enrichment.shape

    # Each column of enrichment is a feature's CWE-profile vector
    profiles = enrichment.T                         # (n_features, n_cwes)

    # Filter to features with any meaningful enrichment signal
    max_abs = np.abs(profiles).max(axis=1)          # (n_features,)
    active_mask = max_abs >= min_enrichment
    active_idx = np.where(active_mask)[0]           # indices of active features
    n_active = len(active_idx)

    print(f"  Active features (|enrichment| ≥ {min_enrichment}): {n_active} / {n_features}")

    if n_active < n_clusters:
        raise ValueError(
            f"Only {n_active} active features, but n_clusters={n_clusters}. "
            f"Lower min_enrichment or n_clusters."
        )

    # Step 1: z-score each CWE dimension so no single CWE dominates by amplitude
    X = StandardScaler().fit_transform(profiles[active_idx])  # (n_active, n_cwes)
    # Step 2: L2-normalise so clustering focuses on profile *shape*, not magnitude
    X = normalize(X, norm="l2")

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels_active = km.fit_predict(X)               # (n_active,)

    # Build full label array (-1 for background features)
    cluster_labels = np.full(n_features, -1, dtype=np.int32)
    cluster_labels[active_idx] = labels_active

    # Compute centroid enrichment profiles (un-normalised, for interpretability)
    centroids = np.zeros((n_clusters, n_cwes), dtype=np.float32)
    for k in range(n_clusters):
        feat_idx = active_idx[labels_active == k]
        if len(feat_idx):
            centroids[k] = enrichment[:, feat_idx].mean(axis=1)

    # Identify dominant CWEs per cluster: centroid > mean + 0.5*std
    dominant_cwes: dict[int, list[str]] = {}
    for k in range(n_clusters):
        c = centroids[k]
        threshold = c.mean() + 0.5 * c.std()
        dominant = [result.cwe_names[i] for i in range(n_cwes) if c[i] > threshold]
        dominant_cwes[k] = dominant if dominant else [result.cwe_names[c.argmax()]]

    cluster_sizes = {
        k: int((cluster_labels == k).sum()) for k in range(n_clusters)
    }
    feature_to_cluster = {
        int(f): int(cluster_labels[f]) for f in active_idx
    }

    return FeatureClusterResult(
        cluster_labels=cluster_labels,
        centroids=centroids,
        cwe_names=result.cwe_names,
        dominant_cwes=dominant_cwes,
        cluster_sizes=cluster_sizes,
        feature_to_cluster=feature_to_cluster,
        n_clusters=n_clusters,
        n_active_features=n_active,
    )


def plot_cluster_centroids(
    clusters: FeatureClusterResult,
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (14, 8),
) -> None:
    """
    Heatmap of cluster centroids.

    Rows = clusters (sorted by dominant CWE), columns = CWE types.
    The colour shows the mean enrichment of the cluster's features in each CWE,
    making it easy to see which vulnerability type(s) each cluster signals.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from scipy.cluster.hierarchy import linkage, leaves_list

    rcParams.update({"font.family": "serif", "font.size": 10})

    C = clusters.centroids                          # (n_clusters, n_cwes)

    # Order rows by hierarchical clustering of centroids
    try:
        Z = linkage(C, method="ward")
        row_order = leaves_list(Z)
    except Exception:
        row_order = np.arange(clusters.n_clusters)

    C_ord = C[row_order]

    # Row labels: cluster id + dominant CWE(s) + size
    row_labels = []
    for k in row_order:
        cwes = ", ".join(clusters.dominant_cwes.get(k, ["?"]))
        row_labels.append(f"C{k} ({clusters.cluster_sizes[k]} feat.)  {cwes}")

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    vmax = max(np.abs(C_ord).max(), 0.5)
    im = ax.imshow(C_ord, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_yticks(range(clusters.n_clusters))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xticks(range(len(clusters.cwe_names)))
    ax.set_xticklabels(clusters.cwe_names, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("CWE class")
    ax.set_ylabel("Feature cluster")
    ax.set_title(
        f"Feature cluster centroids  ({clusters.n_clusters} clusters, "
        f"{clusters.n_active_features} active features)\n"
        "Red = cluster fires more on vulnerable code of that CWE type"
    )
    plt.colorbar(im, ax=ax, label="Mean log2 fold-change in cluster", shrink=0.8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def plot_cluster_cwe_scores(
    clusters: FeatureClusterResult,
    delta: np.ndarray,
    cwe_labels: list[str],
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (14, 6),
) -> None:
    """
    Box-plot of cluster activation scores grouped by CWE.

    For each cluster, shows the distribution of mean-delta scores across
    samples, split by CWE.  Clusters that are truly CWE-specific will show
    a high score only for one CWE row.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({"font.family": "serif", "font.size": 9})

    scores = clusters.cluster_activation_scores(delta)  # (n_samples, n_clusters)
    labels_arr = np.array(cwe_labels)
    unique_cwes = sorted(set(cwe_labels))

    # Only show the top-8 clusters by inter-CWE variance
    inter_var = np.array([
        np.var([scores[labels_arr == c, k].mean() for c in unique_cwes if (labels_arr == c).sum() > 5])
        for k in range(clusters.n_clusters)
    ])
    top_k_idx = np.argsort(-inter_var)[:8]

    fig, axes = plt.subplots(1, len(top_k_idx), figsize=figsize, dpi=150, sharey=False)
    if len(top_k_idx) == 1:
        axes = [axes]

    for ax, k in zip(axes, top_k_idx):
        cwes_with_data = [c for c in unique_cwes if (labels_arr == c).sum() >= 5]
        data = [scores[labels_arr == c, k] for c in cwes_with_data]
        ax.boxplot(data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor="#d0e8f7", alpha=0.8),
                   medianprops=dict(color="navy"))
        ax.set_xticks(range(1, len(cwes_with_data) + 1))
        ax.set_xticklabels(cwes_with_data, rotation=90, fontsize=7)
        dom = ", ".join(clusters.dominant_cwes.get(k, ["?"]))
        ax.set_title(f"C{k}\n{dom}", fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    fig.suptitle(
        "Cluster activation scores by CWE  (top 8 clusters by inter-CWE variance)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Cluster-based CWE prediction
# ---------------------------------------------------------------------------

@dataclass
class ClusterProbeResult:
    """
    Results of training a linear probe to predict CWE type from cluster scores.

    Attributes
    ----------
    accuracy : float
        Mean cross-validated accuracy.
    macro_f1 : float
        Mean cross-validated macro-averaged F1.
    per_cwe_f1 : dict[str, float]
        Per-CWE F1 averaged over folds.
    confusion_matrix : ndarray, shape (n_cwes, n_cwes)
        Accumulated (summed) confusion matrix over all folds.
    cwe_names : list[str]
        CWE labels (rows/columns of confusion_matrix).
    coef : ndarray, shape (n_cwes, n_clusters)
        Logistic regression coefficients.  coef[i, k] > 0 means cluster k
        pushes the model towards predicting CWE i.  NaN if only 2 classes.
    n_folds : int
    n_samples : int
    n_clusters : int
    chance_level : float
        Expected accuracy under a majority-class baseline.
    """
    accuracy: float
    macro_f1: float
    per_cwe_f1: dict[str, float]
    confusion_matrix: np.ndarray
    cwe_names: list[str]
    coef: np.ndarray
    n_folds: int
    n_samples: int
    n_clusters: int
    chance_level: float

    def print_summary(self) -> None:
        """Print a concise text report."""
        print("=" * 60)
        print("Cluster-based CWE Linear Probe")
        print(f"  Samples:   {self.n_samples}")
        print(f"  Clusters:  {self.n_clusters}")
        print(f"  CWE types: {len(self.cwe_names)}")
        print(f"  CV folds:  {self.n_folds}")
        print("-" * 60)
        print(f"  Accuracy:  {self.accuracy:.3f}   (chance ≈ {self.chance_level:.3f})")
        print(f"  Macro F1:  {self.macro_f1:.3f}")
        print()
        print(f"  {'CWE':<15}  {'F1':>6}")
        print(f"  {'-'*15}  {'-'*6}")
        for cwe in sorted(self.per_cwe_f1, key=lambda c: -self.per_cwe_f1[c]):
            print(f"  {cwe:<15}  {self.per_cwe_f1[cwe]:>6.3f}")
        print("=" * 60)


def predict_cwe_from_clusters(
    clusters: FeatureClusterResult,
    delta: np.ndarray,
    cwe_labels: list[str],
    n_folds: int = 5,
    min_samples: int = 10,
    C: float = 1.0,
    random_state: int = 42,
) -> ClusterProbeResult:
    """
    Train a linear probe to predict CWE type from cluster activation scores.

    Cluster activation scores (one per cluster per sample) are used as a
    low-dimensional vulnerability-type representation.  A logistic regression
    classifier is evaluated with stratified k-fold cross-validation.

    Parameters
    ----------
    clusters : FeatureClusterResult
        Output of :func:`compute_feature_clusters`.
    delta : ndarray, shape (n_samples, n_features)
        Feature activation delta (vulnerable − secure).
    cwe_labels : list[str]
        CWE label for each sample.
    n_folds : int
        Number of cross-validation folds.
    min_samples : int
        CWEs with fewer samples are excluded from classification.
    C : float
        Logistic regression regularisation strength (inverse of lambda).
    random_state : int

    Returns
    -------
    ClusterProbeResult
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    # ------------------------------------------------------------------
    # 1. Build feature matrix X and filter to CWEs with enough samples
    # ------------------------------------------------------------------
    scores = clusters.cluster_activation_scores(delta)   # (n_samples, n_clusters)
    labels_arr = np.array(cwe_labels)

    # Keep only CWEs with enough samples for stratified CV
    cwe_counts = Counter(cwe_labels)
    keep_cwes = sorted(c for c, n in cwe_counts.items() if n >= min_samples)

    if len(keep_cwes) < 2:
        raise ValueError(
            f"Need at least 2 CWEs with ≥ {min_samples} samples. "
            f"Found: {keep_cwes}"
        )

    mask = np.isin(labels_arr, keep_cwes)
    X = scores[mask]                                     # (n_kept, n_clusters)
    y_raw = labels_arr[mask]

    # Encode labels as integers
    le = LabelEncoder().fit(keep_cwes)
    y = le.transform(y_raw)

    n_samples, n_clusters_used = X.shape
    n_classes = len(keep_cwes)

    # Chance level = largest class proportion
    _, counts = np.unique(y, return_counts=True)
    chance_level = counts.max() / n_samples

    # ------------------------------------------------------------------
    # 2. Stratified k-fold cross-validation
    # ------------------------------------------------------------------
    # Ensure n_folds ≤ min class size
    min_class_size = counts.min()
    actual_folds = min(n_folds, min_class_size)
    if actual_folds < n_folds:
        print(f"  Warning: reduced CV folds to {actual_folds} (smallest class has {min_class_size} samples)")

    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=random_state)

    fold_accuracies: list[float] = []
    fold_macro_f1s: list[float] = []
    per_cwe_f1_accum: dict[str, list[float]] = {c: [] for c in keep_cwes}
    cm_total = np.zeros((n_classes, n_classes), dtype=np.int64)
    coef_total = np.zeros((n_classes, n_clusters_used), dtype=np.float64)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features (important for logistic regression convergence)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(
            C=C,
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        fold_accuracies.append(accuracy_score(y_test, y_pred))
        fold_macro_f1s.append(f1_score(y_test, y_pred, average="macro", zero_division=0))

        # Per-class F1 for this fold
        fold_f1s = f1_score(y_test, y_pred, average=None, labels=range(n_classes), zero_division=0)
        for i, cwe in enumerate(keep_cwes):
            per_cwe_f1_accum[cwe].append(float(fold_f1s[i]))

        cm_total += confusion_matrix(y_test, y_pred, labels=range(n_classes))
        coef_total += clf.coef_  # (n_classes, n_clusters)

    # ------------------------------------------------------------------
    # 3. Aggregate results
    # ------------------------------------------------------------------
    per_cwe_f1 = {cwe: float(np.mean(vals)) for cwe, vals in per_cwe_f1_accum.items()}
    coef_mean = coef_total / actual_folds

    return ClusterProbeResult(
        accuracy=float(np.mean(fold_accuracies)),
        macro_f1=float(np.mean(fold_macro_f1s)),
        per_cwe_f1=per_cwe_f1,
        confusion_matrix=cm_total,
        cwe_names=keep_cwes,
        coef=coef_mean,
        n_folds=actual_folds,
        n_samples=n_samples,
        n_clusters=n_clusters_used,
        chance_level=float(chance_level),
    )


def plot_probe_confusion_matrix(
    probe: ClusterProbeResult,
    normalise: bool = True,
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """
    Plot the cross-validation confusion matrix.

    Parameters
    ----------
    normalise : bool
        If True, show row-normalised (recall per class) values.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({"font.family": "serif", "font.size": 10})

    cm = probe.confusion_matrix.astype(float)
    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.where(row_sums > 0, cm / row_sums, 0.0)
        cbar_label = "Recall (row-normalised)"
        fmt = ".2f"
    else:
        cbar_label = "Count"
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalise else None, aspect="auto")
    plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)

    n = len(probe.cwe_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(probe.cwe_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(probe.cwe_names, fontsize=9)
    ax.set_xlabel("Predicted CWE")
    ax.set_ylabel("True CWE")
    ax.set_title(
        f"Cluster-probe confusion matrix  "
        f"(acc={probe.accuracy:.3f}, macro-F1={probe.macro_f1:.3f}, "
        f"chance={probe.chance_level:.3f})"
    )

    # Annotate cells
    thresh = cm.max() / 2.0
    raw_cm = probe.confusion_matrix
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            raw = raw_cm[i, j]
            text = f"{val:{fmt}}" if not normalise else f"{val:.2f}\n({raw})"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=6, color="white" if val > thresh else "black")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def plot_probe_cluster_importance(
    probe: ClusterProbeResult,
    clusters: FeatureClusterResult,
    top_k: int = 10,
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (14, 7),
) -> None:
    """
    Heatmap of logistic regression coefficients: which clusters drive each CWE prediction.

    Rows = CWE classes, columns = clusters.  A large positive coefficient means
    that cluster's activation score strongly predicts the CWE.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({"font.family": "serif", "font.size": 9})

    coef = probe.coef                                    # (n_cwes, n_clusters)

    # Column labels: cluster id + dominant CWE(s)
    col_labels = []
    for k in range(probe.n_clusters):
        dom = clusters.dominant_cwes.get(k, ["?"])
        col_labels.append(f"C{k}\n{', '.join(dom[:2])}")

    # Optionally restrict to top-k clusters by max |coef| across CWEs
    if top_k < probe.n_clusters:
        max_abs_col = np.abs(coef).max(axis=0)
        top_col_idx = np.argsort(-max_abs_col)[:top_k]
    else:
        top_col_idx = np.arange(probe.n_clusters)

    coef_sub = coef[:, top_col_idx]
    col_labels_sub = [col_labels[i] for i in top_col_idx]

    vmax = max(np.abs(coef_sub).max(), 0.1)
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    im = ax.imshow(coef_sub, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_yticks(range(len(probe.cwe_names)))
    ax.set_yticklabels(probe.cwe_names, fontsize=9)
    ax.set_xticks(range(len(col_labels_sub)))
    ax.set_xticklabels(col_labels_sub, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Feature cluster (label = dominant CWE)")
    ax.set_ylabel("Predicted CWE class")
    ax.set_title(
        f"Linear probe coefficients: cluster → CWE  (top {len(col_labels_sub)} clusters)\n"
        "Red = cluster activation pushes prediction towards this CWE"
    )
    plt.colorbar(im, ax=ax, label="LR coefficient", shrink=0.8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Binary per-CWE detection probes
# ---------------------------------------------------------------------------

@dataclass
class BinaryProbeResult:
    """
    Results of binary per-CWE detection probes.

    For each CWE a logistic regression is trained to distinguish
    samples with that CWE from all others, using the full feature
    delta as input.

    Attributes
    ----------
    auroc : dict[str, float]
        Cross-validated AUC-ROC per CWE.
    avg_precision : dict[str, float]
        Cross-validated average precision (area under PR curve) per CWE.
    coef : dict[str, ndarray]
        Mean logistic regression weight vector per CWE (shape n_features).
    cwe_names : list[str]
        CWEs that were evaluated.
    n_folds : int
    prevalence : dict[str, float]
        Fraction of samples positive for each CWE (= chance AUC baseline).
    """
    auroc: dict[str, float]
    avg_precision: dict[str, float]
    coef: dict[str, np.ndarray]
    cwe_names: list[str]
    n_folds: int
    prevalence: dict[str, float]

    def print_summary(self, min_auroc: float = 0.0) -> None:
        """Print per-CWE AUC sorted descending."""
        print("=" * 55)
        print("Binary per-CWE Detection Probes")
        print(f"  {'CWE':<15}  {'AUROC':>6}  {'AvgPrec':>8}  {'Prev':>6}")
        print(f"  {'-'*15}  {'-'*6}  {'-'*8}  {'-'*6}")
        for cwe in sorted(self.cwe_names, key=lambda c: -self.auroc[c]):
            if self.auroc[cwe] < min_auroc:
                continue
            print(
                f"  {cwe:<15}  {self.auroc[cwe]:>6.3f}  "
                f"{self.avg_precision[cwe]:>8.3f}  "
                f"{self.prevalence[cwe]:>6.3f}"
            )
        print("=" * 55)


def predict_cwe_binary_probes(
    delta: np.ndarray,
    cwe_labels: list[str],
    n_folds: int = 5,
    min_samples: int = 10,
    C: float = 0.1,
    random_state: int = 42,
    use_selective_features: bool = False,
    result: Optional["FeatureCWECorrelations"] = None,
    enrichment: Optional[np.ndarray] = None,
    top_k_features: int = 50,
) -> "BinaryProbeResult":
    """
    Train one binary logistic regression probe per CWE.

    For each CWE c, the probe answers: "does this sample contain CWE-c?"
    The positive class is ``cwe_labels == c``, negatives are all other CWEs.
    Performance is measured by AUC-ROC (threshold-free, imbalance-robust).

    This is the most appropriate framing for vulnerability detection:
    each CWE is treated as a separate binary signal rather than competing
    with 24 other classes in a 25-way classification.

    Parameters
    ----------
    delta : ndarray, shape (n_samples, n_features)
        Feature activation delta (vulnerable − secure).
    cwe_labels : list[str]
        CWE label per sample.
    n_folds : int
        Stratified CV folds.
    min_samples : int
        CWEs with fewer samples are skipped.
    C : float
        LR regularisation (small C = strong L2 regularisation; recommended
        for high-dimensional delta vectors to avoid overfitting).
    use_selective_features : bool
        If True, restrict each probe to the top-k most selective features for
        that CWE (requires *result* and *enrichment*).  Much faster and often
        more accurate for high-dim inputs.
    result : FeatureCWECorrelations, optional
        Required when ``use_selective_features=True``.
    enrichment : ndarray, optional
        Required when ``use_selective_features=True``.
    top_k_features : int
        Number of features to use per CWE when ``use_selective_features=True``.

    Returns
    -------
    BinaryProbeResult
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import StandardScaler

    labels_arr = np.array(cwe_labels)
    cwe_counts = Counter(cwe_labels)
    eval_cwes = sorted(c for c, n in cwe_counts.items() if n >= min_samples)
    n_samples = len(labels_arr)

    auroc: dict[str, float] = {}
    avg_prec: dict[str, float] = {}
    coef: dict[str, np.ndarray] = {}
    prevalence: dict[str, float] = {}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for cwe in eval_cwes:
        y = (labels_arr == cwe).astype(np.int32)
        prev = y.mean()
        prevalence[cwe] = float(prev)

        # Select features for this CWE
        if use_selective_features and result is not None and enrichment is not None:
            if cwe in result.cwe_names:
                sel_list = result.top_selective_features(cwe, enrichment, k=top_k_features)
                feat_idx = np.array([f for f, _, _ in sel_list])
            else:
                feat_idx = np.arange(delta.shape[1])
        else:
            feat_idx = np.arange(delta.shape[1])

        X = delta[:, feat_idx]

        fold_auroc: list[float] = []
        fold_ap: list[float] = []
        fold_coef: list[np.ndarray] = []

        # Ensure both classes present in each fold
        actual_folds = min(n_folds, int(y.sum()), int((1 - y).sum()))
        if actual_folds < 2:
            continue

        cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=random_state)
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            clf = LogisticRegression(C=C, max_iter=500, solver="lbfgs", random_state=random_state)
            clf.fit(X_train, y_train)
            proba = clf.predict_proba(X_test)[:, 1]

            if y_test.sum() == 0 or y_test.sum() == len(y_test):
                continue
            fold_auroc.append(roc_auc_score(y_test, proba))
            fold_ap.append(average_precision_score(y_test, proba))
            fold_coef.append(clf.coef_[0])

        if not fold_auroc:
            continue

        auroc[cwe] = float(np.mean(fold_auroc))
        avg_prec[cwe] = float(np.mean(fold_ap))
        coef[cwe] = np.mean(fold_coef, axis=0)

    return BinaryProbeResult(
        auroc=auroc,
        avg_precision=avg_prec,
        coef=coef,
        cwe_names=sorted(auroc.keys()),
        n_folds=n_folds,
        prevalence=prevalence,
    )


def plot_binary_probe_auroc(
    probe: "BinaryProbeResult",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (8, 7),
) -> None:
    """
    Horizontal bar chart of per-CWE AUC-ROC.

    Bars are coloured by whether AUC is above 0.6 (light signal) or 0.7 (strong).
    A dashed vertical line at 0.5 marks the random baseline.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({"font.family": "serif", "font.size": 10})

    cwes = sorted(probe.cwe_names, key=lambda c: probe.auroc[c])
    aucs = [probe.auroc[c] for c in cwes]

    colors = []
    for a in aucs:
        if a >= 0.70:
            colors.append("#2ecc71")    # strong
        elif a >= 0.60:
            colors.append("#f39c12")    # moderate
        else:
            colors.append("#95a5a6")    # weak / near chance

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    bars = ax.barh(cwes, aucs, color=colors, edgecolor="white", linewidth=0.5)

    # Annotate with sample count
    for bar, cwe in zip(bars, cwes):
        n = int(probe.prevalence[cwe] * sum(
            1 for c in probe.cwe_names  # use total via prevalence inverse
        ))
        ax.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"prev={probe.prevalence[cwe]:.2f}",
            va="center", fontsize=8,
        )

    ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="Random (0.5)")
    ax.axvline(0.6, color="orange", linestyle=":", linewidth=1, label="Moderate (0.6)")
    ax.axvline(0.7, color="green", linestyle=":", linewidth=1, label="Strong (0.7)")
    ax.set_xlim(0.4, min(1.05, max(aucs) + 0.12))
    ax.set_xlabel("AUC-ROC (cross-validated)")
    ax.set_ylabel("CWE")
    ax.set_title("Binary per-CWE detection probe\n(delta = vulnerable − secure activations)")
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Two-stage vulnerability detection + CWE typing
# ---------------------------------------------------------------------------

@dataclass
class ActivationDensityResult:
    """
    Empirical test of the hypothesis:
    'Vulnerable code activates more features than secure code.'

    If this holds, raw activation magnitude on a single sample (no paired
    comparison needed) is a valid unsupervised vulnerability detector.

    Attributes
    ----------
    secure_density : ndarray, shape (n_samples,)
        Fraction of features active (> 0) per secure sample.
    vuln_density : ndarray, shape (n_samples,)
        Fraction of features active per vulnerable sample.
    secure_l1 : ndarray
        Mean absolute activation per secure sample.
    vuln_l1 : ndarray
        Mean absolute activation per vulnerable sample.
    density_wilcoxon_pvalue : float
        Paired Wilcoxon p-value: H0 = vuln_density == secure_density.
    l1_wilcoxon_pvalue : float
        Paired Wilcoxon p-value on L1 activation.
    density_mean_diff : float
        mean(vuln_density - secure_density).
    l1_mean_diff : float
        mean(vuln_l1 - secure_l1).
    p_vuln_higher_density : float
        Fraction of pairs where vulnerable has strictly more active features.
    p_vuln_higher_l1 : float
        Fraction of pairs where vulnerable has strictly higher L1 norm.
    """
    secure_density: np.ndarray
    vuln_density: np.ndarray
    secure_l1: np.ndarray
    vuln_l1: np.ndarray
    density_wilcoxon_pvalue: float
    l1_wilcoxon_pvalue: float
    density_mean_diff: float
    l1_mean_diff: float
    p_vuln_higher_density: float
    p_vuln_higher_l1: float

    def print_summary(self) -> None:
        print("=" * 62)
        print("Activation Density Hypothesis Test")
        print("  H1: vulnerable code activates more SAE features than secure code")
        print("-" * 62)
        print(f"  {'Metric':<30}  {'Secure':>8}  {'Vuln':>8}  {'Diff':>8}")
        print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}")
        print(
            f"  {'Active feature fraction':<30}  "
            f"{self.secure_density.mean():>8.4f}  "
            f"{self.vuln_density.mean():>8.4f}  "
            f"{self.density_mean_diff:>+8.4f}"
        )
        print(
            f"  {'Mean |activation| (L1)':<30}  "
            f"{self.secure_l1.mean():>8.4f}  "
            f"{self.vuln_l1.mean():>8.4f}  "
            f"{self.l1_mean_diff:>+8.4f}"
        )
        print("-" * 62)
        print(f"  P(vuln density > secure density): {self.p_vuln_higher_density:.3f}")
        print(f"  P(vuln L1     > secure L1):       {self.p_vuln_higher_l1:.3f}")
        print(f"  Wilcoxon p-value (density): {self.density_wilcoxon_pvalue:.2e}")
        print(f"  Wilcoxon p-value (L1):      {self.l1_wilcoxon_pvalue:.2e}")
        if self.p_vuln_higher_density > 0.6 and self.density_wilcoxon_pvalue < 0.05:
            verdict = "SUPPORTED — vulnerable code activates more features"
        elif self.p_vuln_higher_density < 0.4:
            verdict = "REJECTED — secure code activates more features"
        else:
            verdict = "INCONCLUSIVE — no clear density difference"
        print(f"\n  Hypothesis: {verdict}")
        print("=" * 62)


def analyze_activation_density(
    secure: np.ndarray,
    vulnerable: np.ndarray,
    threshold: float = 0.0,
) -> ActivationDensityResult:
    """
    Empirically test whether vulnerable code activates more SAE features.

    Parameters
    ----------
    secure : ndarray, shape (n_samples, n_features)
    vulnerable : ndarray, shape (n_samples, n_features)
    threshold : float
        A feature is considered 'active' if its value exceeds this threshold.
        For ReLU-activated SAE features, 0 is the natural threshold.

    Returns
    -------
    ActivationDensityResult
    """
    from scipy.stats import wilcoxon

    secure_density = (secure > threshold).mean(axis=1).astype(np.float64)
    vuln_density   = (vulnerable > threshold).mean(axis=1).astype(np.float64)
    secure_l1 = np.abs(secure).mean(axis=1).astype(np.float64)
    vuln_l1   = np.abs(vulnerable).mean(axis=1).astype(np.float64)

    density_diff = vuln_density - secure_density
    l1_diff      = vuln_l1 - secure_l1

    try:
        _, density_pval = wilcoxon(density_diff)
    except ValueError:
        density_pval = 1.0
    try:
        _, l1_pval = wilcoxon(l1_diff)
    except ValueError:
        l1_pval = 1.0

    return ActivationDensityResult(
        secure_density=secure_density,
        vuln_density=vuln_density,
        secure_l1=secure_l1,
        vuln_l1=vuln_l1,
        density_wilcoxon_pvalue=float(density_pval),
        l1_wilcoxon_pvalue=float(l1_pval),
        density_mean_diff=float(density_diff.mean()),
        l1_mean_diff=float(l1_diff.mean()),
        p_vuln_higher_density=float((density_diff > 0).mean()),
        p_vuln_higher_l1=float((l1_diff > 0).mean()),
    )


def plot_activation_density(
    density_result: ActivationDensityResult,
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (12, 4),
) -> None:
    """
    Two-panel plot: distribution of active-feature fraction and L1 norm,
    secure vs vulnerable, to visualise the density hypothesis.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({"font.family": "serif", "font.size": 10})

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=150)

    for ax, (sec, vuln, label, pval) in zip(axes, [
        (density_result.secure_density, density_result.vuln_density,
         "Active feature fraction\n(features > 0)",
         density_result.density_wilcoxon_pvalue),
        (density_result.secure_l1, density_result.vuln_l1,
         "Mean |activation| (L1 norm)",
         density_result.l1_wilcoxon_pvalue),
    ]):
        bins = np.linspace(
            min(sec.min(), vuln.min()),
            max(sec.max(), vuln.max()),
            50,
        )
        ax.hist(sec,  bins=bins, alpha=0.6, color="#3498db", label="Secure",     density=True)
        ax.hist(vuln, bins=bins, alpha=0.6, color="#e74c3c", label="Vulnerable", density=True)
        ax.axvline(sec.mean(),  color="#2980b9", linestyle="--", linewidth=1.5)
        ax.axvline(vuln.mean(), color="#c0392b", linestyle="--", linewidth=1.5)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "n.s."))
        ax.set_title(f"Wilcoxon p={pval:.2e}  {sig}")
        ax.legend()

    fig.suptitle(
        "Do vulnerable code samples activate more SAE features?",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def vulnerability_anomaly_scores(delta: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute unsupervised anomaly scores from the activation delta.

    These scores measure the *excess* activation of vulnerable code over the
    secure baseline, implementing the hypothesis that vulnerable code activates
    more features than secure code.  No labels are required.

    Returns a dict with three score variants per sample (shape (n_samples,)):

    ``pos_mean``
        Mean of positively-spiking deltas:
        ``mean(max(delta, 0))``.
        Captures features that fire *more* in vulnerable code (positive excess).
    ``pos_l1``
        L1 norm of positive delta:
        ``sum(max(delta, 0))``.
        Total positive activation shift — vulnerable fires these extra features.
    ``abs_mean``
        Mean absolute delta:
        ``mean(|delta|)``.
        Captures any deviation (more or fewer activations) from the secure baseline.
    """
    pos = np.maximum(delta, 0.0)          # (n, d) — positive spikes only
    return {
        "pos_mean": pos.mean(axis=1),
        "pos_l1":   pos.sum(axis=1),
        "abs_mean": np.abs(delta).mean(axis=1),
    }


@dataclass
class TwoStageResult:
    """
    Results of the two-stage vulnerability detection + CWE typing pipeline.

    Stage 1: binary detection — vulnerable vs secure (using anomaly scores).
    Stage 2: CWE typing — given vulnerable, which CWE?

    Attributes
    ----------
    stage1_auroc : dict[str, float]
        AUC-ROC for each anomaly score variant on the detection task.
    stage2_auroc : dict[str, float]
        Per-CWE AUC-ROC for the typing task (among vulnerable samples only).
    stage2_avg_precision : dict[str, float]
        Per-CWE average precision for the typing task.
    detection_score_name : str
        Which anomaly score variant performed best in Stage 1.
    n_samples : int
    n_positive : int
        Number of vulnerable samples used for Stage 2.
    """
    stage1_auroc: dict[str, float]
    stage2_auroc: dict[str, float]
    stage2_avg_precision: dict[str, float]
    detection_score_name: str
    n_samples: int
    n_positive: int

    def print_summary(self) -> None:
        print("=" * 60)
        print("Two-Stage Detection + CWE Typing")
        print()
        print("Stage 1 — Vulnerability Detection (vuln vs secure baseline)")
        print(f"  {'Score variant':<15}  {'AUROC':>6}")
        print(f"  {'-'*15}  {'-'*6}")
        for name, auc in sorted(self.stage1_auroc.items(), key=lambda x: -x[1]):
            mark = "  <-- best" if name == self.detection_score_name else ""
            print(f"  {name:<15}  {auc:>6.3f}{mark}")
        print()
        print(f"Stage 2 — CWE Typing  ({self.n_positive} vulnerable samples)")
        print(f"  {'CWE':<15}  {'AUROC':>6}  {'AvgPrec':>8}")
        print(f"  {'-'*15}  {'-'*6}  {'-'*8}")
        for cwe in sorted(self.stage2_auroc, key=lambda c: -self.stage2_auroc[c]):
            print(
                f"  {cwe:<15}  {self.stage2_auroc[cwe]:>6.3f}  "
                f"{self.stage2_avg_precision[cwe]:>8.3f}"
            )
        print("=" * 60)


def run_two_stage_analysis(
    secure: np.ndarray,
    vulnerable: np.ndarray,
    cwe_labels: list[str],
    n_folds: int = 5,
    min_samples: int = 10,
    C: float = 0.1,
    random_state: int = 42,
    result: Optional["FeatureCWECorrelations"] = None,
    enrichment: Optional[np.ndarray] = None,
    top_k_features: int = 50,
) -> "TwoStageResult":
    """
    Run the two-stage vulnerability detection + CWE typing pipeline.

    Stage 1 — Vulnerability detection (trained probe on delta)
    -----------------------------------------------------------
    Raw activation magnitude was empirically found to be non-discriminative
    (secure ≈ vulnerable in density/L1).  The vulnerability signal lives in
    the *pattern* of the delta, not its magnitude.

    Stage 1 therefore uses a trained binary logistic regression probe on the
    full activation delta (vulnerable − secure).  This requires the paired
    comparison at inference time but is the only approach that actually works.

    The probe is trained with synthetic negatives: for every real delta
    (vulnerable − secure), we add a permuted-pairing delta as a "null"
    (randomly shuffled secure sample subtracted from same vulnerable sample).
    This tests whether the *specific* pairing carries information beyond
    the general activation level difference.

    Stage 2 — CWE typing
    ---------------------
    Uses the real deltas and runs binary logistic regression probes, one per
    CWE.  Top selective features are used as input if *result* and
    *enrichment* are provided.

    Parameters
    ----------
    secure : ndarray, shape (n_samples, n_features)
    vulnerable : ndarray, shape (n_samples, n_features)
    cwe_labels : list[str]
    n_folds, min_samples, C, random_state : CV / LR hyperparameters
    result, enrichment, top_k_features : optional, for selective features in Stage 2

    Returns
    -------
    TwoStageResult
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    delta = (vulnerable - secure).astype(np.float32)
    n_samples = len(cwe_labels)
    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Stage 1: trained binary probe on delta pattern (real vs permuted)
    # ------------------------------------------------------------------
    # Real delta: vulnerable[i] − secure[i]   → label 1
    # Null delta: vulnerable[i] − secure[j]   → label 0  (shuffled pairing)
    # The probe tests whether the *specific co-occurrence* of (vuln, sec)
    # carries information — i.e. whether there is a consistent vulnerability
    # signature in the delta beyond just the individual activation levels.
    perm = rng.permutation(n_samples)
    delta_null = (vulnerable - secure[perm]).astype(np.float32)

    X_stage1 = np.vstack([delta, delta_null])
    y_detection = np.array([1] * n_samples + [0] * n_samples, dtype=np.int32)

    # Use a small random subset of features for speed (still captures signal)
    feat_sample = rng.choice(delta.shape[1], size=min(512, delta.shape[1]), replace=False)
    X_stage1_sub = X_stage1[:, feat_sample]

    stage1_fold_aucs: list[float] = []
    skf1 = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for tr, te in skf1.split(X_stage1_sub, y_detection):
        sc1 = StandardScaler().fit(X_stage1_sub[tr])
        clf1 = LogisticRegression(C=C, max_iter=500, solver="lbfgs", random_state=random_state)
        clf1.fit(sc1.transform(X_stage1_sub[tr]), y_detection[tr])
        proba1 = clf1.predict_proba(sc1.transform(X_stage1_sub[te]))[:, 1]
        stage1_fold_aucs.append(float(roc_auc_score(y_detection[te], proba1)))

    stage1_auroc = {"delta_probe": float(np.mean(stage1_fold_aucs))}
    best_score_name = "delta_probe"

    # ------------------------------------------------------------------
    # Stage 2: CWE typing using vulnerable-side delta
    # ------------------------------------------------------------------
    stage2_probe = predict_cwe_binary_probes(
        delta=delta,
        cwe_labels=cwe_labels,
        n_folds=n_folds,
        min_samples=min_samples,
        C=C,
        random_state=random_state,
        use_selective_features=(result is not None and enrichment is not None),
        result=result,
        enrichment=enrichment,
        top_k_features=top_k_features,
    )

    return TwoStageResult(
        stage1_auroc=stage1_auroc,
        stage2_auroc=stage2_probe.auroc,
        stage2_avg_precision=stage2_probe.avg_precision,
        detection_score_name=best_score_name,
        n_samples=n_samples,
        n_positive=n_samples,
    )


def plot_two_stage_results(
    ts: "TwoStageResult",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (14, 6),
) -> None:
    """
    Side-by-side bar charts for Stage 1 (detection) and Stage 2 (CWE typing).
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({"font.family": "serif", "font.size": 10})

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=150,
                             gridspec_kw={"width_ratios": [1, 2], "wspace": 0.35})

    # --- Left: Stage 1 bar chart ---
    ax = axes[0]
    names = list(ts.stage1_auroc.keys())
    aucs = [ts.stage1_auroc[n] for n in names]
    colors1 = ["#2ecc71" if n == ts.detection_score_name else "#95a5a6" for n in names]
    ax.barh(names, aucs, color=colors1, edgecolor="white")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1)
    ax.set_xlim(0.4, 1.02)
    ax.set_xlabel("AUC-ROC")
    ax.set_title("Stage 1\nVulnerability Detection\n(vuln delta vs zero baseline)")

    # --- Right: Stage 2 CWE typing ---
    ax2 = axes[1]
    cwes = sorted(ts.stage2_auroc, key=lambda c: ts.stage2_auroc[c])
    aucs2 = [ts.stage2_auroc[c] for c in cwes]
    colors2 = []
    for a in aucs2:
        if a >= 0.70:
            colors2.append("#2ecc71")
        elif a >= 0.60:
            colors2.append("#f39c12")
        else:
            colors2.append("#95a5a6")

    ax2.barh(cwes, aucs2, color=colors2, edgecolor="white", linewidth=0.5)
    ax2.axvline(0.5, color="red", linestyle="--", linewidth=1, label="Random")
    ax2.axvline(0.6, color="orange", linestyle=":", linewidth=1, label="Moderate")
    ax2.axvline(0.7, color="green", linestyle=":", linewidth=1, label="Strong")
    ax2.set_xlim(0.4, min(1.05, max(aucs2) + 0.12))
    ax2.set_xlabel("AUC-ROC")
    ax2.set_title("Stage 2\nCWE Typing\n(given vulnerability present)")
    ax2.legend(loc="lower right", fontsize=8)

    fig.suptitle("Two-Stage Pipeline: Detect → Type", fontsize=11, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def save_correlations(result: FeatureCWECorrelations, output_dir: str | Path) -> None:
    """
    Save the correlation matrix and metadata to *output_dir*.

    Files written:
        feature_cwe_corr_matrix.npy   – (n_cwes, n_features) float32 array
        feature_cwe_pval_matrix.npy   – (n_cwes, n_features) float32 array
        feature_cwe_meta.json         – CWE names, counts, dimensions
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "feature_cwe_corr_matrix.npy", result.correlation_matrix)
    np.save(out / "feature_cwe_pval_matrix.npy", result.pvalue_matrix)

    meta = {
        "cwe_names": result.cwe_names,
        "cwe_counts": result.cwe_counts,
        "n_samples": result.n_samples,
        "n_features": result.n_features,
    }
    with open(out / "feature_cwe_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved correlation matrices to {out}")


def load_correlations(output_dir: str | Path) -> FeatureCWECorrelations:
    """Load previously saved correlation results."""
    out = Path(output_dir)

    corr = np.load(out / "feature_cwe_corr_matrix.npy")
    pval = np.load(out / "feature_cwe_pval_matrix.npy")
    with open(out / "feature_cwe_meta.json") as f:
        meta = json.load(f)

    return FeatureCWECorrelations(
        correlation_matrix=corr,
        pvalue_matrix=pval,
        cwe_names=meta["cwe_names"],
        cwe_counts=meta["cwe_counts"],
        n_samples=meta["n_samples"],
        n_features=meta["n_features"],
    )


# ---------------------------------------------------------------------------
# CI/CD vulnerability scanner
# ---------------------------------------------------------------------------

@dataclass
class VulnerabilityReport:
    """
    Output of a single CI/CD scan.

    Attributes
    ----------
    cwe_risks : dict[str, float]
        Per-CWE risk score in [0, 1].  These are classifier probabilities,
        not calibrated — use them for ranking, not as literal probabilities.
    flagged : list[str]
        CWEs whose risk score exceeds the scanner's threshold.
    top_cwe : str
        CWE with the highest risk score.
    top_risk : float
        Risk score of the top CWE.
    threshold : float
        Decision threshold used.
    """
    cwe_risks: dict[str, float]
    flagged: list[str]
    top_cwe: str
    top_risk: float
    threshold: float

    def print(self) -> None:
        print("=" * 50)
        print("Vulnerability Scan Report")
        print(f"  Threshold: {self.threshold:.2f}")
        print("-" * 50)
        if self.flagged:
            print(f"  FLAGGED: {', '.join(self.flagged)}")
        else:
            print("  CLEAN — no CWEs above threshold")
        print()
        print(f"  {'CWE':<15}  {'Risk':>6}  {'Flag':>5}")
        print(f"  {'-'*15}  {'-'*6}  {'-'*5}")
        for cwe, score in sorted(self.cwe_risks.items(), key=lambda x: -x[1]):
            flag = "  <--" if cwe in self.flagged else ""
            print(f"  {cwe:<15}  {score:>6.3f}{flag}")
        print("=" * 50)

    def to_dict(self) -> dict:
        """Serialisable summary for embedding in CI/CD pipeline output."""
        return {
            "flagged": self.flagged,
            "top_cwe": self.top_cwe,
            "top_risk": round(self.top_risk, 4),
            "threshold": self.threshold,
            "risks": {k: round(v, 4) for k, v in self.cwe_risks.items()},
        }


class CIVulnerabilityScanner:
    """
    Lightweight vulnerability scanner for use in CI/CD pipelines.

    Wraps one binary logistic regression probe per CWE.  At inference time,
    a pull-request author's code change is compared against a secure baseline
    (e.g. the previous commit) to produce a delta of SAE activations.  The
    scanner scores the delta against each CWE probe and returns a
    :class:`VulnerabilityReport`.

    Typical workflow
    ----------------
    **Offline (once, on labelled data):**

    .. code-block:: python

        scanner = CIVulnerabilityScanner.train(
            secure, vulnerable, cwe_labels,
            result=result, enrichment=enrichment,
        )
        scanner.save("scanner.pkl")

    **CI/CD (per PR):**

    .. code-block:: python

        scanner = CIVulnerabilityScanner.load("scanner.pkl")
        report  = scanner.scan(baseline_activations, new_code_activations)
        if report.flagged:
            sys.exit(1)   # fail the CI check
    """

    def __init__(
        self,
        probes: dict,           # cwe → {"clf": LR, "scaler": SS, "features": ndarray}
        threshold: float = 0.5,
        n_features: int = 16384,
    ):
        self._probes = probes
        self.threshold = threshold
        self.n_features = n_features
        self.cwe_names = sorted(probes.keys())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        secure: np.ndarray,
        vulnerable: np.ndarray,
        cwe_labels: list[str],
        result: Optional["FeatureCWECorrelations"] = None,
        enrichment: Optional[np.ndarray] = None,
        top_k_features: int = 100,
        min_samples: int = 10,
        C: float = 0.1,
        threshold: float = 0.5,
        random_state: int = 42,
    ) -> "CIVulnerabilityScanner":
        """
        Train one binary probe per CWE and return a ready-to-use scanner.

        Parameters
        ----------
        secure, vulnerable : ndarray, shape (n_samples, n_features)
        cwe_labels : list[str]
        result, enrichment : optional — used to select per-CWE selective features.
            When provided, each probe uses only its most selective features
            (faster inference, often better accuracy).
        top_k_features : int
            Features per CWE probe when using selective features.
        min_samples : int
            CWEs with fewer samples are skipped.
        C : float
            LR regularisation strength.
        threshold : float
            Risk threshold above which a CWE is flagged.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        delta = (vulnerable - secure).astype(np.float32)
        labels_arr = np.array(cwe_labels)
        cwe_counts = Counter(cwe_labels)
        eval_cwes = sorted(c for c, n in cwe_counts.items() if n >= min_samples)

        probes: dict = {}
        for cwe in eval_cwes:
            y = (labels_arr == cwe).astype(np.int32)
            if y.sum() < 2 or (1 - y).sum() < 2:
                continue

            # Select features
            if result is not None and enrichment is not None and cwe in result.cwe_names:
                sel = result.top_selective_features(cwe, enrichment, k=top_k_features)
                feat_idx = np.array([f for f, _, _ in sel])
            else:
                feat_idx = np.arange(delta.shape[1])

            X = delta[:, feat_idx]
            scaler = StandardScaler().fit(X)
            X_scaled = scaler.transform(X)

            clf = LogisticRegression(C=C, max_iter=500, solver="lbfgs", random_state=random_state)
            clf.fit(X_scaled, y)

            probes[cwe] = {"clf": clf, "scaler": scaler, "features": feat_idx}
            print(f"  Trained probe for {cwe}  (n_pos={y.sum()}, n_feat={len(feat_idx)})")

        print(f"\nScanner trained with {len(probes)} CWE probes.")
        return cls(probes=probes, threshold=threshold, n_features=delta.shape[1])

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def scan(
        self,
        baseline_activations: np.ndarray,
        new_code_activations: np.ndarray,
    ) -> VulnerabilityReport:
        """
        Scan a single code change for potential vulnerabilities.

        Parameters
        ----------
        baseline_activations : ndarray, shape (n_features,) or (1, n_features)
            SAE activations for the secure / previous version of the code.
        new_code_activations : ndarray, shape (n_features,) or (1, n_features)
            SAE activations for the new / changed version of the code.

        Returns
        -------
        VulnerabilityReport
        """
        sec = np.atleast_2d(baseline_activations).astype(np.float32)
        new = np.atleast_2d(new_code_activations).astype(np.float32)
        delta = (new - sec)                             # (1, n_features)

        risks: dict[str, float] = {}
        for cwe, probe in self._probes.items():
            X = delta[:, probe["features"]]
            X_scaled = probe["scaler"].transform(X)
            prob = float(probe["clf"].predict_proba(X_scaled)[0, 1])
            risks[cwe] = prob

        flagged = sorted(c for c, s in risks.items() if s >= self.threshold)
        top_cwe = max(risks, key=lambda c: risks[c]) if risks else "none"

        return VulnerabilityReport(
            cwe_risks=risks,
            flagged=flagged,
            top_cwe=top_cwe,
            top_risk=risks.get(top_cwe, 0.0),
            threshold=self.threshold,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle the scanner to *path* for use in CI/CD."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Scanner saved to {path}  ({len(self._probes)} CWE probes)")

    @classmethod
    def load(cls, path: str | Path) -> "CIVulnerabilityScanner":
        """Load a previously saved scanner."""
        import pickle
        with open(path, "rb") as f:
            scanner = pickle.load(f)
        print(f"Scanner loaded from {path}  ({len(scanner._probes)} CWE probes)")
        return scanner

    # ------------------------------------------------------------------
    # Integrated scan: raw code strings → report (uses ActivationExtractor)
    # ------------------------------------------------------------------

    def scan_code(
        self,
        baseline_code: str,
        new_code: str,
        extractor: "ActivationExtractor",
    ) -> "VulnerabilityReport":
        """
        Scan a code change end-to-end: raw strings → SAE activations → report.

        This is the primary entry point for CI/CD integration when the model
        and SAE are loaded in the same process.

        Parameters
        ----------
        baseline_code : str
            The previous (secure) version of the function, e.g. from git HEAD.
        new_code : str
            The new version from the pull request.
        extractor : ActivationExtractor
            A loaded :class:`~sae_java_bug.evaluation.activation_extractor.ActivationExtractor`
            instance.  Must be initialised with the same SAE config that was
            used to train the scanner probes.

        Returns
        -------
        VulnerabilityReport

        Example
        -------
        .. code-block:: python

            from sae_java_bug.evaluation.activation_extractor import ActivationExtractor
            from sae_java_bug.sparse_autoencoders.schemas import (
                QWEN_CODER_7B_VULNEABLE_CODE_STD_10M_CONFIG,
            )

            extractor = ActivationExtractor.from_config(
                QWEN_CODER_7B_VULNEABLE_CODE_STD_10M_CONFIG
            ).load()

            scanner = CIVulnerabilityScanner.load("scanner.pkl")
            report  = scanner.scan_code(old_function, new_function, extractor)
            report.print()
        """
        baseline_acts, new_acts = extractor.get_delta(baseline_code, new_code)
        return self.scan(baseline_acts, new_acts)


# ------------------------------------------------------------------
# End-to-end demo with mock activations
# ------------------------------------------------------------------

def demo_cicd_scanner(n_features: int = 256, n_samples: int = 200) -> None:
    """
    End-to-end CI/CD demo using synthetic SAE activations.

    This shows exactly how the scanner would be wired into a pipeline:

        1. Offline: train on historical labelled data and save.
        2. CI/CD:   load + scan each PR diff.

    The mock data injects a small but consistent signal for CWE-78 and
    CWE-79 so those probes should fire above the threshold.
    """
    rng = np.random.default_rng(0)

    # ---- 1. Synthetic training data ----------------------------------
    # Each sample: a (secure, vulnerable) pair with a CWE label.
    # Secure activations: sparse random, values in [0, 1].
    # Vulnerable activations: same base + a CWE-specific additive spike.
    cwe_pool = ["CWE-78", "CWE-79", "CWE-89", "CWE-476", "CWE-400"]
    cwe_signal_features = {          # which features carry the CWE signal
        "CWE-78":  [0, 1, 2, 3],
        "CWE-79":  [4, 5, 6, 7],
        "CWE-89":  [8, 9, 10, 11],
        "CWE-476": [12, 13, 14, 15],
        "CWE-400": [16, 17, 18, 19],
    }
    signal_strength = 1.5

    cwe_labels_train = rng.choice(cwe_pool, size=n_samples).tolist()
    secure_train  = rng.random((n_samples, n_features)).astype(np.float32) * 0.5
    vuln_train    = secure_train.copy()
    for i, cwe in enumerate(cwe_labels_train):
        for feat in cwe_signal_features[cwe]:
            vuln_train[i, feat] += signal_strength + rng.normal(0, 0.2)

    # ---- 2. Build lightweight FeatureCWECorrelations for selective features
    result_mock = compute_feature_cwe_correlations(secure_train, vuln_train, cwe_labels_train)
    enrichment_mock = result_mock.feature_enrichment(secure_train, vuln_train, cwe_labels_train)

    # ---- 3. Train and save scanner -----------------------------------
    print("Training scanner on mock data...")
    scanner = CIVulnerabilityScanner.train(
        secure_train, vuln_train, cwe_labels_train,
        result=result_mock,
        enrichment=enrichment_mock,
        top_k_features=20,
        min_samples=10,
        threshold=0.5,
    )
    scanner.save("/tmp/mock_vulnerability_scanner.pkl")

    # ---- 4. CI/CD simulation: scan three mock PRs --------------------
    print("\n" + "=" * 55)
    print("CI/CD Simulation — scanning 3 mock pull requests")
    print("=" * 55)

    scanner_loaded = CIVulnerabilityScanner.load("/tmp/mock_vulnerability_scanner.pkl")

    test_cases = [
        ("PR #101: no change",
         rng.random(n_features).astype(np.float32) * 0.5,   # baseline
         None,  # same as baseline → zero delta
         "clean"),
        ("PR #102: CWE-78 (OS command injection)",
         rng.random(n_features).astype(np.float32) * 0.5,
         "CWE-78",
         "should flag CWE-78"),
        ("PR #103: CWE-79 (XSS)",
         rng.random(n_features).astype(np.float32) * 0.5,
         "CWE-79",
         "should flag CWE-79"),
    ]

    for title, baseline, inject_cwe, expected in test_cases:
        print(f"\n{title}  ({expected})")
        new_code = baseline.copy()
        if inject_cwe:
            for feat in cwe_signal_features[inject_cwe]:
                new_code[feat] += signal_strength

        report = scanner_loaded.scan(baseline, new_code)
        report.print()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute feature–CWE correlations")
    parser.add_argument("--run_id", required=True, help="Activation run directory name")
    parser.add_argument("--layer", type=int, default=0, help="Transformer layer index")
    parser.add_argument("--min_samples", type=int, default=5, help="Min samples per CWE")
    parser.add_argument("--top_k", type=int, default=10, help="Top features per CWE in summary")
    parser.add_argument("--save_dir", default=None, help="Directory to save output matrices")
    parser.add_argument("--plot_heatmap", action="store_true", help="Plot correlation heatmap")
    args = parser.parse_args()

    result = compute_from_run(args.run_id, layer=args.layer, min_samples=args.min_samples)
    result.print_summary(k=args.top_k, min_samples=args.min_samples)

    if args.save_dir:
        save_correlations(result, args.save_dir)

    if args.plot_heatmap:
        plot_top_features_heatmap(result, min_samples=args.min_samples)
