"""
Global Anomaly Detection Baselines

This module implements multiple global anomaly detection methods to demonstrate
that vulnerable code does NOT form a globally anomalous population.

Methods:
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- PCA Reconstruction Error
- Cosine Similarity
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import ks_2samp, mannwhitneyu, wilcoxon
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


@dataclass
class BaselineResult:
    """Results from a single baseline method."""
    method_name: str
    auroc: float
    accuracy: float  # at optimal threshold
    ks_statistic: float
    ks_pvalue: float
    cohens_d: float
    secure_scores_mean: float
    secure_scores_std: float
    vulnerable_scores_mean: float
    vulnerable_scores_std: float
    # For paired analysis
    paired_delta_positive_rate: Optional[float] = None  # P(vuln_score > sec_score)
    wilcoxon_pvalue: Optional[float] = None


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.

    Cohen's d = (mean1 - mean2) / pooled_std

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def evaluate_anomaly_scores(
    secure_scores: np.ndarray,
    vulnerable_scores: np.ndarray,
    method_name: str,
    higher_is_anomalous: bool = True,
    paired_secure_scores: Optional[np.ndarray] = None,
    paired_vulnerable_scores: Optional[np.ndarray] = None,
) -> BaselineResult:
    """
    Evaluate anomaly scores for secure vs vulnerable samples.

    Args:
        secure_scores: Anomaly scores for secure samples
        vulnerable_scores: Anomaly scores for vulnerable samples
        method_name: Name of the method for reporting
        higher_is_anomalous: If True, higher scores indicate anomalies
        paired_secure_scores: If provided, compute paired statistics
        paired_vulnerable_scores: If provided, compute paired statistics

    Returns:
        BaselineResult with all metrics
    """
    # Flip scores if lower means more anomalous
    if not higher_is_anomalous:
        secure_scores = -secure_scores
        vulnerable_scores = -vulnerable_scores
        if paired_secure_scores is not None:
            paired_secure_scores = -paired_secure_scores
            paired_vulnerable_scores = -paired_vulnerable_scores

    # Prepare for AUROC (vulnerable = positive class)
    y_true = np.concatenate([
        np.zeros(len(secure_scores)),
        np.ones(len(vulnerable_scores))
    ])
    y_scores = np.concatenate([secure_scores, vulnerable_scores])

    # AUROC
    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = 0.5  # If all scores are identical

    # Optimal accuracy (find best threshold)
    thresholds = np.percentile(y_scores, np.arange(0, 101, 5))
    best_acc = 0.5
    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        acc = accuracy_score(y_true, preds)
        best_acc = max(best_acc, acc)

    # KS test
    ks_stat, ks_pval = ks_2samp(secure_scores, vulnerable_scores)

    # Cohen's d
    cohens_d = compute_cohens_d(vulnerable_scores, secure_scores)

    # Paired analysis
    paired_delta_positive_rate = None
    wilcoxon_pval = None
    if paired_secure_scores is not None and paired_vulnerable_scores is not None:
        deltas = paired_vulnerable_scores - paired_secure_scores
        paired_delta_positive_rate = np.mean(deltas > 0)
        try:
            _, wilcoxon_pval = wilcoxon(deltas)
        except ValueError:
            wilcoxon_pval = 1.0

    return BaselineResult(
        method_name=method_name,
        auroc=auroc,
        accuracy=best_acc,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pval,
        cohens_d=cohens_d,
        secure_scores_mean=np.mean(secure_scores),
        secure_scores_std=np.std(secure_scores),
        vulnerable_scores_mean=np.mean(vulnerable_scores),
        vulnerable_scores_std=np.std(vulnerable_scores),
        paired_delta_positive_rate=paired_delta_positive_rate,
        wilcoxon_pvalue=wilcoxon_pval,
    )


class GlobalAnomalyBaselines:
    """
    Run multiple global anomaly detection baselines.

    These methods train on secure code and try to detect vulnerable code
    as anomalies. We expect all methods to fail, demonstrating that
    vulnerabilities are NOT globally anomalous.
    """

    def __init__(
        self,
        secure_activations: torch.Tensor,
        vulnerable_activations: torch.Tensor,
        paired: bool = True,
        random_state: int = 42,
    ):
        """
        Initialize with activation tensors.

        Args:
            secure_activations: Shape (n_samples, n_features)
            vulnerable_activations: Shape (n_samples, n_features)
            paired: Whether secure[i] and vulnerable[i] are paired samples
            random_state: Random seed for reproducibility
        """
        self.secure_np = secure_activations.numpy() if isinstance(
            secure_activations, torch.Tensor
        ) else secure_activations
        self.vulnerable_np = vulnerable_activations.numpy() if isinstance(
            vulnerable_activations, torch.Tensor
        ) else vulnerable_activations
        self.paired = paired
        self.random_state = random_state

        # Standardize features
        self.scaler = StandardScaler()
        self.secure_scaled = self.scaler.fit_transform(self.secure_np)
        self.vulnerable_scaled = self.scaler.transform(self.vulnerable_np)

        self.results: Dict[str, BaselineResult] = {}

    def run_isolation_forest(self, contamination: float = 0.01) -> BaselineResult:
        """
        Isolation Forest: Anomaly detection via random forests.

        Lower scores = more anomalous in sklearn's implementation.
        """
        print("Running Isolation Forest...")

        model = IsolationForest(
            contamination=contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(self.secure_scaled)

        secure_scores = model.score_samples(self.secure_scaled)
        vulnerable_scores = model.score_samples(self.vulnerable_scaled)

        result = evaluate_anomaly_scores(
            secure_scores,
            vulnerable_scores,
            method_name="Isolation Forest",
            higher_is_anomalous=False,  # Lower = more anomalous
            paired_secure_scores=secure_scores if self.paired else None,
            paired_vulnerable_scores=vulnerable_scores if self.paired else None,
        )

        self.results["isolation_forest"] = result
        return result

    def run_lof(self, n_neighbors: int = 20) -> BaselineResult:
        """
        Local Outlier Factor: Density-based anomaly detection.

        Higher negative outlier factor = more anomalous.
        """
        print("Running Local Outlier Factor...")

        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,  # Enable predict on new data
            n_jobs=-1,
        )
        model.fit(self.secure_scaled)

        secure_scores = model.score_samples(self.secure_scaled)
        vulnerable_scores = model.score_samples(self.vulnerable_scaled)

        result = evaluate_anomaly_scores(
            secure_scores,
            vulnerable_scores,
            method_name="Local Outlier Factor",
            higher_is_anomalous=False,  # Lower (more negative) = more anomalous
            paired_secure_scores=secure_scores if self.paired else None,
            paired_vulnerable_scores=vulnerable_scores if self.paired else None,
        )

        self.results["lof"] = result
        return result

    def run_ocsvm(self, kernel: str = "rbf", nu: float = 0.01) -> BaselineResult:
        """
        One-Class SVM: Classic one-class classification.

        Note: Can be slow for large datasets. Consider subsampling.
        """
        print("Running One-Class SVM...")

        # Subsample for speed if needed
        max_samples = 5000
        if len(self.secure_scaled) > max_samples:
            idx = np.random.choice(len(self.secure_scaled), max_samples, replace=False)
            train_data = self.secure_scaled[idx]
        else:
            train_data = self.secure_scaled

        model = OneClassSVM(kernel=kernel, nu=nu)
        model.fit(train_data)

        secure_scores = model.score_samples(self.secure_scaled)
        vulnerable_scores = model.score_samples(self.vulnerable_scaled)

        result = evaluate_anomaly_scores(
            secure_scores,
            vulnerable_scores,
            method_name="One-Class SVM",
            higher_is_anomalous=False,  # Lower = more anomalous
            paired_secure_scores=secure_scores if self.paired else None,
            paired_vulnerable_scores=vulnerable_scores if self.paired else None,
        )

        self.results["ocsvm"] = result
        return result

    def run_pca_reconstruction(self, n_components: int = 50) -> BaselineResult:
        """
        PCA Reconstruction Error: Anomalies have high reconstruction error.

        Train PCA on secure code, measure reconstruction error on all samples.
        """
        print("Running PCA Reconstruction Error...")

        pca = PCA(n_components=min(n_components, self.secure_scaled.shape[1]))
        pca.fit(self.secure_scaled)

        # Reconstruction error = ||x - reconstruct(x)||^2
        secure_reconstructed = pca.inverse_transform(pca.transform(self.secure_scaled))
        vulnerable_reconstructed = pca.inverse_transform(pca.transform(self.vulnerable_scaled))

        secure_scores = np.sum((self.secure_scaled - secure_reconstructed) ** 2, axis=1)
        vulnerable_scores = np.sum((self.vulnerable_scaled - vulnerable_reconstructed) ** 2, axis=1)

        result = evaluate_anomaly_scores(
            secure_scores,
            vulnerable_scores,
            method_name="PCA Reconstruction Error",
            higher_is_anomalous=True,  # Higher error = more anomalous
            paired_secure_scores=secure_scores if self.paired else None,
            paired_vulnerable_scores=vulnerable_scores if self.paired else None,
        )

        self.results["pca_reconstruction"] = result
        return result

    def run_cosine_distance_to_centroid(self) -> BaselineResult:
        """
        Cosine Distance to Centroid: Simple centroid-based detection.

        Compute centroid of secure samples, measure cosine distance.
        """
        print("Running Cosine Distance to Centroid...")

        from sklearn.metrics.pairwise import cosine_distances

        centroid = np.mean(self.secure_scaled, axis=0, keepdims=True)

        secure_scores = cosine_distances(self.secure_scaled, centroid).flatten()
        vulnerable_scores = cosine_distances(self.vulnerable_scaled, centroid).flatten()

        result = evaluate_anomaly_scores(
            secure_scores,
            vulnerable_scores,
            method_name="Cosine Distance to Centroid",
            higher_is_anomalous=True,  # Higher distance = more anomalous
            paired_secure_scores=secure_scores if self.paired else None,
            paired_vulnerable_scores=vulnerable_scores if self.paired else None,
        )

        self.results["cosine_centroid"] = result
        return result

    def run_mahalanobis_distance(self) -> BaselineResult:
        """
        Mahalanobis Distance: Accounts for feature correlations.

        Measures distance from secure distribution accounting for covariance.
        """
        print("Running Mahalanobis Distance...")

        # Compute mean and covariance of secure samples
        mean = np.mean(self.secure_scaled, axis=0)
        cov = np.cov(self.secure_scaled, rowvar=False)

        # Regularize covariance for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-6

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        def mahalanobis(x: np.ndarray) -> np.ndarray:
            diff = x - mean
            return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

        secure_scores = mahalanobis(self.secure_scaled)
        vulnerable_scores = mahalanobis(self.vulnerable_scaled)

        result = evaluate_anomaly_scores(
            secure_scores,
            vulnerable_scores,
            method_name="Mahalanobis Distance",
            higher_is_anomalous=True,
            paired_secure_scores=secure_scores if self.paired else None,
            paired_vulnerable_scores=vulnerable_scores if self.paired else None,
        )

        self.results["mahalanobis"] = result
        return result

    def run_all(self) -> Dict[str, BaselineResult]:
        """Run all baseline methods and return results."""
        self.run_isolation_forest()
        self.run_lof()
        self.run_ocsvm()
        self.run_pca_reconstruction()
        self.run_cosine_distance_to_centroid()
        # self.run_mahalanobis_distance()
        return self.results

    def print_summary(self) -> str:
        """Print a formatted summary table of all results."""
        if not self.results:
            return "No results yet. Run run_all() first."

        lines = []
        lines.append("=" * 100)
        lines.append("GLOBAL ANOMALY DETECTION BASELINES - SUMMARY")
        lines.append("=" * 100)
        lines.append("")
        lines.append("Claim: Vulnerable code does NOT form a globally anomalous population.")
        lines.append("Evidence: All methods should show AUROC ~ 0.5 and small effect sizes.")
        lines.append("")
        lines.append("-" * 100)
        lines.append(f"{'Method':<30} {'AUROC':<10} {'Accuracy':<10} {'Cohen d':<10} {'KS stat':<10} {'KS p-val':<12} {'P(d>0)':<10}")
        lines.append("-" * 100)

        for name, result in self.results.items():
            p_delta = f"{result.paired_delta_positive_rate:.3f}" if result.paired_delta_positive_rate else "N/A"
            lines.append(
                f"{result.method_name:<30} "
                f"{result.auroc:<10.3f} "
                f"{result.accuracy:<10.3f} "
                f"{result.cohens_d:<10.3f} "
                f"{result.ks_statistic:<10.3f} "
                f"{result.ks_pvalue:<12.2e} "
                f"{p_delta:<10}"
            )

        lines.append("-" * 100)
        lines.append("")
        lines.append("Interpretation:")
        lines.append("  - AUROC ~ 0.5 = random classifier (no separation)")
        lines.append("  - |Cohen's d| < 0.2 = negligible effect size")
        lines.append("  - P(d>0) ~ 0.5 = no consistent direction of difference")
        lines.append("=" * 100)

        summary = "\n".join(lines)
        print(summary)
        return summary

    def to_latex_table(self) -> str:
        """Generate a LaTeX table of results."""
        if not self.results:
            return "No results yet."

        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{Global anomaly detection baselines fail to distinguish vulnerable from secure code.}")
        lines.append(r"\label{tab:global_baselines}")
        lines.append(r"\begin{tabular}{lccccc}")
        lines.append(r"\toprule")
        lines.append(r"Method & AUROC & Accuracy & Cohen's $d$ & KS stat & $p$-value \\")
        lines.append(r"\midrule")

        for result in self.results.values():
            lines.append(
                f"{result.method_name} & "
                f"{result.auroc:.3f} & "
                f"{result.accuracy:.3f} & "
                f"{result.cohens_d:.3f} & "
                f"{result.ks_statistic:.3f} & "
                f"{result.ks_pvalue:.2e} \\\\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        return "\n".join(lines)


def run_global_baselines(
    secure_activations: torch.Tensor,
    vulnerable_activations: torch.Tensor,
    paired: bool = True,
) -> Tuple[Dict[str, BaselineResult], str, str]:
    """
    Convenience function to run all global baselines.

    Returns:
        results: Dictionary of BaselineResult objects
        summary: Formatted text summary
        latex: LaTeX table code
    """
    baselines = GlobalAnomalyBaselines(
        secure_activations,
        vulnerable_activations,
        paired=paired,
    )
    results = baselines.run_all()
    summary = baselines.print_summary()
    latex = baselines.to_latex_table()

    return results, summary, latex
