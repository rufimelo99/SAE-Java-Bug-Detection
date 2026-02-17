"""
Per-CWE Analysis Module

Analyzes detection performance broken down by CWE (Common Weakness Enumeration) type.
Some vulnerability types may be more detectable than others.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
from scipy.stats import wilcoxon, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams


@dataclass
class CWEResult:
    """Results for a single CWE type."""
    cwe_id: str
    n_samples: int
    # Paired metrics
    mean_delta: float  # mean(vuln_score - secure_score)
    std_delta: float
    p_delta_positive: float  # P(vuln_score > secure_score)
    wilcoxon_pvalue: float
    # Unpaired metrics
    auroc: float
    cohens_d: float


def compute_auroc_simple(
    secure_scores: np.ndarray,
    vulnerable_scores: np.ndarray,
) -> float:
    """Compute AUROC for binary classification (vulnerable = positive)."""
    from sklearn.metrics import roc_auc_score

    y_true = np.concatenate([
        np.zeros(len(secure_scores)),
        np.ones(len(vulnerable_scores))
    ])
    y_scores = np.concatenate([secure_scores, vulnerable_scores])

    try:
        return roc_auc_score(y_true, y_scores)
    except ValueError:
        return 0.5


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


class PerCWEAnalyzer:
    """
    Analyze detection performance per CWE type.

    This helps identify:
    1. Which vulnerability types are most detectable
    2. Whether certain CWE categories cluster together
    3. What types of bugs the SAE approach captures
    """

    def __init__(
        self,
        secure_scores: np.ndarray,
        vulnerable_scores: np.ndarray,
        cwe_labels: List[str],
        paired: bool = True,
    ):
        """
        Initialize analyzer.

        Args:
            secure_scores: Violation/anomaly scores for secure samples
            vulnerable_scores: Violation/anomaly scores for vulnerable samples
            cwe_labels: CWE label for each sample (len = n_samples)
            paired: Whether secure[i] and vulnerable[i] are paired
        """
        self.secure_scores = np.array(secure_scores)
        self.vulnerable_scores = np.array(vulnerable_scores)
        self.cwe_labels = cwe_labels
        self.paired = paired

        assert len(self.secure_scores) == len(self.vulnerable_scores) == len(cwe_labels)

        self.unique_cwes = sorted(set(cwe_labels))
        self.results: Dict[str, CWEResult] = {}

    def analyze_cwe(self, cwe_id: str) -> CWEResult:
        """Analyze a single CWE type."""
        mask = np.array([c == cwe_id for c in self.cwe_labels])
        n_samples = mask.sum()

        if n_samples < 2:
            return CWEResult(
                cwe_id=cwe_id,
                n_samples=n_samples,
                mean_delta=0.0,
                std_delta=0.0,
                p_delta_positive=0.0,
                wilcoxon_pvalue=1.0,
                auroc=0.5,
                cohens_d=0.0,
            )

        secure = self.secure_scores[mask]
        vulnerable = self.vulnerable_scores[mask]

        # Paired analysis
        deltas = vulnerable - secure
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        p_delta_positive = np.mean(deltas > 0)

        try:
            _, wilcoxon_pval = wilcoxon(deltas)
        except ValueError:
            wilcoxon_pval = 1.0

        # Unpaired metrics
        auroc = compute_auroc_simple(secure, vulnerable)
        cohens_d = compute_cohens_d(vulnerable, secure)

        return CWEResult(
            cwe_id=cwe_id,
            n_samples=n_samples,
            mean_delta=mean_delta,
            std_delta=std_delta,
            p_delta_positive=p_delta_positive,
            wilcoxon_pvalue=wilcoxon_pval,
            auroc=auroc,
            cohens_d=cohens_d,
        )

    def analyze_all(self) -> Dict[str, CWEResult]:
        """Analyze all CWE types."""
        for cwe in self.unique_cwes:
            self.results[cwe] = self.analyze_cwe(cwe)
        return self.results

    def get_top_cwes(
        self,
        metric: str = "p_delta_positive",
        n: int = 10,
        min_samples: int = 10,
    ) -> List[CWEResult]:
        """
        Get top N CWEs by a given metric.

        Args:
            metric: One of 'p_delta_positive', 'auroc', 'cohens_d', 'mean_delta'
            n: Number of top CWEs to return
            min_samples: Minimum samples required for inclusion
        """
        if not self.results:
            self.analyze_all()

        # Filter by minimum samples
        filtered = [r for r in self.results.values() if r.n_samples >= min_samples]

        # Sort by metric (descending for most metrics)
        if metric == "wilcoxon_pvalue":
            # Lower p-value = more significant
            filtered.sort(key=lambda x: getattr(x, metric))
        else:
            # Higher = better for other metrics
            filtered.sort(key=lambda x: getattr(x, metric), reverse=True)

        return filtered[:n]

    def print_summary(self, min_samples: int = 5) -> str:
        """Print formatted summary of per-CWE results."""
        if not self.results:
            self.analyze_all()

        lines = []
        lines.append("=" * 110)
        lines.append("PER-CWE ANALYSIS SUMMARY")
        lines.append("=" * 110)
        lines.append("")
        lines.append(f"{'CWE':<15} {'N':<8} {'P(d>0)':<10} {'Mean d':<12} {'AUROC':<10} {'Cohen d':<10} {'Wilcox p':<12}")
        lines.append("-" * 110)

        # Sort by P(delta > 0) descending
        sorted_results = sorted(
            self.results.values(),
            key=lambda x: x.p_delta_positive,
            reverse=True,
        )

        for r in sorted_results:
            if r.n_samples < min_samples:
                continue

            sig = "*" if r.wilcoxon_pvalue < 0.05 else " "
            sig += "*" if r.wilcoxon_pvalue < 0.01 else " "
            sig += "*" if r.wilcoxon_pvalue < 0.001 else " "

            lines.append(
                f"{r.cwe_id:<15} "
                f"{r.n_samples:<8} "
                f"{r.p_delta_positive:<10.3f} "
                f"{r.mean_delta:<12.4f} "
                f"{r.auroc:<10.3f} "
                f"{r.cohens_d:<10.3f} "
                f"{r.wilcoxon_pvalue:<10.2e} {sig}"
            )

        lines.append("-" * 110)
        lines.append("Significance: * p<0.05, ** p<0.01, *** p<0.001")
        lines.append("=" * 110)

        summary = "\n".join(lines)
        print(summary)
        return summary

    def plot_per_cwe_comparison(
        self,
        metric: str = "p_delta_positive",
        min_samples: int = 10,
        save_path: Optional[str] = None,
    ):
        """
        Create a bar plot comparing CWEs by a metric.

        Args:
            metric: Metric to plot
            min_samples: Minimum samples for inclusion
            save_path: Path to save figure (optional)
        """
        if not self.results:
            self.analyze_all()

        # Filter and sort
        data = [
            (r.cwe_id, getattr(r, metric), r.n_samples)
            for r in self.results.values()
            if r.n_samples >= min_samples
        ]
        data.sort(key=lambda x: x[1], reverse=True)

        if not data:
            print("No CWEs with enough samples")
            return

        cwes = [d[0] for d in data]
        values = [d[1] for d in data]
        counts = [d[2] for d in data]

        # Set up figure
        rcParams.update({
            "font.family": "serif",
            "font.size": 12,
        })

        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

        # Color by whether significant
        colors = []
        for r in [self.results[c] for c in cwes]:
            if r.wilcoxon_pvalue < 0.05:
                colors.append("#2ecc71")  # Green for significant
            else:
                colors.append("#95a5a6")  # Gray for not significant

        bars = ax.barh(cwes, values, color=colors, edgecolor="white", linewidth=0.5)

        # Add sample counts as annotations
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"n={count}",
                va="center",
                fontsize=9,
            )

        # Reference line at 0.5 for P(delta > 0)
        if metric == "p_delta_positive":
            ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="Random (0.5)")
            ax.set_xlabel("P(Vulnerable Score > Secure Score)")
        elif metric == "auroc":
            ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="Random (0.5)")
            ax.set_xlabel("AUROC")
        elif metric == "cohens_d":
            ax.axvline(0, color="red", linestyle="--", linewidth=1)
            ax.axvline(0.2, color="orange", linestyle=":", linewidth=1, label="Small effect")
            ax.axvline(0.5, color="orange", linestyle="--", linewidth=1, label="Medium effect")
            ax.set_xlabel("Cohen's d")
        else:
            ax.set_xlabel(metric)

        ax.set_ylabel("CWE Type")
        ax.set_title(f"Detection Performance by CWE Type ({metric})")
        ax.legend(loc="lower right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", format="pdf")

        plt.show()

    def to_latex_table(self, min_samples: int = 10, top_n: int = 15) -> str:
        """Generate LaTeX table of per-CWE results."""
        if not self.results:
            self.analyze_all()

        # Get top CWEs
        top = self.get_top_cwes(metric="p_delta_positive", n=top_n, min_samples=min_samples)

        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{Detection performance by CWE type. Higher P($\Delta > 0$) indicates vulnerable samples have higher violation scores.}")
        lines.append(r"\label{tab:per_cwe}")
        lines.append(r"\begin{tabular}{llccccc}")
        lines.append(r"\toprule")
        lines.append(r"CWE & Description & $n$ & P($\Delta > 0$) & AUROC & Cohen's $d$ & $p$-value \\")
        lines.append(r"\midrule")

        # CWE descriptions (abbreviated)
        cwe_desc = {
            "CWE-79": "XSS",
            "CWE-89": "SQL Injection",
            "CWE-78": "OS Command Inj.",
            "CWE-22": "Path Traversal",
            "CWE-119": "Buffer Overflow",
            "CWE-125": "OOB Read",
            "CWE-787": "OOB Write",
            "CWE-476": "NULL Deref",
            "CWE-416": "Use-After-Free",
            "CWE-190": "Integer Overflow",
            "CWE-20": "Input Validation",
            "CWE-200": "Info Exposure",
            "CWE-352": "CSRF",
            "CWE-400": "Resource Exhaust.",
            "CWE-94": "Code Injection",
            "CWE-287": "Auth Bypass",
            "CWE-362": "Race Condition",
            "CWE-434": "File Upload",
            "CWE-401": "Memory Leak",
            "CWE-415": "Double Free",
            "CWE-369": "Divide by Zero",
            "CWE-120": "Buffer Copy",
        }

        for r in top:
            desc = cwe_desc.get(r.cwe_id, "-")
            sig = ""
            if r.wilcoxon_pvalue < 0.001:
                sig = "***"
            elif r.wilcoxon_pvalue < 0.01:
                sig = "**"
            elif r.wilcoxon_pvalue < 0.05:
                sig = "*"

            lines.append(
                f"{r.cwe_id} & {desc} & {r.n_samples} & "
                f"{r.p_delta_positive:.3f} & {r.auroc:.3f} & "
                f"{r.cohens_d:.3f} & {r.wilcoxon_pvalue:.2e}{sig} \\\\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        return "\n".join(lines)


def analyze_per_cwe(
    secure_scores: np.ndarray,
    vulnerable_scores: np.ndarray,
    cwe_labels: List[str],
    min_samples: int = 10,
    save_path: Optional[str] = None,
) -> Tuple[Dict[str, CWEResult], str, str]:
    """
    Convenience function to run per-CWE analysis.

    Returns:
        results: Dictionary of CWEResult objects
        summary: Formatted text summary
        latex: LaTeX table code
    """
    analyzer = PerCWEAnalyzer(
        secure_scores=secure_scores,
        vulnerable_scores=vulnerable_scores,
        cwe_labels=cwe_labels,
        paired=True,
    )

    results = analyzer.analyze_all()
    summary = analyzer.print_summary(min_samples=min_samples)
    latex = analyzer.to_latex_table(min_samples=min_samples)

    if save_path:
        analyzer.plot_per_cwe_comparison(save_path=save_path)

    return results, summary, latex


# CWE Category mapping for higher-level analysis
CWE_CATEGORIES = {
    "Memory Safety": ["CWE-119", "CWE-120", "CWE-125", "CWE-787", "CWE-416", "CWE-415", "CWE-401", "CWE-476"],
    "Injection": ["CWE-79", "CWE-89", "CWE-78", "CWE-94"],
    "Input Validation": ["CWE-20", "CWE-22"],
    "Authentication": ["CWE-287", "CWE-352"],
    "Resource Management": ["CWE-400", "CWE-399", "CWE-362"],
    "Numeric": ["CWE-190", "CWE-369", "CWE-189"],
    "Information Disclosure": ["CWE-200"],
    "Other": ["CWE-434", "CWE-264"],
}


def get_cwe_category(cwe_id: str) -> str:
    """Get the category for a CWE ID."""
    for category, cwes in CWE_CATEGORIES.items():
        if cwe_id in cwes:
            return category
    return "Other"
