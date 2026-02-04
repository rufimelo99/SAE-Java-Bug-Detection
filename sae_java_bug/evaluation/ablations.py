"""
Ablation Studies Module

Systematically varies hyperparameters of the invariant-based detection method
to find optimal settings and understand method sensitivity.

Key hyperparameters:
- tau (threshold for invariant confidence): {0.90, 0.95, 0.99, 0.995}
- min_support (minimum feature firing count): {50, 100, 200, 500}
- scoring method: {weighted, unweighted, max}
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm


@dataclass
class AblationResult:
    """Results from a single ablation configuration."""
    tau: float
    min_support: int
    scoring_method: str
    n_invariants: int

    # Core metrics
    auroc: float
    p_delta_positive: float  # P(vuln_score > secure_score)
    wilcoxon_pvalue: float
    mean_delta: float
    std_delta: float

    # Distribution stats
    secure_mean: float
    secure_std: float
    vulnerable_mean: float
    vulnerable_std: float


def binarize(activations: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binarize activations (feature fired or not)."""
    return activations > eps


def compute_firing_stats(bins: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute firing statistics.

    Returns:
        fire_count: (d,) - number of samples where each feature fires
        cofire_count: (d, d) - co-firing matrix
    """
    bins_f = bins.float()
    fire_count = bins_f.sum(dim=0)
    cofire_count = bins_f.T @ bins_f
    return fire_count, cofire_count


def compute_conditional_probs(
    fire_count: torch.Tensor,
    cofire_count: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute P(j fires | i fires) matrix."""
    fire_i = fire_count.unsqueeze(1)
    P_ij = cofire_count / (fire_i + eps)
    return P_ij


def extract_invariants(
    P_ij: torch.Tensor,
    fire_count: torch.Tensor,
    tau: float = 0.99,
    min_support: int = 100,
) -> List[Tuple[int, int, float]]:
    """
    Extract invariants: pairs (i, j) where P(j|i) >= tau.

    Returns list of (i, j, P_ij) tuples.
    """
    d = P_ij.shape[0]
    invariants = []

    for i in range(d):
        if fire_count[i] < min_support:
            continue
        js = torch.where(P_ij[i] >= tau)[0]
        for j in js.tolist():
            if i == j:
                continue
            invariants.append((i, j, float(P_ij[i, j])))

    return invariants


def compute_violation_score_weighted(
    sample_bin: torch.Tensor,
    invariants: List[Tuple[int, int, float]],
    max_weight: float = 10.0,
) -> float:
    """Weighted violation score: sum of -log(1-P_ij) for violated invariants."""
    score = 0.0
    for i, j, pij in invariants:
        if sample_bin[i] and not sample_bin[j]:
            weight = min(-np.log(1.0 - pij + 1e-10), max_weight)
            score += weight
    return score


def compute_violation_score_unweighted(
    sample_bin: torch.Tensor,
    invariants: List[Tuple[int, int, float]],
) -> float:
    """Unweighted violation score: count of violated invariants."""
    count = 0
    for i, j, pij in invariants:
        if sample_bin[i] and not sample_bin[j]:
            count += 1
    return float(count)


def compute_violation_score_max(
    sample_bin: torch.Tensor,
    invariants: List[Tuple[int, int, float]],
    max_weight: float = 10.0,
) -> float:
    """Max violation score: maximum weight among violated invariants."""
    max_score = 0.0
    for i, j, pij in invariants:
        if sample_bin[i] and not sample_bin[j]:
            weight = min(-np.log(1.0 - pij + 1e-10), max_weight)
            max_score = max(max_score, weight)
    return max_score


SCORING_METHODS = {
    "weighted": compute_violation_score_weighted,
    "unweighted": compute_violation_score_unweighted,
    "max": compute_violation_score_max,
}


def batch_violation_scores(
    bins: torch.Tensor,
    invariants: List[Tuple[int, int, float]],
    method: str = "weighted",
) -> np.ndarray:
    """Compute violation scores for all samples."""
    score_fn = SCORING_METHODS[method]
    scores = []
    for n in range(bins.shape[0]):
        if method == "weighted" or method == "max":
            scores.append(score_fn(bins[n], invariants))
        else:
            scores.append(score_fn(bins[n], invariants))
    return np.array(scores)


class InvariantAblation:
    """
    Run ablation studies on invariant-based detection.

    Varies:
    - tau: Confidence threshold for invariants
    - min_support: Minimum feature firing count
    - scoring_method: How to aggregate violation scores
    """

    def __init__(
        self,
        secure_activations: torch.Tensor,
        vulnerable_activations: torch.Tensor,
        paired: bool = True,
    ):
        """
        Initialize with activation tensors.

        Args:
            secure_activations: Shape (n_samples, n_features)
            vulnerable_activations: Shape (n_samples, n_features)
            paired: Whether samples are paired
        """
        self.secure_act = secure_activations
        self.vulnerable_act = vulnerable_activations
        self.paired = paired

        # Binarize
        self.secure_bin = binarize(self.secure_act)
        self.vulnerable_bin = binarize(self.vulnerable_act)

        # Compute base statistics from secure code
        self.fire_count, self.cofire_count = compute_firing_stats(self.secure_bin)
        self.P_ij = compute_conditional_probs(self.fire_count, self.cofire_count)

        self.results: List[AblationResult] = []

    def run_single(
        self,
        tau: float,
        min_support: int,
        scoring_method: str,
    ) -> AblationResult:
        """Run a single ablation configuration."""
        # Extract invariants
        invariants = extract_invariants(
            self.P_ij,
            self.fire_count,
            tau=tau,
            min_support=min_support,
        )

        if len(invariants) == 0:
            # No invariants - return null result
            return AblationResult(
                tau=tau,
                min_support=min_support,
                scoring_method=scoring_method,
                n_invariants=0,
                auroc=0.5,
                p_delta_positive=0.5,
                wilcoxon_pvalue=1.0,
                mean_delta=0.0,
                std_delta=0.0,
                secure_mean=0.0,
                secure_std=0.0,
                vulnerable_mean=0.0,
                vulnerable_std=0.0,
            )

        # Compute scores
        secure_scores = batch_violation_scores(self.secure_bin, invariants, scoring_method)
        vulnerable_scores = batch_violation_scores(self.vulnerable_bin, invariants, scoring_method)

        # Metrics
        deltas = vulnerable_scores - secure_scores
        p_delta_positive = np.mean(deltas > 0)

        try:
            _, wilcoxon_pval = wilcoxon(deltas)
        except ValueError:
            wilcoxon_pval = 1.0

        # AUROC
        y_true = np.concatenate([
            np.zeros(len(secure_scores)),
            np.ones(len(vulnerable_scores))
        ])
        y_scores = np.concatenate([secure_scores, vulnerable_scores])
        try:
            auroc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auroc = 0.5

        return AblationResult(
            tau=tau,
            min_support=min_support,
            scoring_method=scoring_method,
            n_invariants=len(invariants),
            auroc=auroc,
            p_delta_positive=p_delta_positive,
            wilcoxon_pvalue=wilcoxon_pval,
            mean_delta=np.mean(deltas),
            std_delta=np.std(deltas),
            secure_mean=np.mean(secure_scores),
            secure_std=np.std(secure_scores),
            vulnerable_mean=np.mean(vulnerable_scores),
            vulnerable_std=np.std(vulnerable_scores),
        )

    def run_grid_search(
        self,
        tau_values: List[float] = [0.90, 0.95, 0.99, 0.995],
        min_support_values: List[int] = [50, 100, 200, 500],
        scoring_methods: List[str] = ["weighted", "unweighted", "max"],
    ) -> List[AblationResult]:
        """Run ablation over all hyperparameter combinations."""
        total = len(tau_values) * len(min_support_values) * len(scoring_methods)

        with tqdm(total=total, desc="Ablation sweep") as pbar:
            for tau in tau_values:
                for min_support in min_support_values:
                    for method in scoring_methods:
                        result = self.run_single(tau, min_support, method)
                        self.results.append(result)
                        pbar.update(1)

        return self.results

    def get_best_config(self, metric: str = "p_delta_positive") -> AblationResult:
        """Get the configuration with best performance on given metric."""
        if not self.results:
            raise ValueError("No results yet. Run run_grid_search() first.")

        if metric == "wilcoxon_pvalue":
            return min(self.results, key=lambda x: x.wilcoxon_pvalue)
        else:
            return max(self.results, key=lambda x: getattr(x, metric))

    def print_summary(self) -> str:
        """Print formatted summary of ablation results."""
        if not self.results:
            return "No results yet."

        lines = []
        lines.append("=" * 120)
        lines.append("INVARIANT ABLATION STUDY - SUMMARY")
        lines.append("=" * 120)
        lines.append("")
        lines.append(f"{'tau':<8} {'min_sup':<10} {'method':<12} {'#inv':<10} {'P(d>0)':<10} {'AUROC':<10} {'Mean d':<12} {'Wilcox p':<12}")
        lines.append("-" * 120)

        # Sort by P(delta > 0)
        sorted_results = sorted(self.results, key=lambda x: x.p_delta_positive, reverse=True)

        for r in sorted_results:
            sig = "*" if r.wilcoxon_pvalue < 0.05 else ""
            lines.append(
                f"{r.tau:<8.3f} "
                f"{r.min_support:<10} "
                f"{r.scoring_method:<12} "
                f"{r.n_invariants:<10} "
                f"{r.p_delta_positive:<10.3f} "
                f"{r.auroc:<10.3f} "
                f"{r.mean_delta:<12.4f} "
                f"{r.wilcoxon_pvalue:<10.2e} {sig}"
            )

        lines.append("-" * 120)

        # Best configs
        best = self.get_best_config("p_delta_positive")
        lines.append("")
        lines.append(f"Best by P(d>0): tau={best.tau}, min_support={best.min_support}, "
                     f"method={best.scoring_method}, P(d>0)={best.p_delta_positive:.3f}")

        best_auroc = self.get_best_config("auroc")
        lines.append(f"Best by AUROC: tau={best_auroc.tau}, min_support={best_auroc.min_support}, "
                     f"method={best_auroc.scoring_method}, AUROC={best_auroc.auroc:.3f}")

        lines.append("=" * 120)

        summary = "\n".join(lines)
        print(summary)
        return summary

    def plot_heatmap(
        self,
        scoring_method: str = "weighted",
        metric: str = "p_delta_positive",
        save_path: Optional[str] = None,
    ):
        """
        Plot heatmap of results for a specific scoring method.

        Args:
            scoring_method: One of 'weighted', 'unweighted', 'max'
            metric: Metric to visualize
            save_path: Path to save figure
        """
        if not self.results:
            return

        # Filter results for this scoring method
        filtered = [r for r in self.results if r.scoring_method == scoring_method]

        # Get unique values
        tau_vals = sorted(set(r.tau for r in filtered))
        sup_vals = sorted(set(r.min_support for r in filtered))

        # Build matrix
        matrix = np.zeros((len(tau_vals), len(sup_vals)))
        for r in filtered:
            i = tau_vals.index(r.tau)
            j = sup_vals.index(r.min_support)
            matrix[i, j] = getattr(r, metric)

        # Plot
        rcParams.update({"font.family": "serif", "font.size": 12})
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric)

        # Labels
        ax.set_xticks(range(len(sup_vals)))
        ax.set_xticklabels(sup_vals)
        ax.set_yticks(range(len(tau_vals)))
        ax.set_yticklabels([f"{t:.3f}" for t in tau_vals])
        ax.set_xlabel("min_support")
        ax.set_ylabel("tau (threshold)")
        ax.set_title(f"Ablation: {scoring_method} scoring, {metric}")

        # Annotate cells
        for i in range(len(tau_vals)):
            for j in range(len(sup_vals)):
                text = ax.text(j, i, f"{matrix[i, j]:.3f}",
                               ha="center", va="center", fontsize=10,
                               color="white" if matrix[i, j] > 0.5 else "black")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", format="pdf")

        plt.show()

    def to_latex_table(self, top_n: int = 10) -> str:
        """Generate LaTeX table of top N configurations."""
        if not self.results:
            return "No results yet."

        # Sort by P(delta > 0)
        sorted_results = sorted(self.results, key=lambda x: x.p_delta_positive, reverse=True)[:top_n]

        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{Ablation study: Top configurations by detection rate.}")
        lines.append(r"\label{tab:ablations}")
        lines.append(r"\begin{tabular}{cclcccc}")
        lines.append(r"\toprule")
        lines.append(r"$\tau$ & min\_support & Scoring & \#Inv & P($\Delta > 0$) & AUROC & $p$-value \\")
        lines.append(r"\midrule")

        for r in sorted_results:
            sig = ""
            if r.wilcoxon_pvalue < 0.001:
                sig = "***"
            elif r.wilcoxon_pvalue < 0.01:
                sig = "**"
            elif r.wilcoxon_pvalue < 0.05:
                sig = "*"

            lines.append(
                f"{r.tau:.3f} & {r.min_support} & {r.scoring_method} & "
                f"{r.n_invariants} & {r.p_delta_positive:.3f} & "
                f"{r.auroc:.3f} & {r.wilcoxon_pvalue:.2e}{sig} \\\\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        return "\n".join(lines)


def run_ablation_study(
    secure_activations: torch.Tensor,
    vulnerable_activations: torch.Tensor,
    tau_values: List[float] = [0.90, 0.95, 0.99, 0.995],
    min_support_values: List[int] = [50, 100, 200, 500],
    scoring_methods: List[str] = ["weighted", "unweighted", "max"],
    save_path: Optional[str] = None,
) -> Tuple[List[AblationResult], str, str]:
    """
    Convenience function to run full ablation study.

    Returns:
        results: List of AblationResult objects
        summary: Formatted text summary
        latex: LaTeX table code
    """
    ablation = InvariantAblation(
        secure_activations,
        vulnerable_activations,
        paired=True,
    )

    results = ablation.run_grid_search(
        tau_values=tau_values,
        min_support_values=min_support_values,
        scoring_methods=scoring_methods,
    )

    summary = ablation.print_summary()
    latex = ablation.to_latex_table()

    if save_path:
        ablation.plot_heatmap(save_path=save_path)

    return results, summary, latex
