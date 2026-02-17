"""
Case Studies Module

Identifies and analyzes the best-detected vulnerabilities for
qualitative analysis and paper examples.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import base64


@dataclass
class CaseStudy:
    """A single case study for a detected vulnerability."""
    sample_idx: int
    vuln_id: str
    cwe: str

    # Scores
    secure_score: float
    vulnerable_score: float
    delta: float  # vulnerable - secure

    # Code snippets
    secure_code: str
    vulnerable_code: str

    # Feature analysis
    n_total_violations: int
    top_violated_invariants: List[Tuple[int, int, float]]  # (i, j, P_ij)
    violated_feature_indices: List[int]


@dataclass
class FeatureInfo:
    """Information about a violated feature."""
    feature_idx: int
    secure_value: float
    vulnerable_value: float
    delta: float
    top_activating_samples: List[str]  # Code snippets that maximally activate this feature


class CaseStudyFinder:
    """
    Identifies and analyzes case studies for the paper.

    Finds samples where:
    1. Delta > 0 (vulnerable has higher violation score)
    2. The violation is interpretable
    """

    def __init__(
        self,
        secure_activations: torch.Tensor,
        vulnerable_activations: torch.Tensor,
        secure_scores: np.ndarray,
        vulnerable_scores: np.ndarray,
        invariants: List[Tuple[int, int, float]],
        vuln_ids: List[str],
        cwes: List[str],
        secure_codes: List[str],
        vulnerable_codes: List[str],
    ):
        """
        Initialize case study finder.

        Args:
            secure_activations: Shape (n_samples, n_features)
            vulnerable_activations: Shape (n_samples, n_features)
            secure_scores: Violation scores for secure samples
            vulnerable_scores: Violation scores for vulnerable samples
            invariants: List of (i, j, P_ij) tuples
            vuln_ids: Vulnerability IDs for each sample
            cwes: CWE labels for each sample
            secure_codes: Secure code snippets
            vulnerable_codes: Vulnerable code snippets
        """
        self.secure_act = secure_activations
        self.vulnerable_act = vulnerable_activations
        self.secure_scores = np.array(secure_scores)
        self.vulnerable_scores = np.array(vulnerable_scores)
        self.invariants = invariants
        self.vuln_ids = vuln_ids
        self.cwes = cwes
        self.secure_codes = secure_codes
        self.vulnerable_codes = vulnerable_codes

        self.deltas = self.vulnerable_scores - self.secure_scores

        # Build invariant lookup
        self.invariant_by_trigger = {}  # i -> list of (j, P_ij)
        for i, j, pij in invariants:
            if i not in self.invariant_by_trigger:
                self.invariant_by_trigger[i] = []
            self.invariant_by_trigger[i].append((j, pij))

    def find_positive_deltas(self, min_delta: float = 0.0) -> List[int]:
        """Find indices where vulnerable score > secure score."""
        return np.where(self.deltas > min_delta)[0].tolist()

    def get_top_cases(
        self,
        n: int = 20,
        min_delta: float = 0.0,
        unique_cwes: bool = True,
    ) -> List[int]:
        """
        Get indices of top N cases by delta.

        Args:
            n: Number of cases to return
            min_delta: Minimum delta threshold
            unique_cwes: If True, select diverse CWEs
        """
        # Sort by delta descending
        indices = np.argsort(self.deltas)[::-1]

        selected = []
        seen_cwes = set()

        for idx in indices:
            if self.deltas[idx] <= min_delta:
                break

            if unique_cwes:
                cwe = self.cwes[idx]
                if cwe in seen_cwes:
                    continue
                seen_cwes.add(cwe)

            selected.append(idx)
            if len(selected) >= n:
                break

        return selected

    def analyze_case(self, idx: int) -> CaseStudy:
        """
        Analyze a single case in detail.

        Identifies which invariants were violated and why.
        """
        secure_bin = (self.secure_act[idx] > 0)
        vulnerable_bin = (self.vulnerable_act[idx] > 0)

        # Find violated invariants
        violations = []
        violated_features = set()

        for i, j, pij in self.invariants:
            # Invariant violated if: i fires but j doesn't (in vulnerable)
            # AND this is different from secure
            vuln_violated = vulnerable_bin[i].item() and not vulnerable_bin[j].item()
            sec_violated = secure_bin[i].item() and not secure_bin[j].item()

            if vuln_violated and not sec_violated:
                violations.append((i, j, pij))
                violated_features.add(i)
                violated_features.add(j)

        # Sort violations by weight
        violations.sort(key=lambda x: -np.log(1 - x[2] + 1e-10), reverse=True)

        return CaseStudy(
            sample_idx=idx,
            vuln_id=self.vuln_ids[idx],
            cwe=self.cwes[idx],
            secure_score=float(self.secure_scores[idx]),
            vulnerable_score=float(self.vulnerable_scores[idx]),
            delta=float(self.deltas[idx]),
            secure_code=self.secure_codes[idx],
            vulnerable_code=self.vulnerable_codes[idx],
            n_total_violations=len(violations),
            top_violated_invariants=violations[:10],  # Top 10 by weight
            violated_feature_indices=list(violated_features),
        )

    def get_case_studies(
        self,
        n: int = 5,
        unique_cwes: bool = True,
    ) -> List[CaseStudy]:
        """Get top N case studies with full analysis."""
        top_indices = self.get_top_cases(n=n * 2, unique_cwes=unique_cwes)  # Get extra to filter

        cases = []
        for idx in top_indices:
            case = self.analyze_case(idx)
            # Filter out cases with no clear violations
            if case.n_total_violations > 0:
                cases.append(case)
            if len(cases) >= n:
                break

        return cases

    def print_case_study(self, case: CaseStudy, max_code_lines: int = 30) -> str:
        """Format a case study for display."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"CASE STUDY: {case.vuln_id}")
        lines.append("=" * 80)
        lines.append(f"CWE: {case.cwe}")
        lines.append(f"Secure Score: {case.secure_score:.4f}")
        lines.append(f"Vulnerable Score: {case.vulnerable_score:.4f}")
        lines.append(f"Delta: {case.delta:.4f}")
        lines.append(f"Invariants Violated: {case.n_total_violations}")
        lines.append("")

        # Code comparison
        lines.append("-" * 40 + " SECURE CODE " + "-" * 40)
        secure_lines = case.secure_code.split("\n")[:max_code_lines]
        lines.extend(secure_lines)
        if len(case.secure_code.split("\n")) > max_code_lines:
            lines.append(f"... ({len(case.secure_code.split(chr(10))) - max_code_lines} more lines)")
        lines.append("")

        lines.append("-" * 40 + " VULNERABLE CODE " + "-" * 37)
        vulnerable_lines = case.vulnerable_code.split("\n")[:max_code_lines]
        lines.extend(vulnerable_lines)
        if len(case.vulnerable_code.split("\n")) > max_code_lines:
            lines.append(f"... ({len(case.vulnerable_code.split(chr(10))) - max_code_lines} more lines)")
        lines.append("")

        # Top violated invariants
        lines.append("-" * 40 + " TOP VIOLATIONS " + "-" * 38)
        for i, (feat_i, feat_j, pij) in enumerate(case.top_violated_invariants[:5]):
            weight = -np.log(1 - pij + 1e-10)
            lines.append(f"  {i+1}. Feature {feat_i} -> Feature {feat_j}")
            lines.append(f"     P(j|i) = {pij:.4f}, weight = {weight:.2f}")

        lines.append("=" * 80)

        text = "\n".join(lines)
        print(text)
        return text

    def print_summary(self, cases: List[CaseStudy]) -> str:
        """Print summary of all case studies."""
        lines = []
        lines.append("=" * 100)
        lines.append("CASE STUDIES SUMMARY")
        lines.append("=" * 100)
        lines.append("")
        lines.append(f"Found {len(self.find_positive_deltas())} samples with Delta > 0 "
                     f"({100*len(self.find_positive_deltas())/len(self.deltas):.1f}% of {len(self.deltas)} samples)")
        lines.append("")
        lines.append(f"{'#':<4} {'Vuln ID':<35} {'CWE':<12} {'Delta':<10} {'#Violations':<12}")
        lines.append("-" * 100)

        for i, case in enumerate(cases):
            lines.append(
                f"{i+1:<4} "
                f"{case.vuln_id[:33]:<35} "
                f"{case.cwe:<12} "
                f"{case.delta:<10.4f} "
                f"{case.n_total_violations:<12}"
            )

        lines.append("=" * 100)

        summary = "\n".join(lines)
        print(summary)
        return summary

    def to_latex_examples(self, cases: List[CaseStudy], n: int = 3) -> str:
        """Generate LaTeX code for case study examples."""
        lines = []

        for i, case in enumerate(cases[:n]):
            lines.append(r"\begin{figure}[t]")
            lines.append(r"\centering")
            lines.append(r"\begin{subfigure}[b]{0.48\textwidth}")
            lines.append(r"\begin{lstlisting}[language=Java,basicstyle=\tiny\ttfamily]")

            # Truncate code
            secure_preview = "\n".join(case.secure_code.split("\n")[:15])
            lines.append(secure_preview)

            lines.append(r"\end{lstlisting}")
            lines.append(r"\caption{Secure version}")
            lines.append(r"\end{subfigure}")
            lines.append(r"\hfill")
            lines.append(r"\begin{subfigure}[b]{0.48\textwidth}")
            lines.append(r"\begin{lstlisting}[language=Java,basicstyle=\tiny\ttfamily]")

            vulnerable_preview = "\n".join(case.vulnerable_code.split("\n")[:15])
            lines.append(vulnerable_preview)

            lines.append(r"\end{lstlisting}")
            lines.append(r"\caption{Vulnerable version}")
            lines.append(r"\end{subfigure}")
            lines.append(f"\\caption{{Case Study {i+1}: {case.cwe} ({case.vuln_id[:20]}...). "
                         f"Violation score increased by {case.delta:.2f}.}}")
            lines.append(f"\\label{{fig:case_{i+1}}}")
            lines.append(r"\end{figure}")
            lines.append("")

        return "\n".join(lines)


def find_case_studies(
    secure_activations: torch.Tensor,
    vulnerable_activations: torch.Tensor,
    secure_scores: np.ndarray,
    vulnerable_scores: np.ndarray,
    invariants: List[Tuple[int, int, float]],
    vuln_ids: List[str],
    cwes: List[str],
    secure_codes: List[str],
    vulnerable_codes: List[str],
    n_cases: int = 5,
) -> Tuple[List[CaseStudy], str]:
    """
    Convenience function to find and analyze case studies.

    Returns:
        cases: List of CaseStudy objects
        summary: Formatted text summary
    """
    finder = CaseStudyFinder(
        secure_activations=secure_activations,
        vulnerable_activations=vulnerable_activations,
        secure_scores=secure_scores,
        vulnerable_scores=vulnerable_scores,
        invariants=invariants,
        vuln_ids=vuln_ids,
        cwes=cwes,
        secure_codes=secure_codes,
        vulnerable_codes=vulnerable_codes,
    )

    cases = finder.get_case_studies(n=n_cases, unique_cwes=True)
    summary = finder.print_summary(cases)

    # Print individual cases
    for case in cases:
        finder.print_case_study(case)

    return cases, summary


def decode_code(encoded: str) -> str:
    """Decode base64-encoded code if needed."""
    try:
        return base64.b64decode(encoded).decode("utf-8")
    except:
        return encoded  # Already decoded
