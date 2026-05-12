#!/usr/bin/env python3
"""
External validation of steering results using SAST tools and static analysis.

Mitigates circularity in probe-based steering evaluation by testing generated code
against external tools:

1. Semgrep: check for guard-related rules
2. clang-tidy: compiler warnings and issues
3. cppcheck: C-specific static analysis
4. Compilation success rate
5. Explicit guard-token counts (length-normalized)
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


class ExternalValidation:
    """Validate steering results with external tools."""

    @staticmethod
    def run_semgrep(code_file: Path, rules: str = "p/security-audit") -> Dict:
        """
        Run Semgrep on generated code.

        Args:
            code_file: path to C code file
            rules: Semgrep rule pattern (e.g., 'p/security-audit')

        Returns:
            dict with issue counts by severity
        """
        try:
            result = subprocess.run(
                ["semgrep", "--json", "--config", rules, str(code_file)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = json.loads(result.stdout) if result.stdout else {}
            findings = output.get("results", [])

            severities = {}
            for finding in findings:
                severity = finding.get("extra", {}).get("severity", "unknown")
                severities[severity] = severities.get(severity, 0) + 1

            return {
                "total_issues": len(findings),
                "by_severity": severities,
                "success": True,
            }
        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            json.JSONDecodeError,
        ) as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def run_clang_tidy(code_file: Path, checks: str = "*") -> Dict:
        """
        Run clang-tidy on generated code.

        Args:
            code_file: path to C code file
            checks: clang-tidy checks to run

        Returns:
            dict with warning counts
        """
        try:
            result = subprocess.run(
                ["clang-tidy", "-checks", checks, "--", str(code_file)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout + result.stderr

            # Count warnings
            warning_count = len(re.findall(r"warning:", output))
            error_count = len(re.findall(r"error:", output))

            return {
                "warnings": warning_count,
                "errors": error_count,
                "success": True,
            }
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def run_cppcheck(code_file: Path, std: str = "c11") -> Dict:
        """
        Run cppcheck on generated code.

        Args:
            code_file: path to C code file
            std: C standard (e.g., 'c11')

        Returns:
            dict with issue counts by type
        """
        try:
            result = subprocess.run(
                ["cppcheck", "--std=" + std, "--json", str(code_file)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = json.loads(result.stdout) if result.stdout else {}
            results = output.get("results", [])

            issue_types = {}
            for issue in results:
                severity = issue.get("severity", "unknown")
                issue_types[severity] = issue_types.get(severity, 0) + 1

            return {
                "total_issues": len(results),
                "by_type": issue_types,
                "success": True,
            }
        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            json.JSONDecodeError,
        ) as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def test_compilation(code_file: Path, compiler: str = "gcc") -> Dict:
        """
        Test whether generated code compiles.

        Args:
            code_file: path to C code file
            compiler: compiler to use (gcc or clang)

        Returns:
            dict with compile status
        """
        try:
            result = subprocess.run(
                [
                    compiler,
                    "-c",
                    "-Wall",
                    "-Wextra",
                    str(code_file),
                    "-o",
                    "/tmp/test.o",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            errors = len(re.findall(r"error:", result.stderr))
            warnings = len(re.findall(r"warning:", result.stderr))

            return {
                "compiles": result.returncode == 0,
                "errors": errors,
                "warnings": warnings,
                "success": True,
            }
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def count_guard_tokens(code: str, tokenizer) -> Dict:
        """
        Count defensive constructs (guards, checks) in code.

        Args:
            code: C code string
            tokenizer: tokenizer for consistent counting

        Returns:
            dict with guard counts and ratios
        """
        guard_patterns = {
            "if_checks": len(re.findall(r"\bif\s*\(", code)),
            "null_checks": len(re.findall(r"\bNULL\b|\bnull\b", code)),
            "size_checks": len(re.findall(r"\bsizeof\s*\(|\bstrlen\s*\(", code)),
            "assertions": len(re.findall(r"\bassert\s*\(", code)),
            "bounds_checks": len(re.findall(r"<\s*\w+\s*\|\s*>\s*\w+", code)),
        }

        total_guards = sum(guard_patterns.values())
        total_tokens = len(tokenizer.encode(code))

        return {
            "counts": guard_patterns,
            "total": total_guards,
            "tokens": total_tokens,
            "guards_per_token": total_guards / (total_tokens + 1e-8),
        }


def validate_steering_quality(
    steered_codes: List[str],
    baseline_codes: List[str],
    tokenizer,
    output_dir: Path = Path("/tmp/validation_results"),
) -> Dict:
    """
    Compare steered and baseline code using external validation.

    Args:
        steered_codes: list of steering-influenced generated code snippets
        baseline_codes: list of unsteered baseline code snippets
        tokenizer: tokenizer
        output_dir: directory to write temporary code files and results

    Returns:
        dict with validation results
    """
    output_dir.mkdir(exist_ok=True)
    results = {
        "semgrep": {"baseline": [], "steered": []},
        "clang_tidy": {"baseline": [], "steered": []},
        "cppcheck": {"baseline": [], "steered": []},
        "compilation": {"baseline": [], "steered": []},
        "guard_tokens": {"baseline": [], "steered": []},
    }

    # Validate baseline
    for i, code in enumerate(baseline_codes):
        code_file = output_dir / f"baseline_{i}.c"
        code_file.write_text(code)

        results["semgrep"]["baseline"].append(ExternalValidation.run_semgrep(code_file))
        results["clang_tidy"]["baseline"].append(
            ExternalValidation.run_clang_tidy(code_file)
        )
        results["cppcheck"]["baseline"].append(
            ExternalValidation.run_cppcheck(code_file)
        )
        results["compilation"]["baseline"].append(
            ExternalValidation.test_compilation(code_file)
        )
        results["guard_tokens"]["baseline"].append(
            ExternalValidation.count_guard_tokens(code, tokenizer)
        )

    # Validate steered
    for i, code in enumerate(steered_codes):
        code_file = output_dir / f"steered_{i}.c"
        code_file.write_text(code)

        results["semgrep"]["steered"].append(ExternalValidation.run_semgrep(code_file))
        results["clang_tidy"]["steered"].append(
            ExternalValidation.run_clang_tidy(code_file)
        )
        results["cppcheck"]["steered"].append(
            ExternalValidation.run_cppcheck(code_file)
        )
        results["compilation"]["steered"].append(
            ExternalValidation.test_compilation(code_file)
        )
        results["guard_tokens"]["steered"].append(
            ExternalValidation.count_guard_tokens(code, tokenizer)
        )

    # Summarize
    summary = {
        "num_samples": len(steered_codes),
        "semgrep_issue_reduction": (
            np.mean([r.get("total_issues", 0) for r in results["semgrep"]["baseline"]])
            - np.mean([r.get("total_issues", 0) for r in results["semgrep"]["steered"]])
        ),
        "compilation_rate_baseline": sum(
            1 for r in results["compilation"]["baseline"] if r.get("compiles", False)
        )
        / len(results["compilation"]["baseline"]),
        "compilation_rate_steered": sum(
            1 for r in results["compilation"]["steered"] if r.get("compiles", False)
        )
        / len(results["compilation"]["steered"]),
        "guard_tokens_increase": (
            np.mean([r["total"] for r in results["guard_tokens"]["steered"]])
            - np.mean([r["total"] for r in results["guard_tokens"]["baseline"]])
        ),
        "guard_per_token_baseline": np.mean(
            [r["guards_per_token"] for r in results["guard_tokens"]["baseline"]]
        ),
        "guard_per_token_steered": np.mean(
            [r["guards_per_token"] for r in results["guard_tokens"]["steered"]]
        ),
    }

    return {
        "detailed_results": results,
        "summary": summary,
    }


def main():
    """Run external validation."""
    print("=" * 70)
    print("EXTERNAL VALIDATION OF STEERING")
    print("=" * 70)

    print("\nThis script validates steered code generation using external tools:")
    print("  - Semgrep: security rule violations")
    print("  - clang-tidy: compiler warnings")
    print("  - cppcheck: C-specific static analysis")
    print("  - Compilation: does it compile?")
    print("  - Guard tokens: explicit defensive construct counts")

    print("\nUsage:")
    print(
        """
    from pathlib import Path
    import json

    steered_codes = [
        "int main() { if (x != NULL) { process(x); } }",
        "int main() { if (len < MAX) { strcpy(dest, src); } }",
    ]
    baseline_codes = [
        "int main() { process(x); }",
        "int main() { strcpy(dest, src); }",
    ]

    results = validate_steering_quality(
        steered_codes, baseline_codes, tokenizer, output_dir=Path('/tmp/validation')
    )

    print(f"Compilation rate - baseline: {results['summary']['compilation_rate_baseline']:.1%}")
    print(f"Compilation rate - steered: {results['summary']['compilation_rate_steered']:.1%}")
    print(f"Guard tokens increase: {results['summary']['guard_tokens_increase']:.1f}")
    print(f"Issue reduction: {results['summary']['semgrep_issue_reduction']:.1f}")
    """
    )

    print("\nRequired tools (install with pip/apt):")
    print("  - semgrep: pip install semgrep")
    print("  - clang-tidy: apt install clang-tools")
    print("  - cppcheck: apt install cppcheck")
    print("  - gcc/clang: apt install build-essential")


if __name__ == "__main__":
    main()
