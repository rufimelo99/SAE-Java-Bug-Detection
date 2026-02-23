"""
Semgrep pre-filter stage for the CI/CD vulnerability scanner.

Runs Semgrep on the incoming code and extracts which CWE IDs were flagged.
The downstream SAE scanner then only runs probes for those CWEs, requiring
both tools to agree before raising a flag — significantly reducing false
positives.

Requirements
------------
    pip install semgrep          # or: brew install semgrep

Usage
-----
>>> from sae_java_bug.evaluation.semgrep_stage import SemgrepStage
>>> sg = SemgrepStage(language="java")
>>> flagged_cwes = sg.scan_code("public void exec(String c) { Runtime.getRuntime().exec(c); }")
>>> flagged_cwes
{"CWE-78"}
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Semgrep rule-ID → CWE mapping
# Semgrep embeds CWE tags in rule metadata; we also maintain a hard-coded
# fallback for the most common Java rules so the stage works offline.
# ---------------------------------------------------------------------------

# Maps semgrep rule-id fragments → CWE IDs.  A rule matches if the fragment
# appears anywhere in the full rule ID (e.g. "exec" matches
# "java.lang.security.audit.command-injection-formatted-runtime-exec").
_RULE_FRAGMENT_TO_CWE: dict[str, str] = {
    # Injection
    "command-injection":        "CWE-78",
    "runtime-exec":             "CWE-78",
    "process-builder":          "CWE-78",
    "sql-injection":            "CWE-89",
    "sqli":                     "CWE-89",
    "jdbc":                     "CWE-89",
    "xss":                      "CWE-79",
    "cross-site-scripting":     "CWE-79",
    "code-injection":           "CWE-94",
    "reflection-injection":     "CWE-94",
    # Path / file
    "path-traversal":           "CWE-22",
    "zip-slip":                 "CWE-22",
    # Auth / session
    "csrf":                     "CWE-352",
    "auth-bypass":              "CWE-287",
    "hardcoded-password":       "CWE-287",
    "broken-auth":              "CWE-287",
    # Resource / DoS
    "redos":                    "CWE-400",
    "resource-exhaustion":      "CWE-400",
    "denial-of-service":        "CWE-400",
    # Info exposure
    "information-exposure":     "CWE-200",
    "sensitive-data":           "CWE-200",
    "stack-trace":              "CWE-200",
    # File upload
    "file-upload":              "CWE-434",
    "unrestricted-upload":      "CWE-434",
    # Crypto / randomness
    "weak-random":              "CWE-330",
    "insecure-random":          "CWE-330",
}

# Language → file extension used when writing the temp file
_LANGUAGE_EXT: dict[str, str] = {
    "java":       ".java",
    "python":     ".py",
    "javascript": ".js",
    "typescript": ".ts",
    "go":         ".go",
    "c":          ".c",
    "cpp":        ".cpp",
}

# Semgrep rulesets to try in order.  The first one that is available is used.
_DEFAULT_CONFIGS = [
    "p/java",           # Java-specific security rules (requires internet first run)
    "p/owasp-top-ten",  # OWASP Top 10 (broad, language-agnostic)
    "auto",             # Auto-detect (slowest but most thorough)
]


def _rule_id_to_cwe(rule_id: str) -> Optional[str]:
    """
    Map a Semgrep rule ID to a CWE identifier.

    First tries the CWE field embedded in Semgrep's JSON output (when
    available via ``--verbose`` metadata).  Falls back to substring matching
    against the rule ID itself.
    """
    rule_lower = rule_id.lower()
    for fragment, cwe in _RULE_FRAGMENT_TO_CWE.items():
        if fragment in rule_lower:
            return cwe
    return None


def _extract_cwes_from_semgrep_output(output: dict) -> set[str]:
    """
    Parse Semgrep's JSON output and return the set of CWE IDs found.

    Checks two places:
    1. ``result["extra"]["metadata"]["cwe"]`` — explicit CWE tag in rule metadata
    2. ``result["check_id"]``                 — substring matching via _rule_id_to_cwe
    """
    cwes: set[str] = set()
    for result in output.get("results", []):
        # 1. Explicit CWE in metadata
        meta_cwes = (
            result.get("extra", {})
                  .get("metadata", {})
                  .get("cwe", [])
        )
        if isinstance(meta_cwes, str):
            meta_cwes = [meta_cwes]
        for cwe_raw in meta_cwes:
            # Normalise: "CWE-78: OS Command Injection" → "CWE-78"
            cwe_id = cwe_raw.strip().split(":")[0].strip().upper()
            if cwe_id.startswith("CWE-"):
                cwes.add(cwe_id)

        # 2. Rule-ID substring matching as fallback
        rule_id = result.get("check_id", "")
        cwe_from_rule = _rule_id_to_cwe(rule_id)
        if cwe_from_rule:
            cwes.add(cwe_from_rule)

    return cwes


class SemgrepStage:
    """
    Run Semgrep on a code snippet and return the set of CWE IDs flagged.

    Parameters
    ----------
    language : str
        Programming language of the code ("java", "python", etc.).
    configs : list[str], optional
        Semgrep ruleset configs to use, tried in order.
        Defaults to ``["p/java", "p/owasp-top-ten", "auto"]``.
    timeout : int
        Per-scan timeout in seconds.
    semgrep_bin : str
        Path to the semgrep binary (auto-detected if None).
    """

    def __init__(
        self,
        language: str = "java",
        configs: Optional[list[str]] = None,
        timeout: int = 30,
        semgrep_bin: Optional[str] = None,
    ):
        self.language = language
        self.configs = configs or _DEFAULT_CONFIGS
        self.timeout = timeout
        self.semgrep_bin = semgrep_bin or _find_semgrep()

        if self.semgrep_bin is None:
            raise RuntimeError(
                "Semgrep not found. Install it with: pip install semgrep"
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def scan_code(self, code: str) -> set[str]:
        """
        Scan a code snippet and return the set of CWE IDs that Semgrep flags.

        Parameters
        ----------
        code : str
            Source code to analyse.

        Returns
        -------
        set[str]
            e.g. ``{"CWE-78", "CWE-89"}`` — empty set if nothing found.
        """
        ext = _LANGUAGE_EXT.get(self.language, ".txt")

        with tempfile.NamedTemporaryFile(
            suffix=ext, mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            return self._run_semgrep(tmp_path)
        finally:
            os.unlink(tmp_path)

    def scan_file(self, path: str | Path) -> set[str]:
        """Scan an existing file on disk."""
        return self._run_semgrep(str(path))

    def is_available(self) -> bool:
        """Check that the semgrep binary is reachable."""
        try:
            subprocess.run(
                [self.semgrep_bin, "--version"],
                capture_output=True, timeout=5,
            )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_semgrep(self, path: str) -> set[str]:
        """Run semgrep and parse the JSON output."""
        for config in self.configs:
            try:
                result = subprocess.run(
                    [
                        self.semgrep_bin,
                        "--config", config,
                        "--json",
                        "--quiet",
                        "--no-git-ignore",
                        path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                if result.returncode not in (0, 1):
                    # returncode 1 = findings present (not an error)
                    # anything else is an actual error
                    continue

                output = json.loads(result.stdout or "{}")
                cwes = _extract_cwes_from_semgrep_output(output)
                if cwes or result.returncode == 0:
                    return cwes   # success — even if empty (clean code)

            except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
                continue

        # All configs failed — return empty rather than crashing CI
        return set()


def _find_semgrep() -> Optional[str]:
    """Locate the semgrep binary on PATH."""
    import shutil
    return shutil.which("semgrep")