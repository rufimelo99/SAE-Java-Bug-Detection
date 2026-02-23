# ── Offline: train the scanner once (on your cached JSONL activations) ─────
from sae_java_bug.evaluation.feature_cwe_correlation import (
    load_activations_from_jsonl,
    compute_feature_cwe_correlations,
    CIVulnerabilityScanner,
)

import os
if not os.path.exists("scanner.pkl"):
    secure, vuln, cwe_labels = load_activations_from_jsonl(
        "run_20260218_134529_vulnerable_code_qwen_coder_standard_16384_10M",
        layer=11,
    )
    result     = compute_feature_cwe_correlations(secure, vuln, cwe_labels)
    enrichment = result.feature_enrichment(secure, vuln, cwe_labels)

    scanner = CIVulnerabilityScanner.train(
        secure, vuln, cwe_labels,
        result=result, enrichment=enrichment,
        top_k_features=100, threshold=0.5,
    )
    scanner.save("scanner.pkl")


# ── CI/CD: load model once per pipeline run ──────────────────────────────
from sae_java_bug.evaluation.activation_extractor import ActivationExtractor
from sae_java_bug.sparse_autoencoders.schemas import QWEN_CODER_7B_VULNEABLE_CODE_STD_10M_CONFIG

extractor = ActivationExtractor.from_config(
    QWEN_CODER_7B_VULNEABLE_CODE_STD_10M_CONFIG   # layer=11, Qwen2.5-7B, SAE from HF
).load()

scanner = CIVulnerabilityScanner.load("scanner.pkl")


# ── Per-PR: one call per changed function ────────────────────────────────
safe_fn = """
import org.apache.commons.text.StringEscapeUtils;

public String renderGreeting(String name) {
    String safeName = StringEscapeUtils.escapeHtml4(name);
    return "<html><body>Welcome " + safeName + "</body></html>";
}
"""
vuln = """
public String renderGreeting(String name) {
    return "<html><body>Welcome " + name + "</body></html>";
}
"""

report = scanner.scan_code(safe_fn, vuln, extractor)
report.print()

import json, sys
print(json.dumps(report.to_dict(), indent=2))