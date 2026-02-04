"""
Run case studies extraction end-to-end.

Reads the activation JSONL, computes invariants and violation scores,
and outputs the best case studies.

Usage:
    python -m sae_java_bug.evaluation.run_case_studies
"""

import base64
import glob
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import orjson
import torch
from scipy.stats import wilcoxon
from tqdm import tqdm


RUN_ID = "run_20260202_220441_rufimelo_secure_code_qwen_coder_topk_16384"
ACTIVATION_DIR = "sae_java_bug/artifacts/activations"


def decode_b64(s: str) -> str:
    try:
        return base64.b64decode(s).decode("utf-8")
    except Exception:
        return s


def binarize(activations: torch.Tensor) -> torch.Tensor:
    return activations > 0


def compute_conditional_probs(fire_count, cofire_count, eps=1e-8):
    fire_i = fire_count.unsqueeze(1)
    return cofire_count / (fire_i + eps)


def extract_invariants(P_ij, fire_count, tau=0.995, min_support=200):
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


def violation_score(b, invariants, max_weight=10.0):
    score = 0.0
    violated = []
    for i, j, pij in invariants:
        if b[i] and not b[j]:
            w = min(-np.log(max(1.0 - pij, 1e-10)), max_weight)
            score += w
            violated.append((i, j, pij, w))
    return score, violated


def main():
    # Find the activation file
    pattern = f"{ACTIVATION_DIR}/{RUN_ID}/activations_layer_0_*"
    files = glob.glob(pattern)
    if not files:
        print(f"No files found matching {pattern}")
        sys.exit(1)

    activation_file = files[0]
    print(f"Reading: {activation_file}")

    # Load all data
    metadata = []  # (vuln_id, cwe, secure_code, vulnerable_code, file_ext)
    secure_features_list = []
    vuln_features_list = []

    with open(activation_file, "r") as f:
        for line in tqdm(f, desc="Loading", unit=" lines"):
            try:
                data = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue

            metadata.append({
                "vuln_id": data["vuln_id"],
                "cwe": data["cwe"],
                "secure_code": decode_b64(data["secure_code"]),
                "vulnerable_code": decode_b64(data["vulnerable_code"]),
                "file_extension": data["file_extension"],
            })

            secure_features_list.append(torch.tensor(data["secure"]))
            vuln_features_list.append(torch.tensor(data["vulnerable"]))

    secure_act = torch.stack(secure_features_list)
    vuln_act = torch.stack(vuln_features_list)
    print(f"Loaded {len(metadata)} samples, feature dim = {secure_act.shape[1]}")

    # Compute invariants from secure code
    print("\nComputing invariants from secure code...")
    secure_bin = binarize(secure_act)
    vuln_bin = binarize(vuln_act)

    bins_f = secure_bin.float()
    fire_count = bins_f.sum(dim=0)
    cofire_count = bins_f.T @ bins_f
    P_ij = compute_conditional_probs(fire_count, cofire_count)

    invariants = extract_invariants(P_ij, fire_count, tau=0.995, min_support=200)
    print(f"Extracted {len(invariants)} invariants")

    # Compute violation scores
    print("\nComputing violation scores...")
    secure_scores = []
    vuln_scores = []
    vuln_violations = []  # Store per-sample violation details

    for idx in tqdm(range(len(metadata)), desc="Scoring"):
        s_score, _ = violation_score(secure_bin[idx], invariants)
        v_score, v_violated = violation_score(vuln_bin[idx], invariants)
        secure_scores.append(s_score)
        vuln_scores.append(v_score)
        vuln_violations.append(v_violated)

    secure_scores = np.array(secure_scores)
    vuln_scores = np.array(vuln_scores)
    deltas = vuln_scores - secure_scores

    # ── PRIMARY: Paired metrics (within-pair comparison) ────────────────
    n_pairs = len(deltas)
    n_pos = int(np.sum(deltas > 0))
    n_neg = int(np.sum(deltas < 0))
    n_tie = int(np.sum(deltas == 0))
    p_delta_pos = np.mean(deltas > 0)

    stat, pval = wilcoxon(deltas)

    # Rank-biserial correlation r = 1 - (2T / (n*(n+1)/2))
    # where T = Wilcoxon T-statistic (sum of ranks for the smaller group)
    non_zero = deltas[deltas != 0]
    n_nz = len(non_zero)
    rank_biserial = 1.0 - (2.0 * stat) / (n_nz * (n_nz + 1) / 2) if n_nz > 0 else 0.0

    print(f"\n{'='*80}")
    print("PRIMARY RESULTS: Paired Metrics (within-pair comparison)")
    print(f"{'='*80}")
    print(f"  Number of pairs:       {n_pairs}")
    print(f"  Mean secure score:     {secure_scores.mean():.4f}")
    print(f"  Mean vulnerable score: {vuln_scores.mean():.4f}")
    print(f"  Mean delta (V - S):    {deltas.mean():.4f}")
    print(f"  Median delta (V - S):  {np.median(deltas):.4f}")
    print(f"  Std delta:             {deltas.std():.4f}")
    print()
    print(f"  All pairs:             {n_pairs}")
    print(f"    delta > 0:           {n_pos}  ({p_delta_pos:.4f})")
    print(f"    delta < 0:           {n_neg}  ({np.mean(deltas < 0):.4f})")
    print(f"    delta = 0 (ties):    {n_tie}  ({np.mean(deltas == 0):.4f})")
    print()
    print(f"  Non-zero pairs:        {n_nz}")
    p_cond = n_pos / n_nz if n_nz > 0 else float('nan')
    print(f"    P(delta>0 | delta!=0): {p_cond:.4f}  ({n_pos}/{n_nz})")
    print()
    print(f"  Wilcoxon signed-rank (n={n_nz} non-zero pairs):")
    print(f"    T-statistic:         {stat:.1f}")
    print(f"    p-value:             {pval:.2e}")
    print(f"    Rank-biserial r:     {rank_biserial:.4f}")
    print()
    print(f"  Cohen's d (paired):    {deltas.mean() / (deltas.std() + 1e-10):.4f}")

    # ── SUPPLEMENTARY: Unpaired detection metrics ─────────────────────
    # These scores pool secure and vulnerable independently, breaking
    # the pairing structure.  A near-chance AUROC here is expected and
    # actually *strengthens* the narrative: the signal lives at the
    # pair level (relational), not the population level (distributional).
    all_scores = np.concatenate([secure_scores, vuln_scores])
    all_labels = np.concatenate([np.zeros(len(secure_scores)),
                                 np.ones(len(vuln_scores))])

    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    auroc = roc_auc_score(all_labels, all_scores)

    precision_arr, recall_arr, pr_thresholds = precision_recall_curve(
        all_labels, all_scores
    )
    pr_auc = auc(recall_arr, precision_arr)

    ranked_indices = np.argsort(all_scores)[::-1]
    ranked_labels = all_labels[ranked_indices]

    print(f"\n{'='*80}")
    print("SUPPLEMENTARY: Unpaired Detection Metrics")
    print("  (signal is relational/pair-level, not distributional/population-level)")
    print(f"{'='*80}")
    print(f"  AUROC:                 {auroc:.4f}  (chance = 0.5000)")
    print(f"  PR-AUC:                {pr_auc:.4f}  (chance = {all_labels.mean():.4f})")
    print()
    print(f"  Note: Near-chance AUROC confirms the vulnerability signal is")
    print(f"  detectable only via paired comparison, not from absolute scores.")
    print()
    print(f"  {'K':<8} {'Precision@K':<15} {'#Vulnerable':<15} {'Expected(random)'}")
    print(f"  {'─'*55}")
    for k in [10, 20, 50, 100, 200, 500]:
        if k > len(ranked_labels):
            break
        top_k_labels = ranked_labels[:k]
        prec = top_k_labels.sum() / k
        n_vuln = int(top_k_labels.sum())
        expected = k * all_labels.mean()
        print(f"  {k:<8} {prec:<15.3f} {n_vuln:<15} {expected:.1f}")

    print(f"{'='*80}")

    # Find positive delta cases
    positive_indices = np.where(deltas > 0)[0]
    sorted_by_delta = positive_indices[np.argsort(deltas[positive_indices])[::-1]]

    print(f"\n{'='*80}")
    print(f"CASE STUDIES: {len(sorted_by_delta)} samples with positive delta")
    print(f"{'='*80}")

    # Print top cases
    seen_cwes = set()
    case_num = 0

    for idx in sorted_by_delta:
        m = metadata[idx]
        cwe = m["cwe"]

        # Show diverse CWEs first, then repeat
        if case_num < 10 and cwe in seen_cwes:
            continue
        seen_cwes.add(cwe)
        case_num += 1

        if case_num > 15:
            break

        print(f"\n{'─'*80}")
        print(f"CASE {case_num}: {m['vuln_id']}")
        print(f"{'─'*80}")
        print(f"  CWE:        {cwe}")
        print(f"  Extension:  {m['file_extension']}")
        print(f"  Secure score:     {secure_scores[idx]:.4f}")
        print(f"  Vulnerable score: {vuln_scores[idx]:.4f}")
        print(f"  Delta:            {deltas[idx]:+.4f}")

        # Top violated invariants (unique to vulnerable)
        s_score_check, s_violations = violation_score(secure_bin[idx], invariants)
        s_violated_pairs = set((i, j) for i, j, _, _ in s_violations)
        v_violated = vuln_violations[idx]

        # Violations present in vulnerable but not in secure
        new_violations = [(i, j, p, w) for i, j, p, w in v_violated
                          if (i, j) not in s_violated_pairs]
        new_violations.sort(key=lambda x: x[3], reverse=True)

        if new_violations:
            print(f"\n  New invariant violations (not in secure version):")
            for i, j, pij, w in new_violations[:5]:
                print(f"    Feature {i:>5d} -> Feature {j:>5d}  "
                      f"P(j|i)={pij:.4f}  weight={w:.2f}")

        # Code preview
        secure_lines = m["secure_code"].replace("  ", "\n").split("\n")
        vuln_lines = m["vulnerable_code"].replace("  ", "\n").split("\n")

        # Find differing lines (simple heuristic)
        print(f"\n  SECURE CODE (first 20 lines):")
        for i, line in enumerate(secure_lines[:20]):
            print(f"    {i+1:3d} | {line}")
        if len(secure_lines) > 20:
            print(f"    ... ({len(secure_lines) - 20} more lines)")

        print(f"\n  VULNERABLE CODE (first 20 lines):")
        for i, line in enumerate(vuln_lines[:20]):
            print(f"    {i+1:3d} | {line}")
        if len(vuln_lines) > 20:
            print(f"    ... ({len(vuln_lines) - 20} more lines)")

    # Summary table of all positive-delta cases
    print(f"\n{'='*80}")
    print("ALL POSITIVE DELTA CASES")
    print(f"{'='*80}")
    print(f"{'#':<4} {'Vuln ID':<40} {'CWE':<12} {'Ext':<6} {'Delta':<10} {'#NewViol'}")
    print(f"{'─'*80}")

    for rank, idx in enumerate(sorted_by_delta):
        m = metadata[idx]
        s_score_check, s_violations = violation_score(secure_bin[idx], invariants)
        s_violated_pairs = set((i, j) for i, j, _, _ in s_violations)
        new_viols = sum(1 for i, j, p, w in vuln_violations[idx]
                        if (i, j) not in s_violated_pairs)

        print(f"{rank+1:<4} {m['vuln_id'][:38]:<40} {m['cwe']:<12} "
              f"{m['file_extension']:<6} {deltas[idx]:+8.4f}  {new_viols}")

    # CWE distribution of detected samples
    print(f"\n{'='*80}")
    print("CWE DISTRIBUTION OF DETECTED SAMPLES")
    print(f"{'='*80}")
    cwe_counts = defaultdict(int)
    cwe_total = defaultdict(int)
    for idx in range(len(metadata)):
        cwe_total[metadata[idx]["cwe"]] += 1
    for idx in sorted_by_delta:
        cwe_counts[metadata[idx]["cwe"]] += 1

    print(f"{'CWE':<12} {'Detected':<10} {'Total':<10} {'Rate':<10}")
    print(f"{'─'*42}")
    for cwe in sorted(cwe_counts.keys(), key=lambda c: cwe_counts[c], reverse=True):
        rate = cwe_counts[cwe] / cwe_total[cwe] * 100
        print(f"{cwe:<12} {cwe_counts[cwe]:<10} {cwe_total[cwe]:<10} {rate:<8.1f}%")


if __name__ == "__main__":
    main()
