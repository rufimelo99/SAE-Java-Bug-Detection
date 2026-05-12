"""
Attention Head Analysis: Do Mid-Layer Heads Focus on Body Tokens?

Tests whether certain attention heads systematically attend to body tokens
(middle of sequence) vs. structural tokens (start/end), and whether this
correlates with vulnerability signals.

Key insight: If attention heads focus body → final token, they might propagate
vulnerability information through the network.

Usage:
    conda run -n sae python attention_analysis.py

Output:
    artifacts/analysis/attention/
        attention_statistics.json        per-layer, per-head statistics
        body_vs_structural_focus.json    head classifications
        attention_summary_table.txt      human-readable summary
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
ARTIFACTS = Path(__file__).parents[2] / "artifacts"
OUTPUT_DIR = ARTIFACTS / "analysis" / "attention"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_RUN_DIR = (
    ARTIFACTS
    / "activations"
    / "raw_activations"
    / "vulnerable_code_qwen_coder_standard_16384_raw"
)

SOURCE_JSONL = (
    RAW_RUN_DIR / "activations_layer_0_raw_component_hidden_state_last_token.jsonl"
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42

# Sample a subset of pairs for attention analysis (full analysis is expensive)
N_SAMPLE_PAIRS = 50


def load_source_records(jsonl_path: Path, n_sample: int = N_SAMPLE_PAIRS):
    """Load sample vulnerable/secure code pairs."""
    records = []
    with jsonl_path.open() as f:
        for i, line in enumerate(f):
            if i >= n_sample:
                break
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            try:
                import base64

                secure_code = base64.b64decode(r["secure_code"]).decode(
                    "utf-8", errors="replace"
                )
                vuln_code = base64.b64decode(r["vulnerable_code"]).decode(
                    "utf-8", errors="replace"
                )
            except Exception:
                secure_code = r.get("secure_code", "")
                vuln_code = r.get("vulnerable_code", "")

            records.append(
                {
                    "cwe": r["cwe"],
                    "secure_code": secure_code,
                    "vulnerable_code": vuln_code,
                }
            )
    return records


def extract_attention_weights(
    code: str, tokenizer, model, device
) -> Dict[int, np.ndarray]:
    """
    Extract attention weights from model forward pass.

    Returns:
        {layer: [n_heads, seq_len, seq_len]} attention matrix
    """
    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    result = {}
    for layer_idx, attn in enumerate(outputs.attentions):
        # attn shape: [batch=1, n_heads, seq_len, seq_len]
        attn_np = attn[0].float().cpu().numpy()  # [n_heads, seq_len, seq_len]
        result[layer_idx] = attn_np

    return result


def classify_token_position(
    token_idx: int, seq_len: int, threshold: float = 0.2
) -> str:
    """
    Classify token as 'start' (first 20%), 'body' (middle 60%), or 'end' (last 20%).
    """
    pos_ratio = token_idx / seq_len if seq_len > 0 else 0
    if pos_ratio < threshold:
        return "start"
    elif pos_ratio > (1.0 - threshold):
        return "end"
    else:
        return "body"


def analyze_head_focus(attn_matrix: np.ndarray, seq_len: int) -> Dict:
    """
    For each attention head, determine what token positions it focuses on.

    Returns:
        {
            "body_focus": % of attention weight on body tokens,
            "start_focus": % on start tokens,
            "end_focus": % on end tokens,
            "target_categories": distribution of where each head attends from,
        }
    """
    n_heads, _, _ = attn_matrix.shape

    results = {
        "n_heads": n_heads,
        "seq_len": seq_len,
        "per_head": {},
    }

    for head_idx in range(n_heads):
        # attn_matrix[head_idx] is [seq_len, seq_len]
        # attn[i, j] = attention from position i to position j
        attn_head = attn_matrix[head_idx]  # [seq_len, seq_len]

        # For each source position, where does attention go?
        # Average across all source positions
        avg_attention_dist = attn_head.mean(axis=0)  # [seq_len]

        # Classify target positions
        body_focus = 0.0
        start_focus = 0.0
        end_focus = 0.0

        for target_idx in range(seq_len):
            weight = avg_attention_dist[target_idx]
            category = classify_token_position(target_idx, seq_len)
            if category == "body":
                body_focus += weight
            elif category == "start":
                start_focus += weight
            elif category == "end":
                end_focus += weight

        # Normalize
        total = body_focus + start_focus + end_focus
        if total > 0:
            body_focus /= total
            start_focus /= total
            end_focus /= total

        results["per_head"][head_idx] = {
            "body_focus": float(body_focus),
            "start_focus": float(start_focus),
            "end_focus": float(end_focus),
            "is_body_focused": float(body_focus) > 0.4,  # threshold
        }

    # Summary statistics
    body_focuses = [h["body_focus"] for h in results["per_head"].values()]
    results["summary"] = {
        "mean_body_focus": float(np.mean(body_focuses)),
        "std_body_focus": float(np.std(body_focuses)),
        "n_body_focused_heads": sum(
            1 for h in results["per_head"].values() if h["is_body_focused"]
        ),
        "pct_body_focused_heads": float(
            100.0
            * sum(1 for h in results["per_head"].values() if h["is_body_focused"])
            / len(results["per_head"])
        ),
    }

    return results


def main():
    import torch

    print("\n" + "=" * 70)
    print("ATTENTION HEAD ANALYSIS: DO HEADS FOCUS ON BODY TOKENS?")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load model and tokenizer
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map=device, attn_implementation="eager"
    )
    model.eval()

    # Load sample code pairs
    print(f"\nLoading up to {N_SAMPLE_PAIRS} code pairs...")
    records = load_source_records(SOURCE_JSONL, n_sample=N_SAMPLE_PAIRS)
    print(f"Loaded {len(records)} pairs")

    # Analyze attention for vulnerable and secure samples
    results = {
        "description": "Attention head focus analysis: body vs. structural tokens",
        "n_sample_pairs": len(records),
        "vulnerable_analysis": {},
        "secure_analysis": {},
        "comparison": {},
    }

    for code_type in ["vulnerable", "secure"]:
        print(f"\n{code_type.upper()} CODE ANALYSIS")
        print("-" * 70)

        all_layer_results = {}

        for pair_idx, record in enumerate(records[:10]):  # Analyze first 10 pairs
            code = record[f"{code_type}_code"]
            if not code or len(code) < 100:
                continue

            try:
                attn_weights = extract_attention_weights(code, tokenizer, model, device)
            except Exception as e:
                print(f"  Pair {pair_idx}: Error - {e}")
                continue

            seq_len = list(attn_weights.values())[0].shape[1]

            # Analyze each layer
            for layer_idx, attn_matrix in attn_weights.items():
                if layer_idx not in all_layer_results:
                    all_layer_results[layer_idx] = []

                head_focus = analyze_head_focus(attn_matrix, seq_len)
                all_layer_results[layer_idx].append(head_focus["summary"])

        # Aggregate across pairs per layer
        layer_summaries = {}
        for layer_idx in sorted(all_layer_results.keys()):
            summaries = all_layer_results[layer_idx]
            if not summaries:
                continue

            body_focuses = [s["mean_body_focus"] for s in summaries]
            pct_body_heads = [s["pct_body_focused_heads"] for s in summaries]

            layer_summaries[layer_idx] = {
                "layer": layer_idx,
                "mean_body_focus": float(np.mean(body_focuses)),
                "std_body_focus": float(np.std(body_focuses)),
                "mean_pct_body_focused_heads": float(np.mean(pct_body_heads)),
            }

            print(
                f"  Layer {layer_idx:2d}: "
                f"body focus {layer_summaries[layer_idx]['mean_body_focus']:.3f} ± {layer_summaries[layer_idx]['std_body_focus']:.3f}, "
                f"body-focused heads: {layer_summaries[layer_idx]['mean_pct_body_focused_heads']:.1f}%"
            )

        results[f"{code_type}_analysis"] = layer_summaries

    # Compare vulnerable vs. secure
    print("\n" + "=" * 70)
    print("COMPARISON: VULNERABLE vs. SECURE")
    print("=" * 70)

    comparison = {}
    for layer_idx in sorted(results["vulnerable_analysis"].keys()):
        if layer_idx not in results["secure_analysis"]:
            continue

        vuln = results["vulnerable_analysis"][layer_idx]
        secure = results["secure_analysis"][layer_idx]

        delta_body_focus = vuln["mean_body_focus"] - secure["mean_body_focus"]
        comparison[layer_idx] = {
            "layer": layer_idx,
            "vulnerable_body_focus": vuln["mean_body_focus"],
            "secure_body_focus": secure["mean_body_focus"],
            "delta_body_focus": float(delta_body_focus),
            "delta_pct_body_heads": float(
                vuln["mean_pct_body_focused_heads"]
                - secure["mean_pct_body_focused_heads"]
            ),
        }

        direction = (
            "↑ vuln"
            if delta_body_focus > 0.01
            else "↓ vuln" if delta_body_focus < -0.01 else "~"
        )
        print(
            f"  Layer {layer_idx:2d}: "
            f"body focus {vuln['mean_body_focus']:.3f} (vuln) vs {secure['mean_body_focus']:.3f} (secure) "
            f"{direction}"
        )

    results["comparison"] = comparison

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if comparison:
        deltas = [c["delta_body_focus"] for c in comparison.values()]
        mean_delta = np.mean(deltas)
        print(f"\nMean Δ body focus (vuln - secure): {mean_delta:+.4f}")
        print(f"Range: {np.min(deltas):+.4f} to {np.max(deltas):+.4f}")

        if abs(mean_delta) > 0.05:
            print(
                "\n→ Vulnerable and secure code show notably different attention patterns"
            )
        else:
            print(
                "\n→ Attention patterns are largely similar between vulnerable and secure code"
            )

    # Save results
    results_path = OUTPUT_DIR / "attention_statistics.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved results to {results_path}\n")


if __name__ == "__main__":
    main()
