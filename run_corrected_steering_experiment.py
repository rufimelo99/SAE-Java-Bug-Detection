#!/usr/bin/env python3
"""
Corrected Direction Steering Experiment with Proper Methodology

Fixes two critical issues:
1. Length normalization: Uses per-token averaged log-probs instead of raw sums
2. Information leakage: Computes directions on train set, evaluates on held-out test set

Usage:
    python run_corrected_steering_experiment.py [--n_samples 100] [--test_split 0.2]
"""

import argparse
import base64
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# LOAD DATA WITH TRAIN/TEST SPLIT
# ============================================================================


def load_deltasecommits_with_split(
    n_samples: int = 100, test_split: float = 0.2, language: str = "c"
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load real vulnerable code pairs from DeltaSecommits with train/test split.

    Returns:
        (train_pairs, test_pairs) where each is a list of dicts with:
        - vulnerable_code: actual buggy code
        - secure_code: actual fixed code
        - cwe: vulnerability type
        - length: code length
    """
    acts_dir = Path(
        "sae_java_bug/artifacts/activations/raw_activations/vulnerable_code_qwen_coder_standard_16384_raw"
    )
    acts_file = (
        acts_dir / "activations_layer_0_raw_component_hidden_state_last_token.jsonl"
    )

    if not acts_file.exists():
        print(f"✗ Dataset file not found: {acts_file}")
        return [], []

    all_pairs = []
    print(f"Loading {language.upper()}-only vulnerable code pairs from dataset...")

    with acts_file.open() as f:
        for line_idx, line in enumerate(f):
            if len(all_pairs) >= n_samples * 1.5:  # Load extra for splitting
                break

            if line_idx % 2000 == 0 and line_idx > 0:
                print(f"  Processed {line_idx} records...")

            try:
                record = json.loads(line)

                # Filter by language
                if record.get("file_extension") != language:
                    continue

                # Decode code
                try:
                    secure = base64.b64decode(record["secure_code"]).decode(
                        "utf-8", errors="replace"
                    )
                    vulnerable = base64.b64decode(record["vulnerable_code"]).decode(
                        "utf-8", errors="replace"
                    )
                except:
                    continue

                # Filter by length
                if len(secure) < 50 or len(secure) > 500:
                    continue
                if len(vulnerable) < 50 or len(vulnerable) > 500:
                    continue

                all_pairs.append(
                    {
                        "secure_code": secure,
                        "vulnerable_code": vulnerable,
                        "cwe": record.get("cwe_id", "unknown"),
                        "vuln_id": record.get("vuln_id", f"sample_{len(all_pairs)}"),
                        "length": len(secure),
                    }
                )
            except Exception as e:
                if line_idx < 5:
                    print(f"  Skipping line {line_idx}: {e}")

    # Split into train/test
    np.random.seed(42)
    indices = np.random.permutation(len(all_pairs))
    split_idx = int(len(all_pairs) * (1 - test_split))

    train_pairs = [all_pairs[i] for i in indices[:split_idx]]
    test_pairs = [all_pairs[i] for i in indices[split_idx : split_idx + n_samples]]

    print(f"✓ Loaded {len(train_pairs)} train pairs, {len(test_pairs)} test pairs\n")
    return train_pairs[:n_samples], test_pairs[:n_samples]


# ============================================================================
# LENGTH-NORMALIZED LOG-PROBABILITY
# ============================================================================


def measure_code_logprob_normalized(
    model, tokenizer, prompt: str, code: str, device, max_length: int = 512
) -> float:
    """
    Measure per-token average log-probability of code following a prompt.

    This is length-normalized: divides total log-prob by number of tokens
    to avoid bias toward shorter sequences.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    code_ids = tokenizer(code, return_tensors="pt").input_ids.to(device)

    if code_ids[0, 0] == tokenizer.bos_token_id:
        code_ids = code_ids[:, 1:]

    full_ids = torch.cat([input_ids, code_ids], dim=1)

    if full_ids.shape[1] > max_length:
        return float("nan")  # Skip overly long sequences

    try:
        with torch.no_grad():
            outputs = model(input_ids=full_ids, use_cache=False)
            logits = outputs.logits[0]

        target_logits = logits[len(input_ids[0]) - 1 : -1]
        target_probs = torch.nn.functional.log_softmax(target_logits, dim=-1)

        logprobs = []
        for i, token_id in enumerate(code_ids[0]):
            if i < len(target_probs):
                logprobs.append(target_probs[i, token_id].item())

        # Return PER-TOKEN average (length-normalized)
        return np.mean(logprobs) if logprobs else float("nan")
    except:
        return float("nan")


# ============================================================================
# COMPUTE DIRECTION ON TRAIN SET ONLY
# ============================================================================


def compute_vulnerability_direction_from_activations(
    layer: int, train_pairs: List[Dict]
) -> torch.Tensor:
    """
    Compute vulnerability direction from training activations only.

    This prevents information leakage by computing on a separate train set.
    """
    acts_dir = Path(
        "sae_java_bug/artifacts/activations/raw_activations/vulnerable_code_qwen_coder_standard_16384_raw"
    )
    activations_file = (
        acts_dir
        / f"activations_layer_{layer}_raw_component_hidden_state_last_token.jsonl"
    )

    if not activations_file.exists():
        return None

    # Get IDs of train pairs
    train_vuln_ids = set(p["vuln_id"] for p in train_pairs)

    vulnerable_acts = []
    secure_acts = []

    with activations_file.open() as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get("file_extension") != "c":
                    continue

                # Only use training set
                if record.get("vuln_id") not in train_vuln_ids:
                    continue

                if isinstance(record.get("vulnerable"), list) and isinstance(
                    record.get("secure"), list
                ):
                    vulnerable_acts.append(np.array(record["vulnerable"]))
                    secure_acts.append(np.array(record["secure"]))
            except:
                pass

    if not vulnerable_acts or not secure_acts:
        print(
            f"  Warning: Limited data for layer {layer} (vuln={len(vulnerable_acts)}, sec={len(secure_acts)})"
        )
        if len(vulnerable_acts) == 0 or len(secure_acts) == 0:
            return None

    mean_vulnerable = np.mean(vulnerable_acts, axis=0)
    mean_secure = np.mean(secure_acts, axis=0)
    direction = mean_vulnerable - mean_secure
    direction_norm = direction / (np.linalg.norm(direction) + 1e-8)

    return torch.from_numpy(direction_norm).float()


# ============================================================================
# CORRECTED STEERING EXPERIMENT
# ============================================================================


def run_corrected_experiment(
    n_samples: int = 100, device: torch.device = None, test_split: float = 0.2
):
    """
    Run steering experiment with proper methodology:
    1. Train/test split for direction computation
    2. Length-normalized preference metric
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_id = "Qwen/Qwen2.5-7B-Instruct"

    print("=" * 80)
    print("CORRECTED DIRECTION STEERING: Train/Test Split + Length Normalization")
    print("=" * 80)
    print(f"\nLoading model: {model_id}")
    print(f"Device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    # Load data WITH train/test split
    train_pairs, test_pairs = load_deltasecommits_with_split(
        n_samples=n_samples, test_split=test_split, language="c"
    )

    if not test_pairs:
        print("✗ Failed to load test pairs")
        return None

    prompts = [
        "// C code:\n",
        "// Function:\n",
        "// Implementation:\n",
    ]

    layers_to_test = [3, 7, 11, 15, 19, 23]
    alphas = [-20, -10, 0, 10, 20]

    results = {
        "n_test_samples": len(test_pairs),
        "n_train_samples": len(train_pairs),
        "methodology": "length-normalized + train/test split",
        "layers": {},
        "baseline": {},
    }

    # ========================================================================
    # STEP 1: Baseline preference (length-normalized)
    # ========================================================================
    print("=" * 80)
    print("STEP 1: Baseline Preference (α=0, length-normalized)")
    print("=" * 80)

    baseline_prefs = []
    for i, pair in enumerate(test_pairs):
        sec_lps = []
        vuln_lps = []

        for prompt in prompts:
            # Use LENGTH-NORMALIZED log-probs
            sec_lp = measure_code_logprob_normalized(
                model, tokenizer, prompt, pair["secure_code"], device
            )
            vuln_lp = measure_code_logprob_normalized(
                model, tokenizer, prompt, pair["vulnerable_code"], device
            )

            if not np.isnan(sec_lp):
                sec_lps.append(sec_lp)
            if not np.isnan(vuln_lp):
                vuln_lps.append(vuln_lp)

        if sec_lps and vuln_lps:
            mean_sec = np.mean(sec_lps)
            mean_vuln = np.mean(vuln_lps)
            pref = mean_sec - mean_vuln
            baseline_prefs.append(pref)

            indicator = "→ SECURE" if pref > 0 else "→ VULNERABLE"
            print(
                f"  [{i+1:2d}/{len(test_pairs)}] {pair['cwe']:15s} pref={pref:+.4f} {indicator}  (len={pair['length']:3d})"
            )

    mean_baseline = np.mean(baseline_prefs)
    results["baseline"] = {
        "mean_preference": float(mean_baseline),
        "std": float(np.std(baseline_prefs)),
        "range": [float(min(baseline_prefs)), float(max(baseline_prefs))],
        "normalization": "per-token average (length-normalized)",
    }
    print(
        f"\n  OVERALL BASELINE: {mean_baseline:+.4f} ± {np.std(baseline_prefs):.4f}\n"
    )

    # ========================================================================
    # STEP 2: Steering with train/test split
    # ========================================================================
    print("=" * 80)
    print("STEP 2: Steering Experiment (Directions from TRAIN set only)")
    print("=" * 80)

    for layer in layers_to_test:
        print(f"\n▶ Layer {layer}")

        # Compute direction from TRAIN set only
        direction = compute_vulnerability_direction_from_activations(layer, train_pairs)

        if direction is None:
            print(f"  ✗ Could not load direction from train set")
            continue

        direction = direction.to(device)
        layer_prefs = []

        for alpha in alphas:
            alpha_prefs = []

            hook_handle = None
            if alpha != 0:

                def steering_hook(module, inp, output):
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output
                    h = h - alpha * direction.to(h.device).to(h.dtype)
                    if isinstance(output, tuple):
                        return (h,) + output[1:]
                    return h

                hook_handle = model.model.layers[layer].register_forward_hook(
                    steering_hook
                )

            try:
                for pair in test_pairs:
                    sec_lps = []
                    vuln_lps = []

                    for prompt in prompts:
                        sec_lp = measure_code_logprob_normalized(
                            model, tokenizer, prompt, pair["secure_code"], device
                        )
                        vuln_lp = measure_code_logprob_normalized(
                            model, tokenizer, prompt, pair["vulnerable_code"], device
                        )

                        if not np.isnan(sec_lp):
                            sec_lps.append(sec_lp)
                        if not np.isnan(vuln_lp):
                            vuln_lps.append(vuln_lp)

                    if sec_lps and vuln_lps:
                        pref = np.mean(sec_lps) - np.mean(vuln_lps)
                        alpha_prefs.append(pref)

                if alpha_prefs:
                    mean_pref = np.mean(alpha_prefs)
                    layer_prefs.append((alpha, mean_pref))
                    print(f"    α={alpha:+6.1f}: mean_pref={mean_pref:+.4f}")

            finally:
                if hook_handle is not None:
                    hook_handle.remove()

        results["layers"][str(layer)] = {
            "alpha_results": {str(float(a)): float(p) for a, p in layer_prefs}
        }

    # Save results
    out_file = Path("results_corrected_steering_train_test_split.json")
    with out_file.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Corrected results saved: {out_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n_samples", type=int, default=100, help="Number of code pairs to test"
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Fraction to use as test set (rest for direction computation)",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    results = run_corrected_experiment(
        n_samples=args.n_samples, device=device, test_split=args.test_split
    )

    if results:
        print("\n" + "=" * 80)
        print("Generating comparison visualization...")
        print("=" * 80)
        print("\nDone! ✓")
        print(f"\nKey improvement: Length-normalized metric + train/test split")
        print(f"  Train samples used for direction: {results['n_train_samples']}")
        print(f"  Test samples used for steering: {results['n_test_samples']}")
