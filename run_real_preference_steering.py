#!/usr/bin/env python3
"""
Direction Steering with Real DeltaSecommits Code Pairs

Loads actual vulnerable code from DeltaSecommits dataset and tests whether
steering the model along the vulnerability direction changes preference between
the secure (fixed) and vulnerable (buggy) versions.

Usage:
    python run_real_preference_steering.py [--n_samples 10] [--quick]
"""

import argparse
import base64
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# LOAD REAL VULNERABLE CODE FROM DATASET
# ============================================================================


def load_real_vulnerable_pairs(n_samples=10, language="c"):
    """
    Load real vulnerable code pairs from DeltaSecommits dataset.

    Returns list of dicts with:
    - vulnerable_code: actual buggy code
    - secure_code: actual fixed code
    - cwe: vulnerability type
    - length: code length (for filtering)
    """
    acts_dir = Path(
        "sae_java_bug/artifacts/activations/raw_activations/vulnerable_code_qwen_coder_standard_16384_raw"
    )
    acts_file = (
        acts_dir / "activations_layer_0_raw_component_hidden_state_last_token.jsonl"
    )

    if not acts_file.exists():
        print(f"✗ Dataset file not found: {acts_file}")
        return []

    pairs = []
    print(f"Loading {language.upper()}-only vulnerable code pairs from dataset...")

    with acts_file.open() as f:
        for line_idx, line in enumerate(f):
            if len(pairs) >= n_samples:
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

                # Filter by length (reasonable code snippets)
                if len(secure) < 50 or len(secure) > 500:
                    continue
                if len(vulnerable) < 50 or len(vulnerable) > 500:
                    continue

                pairs.append(
                    {
                        "secure_code": secure,
                        "vulnerable_code": vulnerable,
                        "cwe": record.get("cwe_id", "unknown"),
                        "vuln_id": record.get("vuln_id", f"sample_{len(pairs)}"),
                        "length": len(secure),
                    }
                )
            except Exception as e:
                if line_idx < 5:
                    print(f"  Skipping line {line_idx}: {e}")

    print(f"✓ Loaded {len(pairs)} real {language.upper()}-only code pairs\n")
    return pairs


def measure_code_logprob(model, tokenizer, prompt, code, device, max_length=512):
    """Measure log-probability of code following a prompt."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    code_ids = tokenizer(code, return_tensors="pt").input_ids.to(device)

    if code_ids[0, 0] == tokenizer.bos_token_id:
        code_ids = code_ids[:, 1:]

    full_ids = torch.cat([input_ids, code_ids], dim=1)

    if full_ids.shape[1] > max_length:
        return float("-inf")

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

        return np.mean(logprobs) if logprobs else float("-inf")
    except:
        return float("-inf")


def compute_vulnerability_direction(layer):
    """Load or compute vulnerability direction from C-only pairs."""
    acts_dir = Path(
        "sae_java_bug/artifacts/activations/raw_activations/vulnerable_code_qwen_coder_standard_16384_raw"
    )
    activations_file = (
        acts_dir
        / f"activations_layer_{layer}_raw_component_hidden_state_last_token.jsonl"
    )

    if not activations_file.exists():
        return None

    vulnerable_acts = []
    secure_acts = []

    with activations_file.open() as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get("file_extension") != "c":
                    continue

                if isinstance(record.get("vulnerable"), list) and isinstance(
                    record.get("secure"), list
                ):
                    vulnerable_acts.append(np.array(record["vulnerable"]))
                    secure_acts.append(np.array(record["secure"]))
            except:
                pass

    if not vulnerable_acts or not secure_acts:
        return None

    mean_vulnerable = np.mean(vulnerable_acts, axis=0)
    mean_secure = np.mean(secure_acts, axis=0)
    direction = mean_vulnerable - mean_secure
    direction_norm = direction / (np.linalg.norm(direction) + 1e-8)

    return torch.from_numpy(direction_norm).float()


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================


def run_experiment(n_samples=10, device=None, quick=False):
    """Run steering experiment with real code pairs."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_id = "Qwen/Qwen2.5-7B-Instruct"

    print("=" * 80)
    print("DIRECTION STEERING: Real DeltaSecommits Code Pairs")
    print("=" * 80)
    print(f"\nLoading model: {model_id}")
    print(f"Device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    # Load real code pairs
    pairs = load_real_vulnerable_pairs(n_samples=n_samples, language="c")
    if not pairs:
        print("✗ Failed to load code pairs")
        return None

    # Generic prompts
    prompts = [
        "// C code:\n",
        "// Function:\n",
        "// Implementation:\n",
    ]

    layers_to_test = [3] if quick else [3, 7, 11, 15, 19, 23]
    alphas = [-20, 0, 20] if quick else [-20, -10, 0, 10, 20]

    results = {
        "n_samples": len(pairs),
        "n_prompts": len(prompts),
        "layers": {},
        "baseline": {},
    }

    # ========================================================================
    # STEP 1: Baseline preference
    # ========================================================================
    print("=" * 80)
    print("STEP 1: Baseline Preference (α=0, no steering)")
    print("=" * 80)

    baseline_prefs = []
    for i, pair in enumerate(pairs):
        sec_lps = []
        vuln_lps = []

        for prompt in prompts:
            sec_lp = measure_code_logprob(
                model, tokenizer, prompt, pair["secure_code"], device
            )
            vuln_lp = measure_code_logprob(
                model, tokenizer, prompt, pair["vulnerable_code"], device
            )

            sec_lps.append(sec_lp)
            vuln_lps.append(vuln_lp)

        mean_sec = np.mean(sec_lps)
        mean_vuln = np.mean(vuln_lps)
        pref = mean_sec - mean_vuln
        baseline_prefs.append(pref)

        indicator = "→ SECURE" if pref > 0 else "→ VULNERABLE"
        print(
            f"  [{i+1:2d}/{len(pairs)}] {pair['cwe']:15s} pref={pref:+.4f} {indicator}  (len={pair['length']:3d})"
        )

    mean_baseline = np.mean(baseline_prefs)
    results["baseline"] = {
        "mean_preference": float(mean_baseline),
        "std": float(np.std(baseline_prefs)),
        "range": [float(min(baseline_prefs)), float(max(baseline_prefs))],
    }
    print(
        f"\n  OVERALL BASELINE: {mean_baseline:+.4f} ± {np.std(baseline_prefs):.4f}\n"
    )

    # ========================================================================
    # STEP 2: Steering experiment
    # ========================================================================
    print("=" * 80)
    print("STEP 2: Steering Experiment")
    print("=" * 80)

    for layer in layers_to_test:
        print(f"\n▶ Layer {layer}")
        direction = compute_vulnerability_direction(layer)

        if direction is None:
            print(f"  ✗ Could not load direction")
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
                for pair in pairs:
                    sec_lps = []
                    vuln_lps = []

                    for prompt in prompts:
                        sec_lp = measure_code_logprob(
                            model, tokenizer, prompt, pair["secure_code"], device
                        )
                        vuln_lp = measure_code_logprob(
                            model, tokenizer, prompt, pair["vulnerable_code"], device
                        )

                        sec_lps.append(sec_lp)
                        vuln_lps.append(vuln_lp)

                    pref = np.mean(sec_lps) - np.mean(vuln_lps)
                    alpha_prefs.append(pref)

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
    out_file = Path("results_real_preference_steering.json")
    with out_file.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved: {out_file}")

    return results


def plot_results(results):
    """Visualize results."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "figure.dpi": 150,
        }
    )

    layers = sorted([int(k) for k in results["layers"].keys()])
    fig, axes = plt.subplots(len(layers), 1, figsize=(10, 3 * len(layers)))
    if len(layers) == 1:
        axes = [axes]

    for ax_idx, layer in enumerate(layers):
        ax = axes[ax_idx]
        layer_data = results["layers"][str(layer)]["alpha_results"]

        alphas = sorted([float(a) for a in layer_data.keys()])
        prefs = [layer_data[str(float(a))] for a in alphas]

        ax.plot(
            alphas,
            prefs,
            marker="o",
            linewidth=2.5,
            markersize=8,
            label="Mean preference",
        )
        ax.axhline(
            results["baseline"]["mean_preference"],
            color="blue",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5,
            label=f"Baseline ({results['baseline']['mean_preference']:+.3f})",
        )
        ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)

        ax.set_xlabel("Steering Strength (α)", fontsize=11)
        ax.set_ylabel(
            "Preference: log-prob(secure) - log-prob(vulnerable)", fontsize=11
        )
        ax.set_title(
            f"Layer {layer} ({results['n_samples']} Real Code Pairs)",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Real Code: Secure vs Vulnerable Preference Under Steering",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    fig_path = Path("figures/fig_real_preference_steering.pdf")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"✓ Figure saved: {fig_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n_samples", type=int, default=10, help="Number of code pairs to test"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick test (Layer 3 only)"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    results = run_experiment(n_samples=args.n_samples, device=device, quick=args.quick)

    if results:
        print("\n" + "=" * 80)
        print("Generating visualization...")
        print("=" * 80)
        plot_results(results)
        print("\nDone! ✓")
