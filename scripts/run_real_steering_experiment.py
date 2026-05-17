#!/usr/bin/env python3
"""
Real direction steering experiments for Qwen, CodeLlama, StarCoder2.

For each model:
  1. Load N code pairs from DeltaSecommits JSONL
  2. 80/20 train/test split
  3. Compute vulnerability direction at each layer from training pairs
  4. For each test pair, apply steering hook (adds alpha * direction to residual stream)
  5. Measure preference = mean per-token log P(secure) - mean per-token log P(vulnerable)
  6. Save per-layer, per-alpha results to results/raw_data/steering_results_{model}_{N}samples.json

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/run_real_steering_experiment.py
    CUDA_VISIBLE_DEVICES=1 python scripts/run_real_steering_experiment.py --models qwen --n_samples 100
"""

import argparse
import base64
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
DATA_JSONL = (
    REPO_ROOT
    / "sae_java_bug/artifacts/activations/TO_UPLOAD"
    / "activations_layer_0_sae_blocks.0.hook_resid_post_component_hook_resid_post.hook_sae_acts_post.jsonl"
)
RESULTS_DIR = REPO_ROOT / "results" / "raw_data"

MODELS_CONFIG = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "codellama": "codellama/CodeLlama-7b-Instruct-hf",
    "starcoder2": "bigcode/starcoder2-7b",
}

LAYERS_TO_TEST = [3, 7, 11, 15, 19, 23]
ALPHA_VALUES = [-20.0, -10.0, 0.0, 10.0, 20.0]
MAX_TOKENS = 512
LANGUAGE_FILTER = "c"
CODE_LENGTH_FILTER = (50, 500)  # characters, matching original experiment
PROMPTS = ["// C code:", "// Function:", "// Implementation:"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _try_decode(s: str) -> str:
    try:
        return base64.b64decode(s).decode("utf-8", errors="replace")
    except Exception:
        return s


def load_pairs(n_samples: int, language: str = "c") -> List[Dict]:
    logger.info(f"Loading code pairs from {DATA_JSONL}")
    if not DATA_JSONL.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_JSONL}")

    pairs = []
    with open(DATA_JSONL) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ext = r.get("file_extension", "").lstrip(".").lower()
            if language and ext != language:
                continue
            vuln = _try_decode(r.get("vulnerable_code", ""))
            sec = _try_decode(r.get("secure_code", ""))
            lo, hi = CODE_LENGTH_FILTER
            if not vuln.strip() or not sec.strip():
                continue
            if not (lo <= len(vuln) <= hi) or not (lo <= len(sec) <= hi):
                continue
            pairs.append({"vulnerable_code": vuln, "secure_code": sec, "cwe": r.get("cwe", "")})
            if len(pairs) >= n_samples:
                break

    logger.info(f"  Loaded {len(pairs)} pairs (language={language})")
    return pairs


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def get_layers(model) -> torch.nn.ModuleList:
    """Return the transformer block list for supported architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Llama / Qwen / StarCoder2-style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT2-style
    raise ValueError(f"Cannot locate transformer layers in {type(model).__name__}")


def make_steering_hook(direction: torch.Tensor, alpha: float):
    """Hook that subtracts alpha * direction from the hidden-state output.

    Convention matches the paper: positive alpha suppresses the vulnerability
    direction, negative alpha amplifies it (h' = h - α·d).

    Handles both old-style tuple outputs (hidden_states, ...) and new-style
    plain tensor outputs (transformers >=4.50 returns the tensor directly).
    """
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0]  # (batch, seq_len, hidden_dim)
            h = h - alpha * direction.to(h.device, h.dtype)
            return (h,) + output[1:]
        else:
            # Plain tensor: (batch, seq_len, hidden_dim)
            return output - alpha * direction.to(output.device, output.dtype)
    return hook_fn


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_per_token_logprob(model, tokenizer, code: str) -> Optional[float]:
    """Mean per-token log probability averaged across PROMPTS prefixes."""
    scores = []
    for prompt in PROMPTS:
        text = prompt + "\n" + code
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
        if enc["input_ids"].shape[1] < 2:
            continue
        input_ids = enc["input_ids"].to(model.device)
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=input_ids)
        scores.append(-out.loss.item())
    return float(np.mean(scores)) if scores else None


def extract_layer_activations(
    model, tokenizer, pairs: List[Dict], layer: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mean-token hidden states at `layer` for all pairs.
    Returns (vuln_acts, secure_acts) each of shape (N, hidden_dim).
    """
    layers = get_layers(model)
    vuln_acts, sec_acts = [], []

    collected = {"vuln": None, "sec": None}

    def capture_hook(key):
        def fn(module, input, output):
            # Handle both tuple outputs (old HF) and plain tensor (HF >=4.50)
            h = output[0] if isinstance(output, tuple) else output
            # h: (batch, seq_len, hidden_dim) — take first batch item, mean over tokens
            collected[key] = h[0].mean(0).float().cpu().numpy()
        return fn

    for pair in tqdm(pairs, desc=f"  Extracting layer {layer} activations", leave=False):
        for key, code in [("vuln", pair["vulnerable_code"]), ("sec", pair["secure_code"])]:
            enc = tokenizer(code, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
            input_ids = enc["input_ids"].to(model.device)
            if input_ids.shape[1] < 2:
                continue
            h = layers[layer].register_forward_hook(capture_hook(key))
            with torch.no_grad():
                model(input_ids=input_ids)
            h.remove()

        if collected["vuln"] is not None and collected["sec"] is not None:
            vuln_acts.append(collected["vuln"])
            sec_acts.append(collected["sec"])
        collected["vuln"] = collected["sec"] = None

    return np.array(vuln_acts), np.array(sec_acts)


def compute_direction(vuln_acts: np.ndarray, sec_acts: np.ndarray) -> torch.Tensor:
    """Compute normalized vulnerability direction from activation arrays."""
    direction = vuln_acts.mean(0) - sec_acts.mean(0)
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    return torch.tensor(direction, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Steering experiment
# ---------------------------------------------------------------------------

def run_steering_for_model(model_name: str, n_samples: int = 100) -> Dict:
    logger.info(f"\n{'='*70}")
    logger.info(f"  Model: {model_name.upper()}")
    logger.info(f"{'='*70}\n")

    # --- Load data -------------------------------------------------------
    pairs = load_pairs(n_samples, language=LANGUAGE_FILTER)
    if len(pairs) < 4:
        logger.error(f"Not enough pairs (got {len(pairs)})")
        return {}

    n_train = max(4, int(len(pairs) * 0.8))
    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:]
    logger.info(f"  Train: {len(train_pairs)}  Test: {len(test_pairs)}")

    # --- Load model -------------------------------------------------------
    hf_id = MODELS_CONFIG[model_name]
    logger.info(f"  Loading tokenizer: {hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"  Loading model: {hf_id}")
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    layers = get_layers(model)
    logger.info(f"  Model has {len(layers)} transformer layers")

    results = {
        "model": model_name,
        "n_samples": len(test_pairs),
        "n_prompts": len(PROMPTS),
        "n_train_samples": len(train_pairs),
        "n_test_samples": len(test_pairs),
        "methodology": "prompt-prefixed per-token log prob (h' = h - α·d), train/test split",
        "layers": {},
        "baseline": {},
    }

    # --- Baseline (no steering, alpha=0) ----------------------------------
    logger.info("\n  Computing baseline preference (no steering)...")
    baseline_prefs = []
    for pair in tqdm(test_pairs, desc="  Baseline", leave=False):
        lp_v = compute_per_token_logprob(model, tokenizer, pair["vulnerable_code"])
        lp_s = compute_per_token_logprob(model, tokenizer, pair["secure_code"])
        if lp_v is not None and lp_s is not None:
            baseline_prefs.append(lp_s - lp_v)

    baseline_mean = float(np.mean(baseline_prefs)) if baseline_prefs else 0.0
    baseline_std = float(np.std(baseline_prefs)) if baseline_prefs else 0.0
    results["baseline"] = {
        "mean_preference": baseline_mean,
        "std": baseline_std,
        "range": [float(np.min(baseline_prefs)), float(np.max(baseline_prefs))] if baseline_prefs else [0.0, 0.0],
        "n_valid_pairs": len(baseline_prefs),
    }
    logger.info(f"  Baseline: mean={baseline_mean:.4f}  std={baseline_std:.4f}")

    # --- Per-layer steering -----------------------------------------------
    for layer in LAYERS_TO_TEST:
        if layer >= len(layers):
            logger.warning(f"  Layer {layer} out of range ({len(layers)} layers) — skipping")
            continue

        logger.info(f"\n  Layer {layer} —")

        # Compute direction from training pairs
        logger.info(f"    Computing direction from {len(train_pairs)} training pairs...")
        vuln_acts, sec_acts = extract_layer_activations(model, tokenizer, train_pairs, layer)
        if len(vuln_acts) < 2:
            logger.warning(f"    Not enough training activations for layer {layer}")
            continue
        direction = compute_direction(vuln_acts, sec_acts)
        logger.info(f"    Direction norm: {direction.norm().item():.4f}  shape: {direction.shape}")

        layer_results: Dict[str, float] = {}

        for alpha in ALPHA_VALUES:
            logger.info(f"    alpha={alpha:+.0f}...")
            prefs = []

            hook_handle = layers[layer].register_forward_hook(
                make_steering_hook(direction, alpha)
            )
            for pair in tqdm(test_pairs, desc=f"    alpha={alpha:+.0f}", leave=False):
                lp_v = compute_per_token_logprob(model, tokenizer, pair["vulnerable_code"])
                lp_s = compute_per_token_logprob(model, tokenizer, pair["secure_code"])
                if lp_v is not None and lp_s is not None:
                    prefs.append(lp_s - lp_v)
            hook_handle.remove()

            mean_pref = float(np.mean(prefs)) if prefs else baseline_mean
            layer_results[str(alpha)] = mean_pref
            logger.info(f"      mean_preference={mean_pref:.4f}  n_valid={len(prefs)}")

        results["layers"][str(layer)] = {"alpha_results": layer_results}

    # --- Save results -----------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"steering_results_{model_name}_{n_samples}samples.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n  ✓ Saved: {out_path}")

    # Free GPU memory before next model
    del model
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="qwen,codellama,starcoder2",
        help="Comma-separated model names",
    )
    parser.add_argument("--n_samples", type=int, default=100)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]

    logger.info("=" * 70)
    logger.info("REAL DIRECTION STEERING EXPERIMENTS")
    logger.info(f"Models: {models}")
    logger.info(f"N samples: {args.n_samples}")
    logger.info("=" * 70)

    for model_name in models:
        if model_name not in MODELS_CONFIG:
            logger.error(f"Unknown model: {model_name}. Choose from {list(MODELS_CONFIG)}")
            continue
        try:
            run_steering_for_model(model_name, args.n_samples)
        except Exception as e:
            logger.error(f"Error for {model_name}: {e}", exc_info=True)

    logger.info("\n" + "=" * 70)
    logger.info("STEERING EXPERIMENTS COMPLETE")
    logger.info(f"Results in: {RESULTS_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
