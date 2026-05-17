#!/usr/bin/env python3
"""
Reconstruct the original real-preference steering experiment.

Matches the methodology that produced results_real_preference_steering_100samples.json:
  - Same data path and filters (C code, 50-500 chars, 100 pairs)
  - Direction computed from all 100 pairs at each layer (no train/test split)
  - Hook: h' = h - alpha * direction  (positive alpha suppresses vulnerability)
  - Preference: mean log P(secure | prompt) - mean log P(vuln | prompt), averaged over 3 prompts
  - Layers: [3, 7, 11, 15, 19, 23]
  - Alpha: [-20, -10, 0, 10, 20]

Extended for all three models (Qwen, CodeLlama, StarCoder2).

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/run_real_preference_steering.py
    CUDA_VISIBLE_DEVICES=1 python scripts/run_real_preference_steering.py --models qwen
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

# Original data path used in the experiment that produced the paper's figure
DATA_JSONL = (
    REPO_ROOT
    / "sae_java_bug/artifacts/activations/raw_activations"
    / "vulnerable_code_qwen_coder_standard_16384_raw"
    / "activations_layer_0_raw_component_hidden_state_last_token.jsonl"
)

RESULTS_DIR = REPO_ROOT / "results" / "raw_data"

MODELS_CONFIG = {
    "qwen":       "Qwen/Qwen2.5-7B-Instruct",
    "codellama":  "codellama/CodeLlama-7b-Instruct-hf",
    "starcoder2": "bigcode/starcoder2-7b",
}

# Exact settings from the original experiment
LAYERS       = [3, 7, 11, 15, 19, 23]
ALPHA_VALUES = [-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
N_PAIRS      = 10_000  # effectively unlimited — use all qualifying pairs
LANGUAGE     = "c"
MIN_LEN      = 50
MAX_LEN      = 500
MAX_TOKENS   = 512
PROMPTS      = ["// C code:", "// Function:", "// Implementation:"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _try_decode(s: str) -> str:
    try:
        return base64.b64decode(s).decode("utf-8", errors="replace")
    except Exception:
        return s


def load_pairs() -> List[Dict]:
    logger.info(f"Loading pairs from {DATA_JSONL}")
    if not DATA_JSONL.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_JSONL}")

    pairs = []
    with open(DATA_JSONL) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("file_extension", "").lstrip(".").lower() != LANGUAGE:
                continue
            vuln = _try_decode(r.get("vulnerable_code", ""))
            sec  = _try_decode(r.get("secure_code", ""))
            if not (MIN_LEN <= len(vuln) <= MAX_LEN):
                continue
            if not (MIN_LEN <= len(sec) <= MAX_LEN):
                continue
            pairs.append({"vulnerable_code": vuln, "secure_code": sec, "cwe": r.get("cwe", "")})
            if len(pairs) >= N_PAIRS:
                break

    logger.info(f"  Loaded {len(pairs)} pairs")
    return pairs


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def get_layers(model) -> torch.nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Cannot locate transformer layers in {type(model).__name__}")


def make_steering_hook(direction: torch.Tensor, alpha: float):
    """h' = h - alpha * d  (positive alpha suppresses vulnerability direction)."""
    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h - alpha * direction.to(h.device, h.dtype)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return hook_fn


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def code_logprob(model, tokenizer, code: str) -> Optional[float]:
    """Mean per-token log prob averaged over all PROMPTS prefixes."""
    scores = []
    for prompt in PROMPTS:
        text = prompt + "\n" + code
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
        if enc["input_ids"].shape[1] < 2:
            continue
        input_ids = enc["input_ids"].to(model.device)
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=input_ids)
        scores.append(-out.loss.item())  # -cross_entropy = mean log prob per token
    return float(np.mean(scores)) if scores else None


def pair_preference(model, tokenizer, pair: Dict) -> Optional[float]:
    """log P(secure) - log P(vulnerable), averaged over PROMPTS."""
    lp_v = code_logprob(model, tokenizer, pair["vulnerable_code"])
    lp_s = code_logprob(model, tokenizer, pair["secure_code"])
    if lp_v is None or lp_s is None:
        return None
    return lp_s - lp_v


def extract_mean_activations(
    model, tokenizer, pairs: List[Dict], layer: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean-token hidden states at `layer` for each pair's vuln and secure code."""
    layers = get_layers(model)
    vuln_acts, sec_acts = [], []
    collected: Dict[str, Optional[np.ndarray]] = {"vuln": None, "sec": None}

    def capture(key):
        def fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            collected[key] = h[0].mean(0).float().cpu().numpy()
        return fn

    for pair in tqdm(pairs, desc=f"  L{layer} direction", leave=False):
        for key, code in [("vuln", pair["vulnerable_code"]), ("sec", pair["secure_code"])]:
            enc = tokenizer(code, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
            if enc["input_ids"].shape[1] < 2:
                collected[key] = None
                continue
            handle = layers[layer].register_forward_hook(capture(key))
            with torch.no_grad():
                model(input_ids=enc["input_ids"].to(model.device))
            handle.remove()

        if collected["vuln"] is not None and collected["sec"] is not None:
            vuln_acts.append(collected["vuln"])
            sec_acts.append(collected["sec"])
        collected["vuln"] = collected["sec"] = None

    return np.array(vuln_acts), np.array(sec_acts)


def compute_direction(vuln_acts: np.ndarray, sec_acts: np.ndarray) -> torch.Tensor:
    d = vuln_acts.mean(0) - sec_acts.mean(0)
    d /= np.linalg.norm(d) + 1e-10
    return torch.tensor(d, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_for_model(model_name: str, pairs: List[Dict]) -> Dict:
    logger.info(f"\n{'='*70}\n  {model_name.upper()}\n{'='*70}")

    hf_id = MODELS_CONFIG[model_name]
    logger.info(f"  Loading {hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    layers = get_layers(model)
    logger.info(f"  {len(layers)} transformer layers")

    # Baseline (alpha = 0, no steering)
    logger.info("\n  Baseline...")
    baseline_prefs = []
    for pair in tqdm(pairs, desc="  Baseline", leave=False):
        p = pair_preference(model, tokenizer, pair)
        if p is not None:
            baseline_prefs.append(p)
    baseline_mean = float(np.mean(baseline_prefs))
    baseline_std  = float(np.std(baseline_prefs))
    logger.info(f"  Baseline: mean={baseline_mean:.4f}  std={baseline_std:.4f}")

    results = {
        "model":     model_name,
        "n_samples": len(pairs),
        "n_prompts": len(PROMPTS),
        "baseline": {
            "mean_preference": baseline_mean,
            "std": baseline_std,
            "range": [float(np.min(baseline_prefs)), float(np.max(baseline_prefs))],
        },
        "layers": {},
    }

    for layer in LAYERS:
        if layer >= len(layers):
            logger.warning(f"  Layer {layer} out of range — skipping")
            continue

        logger.info(f"\n  Layer {layer}")

        # Direction from all pairs (no train/test split — matches original methodology)
        vuln_acts, sec_acts = extract_mean_activations(model, tokenizer, pairs, layer)
        if len(vuln_acts) < 2:
            logger.warning(f"  Not enough activations for layer {layer}")
            continue
        direction = compute_direction(vuln_acts, sec_acts)
        logger.info(f"    direction norm={direction.norm():.4f}  shape={tuple(direction.shape)}")

        alpha_results: Dict[str, float] = {}
        for alpha in ALPHA_VALUES:
            handle = layers[layer].register_forward_hook(make_steering_hook(direction, alpha))
            prefs = []
            for pair in tqdm(pairs, desc=f"    α={alpha:+.0f}", leave=False):
                p = pair_preference(model, tokenizer, pair)
                if p is not None:
                    prefs.append(p)
            handle.remove()

            mean_pref = float(np.mean(prefs)) if prefs else baseline_mean
            alpha_results[str(alpha)] = mean_pref
            effect = mean_pref - baseline_mean
            logger.info(f"    α={alpha:+5.0f}: mean={mean_pref:.4f}  effect={effect:+.4f}")

        results["layers"][str(layer)] = {"alpha_results": alpha_results}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"steering_results_{model_name}_{len(pairs)}samples.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n  ✓ Saved: {out}")

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="qwen,codellama,starcoder2")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]

    pairs = load_pairs()
    logger.info(f"Using {len(pairs)} pairs for all models")

    for model_name in models:
        if model_name not in MODELS_CONFIG:
            logger.error(f"Unknown model '{model_name}'. Choose from {list(MODELS_CONFIG)}")
            continue
        try:
            run_for_model(model_name, pairs)
        except Exception as e:
            logger.error(f"Error for {model_name}: {e}", exc_info=True)

    logger.info("\n" + "="*70)
    logger.info("DONE")
    logger.info(f"Results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
