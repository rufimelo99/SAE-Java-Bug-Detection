#!/usr/bin/env python3
"""
Extract and cache activations for multiple datasets (DeltaSecommits, SVEN, PreciseBugs).

Supports different models and layers. Activations cached as NPZ files for fast reuse.

Usage:
    python -m sae_java_bug.evaluation.multi_dataset_activations \
        --datasets deltasecommits,sven,precisebugs \
        --models qwen-7b,codellama-7b,starcoder2-7b \
        --layers 3,7,11,15,19,23,27

Or with defaults (all datasets, all models, all layers):
    python -m sae_java_bug.evaluation.multi_dataset_activations
"""

import argparse
import base64
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent

# Models configuration
MODELS = {
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "codellama-7b": "codellama/CodeLlama-7b-Instruct-hf",
    "codellama-13b": "codellama/CodeLlama-13b-Instruct-hf",
    "starcoder2-7b": "bigcode/starcoder2-7b",
    "starcoder2-15b": "bigcode/starcoder2-15b",
}

# Dataset configurations (name -> code pairs source)
DATASETS = {
    "deltasecommits": {
        "description": "DeltaSecommits C vulnerabilities",
        "data_source": REPO_ROOT
        / "sae_java_bug/artifacts/activations/TO_UPLOAD/activations_layer_11_sae_blocks.11.hook_resid_post_component_hook_resid_post.hook_sae_acts_post.jsonl",
        "output_dir": REPO_ROOT / "sae_java_bug/artifacts/multi_model_probing",
    },
    "sven": {
        "description": "SVEN C vulnerabilities",
        "data_source": REPO_ROOT
        / "sae_java_bug/artifacts/data/sven_raw/sven_c_pairs.jsonl",
        "output_dir": REPO_ROOT / "sae_java_bug/artifacts/multi_model_probing",
    },
    "precisebugs": {
        "description": "PreciseBugs C vulnerabilities",
        "data_source": REPO_ROOT
        / "sae_java_bug/artifacts/data/precisebugs_raw/precisebugs_c_pairs.jsonl",
        "output_dir": REPO_ROOT / "sae_java_bug/artifacts/multi_model_probing",
    },
}

MAX_TOKENS = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _b64decode(s: str) -> str:
    """Safely decode base64 string."""
    try:
        return base64.b64decode(s).decode("utf-8", errors="replace")
    except Exception:
        return s


def load_deltasecommits_pairs(data_source: Path) -> list[dict]:
    """Load DeltaSecommits code pairs from JSONL."""
    if not data_source.exists():
        raise FileNotFoundError(f"Data file not found: {data_source}")

    logger.info(f"Loading DeltaSecommits pairs from {data_source}")
    pairs = []
    with open(data_source) as f:
        for line in f:
            r = json.loads(line)
            vuln_code = _b64decode(r.get("vulnerable_code", ""))
            sec_code = _b64decode(r.get("secure_code", ""))
            if not vuln_code.strip() or not sec_code.strip():
                continue
            pairs.append(
                {
                    "vuln_id": r.get("vuln_id", ""),
                    "cwe": r.get("cwe", ""),
                    "file_extension": r.get("file_extension", "").lstrip(".").lower(),
                    "vulnerable_code": vuln_code,
                    "secure_code": sec_code,
                }
            )
    logger.info(f"  {len(pairs):,} valid pairs loaded.")
    return pairs


def load_sven_pairs(data_source: Path, code_dir: Path = None) -> list[dict]:
    """Load SVEN code pairs from JSONL file."""
    if not data_source.exists():
        raise FileNotFoundError(f"Data file not found: {data_source}")

    logger.info(f"Loading SVEN pairs from {data_source}")
    pairs = []
    with open(data_source) as f:
        for line in f:
            r = json.loads(line)
            vuln_code = r.get("vulnerable_code", "")
            sec_code = r.get("secure_code", "")
            if not vuln_code.strip() or not sec_code.strip():
                continue
            pairs.append(
                {
                    "vuln_id": r.get("vuln_id", ""),
                    "cwe": r.get("cwe", ""),
                    "file_extension": r.get("file_extension", "c").lstrip(".").lower(),
                    "vulnerable_code": vuln_code,
                    "secure_code": sec_code,
                }
            )

    logger.info(f"  {len(pairs):,} valid pairs loaded.")
    return pairs


def load_precisebugs_pairs(data_source: Path) -> list[dict]:
    """Load PreciseBugs code pairs from JSONL file."""
    if not data_source.exists():
        raise FileNotFoundError(f"Data file not found: {data_source}")

    logger.info(f"Loading PreciseBugs pairs from {data_source}")
    pairs = []
    with open(data_source) as f:
        for line in f:
            r = json.loads(line)
            vuln_code = r.get("vulnerable_code", "")
            sec_code = r.get("secure_code", "")
            if not vuln_code.strip() or not sec_code.strip():
                continue
            pairs.append(
                {
                    "vuln_id": r.get("vuln_id", ""),
                    "cwe": r.get("cwe", ""),
                    "file_extension": r.get("file_extension", "c").lstrip(".").lower(),
                    "vulnerable_code": vuln_code,
                    "secure_code": sec_code,
                }
            )

    logger.info(f"  {len(pairs):,} valid pairs loaded.")
    return pairs


def extract_activations(
    pairs: list[dict],
    model_name: str,
    model_hf_id: str,
    layers: list[int],
    cache_path: Path,
) -> dict:
    """Extract activations for specified layers."""
    if cache_path.exists():
        logger.info(f"Loading cached activations from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        result = {}
        for layer in layers:
            layer_data = {}
            for key in ["vuln_last", "vuln_mean", "secure_last", "secure_mean"]:
                k = f"layer_{layer}_{key}"
                if k in data.files:
                    layer_data[key] = data[k]
            if layer_data:
                result[layer] = layer_data
        return result

    logger.info(f"Loading tokenizer: {model_hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model: {model_hf_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_hf_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Extract activations
    activations = {}
    with torch.no_grad():
        for pair in tqdm(pairs, desc=f"Extracting activations ({model_name})"):
            vuln_code = pair["vulnerable_code"]
            sec_code = pair["secure_code"]

            for code_type, code in [("vuln", vuln_code), ("secure", sec_code)]:
                if not code.strip():
                    continue

                inputs = tokenizer(
                    code,
                    return_tensors="pt",
                    max_length=MAX_TOKENS,
                    truncation=True,
                    padding=True,
                ).to(DEVICE)

                # Forward pass
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states

                for layer in layers:
                    if layer >= len(hidden_states):
                        continue

                    layer_acts = hidden_states[layer]  # (batch, seq_len, hidden)

                    # Last token
                    last_token = layer_acts[0, -1, :].cpu().numpy().astype(np.float32)

                    # Mean token
                    mean_token = layer_acts[0].mean(0).cpu().numpy().astype(np.float32)

                    if layer not in activations:
                        activations[layer] = {
                            "vuln_last": [],
                            "vuln_mean": [],
                            "secure_last": [],
                            "secure_mean": [],
                        }

                    if code_type == "vuln":
                        activations[layer]["vuln_last"].append(last_token)
                        activations[layer]["vuln_mean"].append(mean_token)
                    else:
                        activations[layer]["secure_last"].append(last_token)
                        activations[layer]["secure_mean"].append(mean_token)

    # Convert to arrays and save
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {}
    for layer, acts in activations.items():
        for key in ["vuln_last", "vuln_mean", "secure_last", "secure_mean"]:
            save_data[f"layer_{layer}_{key}"] = np.array(acts[key], dtype=np.float32)

    np.savez(cache_path, **save_data)
    logger.info(f"Saved activations to {cache_path}")

    return activations


def main():
    parser = argparse.ArgumentParser(description="Extract multi-dataset activations")
    parser.add_argument(
        "--datasets",
        default="deltasecommits",
        help="Comma-separated datasets (deltasecommits, sven, precisebugs)",
    )
    parser.add_argument(
        "--models",
        default="qwen-7b,codellama-7b,starcoder2-7b",
        help="Comma-separated models",
    )
    parser.add_argument(
        "--layers", default="3,7,11,15,19,23,27", help="Comma-separated layer indices"
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    layers = [int(l.strip()) for l in args.layers.split(",")]

    logger.info("=" * 70)
    logger.info(f"Multi-Dataset Activation Extraction")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Models: {models}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Device: {DEVICE}")
    logger.info("=" * 70)

    for dataset in datasets:
        if dataset not in DATASETS:
            logger.error(f"Unknown dataset: {dataset}")
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"Dataset: {dataset}")
        logger.info(f"{'='*70}")

        dataset_cfg = DATASETS[dataset]
        data_source = dataset_cfg["data_source"]
        output_dir = dataset_cfg["output_dir"]

        # Load pairs
        try:
            if dataset == "deltasecommits":
                pairs = load_deltasecommits_pairs(data_source)
            elif dataset == "sven":
                pairs = load_sven_pairs(data_source)
            elif dataset == "precisebugs":
                pairs = load_precisebugs_pairs(data_source)
            else:
                continue
        except FileNotFoundError as e:
            logger.warning(f"Skipping {dataset}: data not found ({e})")
            continue
        except Exception as e:
            logger.error(f"Failed to load {dataset}: {e}")
            continue

        if not pairs:
            logger.warning(f"No pairs loaded for {dataset}")
            continue

        # Extract for each model
        for model in models:
            if model not in MODELS:
                logger.warning(f"Unknown model: {model}")
                continue

            logger.info(f"\nModel: {model}")
            model_hf_id = MODELS[model]
            cache_path = output_dir / f"activations_{dataset}_{model}.npz"

            try:
                extract_activations(
                    pairs=pairs,
                    model_name=model,
                    model_hf_id=model_hf_id,
                    layers=layers,
                    cache_path=cache_path,
                )
            except Exception as e:
                logger.error(
                    f"Failed to extract activations for {dataset}/{model}: {e}"
                )

    logger.info("\n" + "=" * 70)
    logger.info("✓ Activation extraction complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
