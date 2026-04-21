"""
Extract Activations with Instruction Prefix Variants

Extracts mean-pooled residual stream activations for secure and vulnerable code
with and without security instruction prefixes.

Compares:
  - Baseline: code presented as-is
  - With instruction: "Analyze the following code for security vulnerabilities:\n\n" + code

Usage:
    conda run -n sae python instruction_activations_extract.py

Saves:
    artifacts/activations/instruction_comparison/<run_ts>/
        baseline_safe_layer_{L}.pt          [N_baseline, d_model]
        baseline_vulnerable_layer_{L}.pt    [N_baseline, d_model]
        instruction_safe_layer_{L}.pt       [N_instruction, d_model]
        instruction_vulnerable_layer_{L}.pt [N_instruction, d_model]
        meta.json                           {run_ts, n_pairs, layers, ...}
"""

import base64
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
ARTIFACTS = Path(__file__).parents[2] / "artifacts" / "activations"

RAW_RUN_DIR = (
    ARTIFACTS / "raw_activations" / "vulnerable_code_qwen_coder_standard_16384_raw"
)
SOURCE_JSONL = (
    RAW_RUN_DIR / "activations_layer_0_raw_component_hidden_state_last_token.jsonl"
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]
MAX_TOKENS = 2048

# Instruction variants
INSTRUCTION_SECURITY = "Analyze the following code for security vulnerabilities:\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────


def load_source_records(jsonl_path: Path):
    """Load (vuln_id, secure_code, vulnerable_code, cwe, file_extension) from JSONL."""
    records = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            try:
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
                    "vuln_id": r["vuln_id"],
                    "cwe": r["cwe"],
                    "file_extension": r["file_extension"],
                    "secure_code": secure_code,
                    "vulnerable_code": vuln_code,
                }
            )
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Activation extraction
# ─────────────────────────────────────────────────────────────────────────────


def extract_activations_with_instruction(
    code_str: str,
    tokenizer,
    model,
    target_layers: list,
    device,
    instruction: str = None,
    max_tokens: int = MAX_TOKENS,
) -> dict[int, np.ndarray]:
    """
    Run code through model with optional instruction prefix.
    Return {layer_idx: mean_pooled_vector [d_model]}.
    """
    # Prepend instruction if provided
    if instruction is not None:
        text = instruction + code_str
    else:
        text = code_str

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    ).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract mean-pooled activations
    result = {}
    for layer_idx in target_layers:
        h = outputs.hidden_states[layer_idx + 1]  # [1, seq_len, d_model]
        mean_vec = h[0].float().mean(dim=0).cpu().numpy()
        result[layer_idx] = mean_vec

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("[*] Loading data...")
    records = load_source_records(SOURCE_JSONL)
    print(f"    Loaded {len(records)} records")

    # Load model and tokenizer
    print("[*] Loading model and tokenizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map=device, torch_dtype=torch.float16
    )
    model.eval()

    # Create output directory
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ARTIFACTS / "instruction_comparison" / run_ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize activation storage
    baseline_safe = {l: [] for l in LAYERS}
    baseline_vuln = {l: [] for l in LAYERS}
    instruction_safe = {l: [] for l in LAYERS}
    instruction_vuln = {l: [] for l in LAYERS}

    # Extract activations
    print("[*] Extracting activations (baseline)...")
    for i, record in enumerate(records):
        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(records)}]")

        # Baseline secure
        try:
            acts = extract_activations_with_instruction(
                record["secure_code"],
                tokenizer,
                model,
                LAYERS,
                device,
                instruction=None,
            )
            for layer in LAYERS:
                baseline_safe[layer].append(acts[layer])
        except Exception as e:
            print(
                f"    Warning: baseline safe extraction failed for {record['vuln_id']}: {e}"
            )
            continue

        # Baseline vulnerable
        try:
            acts = extract_activations_with_instruction(
                record["vulnerable_code"],
                tokenizer,
                model,
                LAYERS,
                device,
                instruction=None,
            )
            for layer in LAYERS:
                baseline_vuln[layer].append(acts[layer])
        except Exception as e:
            print(
                f"    Warning: baseline vuln extraction failed for {record['vuln_id']}: {e}"
            )
            continue

    print(f"    Baseline: {len(baseline_safe[LAYERS[0]])} pairs extracted")

    print("[*] Extracting activations (with instruction)...")
    for i, record in enumerate(records):
        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(records)}]")

        # Instruction secure
        try:
            acts = extract_activations_with_instruction(
                record["secure_code"],
                tokenizer,
                model,
                LAYERS,
                device,
                instruction=INSTRUCTION_SECURITY,
            )
            for layer in LAYERS:
                instruction_safe[layer].append(acts[layer])
        except Exception as e:
            print(
                f"    Warning: instruction safe extraction failed for {record['vuln_id']}: {e}"
            )
            continue

        # Instruction vulnerable
        try:
            acts = extract_activations_with_instruction(
                record["vulnerable_code"],
                tokenizer,
                model,
                LAYERS,
                device,
                instruction=INSTRUCTION_SECURITY,
            )
            for layer in LAYERS:
                instruction_vuln[layer].append(acts[layer])
        except Exception as e:
            print(
                f"    Warning: instruction vuln extraction failed for {record['vuln_id']}: {e}"
            )
            continue

    print(f"    Instruction: {len(instruction_safe[LAYERS[0]])} pairs extracted")

    # Convert to numpy arrays
    print("[*] Converting to arrays...")
    for layer in LAYERS:
        baseline_safe[layer] = np.array(baseline_safe[layer])
        baseline_vuln[layer] = np.array(baseline_vuln[layer])
        instruction_safe[layer] = np.array(instruction_safe[layer])
        instruction_vuln[layer] = np.array(instruction_vuln[layer])

    # Save activations
    print("[*] Saving activations...")
    for layer in LAYERS:
        torch.save(
            torch.from_numpy(baseline_safe[layer]),
            out_dir / f"baseline_safe_layer_{layer}.pt",
        )
        torch.save(
            torch.from_numpy(baseline_vuln[layer]),
            out_dir / f"baseline_vulnerable_layer_{layer}.pt",
        )
        torch.save(
            torch.from_numpy(instruction_safe[layer]),
            out_dir / f"instruction_safe_layer_{layer}.pt",
        )
        torch.save(
            torch.from_numpy(instruction_vuln[layer]),
            out_dir / f"instruction_vulnerable_layer_{layer}.pt",
        )

    # Save metadata
    meta = {
        "run_ts": run_ts,
        "layers": LAYERS,
        "model_id": MODEL_ID,
        "instruction_prefix": INSTRUCTION_SECURITY,
        "n_baseline_pairs": len(baseline_safe[LAYERS[0]]),
        "n_instruction_pairs": len(instruction_safe[LAYERS[0]]),
        "shapes": {
            "baseline_safe": baseline_safe[LAYERS[0]].shape,
            "baseline_vulnerable": baseline_vuln[LAYERS[0]].shape,
            "instruction_safe": instruction_safe[LAYERS[0]].shape,
            "instruction_vulnerable": instruction_vuln[LAYERS[0]].shape,
        },
    }
    meta_path = out_dir / "meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[+] Activations saved to {out_dir}")
    print(f"[+] Metadata saved to {meta_path}")
    print(f"[+] Run timestamp: {run_ts}")

    return out_dir


if __name__ == "__main__":
    main()
