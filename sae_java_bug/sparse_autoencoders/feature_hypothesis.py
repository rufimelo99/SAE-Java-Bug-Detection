"""
Map SAE feature positions to semantic hypotheses using stored activations.

For each active feature in the 16,384-dimensional SAE representation:
  - Collects the 10 highest and 10 lowest non-zero activating code examples
    (combining both secure and vulnerable activations)
  - Asks Claude via AWS Bedrock to formulate a hypothesis about the feature
  - Saves feature_idx → hypothesis mapping to a JSONL file

The script is resumable: already-processed feature indices are skipped.

Usage:
    python feature_hypothesis.py --run_id run_20260218_134529_vulnerable_code_qwen_coder_standard_16384_10M
    python feature_hypothesis.py --run_id <run_id> --layer 11 --output my_hypotheses.jsonl --region us-east-1
"""

import argparse
import base64
import json
from pathlib import Path

import boto3
import numpy as np
from tqdm import tqdm

# ── defaults ─────────────────────────────────────────────────────────────────
BEDROCK_MODEL_ID = "global.anthropic.claude-opus-4-6-v1"
TOP_K = 10
MIN_ACTIVATION = 1e-4   # features whose max is below this are skipped (all-zero)
MAX_CODE_CHARS = 4048    # truncate each code snippet in the prompt


# ── data loading ─────────────────────────────────────────────────────────────

def load_activations(jsonl_path: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Load records and build (n_samples × n_features) matrices for secure and vulnerable."""
    records: list[dict] = []
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    n = len(records)
    d = len(records[0]["secure"])
    secure_mat = np.empty((n, d), dtype=np.float32)
    vuln_mat = np.empty((n, d), dtype=np.float32)

    for i, r in enumerate(records):
        secure_mat[i] = r["secure"]
        vuln_mat[i] = r["vulnerable"]

    return records, secure_mat, vuln_mat


def decode_code(b64_str: str) -> str:
    try:
        return base64.b64decode(b64_str).decode("utf-8", errors="replace")
    except Exception:
        return b64_str


def already_processed(output_path: Path) -> set[int]:
    seen: set[int] = set()
    if not output_path.exists():
        return seen
    with output_path.open() as f:
        for line in f:
            try:
                seen.add(json.loads(line)["feature_idx"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def find_jsonl_file(run_dir: Path, layer: int) -> Path:
    matches = list(run_dir.glob(f"activations_layer_{layer}_*.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No JSONL activation file for layer {layer} in {run_dir}")
    return matches[0]


# ── LLM interaction ───────────────────────────────────────────────────────────

def build_prompt(
    feature_idx: int,
    top_examples: list[dict],
    bottom_examples: list[dict],
) -> str:
    def fmt_block(examples: list[dict]) -> str:
        parts = []
        for ex in examples:
            header = (
                f"[activation={ex['activation']:.4f} | label={ex['label']} | "
                f"cwe={ex['cwe']} | lang={ex['file_extension']}]"
            )
            snippet = ex["code"][:MAX_CODE_CHARS].replace("\n", "\n  ")
            parts.append(f"{header}\n  ```\n  {snippet}\n  ```")
        return "\n\n".join(parts)

    return f"""You are interpreting a Sparse Autoencoder (SAE) trained on a code vulnerability detection model (Qwen2.5-7B-Instruct, layer 11).

Each code snippet below either comes from the secure version or the vulnerable version of a paired commit.
Your task: based on the code patterns, vulnerability labels, CWE types, and activation magnitudes, hypothesise what semantic concept SAE feature #{feature_idx} encodes.

## Top {len(top_examples)} highest activations (feature fires most strongly here):

{fmt_block(top_examples)}

## {len(bottom_examples)} lowest non-zero activations (feature fires weakly here):

{fmt_block(bottom_examples)}

Respond with exactly three lines:
HYPOTHESIS: <one concise sentence describing what this feature likely encodes>
CONFIDENCE: <low|medium|high>
NOTES: <brief observation about patterns, caveats, or mixed signals — or "none">"""


def call_bedrock(client, prompt: str, model_id: str) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"].strip()


def parse_response(text: str) -> dict:
    hypothesis = confidence = notes = ""
    for line in text.splitlines():
        if line.startswith("HYPOTHESIS:"):
            hypothesis = line[len("HYPOTHESIS:"):].strip()
        elif line.startswith("CONFIDENCE:"):
            confidence = line[len("CONFIDENCE:"):].strip()
        elif line.startswith("NOTES:"):
            notes = line[len("NOTES:"):].strip()
    return {"hypothesis": hypothesis, "confidence": confidence, "notes": notes}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Map SAE features to semantic hypotheses via AWS Bedrock."
    )
    parser.add_argument(
        "--run_id", type=str, required=True,
        help="Run directory name under artifacts/activations/"
    )
    parser.add_argument("--layer", type=int, default=11, help="Layer to analyse (default: 11)")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL path (default: <run_dir>/feature_hypotheses_layer_<N>.jsonl)"
    )
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--model", type=str, default=BEDROCK_MODEL_ID, help="Bedrock model ID")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent / "artifacts" / "activations"
    run_dir = base_dir / args.run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    jsonl_path = find_jsonl_file(run_dir, args.layer)
    output_path = (
        Path(args.output) if args.output
        else run_dir / f"feature_hypotheses_layer_{args.layer}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading activations from {jsonl_path.name} ...")
    records, secure_mat, vuln_mat = load_activations(jsonl_path)
    n_samples, n_features = vuln_mat.shape
    print(f"  {n_samples} records  ×  {n_features} features")

    seen = already_processed(output_path)
    if seen:
        print(f"Resuming — {len(seen)} features already written.")

    # Only process features that ever meaningfully activate (vulnerable activations only)
    max_per_feature = vuln_mat.max(axis=0)
    active_features = np.where(max_per_feature >= MIN_ACTIVATION)[0]
    print(f"  {len(active_features)} / {n_features} features active (max >= {MIN_ACTIVATION})")

    bedrock = boto3.client("bedrock-runtime", region_name=args.region)

    with output_path.open("a") as out_f:
        for feat_idx in tqdm(active_features, desc="Hypothesising features"):
            feat_idx = int(feat_idx)
            if feat_idx in seen:
                continue

            acts = vuln_mat[:, feat_idx]
            nonzero_idx = np.where(acts >= MIN_ACTIVATION)[0]

            if len(nonzero_idx) == 0:
                continue

            order_desc = nonzero_idx[np.argsort(acts[nonzero_idx])[::-1]]
            order_asc = nonzero_idx[np.argsort(acts[nonzero_idx])]

            top_idx = order_desc[:TOP_K]
            bottom_idx = order_asc[:TOP_K]

            def make_examples(indices):
                out = []
                for idx in indices:
                    rec = records[idx]
                    out.append({
                        "activation": float(acts[idx]),
                        "label": "vulnerable",
                        "cwe": rec["cwe"],
                        "file_extension": rec["file_extension"],
                        "vuln_id": rec["vuln_id"],
                        "code": decode_code(rec["vulnerable_code"]),
                    })
                return out

            top_examples = make_examples(top_idx)
            bottom_examples = make_examples(bottom_idx)

            prompt = build_prompt(feat_idx, top_examples, bottom_examples)

            try:
                raw_response = call_bedrock(bedrock, prompt, args.model)
                parsed = parse_response(raw_response)
            except Exception as e:
                print(f"\nError for feature {feat_idx}: {e}")
                continue

            result = {
                "feature_idx": feat_idx,
                "n_nonzero": int(len(nonzero_idx)),
                "max_activation": float(acts.max()),
                "hypothesis": parsed["hypothesis"],
                "confidence": parsed["confidence"],
                "notes": parsed["notes"],
                "raw_response": raw_response,
            }
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

    print(f"\nDone. Hypotheses saved to {output_path}")


if __name__ == "__main__":
    main()