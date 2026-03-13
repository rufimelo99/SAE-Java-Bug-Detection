"""
collect_per_token_sae.py
========================
Collects per-token SAE feature activations for all 2493 paired samples
at Layer 11, saving them as a compact JSONL for downstream PCA analysis.

Output record format:
  {
    "vuln_id": str,
    "cwe": str,
    "file_extension": str,
    "n_secure_tokens": int,
    "n_vuln_tokens": int,
    "secure":     [[f0, f1, ...], ...],   # [n_tokens, 16384] — sparse: only nonzero
    "vulnerable": [[f0, f1, ...], ...],   # [n_tokens, 16384]
  }

To keep file size manageable, each per-token vector is stored as a sparse dict
  {"indices": [...], "values": [...]}  (only nonzero SAE features)

Run on VM:
  python collect_per_token_sae.py \
      --out artifacts/activations/per_token_sae_l11.jsonl \
      --layer 11 \
      --batch_size 1
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--out",        default="artifacts/activations/per_token_sae_l11.jsonl")
parser.add_argument("--layer",      type=int, default=11)
parser.add_argument("--sae_release", default="rufimelo/vulnerable_code_qwen_coder_standard_16384_10M")
parser.add_argument("--model",      default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--max_tokens", type=int, default=256,
                    help="Truncate sequences to this many tokens")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

HOOK_NAME = f"blocks.{args.layer}.hook_resid_post"
SAE_ID    = f"blocks.{args.layer}.hook_resid_post"

print(f"Device: {args.device}")
print(f"Layer: {args.layer}  hook: {HOOK_NAME}")
print(f"Output: {args.out}")

# ── Load model + SAE ─────────────────────────────────────────────────────────
print("\nLoading model …")
model = HookedTransformer.from_pretrained(args.model, device=args.device)
model.eval()

print("Loading SAE …")
sae, _, _ = SAE.from_pretrained(
    release=args.sae_release,
    sae_id=SAE_ID,
    device=args.device,
)
sae.eval()

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading dataset …")
ds = load_dataset("rufimelo/DeltaSecommits", split="train")

# ── Helper: encode one code string → per-token SAE activations ───────────────
def encode_code(code: str):
    """Returns list of sparse dicts, one per token (up to max_tokens)."""
    tokens = model.to_tokens(code, prepend_bos=True)[0]   # [T]
    tokens = tokens[:args.max_tokens].unsqueeze(0)         # [1, T]

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=HOOK_NAME,
            device=args.device,
        )
        resid = cache[HOOK_NAME][0]          # [T, d_model]
        acts  = sae.encode(resid)            # [T, d_sae]

    # Store as sparse: only nonzero features per token
    result = []
    for t in range(acts.shape[0]):
        row = acts[t]
        nz  = (row != 0).nonzero(as_tuple=True)[0]
        result.append({
            "indices": nz.tolist(),
            "values":  row[nz].tolist(),
        })
    return result, acts.shape[0]


# ── Main loop ─────────────────────────────────────────────────────────────────
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)

n_done = 0
# Resume support: count already-written lines
if out_path.exists():
    with open(out_path) as f:
        n_done = sum(1 for _ in f)
    print(f"Resuming from record {n_done}")

with open(out_path, "a") as fout:
    for i, rec in enumerate(ds):
        if i < n_done:
            continue

        vuln_id = rec.get("vuln_id") or rec.get("id") or str(i)
        secure_code     = rec.get("prior_version",   rec.get("secure_code",     rec.get("fixed",    "")))
        vulnerable_code = rec.get("after_version",   rec.get("vulnerable_code", rec.get("vulnerable", "")))
        cwe             = rec.get("cwe", "")
        ext             = rec.get("file_extension", "")

        try:
            sec_acts, n_sec = encode_code(secure_code)
            vul_acts, n_vul = encode_code(vulnerable_code)
        except Exception as e:
            print(f"  [skip] {vuln_id}: {e}")
            continue

        out_rec = {
            "vuln_id":        vuln_id,
            "cwe":            cwe,
            "file_extension": ext,
            "n_secure_tokens":     n_sec,
            "n_vulnerable_tokens": n_vul,
            "secure":     sec_acts,
            "vulnerable": vul_acts,
        }
        fout.write(json.dumps(out_rec) + "\n")
        fout.flush()

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(ds)} done")

print(f"\nFinished. Output: {out_path}")
