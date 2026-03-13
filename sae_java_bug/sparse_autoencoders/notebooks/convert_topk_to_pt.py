"""Convert TOPK JSONL activations to stacked .pt tensors."""
import json
import torch
from pathlib import Path

JSONL = Path(__file__).parents[2] / "artifacts/activations/TOPK" / (
    "activations_layer_11_sae_blocks.11.hook_resid_post_"
    "component_hook_resid_post.hook_sae_acts_post.jsonl"
)
OUT = JSONL.parent

safe_vecs, vuln_vecs = [], []
bad = 0
with JSONL.open() as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"  [skip] line {i}: {e}")
            bad += 1
            continue
        safe_vecs.append(r["secure"])
        vuln_vecs.append(r["vulnerable"])

print(f"Loaded {len(safe_vecs)} records ({bad} skipped)")

safe_t = torch.FloatTensor(safe_vecs)
vuln_t = torch.FloatTensor(vuln_vecs)

torch.save(safe_t, OUT / "safe_layer_11.pt")
torch.save(vuln_t, OUT / "vulnerable_layer_11.pt")
print(f"Saved: safe {safe_t.shape}, vuln {vuln_t.shape}")
print(f"Output dir: {OUT}")
