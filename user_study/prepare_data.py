"""
Prepares a compact study dataset from the raw activation and hypothesis files.

Run once before launching the Streamlit app:
    python prepare_data.py

Outputs:
    data/study_data.jsonl   - compact records with decoded code + top features
    data/hypotheses.json    - feature index → hypothesis lookup
"""

import base64
import json
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent / "sae_java_bug"

HYPOTHESES_FILE = ROOT / "sparse_autoencoders" / "my_hypotheses_layer11.jsonl"
ACTIVATIONS_FILE = (
    ROOT
    / "artifacts"
    / "activations"
    / "TO_UPLOAD"
    / "layer11"
    / "activations_layer_11_sae_blocks.11.hook_resid_post_component_hook_resid_post.hook_sae_acts_post.jsonl"
)

OUT_DIR = Path(__file__).parent / "data"
OUT_STUDY = OUT_DIR / "study_data.jsonl"
OUT_HYPOTHESES = OUT_DIR / "hypotheses.json"

TOP_K = 20  # number of top features to keep per sample


def load_hypotheses(path: Path) -> dict[int, dict]:
    hypotheses: dict[int, dict] = {}
    print(f"Loading hypotheses from {path} …")
    with open(path) as f:
        for line in f:
            h = json.loads(line)
            hypotheses[h["feature_idx"]] = {
                "hypothesis": h.get("hypothesis", ""),
                "confidence": h.get("confidence", ""),
                "notes": h.get("notes", ""),
                "n_nonzero": h.get("n_nonzero", 0),
                "max_activation": h.get("max_activation", 0.0),
            }
    print(f"  Loaded {len(hypotheses):,} feature hypotheses.")
    return hypotheses


def decode_code(b64: str) -> str:
    try:
        return base64.b64decode(b64).decode("utf-8", errors="replace")
    except Exception:
        return b64  # return as-is if decoding fails


def top_features(secure: list[float], vulnerable: list[float], k: int) -> list[int]:
    """Return indices of the k features with the largest |vuln - secure| diff."""
    diffs = [abs(v - s) for s, v in zip(secure, vulnerable)]
    return sorted(range(len(diffs)), key=lambda i: diffs[i], reverse=True)[:k]


def process(hypotheses: dict[int, dict]) -> None:
    total = 0
    print(f"Processing activations from {ACTIVATIONS_FILE} …")
    print(f"  Keeping top {TOP_K} features per sample.")

    with open(ACTIVATIONS_FILE) as fin, open(OUT_STUDY, "w") as fout:
        for line in fin:
            record = json.loads(line)

            sec_acts: list[float] = record["secure"]
            vul_acts: list[float] = record["vulnerable"]

            top_idx = top_features(sec_acts, vul_acts, TOP_K)

            features = []
            for i in top_idx:
                h = hypotheses.get(i, {})
                features.append(
                    {
                        "feature_idx": i,
                        "secure_activation": round(sec_acts[i], 6),
                        "vulnerable_activation": round(vul_acts[i], 6),
                        "diff": round(vul_acts[i] - sec_acts[i], 6),
                        "hypothesis": h.get("hypothesis", "No hypothesis available."),
                        "confidence": h.get("confidence", ""),
                        "notes": h.get("notes", ""),
                        "n_nonzero": h.get("n_nonzero", 0),
                        "max_activation": h.get("max_activation", 0.0),
                    }
                )

            compact = {
                "vuln_id": record["vuln_id"],
                "cwe": record["cwe"],
                "file_extension": record.get("file_extension", ""),
                "secure_code": decode_code(record["secure_code"]),
                "vulnerable_code": decode_code(record["vulnerable_code"]),
                "top_features": features,
            }
            fout.write(json.dumps(compact) + "\n")
            total += 1

            if total % 100 == 0:
                print(f"  … {total} records processed", end="\r", flush=True)

    print(f"\n  Done. {total} records written to {OUT_STUDY}")


def save_hypotheses(hypotheses: dict[int, dict]) -> None:
    # Save with string keys for JSON compatibility
    out = {str(k): v for k, v in hypotheses.items()}
    with open(OUT_HYPOTHESES, "w") as f:
        json.dump(out, f)
    print(f"Hypotheses saved to {OUT_HYPOTHESES}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not HYPOTHESES_FILE.exists():
        print(f"ERROR: hypotheses file not found at {HYPOTHESES_FILE}", file=sys.stderr)
        sys.exit(1)
    if not ACTIVATIONS_FILE.exists():
        print(f"ERROR: activations file not found at {ACTIVATIONS_FILE}", file=sys.stderr)
        sys.exit(1)

    hypotheses = load_hypotheses(HYPOTHESES_FILE)
    save_hypotheses(hypotheses)
    process(hypotheses)
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
