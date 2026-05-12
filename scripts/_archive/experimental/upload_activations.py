#!/usr/bin/env python3
"""
Upload activation artifacts to HuggingFace Datasets.
Each subdirectory in TO_UPoad/ becomes a separate config/split.
Base64-encoded secure_code and vulnerable_code fields are decoded before upload.

Usage:
    python scripts/upload_activations.py <hf_repo_id> [--base-dir artifacts/activation/TO_UPoad]

Example:
    python scripts/upload_activations.py rufimelo/my-activations
    python scripts/upload_activations.py rufimelo/my-activations --base-dir /path/to/TO_UPoad --private
"""

import argparse
import base64
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def build_readme(splits: list) -> str:
    """Generate README.md YAML front matter so HF recognises each layer as a named config."""
    lines = ["---", "configs:"]
    for split_dir in splits:
        name = split_dir.name
        lines += [
            f"- config_name: {name}",
            f"  data_files:",
            f"  - split: train",
            f"    path: data/{name}/*.jsonl",
        ]
    lines.append("---")
    return "\n".join(lines) + "\n"


def decode_jsonl(src: Path, dst: Path):
    """Copy a JSONL file, decoding base64 secure_code and vulnerable_code fields."""
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            record = json.loads(line)
            for field in ("secure_code", "vulnerable_code"):
                if field in record and record[field]:
                    try:
                        record[field] = base64.b64decode(record[field]).decode("utf-8")
                    except Exception:
                        pass  # leave as-is if not valid base64
            fout.write(json.dumps(record) + "\n")


def prepare_split(split_dir: Path, tmp_dir: Path) -> Path:
    """Mirror split_dir into tmp_dir, decoding JSONL files along the way."""
    out_dir = tmp_dir / split_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for src_file in split_dir.rglob("*"):
        if not src_file.is_file():
            continue
        dst_file = out_dir / src_file.relative_to(split_dir)
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if src_file.suffix == ".jsonl":
            decode_jsonl(src_file, dst_file)
        else:
            shutil.copy2(src_file, dst_file)

    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Upload activations to HuggingFace Datasets")
    parser.add_argument("repo_id", help="HuggingFace repo ID (e.g., username/dataset-name)")
    parser.add_argument(
        "--base-dir",
        default="artifacts/activation/TO_UPoad",
        help="Base directory containing layer subdirectories (default: artifacts/activation/TO_UPoad)",
    )
    parser.add_argument("--private", action="store_true", help="Make the dataset private")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise SystemExit(f"ERROR: Directory not found: {base_dir}")

    splits = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    if not splits:
        raise SystemExit(f"ERROR: No subdirectories found in {base_dir}")

    print(f"Found {len(splits)} split(s): {[s.name for s in splits]}")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True, private=args.private)
    print(f"Repo: https://huggingface.co/datasets/{args.repo_id}\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # Upload README with config definitions so HF recognises named configs
        readme_path = tmp_dir / "README.md"
        readme_path.write_text(build_readme(splits))
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print("Uploaded README.md with config definitions.\n")

        for split_dir in splits:
            split_name = split_dir.name
            file_count = sum(1 for f in split_dir.rglob("*") if f.is_file())
            print(f"Decoding and uploading split '{split_name}' ({file_count} file(s))...")

            prepared = prepare_split(split_dir, tmp_dir)
            api.upload_folder(
                folder_path=str(prepared),
                repo_id=args.repo_id,
                repo_type="dataset",
                path_in_repo=f"data/{split_name}",
            )
            print(f"  Done: {split_name}")

    print(f"\nAll {len(splits)} split(s) uploaded to: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
