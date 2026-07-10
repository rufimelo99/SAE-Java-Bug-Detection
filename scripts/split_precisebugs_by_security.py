#!/usr/bin/env python3
"""
Split existing PreciseBugs NPZ activation files into two subsets:
  - precisebugs_security_*   : records with a known CWE (security bugs)
  - precisebugs_nonsecurity_*: records with cwe == "unknown" (non-security bugs)

All arrays in each NPZ are sliced along axis 0; dtype and keys are preserved.
Corresponding filtered JSONL metadata files are also written so downstream
scripts can resolve labels without modification.

Usage:
    python scripts/split_precisebugs_by_security.py
    python scripts/split_precisebugs_by_security.py --dry-run
"""

import argparse
import io
import json
import logging
import zipfile
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).parent
_PROJECT_DIR = _SCRIPTS_DIR.parent

ACTIVATIONS_DIR = _PROJECT_DIR / "sae_java_bug" / "artifacts" / "multi_model_probing"
METADATA_JSONL = (
    _PROJECT_DIR
    / "sae_java_bug"
    / "artifacts"
    / "data"
    / "precisebugs_raw"
    / "precisebugs_c_pairs.jsonl"
)
DATA_DIR = METADATA_JSONL.parent


def _normalize_cwe(cwe: str) -> str:
    if cwe.startswith("CWE-"):
        return "CWE-" + str(int(cwe[4:]))
    return cwe


def load_masks():
    """Return (records, security_mask, nonsecurity_mask) from the metadata JSONL."""
    with open(METADATA_JSONL) as f:
        records = [json.loads(l) for l in f if l.strip()]

    cwes = np.array([_normalize_cwe(r.get("cwe", "unknown")) for r in records])
    security_mask = cwes != "unknown"
    nonsecurity_mask = ~security_mask

    logger.info(
        f"Total records: {len(records)} | "
        f"Security (known CWE): {security_mask.sum()} | "
        f"Non-security (unknown): {nonsecurity_mask.sum()}"
    )
    return records, security_mask, nonsecurity_mask


def write_metadata_jsonl(records, mask, out_path: Path, dry_run: bool):
    subset = [r for r, keep in zip(records, mask) if keep]
    if dry_run:
        logger.info(f"[dry-run] Would write {len(subset)} records → {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in subset:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Wrote {len(subset)} records → {out_path}")


def split_npz(npz_path: Path, security_idx, nonsecurity_idx, dry_run: bool):
    stem = npz_path.stem.replace("activations_precisebugs_", "")
    out_sec = ACTIVATIONS_DIR / f"activations_precisebugs_security_{stem}.npz"
    out_non = ACTIVATIONS_DIR / f"activations_precisebugs_nonsecurity_{stem}.npz"

    if dry_run:
        logger.info(f"[dry-run] {npz_path.name} → {out_sec.name} + {out_non.name}")
        return

    logger.info(f"Splitting {npz_path.name} (streaming, one array at a time) ...")

    # NPZ files are ZIP archives of .npy files — stream one key at a time
    # so we never hold more than one array (~67 MB) in memory.
    with zipfile.ZipFile(npz_path, "r") as src, \
         zipfile.ZipFile(out_sec, "w", compression=zipfile.ZIP_STORED) as zsec, \
         zipfile.ZipFile(out_non, "w", compression=zipfile.ZIP_STORED) as znon:

        names = src.namelist()
        for i, name in enumerate(names, 1):
            logger.info(f"  [{i}/{len(names)}] {name}")
            arr = np.load(io.BytesIO(src.read(name)))

            for zout, idx in ((zsec, security_idx), (znon, nonsecurity_idx)):
                buf = io.BytesIO()
                np.save(buf, arr[idx])
                buf.seek(0)
                zout.writestr(name, buf.read())

    logger.info(f"  Done: {out_sec.name} + {out_non.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be created without writing anything")
    args = parser.parse_args()

    records, security_mask, nonsecurity_mask = load_masks()
    security_idx = np.where(security_mask)[0]
    nonsecurity_idx = np.where(nonsecurity_mask)[0]

    # Write filtered metadata JSONL files
    write_metadata_jsonl(
        records, security_mask,
        DATA_DIR / "precisebugs_security_c_pairs.jsonl",
        args.dry_run,
    )
    write_metadata_jsonl(
        records, nonsecurity_mask,
        DATA_DIR / "precisebugs_nonsecurity_c_pairs.jsonl",
        args.dry_run,
    )

    # Split each NPZ
    npz_files = sorted(ACTIVATIONS_DIR.glob("activations_precisebugs_[!s][!e][!c][!u]*.npz"))
    # Exclude already-split files (security_ / nonsecurity_)
    npz_files = [
        p for p in ACTIVATIONS_DIR.glob("activations_precisebugs_*.npz")
        if "_security_" not in p.name and "_nonsecurity_" not in p.name
    ]

    if not npz_files:
        logger.error(f"No precisebugs NPZ files found in {ACTIVATIONS_DIR}")
        return

    logger.info(f"Found {len(npz_files)} NPZ files to split:")
    for p in npz_files:
        logger.info(f"  {p.name}")

    for npz_path in sorted(npz_files):
        split_npz(npz_path, security_idx, nonsecurity_idx, args.dry_run)

    logger.info("\nDone. New files created:")
    if not args.dry_run:
        for p in sorted(ACTIVATIONS_DIR.glob("activations_precisebugs_security_*.npz")):
            size_gb = p.stat().st_size / 1e9
            logger.info(f"  {p.name}  ({size_gb:.2f} GB)")
        for p in sorted(ACTIVATIONS_DIR.glob("activations_precisebugs_nonsecurity_*.npz")):
            size_gb = p.stat().st_size / 1e9
            logger.info(f"  {p.name}  ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
