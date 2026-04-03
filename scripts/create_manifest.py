"""
Create manifest CSV for the DOLOS dataset.

Scans Data/DOLOS-all/resizedVideosNew/ to enumerate valid sample IDs,
parses labels from filenames, and maps each to its original video path.

Usage:
    python scripts/create_manifest.py
"""

import os
import re
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESIZED_DIR = ROOT / "Data" / "DOLOS-all" / "resizedVideosNew"
ORIGINAL_DIR = ROOT / "Data" / "DOLOS-all" / "DOLOS"
OUTPUT_CSV = ROOT / "Data" / "manifest_dolos.csv"


def parse_label(stem: str) -> int:
    """
    Parse label from filename stem.
    - 'lie' or 'deception' → 0
    - 'truth' or 'true' → 1
    Handles variants with spaces (e.g. '_ lie1') and abbreviations ('_true').
    Raises ValueError if neither is found.
    """
    # Normalize: collapse spaces around underscores, lowercase
    lower = re.sub(r'_\s+', '_', stem.lower())
    if re.search(r'_lie[_\d]|_lie$|deception', lower):
        return 0
    if re.search(r'_truth[_\d]|_truth$|_true[_\d]|_true$', lower):
        return 1
    raise ValueError(f"Cannot determine label from stem: {stem!r}")


def main():
    if not RESIZED_DIR.exists():
        raise FileNotFoundError(f"resizedVideosNew not found: {RESIZED_DIR}")

    rows = []
    skipped = []

    for fname in sorted(RESIZED_DIR.iterdir()):
        if not fname.suffix.lower() == ".mp4":
            continue
        # Strip "[R] " prefix
        name = fname.name
        if name.startswith("[R] "):
            stem = name[4:]  # remove "[R] "
        else:
            stem = name
        # Remove extension
        sample_id = Path(stem).stem

        try:
            label = parse_label(sample_id)
        except ValueError as e:
            skipped.append((sample_id, str(e)))
            continue

        original_path = ORIGINAL_DIR / f"{sample_id}.mp4"
        if not original_path.exists():
            skipped.append((sample_id, f"Original not found: {original_path}"))
            continue

        rows.append({
            "sample_id": sample_id,
            "label": label,
            "dataset_source": "dolos",
            "video_path": str(original_path),
        })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "label", "dataset_source", "video_path"])
        writer.writeheader()
        writer.writerows(rows)

    n_lie = sum(1 for r in rows if r["label"] == 0)
    n_truth = sum(1 for r in rows if r["label"] == 1)
    print(f"Manifest written to: {OUTPUT_CSV}")
    print(f"  Total samples : {len(rows)}")
    print(f"  Lie  (label=0): {n_lie}")
    print(f"  Truth(label=1): {n_truth}")
    if skipped:
        print(f"\nSkipped {len(skipped)} files:")
        for sid, reason in skipped:
            print(f"  {sid}: {reason}")


if __name__ == "__main__":
    main()
