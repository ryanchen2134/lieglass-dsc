"""
One-time frame extraction: resizedVideosNew → features/{id}/frames.npz

Reads each [R] {id}.mp4 from resizedVideosNew, uniformly samples n_frames,
detects black frames (speaker not talking), and saves a compressed numpy
archive containing:

  frames.npz:
    'frames'  uint8   (n_frames, H, W, 3)   — RGB uint8, no normalization
    'mask'    bool    (n_frames,)             — True = speaker visible (not black)

Why .npz instead of .mp4?
  - No OpenCV at training time → __getitem__ is pure numpy → num_workers safe
  - Compressed uint8 face crops: ~1–3 MB per sample (vs decoding 64 H.264 seeks)
  - Normalization (uint8 → float32 ImageNet) is a cheap in-process op

Usage (from project root):
    python -m deception_detection.data.preprocessing.extract_frames \\
        --resized_dir Data/DOLOS-all/resizedVideosNew \\
        --feature_dir features \\
        --n_frames 64 \\
        --workers 8
"""

import os
import csv
import multiprocessing
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BLACK_THRESHOLD = 8   # mean pixel value below this → black frame


# ---------------------------------------------------------------------------
# Per-sample extraction
# ---------------------------------------------------------------------------

def _extract(args: tuple) -> tuple[str, bool, str]:
    sample_id, src_str, out_str, n_frames, force = args
    src = Path(src_str)
    out = Path(out_str)

    if not force and out.exists():
        return sample_id, True, ""

    try:
        cap = cv2.VideoCapture(str(src))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            cap.release()
            frames_arr = np.zeros((n_frames, 224, 224, 3), dtype=np.uint8)
            mask_arr   = np.zeros(n_frames, dtype=bool)
            np.savez_compressed(str(out), frames=frames_arr, mask=mask_arr)
            return sample_id, True, ""

        indices = np.linspace(0, total - 1, n_frames, dtype=int)
        frames, mask = [], []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                mask.append(False)
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            is_visible = bool(frame.mean() > BLACK_THRESHOLD)
            frames.append(frame)
            mask.append(is_visible)

        cap.release()
        np.savez_compressed(
            str(out),
            frames=np.stack(frames),            # (n, 224, 224, 3) uint8
            mask=np.array(mask, dtype=bool),    # (n,)
        )
        return sample_id, True, ""

    except Exception:
        import traceback
        return sample_id, False, traceback.format_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract n_frames from resizedVideosNew → features/{id}/frames.npz"
    )
    parser.add_argument("--manifest",    default="Data/manifest_dolos.csv")
    parser.add_argument("--resized_dir", default="Data/DOLOS-all/resizedVideosNew",
                        help="Directory containing [R] {sample_id}.mp4 files")
    parser.add_argument("--feature_dir", default="features")
    parser.add_argument("--n_frames",    type=int, default=64)
    parser.add_argument("--workers",     type=int,
                        default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--force",       action="store_true",
                        help="Re-extract even if frames.npz already exists")
    parser.add_argument("--sample_id",   default=None,
                        help="Process a single sample (for debugging)")
    args = parser.parse_args()

    resized_dir = PROJECT_ROOT / args.resized_dir
    feature_dir = PROJECT_ROOT / args.feature_dir
    manifest    = PROJECT_ROOT / args.manifest

    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    if args.sample_id:
        rows = [r for r in rows if r["sample_id"] == args.sample_id]
        if not rows:
            print(f"Sample not found: {args.sample_id}")
            return

    pending, skipped, missing = [], 0, []

    for row in rows:
        sid = row["sample_id"]
        src = resized_dir / f"[R] {sid}.mp4"
        if not src.exists():
            missing.append(sid)
            continue

        out = feature_dir / sid / "frames.npz"
        (feature_dir / sid).mkdir(parents=True, exist_ok=True)

        if not args.force and out.exists():
            skipped += 1
            continue

        pending.append((sid, str(src), str(out), args.n_frames, args.force))

    if missing:
        print(f"WARNING: {len(missing)} samples missing from {resized_dir}")
        for s in missing[:5]:
            print(f"  {s}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")

    print(f"{skipped} already done, {len(pending)} to extract with {args.workers} worker(s).")
    if not pending:
        print("Nothing to do.")
        return

    errors = []
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        for sid, ok, err in tqdm(
            pool.imap_unordered(_extract, pending),
            total=len(pending),
            desc="extract_frames",
            unit="video",
        ):
            if not ok:
                errors.append((sid, err))
                tqdm.write(f"  [ERROR] {sid}:\n{err}")

    print(f"\nDone. {len(pending) - len(errors)}/{len(pending)} succeeded"
          f" ({skipped} already existed).")
    if errors:
        err_log = PROJECT_ROOT / "extract_frames_errors.txt"
        with open(err_log, "w") as f:
            for sid, err in errors:
                f.write(f"{sid}\t{err}\n")
        print(f"Error log: {err_log}")


if __name__ == "__main__":
    main()
