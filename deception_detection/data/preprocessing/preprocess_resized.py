"""
Preprocessing for resizedVideosNew — face-crop videos with intentional black frames.

The resizedVideosNew/ pipeline already produces face-cropped videos with black frames
for non-speaking segments (active speaker not detected). This script:

  1. Detects black frames by pixel intensity  →  frame_mask.npy
  2. Copies/converts the video                →  features/{id}/video.mp4
  3. Keeps / re-extracts audio               →  features/{id}/audio.wav

Audio note:
  resizedVideosNew videos were created with an OpenCV pipeline that does NOT preserve
  audio. Audio is taken from features/{id}/audio.wav if it already exists (from the
  earlier video_prep.py run). If missing, the script falls back to the original DOLOS
  video (correcting the /home/asdf path) or writes silence.

Usage (from project root):
    python -m deception_detection.data.preprocessing.preprocess_resized \\
        --manifest Data/manifest_dolos.csv \\
        --resized_dir Data/DOLOS-all/resizedVideosNew \\
        --feature_dir features \\
        --workers 4
"""

import os
import csv
import shutil
import multiprocessing
import numpy as np
import cv2
import torchaudio
import torch
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# A frame is considered "black" if its mean pixel value (0-255) is below this.
BLACK_THRESHOLD = 8


# ---------------------------------------------------------------------------
# Frame-mask detection
# ---------------------------------------------------------------------------

def _detect_frame_mask(video_path: Path) -> np.ndarray:
    """
    Read every frame of video_path and return a boolean ndarray (n_frames,)
    where True means the frame is NOT black (speaker is visible).
    """
    cap = cv2.VideoCapture(str(video_path))
    mask = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Mean over all pixels and all channels
        is_visible = bool(frame.mean() > BLACK_THRESHOLD)
        mask.append(is_visible)
    cap.release()
    return np.array(mask, dtype=bool)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _try_extract_audio(source_video: Path, out_wav: Path, target_sr: int = 16000):
    """Extract mono 16 kHz audio from source_video → out_wav."""
    try:
        waveform, sr = torchaudio.load(str(source_video))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        torchaudio.save(str(out_wav), waveform, target_sr)
        return True
    except Exception:
        return False


def _ensure_audio(sample_id: str, sample_dir: Path, row: dict, src_video: Path | None = None):
    """
    Guarantee features/{sample_id}/audio.wav exists.
    Priority:
      1. Already present → skip.
      2. Extract from src_video (the resized source — has original audio track).
      3. Extract from original DOLOS video (path-corrected from manifest).
      4. Write 1-second silence as last resort.
    """
    out_wav = sample_dir / "audio.wav"
    if out_wav.exists():
        return  # already done

    # Try the resized source video first (it carries the original audio track)
    if src_video is not None and src_video.exists():
        if _try_extract_audio(src_video, out_wav):
            return

    # Try original DOLOS path (correct /home/asdf → actual project root)
    raw_path = Path(row.get("video_path", ""))
    corrected = Path(str(raw_path).replace("/home/asdf/", str(PROJECT_ROOT) + "/"))
    if corrected.exists() and _try_extract_audio(corrected, out_wav):
        return

    # Try the canonical DOLOS directory
    dolos_dir = PROJECT_ROOT / "Data" / "DOLOS-all" / "DOLOS"
    candidate = dolos_dir / f"{sample_id}.mp4"
    if candidate.exists() and _try_extract_audio(candidate, out_wav):
        return

    # Last resort: 1 second of silence
    silence = torch.zeros(1, 16000)
    torchaudio.save(str(out_wav), silence, 16000)


# ---------------------------------------------------------------------------
# Per-sample worker
# ---------------------------------------------------------------------------

def _process_sample(args: tuple) -> tuple[str, bool, str]:
    sample_id, src_video_str, sample_dir_str, row_dict, force = args
    sample_dir = Path(sample_dir_str)
    src_video = Path(src_video_str)
    out_video = sample_dir / "video.mp4"
    out_mask = sample_dir / "frame_mask.npy"

    try:
        sample_dir.mkdir(parents=True, exist_ok=True)

        # --- Video ---
        if force or not out_video.exists():
            # Use ffmpeg via subprocess for a clean re-encode to H.264 + correct container.
            # Falls back to shutil.copy if ffmpeg is unavailable.
            _copy_video(src_video, out_video)

        # --- Frame mask ---
        if force or not out_mask.exists():
            mask = _detect_frame_mask(src_video)
            np.save(str(out_mask), mask)

        # --- Audio ---
        _ensure_audio(sample_id, sample_dir, row_dict, src_video=src_video)

        return sample_id, True, ""
    except Exception:
        import traceback
        return sample_id, False, traceback.format_exc()


def _copy_video(src: Path, dst: Path):
    """
    Copy video to dst. Tries ffmpeg re-encode first (for container compatibility),
    falls back to a raw file copy.
    """
    import subprocess
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(src),
                "-c:v", "libx264", "-crf", "18", "-c:a", "aac",
                str(dst),
            ],
            capture_output=True, timeout=120,
        )
        if result.returncode == 0:
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Fallback: raw copy (video stays in original codec)
    shutil.copy2(str(src), str(dst))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess resizedVideosNew → features/ (frame_mask + video + audio)"
    )
    parser.add_argument("--manifest",    default="Data/manifest_dolos.csv",
                        help="CSV with sample_id, label, video_path columns")
    parser.add_argument("--resized_dir", default="Data/DOLOS-all/resizedVideosNew",
                        help="Directory containing [R] {sample_id}.mp4 files")
    parser.add_argument("--feature_dir", default="features",
                        help="Output root directory for feature subdirectories")
    parser.add_argument("--workers",     type=int,
                        default=max(1, (os.cpu_count() or 4) // 2),
                        help="Number of parallel worker processes")
    parser.add_argument("--force",       action="store_true",
                        help="Re-process even if outputs already exist")
    parser.add_argument("--sample_id",   default=None,
                        help="Process a single sample only (for debugging)")
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / args.manifest
    resized_dir   = PROJECT_ROOT / args.resized_dir
    feature_dir   = PROJECT_ROOT / args.feature_dir
    feature_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, newline="") as f:
        all_rows = list(csv.DictReader(f))

    if args.sample_id:
        all_rows = [r for r in all_rows if r["sample_id"] == args.sample_id]
        if not all_rows:
            print(f"Sample not found in manifest: {args.sample_id}")
            return

    # Build work list
    pending = []
    skipped = 0
    missing_src = []

    for row in all_rows:
        sid = row["sample_id"]
        src = resized_dir / f"{sid}.mp4"

        if not src.exists():
            missing_src.append(sid)
            continue

        sample_dir = feature_dir / sid
        if not args.force and (
            (sample_dir / "video.mp4").exists()
            and (sample_dir / "frame_mask.npy").exists()
            and (sample_dir / "audio.wav").exists()
        ):
            skipped += 1
            continue

        pending.append((sid, str(src), str(sample_dir), row, args.force))

    if missing_src:
        print(f"WARNING: {len(missing_src)} samples have no source in {resized_dir}:")
        for sid in missing_src[:10]:
            print(f"  {sid}")
        if len(missing_src) > 10:
            print(f"  ... and {len(missing_src) - 10} more")

    print(f"{skipped} already done, {len(pending)} to process with {args.workers} worker(s).")
    if not pending:
        print("Nothing to do.")
        return

    errors = []
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        for sid, ok, err in tqdm(
            pool.imap_unordered(_process_sample, pending),
            total=len(pending),
            desc="preprocess_resized",
            unit="video",
        ):
            if not ok:
                errors.append((sid, err))
                tqdm.write(f"  [ERROR] {sid}:\n{err}")

    ok_count = len(pending) - len(errors)
    print(f"\nDone. {ok_count}/{len(pending)} succeeded ({skipped} already existed).")
    if errors:
        print(f"\nFailed ({len(errors)}):")
        for sid, _ in errors:
            print(f"  FAILED: {sid}")
        err_log = PROJECT_ROOT / "preprocess_resized_errors.txt"
        with open(err_log, "w") as f:
            for sid, err in errors:
                f.write(f"{sid}\t{err}\n")
        print(f"Error log: {err_log}")


if __name__ == "__main__":
    main()
