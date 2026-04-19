"""
Full-frame preprocessing — start directly from DOLOS original videos.

Differences vs. the original DOLOS pipeline (video_prep.py / preprocess_resized.py):

  * Source   : Data/DOLOS-all/DOLOS/{sample_id}.mp4  (ORIGINAL, un-resized).
  * Temporal : EVERY frame is kept — no uniform sampling, no blacking-out.
  * Spatial  : YOLO face box, then expanded **3×** (side length) around the
               face centre. The crop stays face-centred; it just carries more
               context (shoulders / background).
  * Missing  : When YOLO returns no face in a frame, the last-seen box is
               propagated forward (or the first future box for the head of
               the clip). Frames are NEVER blacked out.

Output per sample (``features/{sample_id}/``):
    frames_full.npz     compressed — keys:
                          'frames'  uint8  (N, 224, 224, 3)  RGB
    audio.wav           16 kHz mono PCM

Note: ``frame_mask.npy`` is intentionally NOT written — every frame is
considered valid under this pipeline.

Usage (from project root):
    python -m deception_detection.data.preprocessing.preprocess_fullframe \\
        --manifest    Data/manifest_dolos.csv \\
        --dolos_dir   Data/DOLOS-all/DOLOS \\
        --feature_dir features \\
        --workers     4
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import traceback
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

CROP_SIZE = (224, 224)              # (W, H) written to frames_full.npz
BOX_SCALE = 3.0                     # expand YOLO side length ×3 (area ×9)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
YOLO_MODEL_PATH = PROJECT_ROOT / "Data" / "Resizer" / "yolov8n-face.pt"

# Per-process YOLO model — spawn context means each worker re-imports this
# module and gets its own instance via the pool initializer.
_worker_yolo = None


# ---------------------------------------------------------------------------
# Worker init / task
# ---------------------------------------------------------------------------

def _init_worker(yolo_model_path: str):
    """Pool initializer: load YOLO once per worker process."""
    from ultralytics import YOLO

    global _worker_yolo
    _worker_yolo = YOLO(yolo_model_path)


def _worker_task(args: tuple) -> tuple[str, bool, str]:
    sample_id, video_path_str, output_dir_str, force = args
    try:
        process_video(
            Path(video_path_str),
            Path(output_dir_str),
            _worker_yolo,
            force=force,
        )
        return sample_id, True, ""
    except Exception:
        return sample_id, False, traceback.format_exc()


# ---------------------------------------------------------------------------
# Box utilities
# ---------------------------------------------------------------------------

def _pick_main_face(xyxy: np.ndarray) -> int:
    """
    From a (K, 4) array of YOLO face boxes in one frame, pick the index of
    the "main" face = largest box area. With no active-speaker tracking we
    fall back to the dominant (typically closest / largest) face.
    """
    if len(xyxy) == 1:
        return 0
    wh = xyxy[:, 2:] - xyxy[:, :2]
    areas = wh[:, 0] * wh[:, 1]
    return int(np.argmax(areas))


def _expand_box(box: np.ndarray, scale: float, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """
    Expand a (x1, y1, x2, y2) box by `scale` along each side length,
    keeping the centre fixed, then squarify (max side) and clamp to the
    frame. Returns an integer (x1, y1, x2, y2) crop rectangle.
    """
    x1, y1, x2, y2 = box.astype(np.float32)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    # Side length after expansion — use the longer of (w, h) so the
    # resulting crop is square and the face never stretches.
    side = max(x2 - x1, y2 - y1) * scale
    half = side * 0.5

    nx1 = int(round(cx - half))
    ny1 = int(round(cy - half))
    nx2 = int(round(cx + half))
    ny2 = int(round(cy + half))

    # Clamp to frame bounds (may shrink the final crop on edges — acceptable,
    # face still dominates because the box was centred).
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(frame_w, nx2)
    ny2 = min(frame_h, ny2)

    # Degenerate fallback: full frame.
    if nx2 <= nx1 or ny2 <= ny1:
        return 0, 0, frame_w, frame_h
    return nx1, ny1, nx2, ny2


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------

def process_video(
    video_path: Path,
    output_dir: Path,
    yolo_model,
    *,
    force: bool = False,
) -> bool:
    """
    Process a single DOLOS video:
      1. Decode every frame.
      2. YOLO-detect faces, pick the main face (largest area) per frame.
      3. Expand box ×3 around centre, square, clamp, crop, resize → uint8.
      4. Propagate last-seen box when a frame has no detection.
      5. Persist frames_full.npz (compressed) and audio.wav.

    Returns True on success. Raises on unrecoverable errors.
    """
    out_frames = output_dir / "frames_full.npz"
    out_audio  = output_dir / "audio.wav"

    if not force and out_frames.exists() and out_audio.exists():
        return True

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: single pass through the video, cropping each frame on the fly.
    # Resizing to 224×224 immediately keeps memory bounded (~150 KB/frame).
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture failed to open {video_path}")

    frames_rgb: list[np.ndarray] = []
    last_box: Optional[tuple[int, int, int, int]] = None

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        h, w = frame_bgr.shape[:2]

        # YOLO detect — verbose=False silences per-frame prints.
        results = yolo_model.predict(
            source=frame_bgr, conf=0.25, save=False, verbose=False
        )[0]
        xyxy = (
            results.boxes.xyxy.cpu().numpy()
            if results.boxes is not None
            else np.empty((0, 4))
        )

        if len(xyxy) > 0:
            idx = _pick_main_face(xyxy)
            box = _expand_box(xyxy[idx], BOX_SCALE, w, h)
            last_box = box
        elif last_box is not None:
            # Propagate the most recent known box — never black the frame.
            box = last_box
        else:
            # No detection yet AND no prior box: fall back to a centre
            # square crop sized to the shorter side. Face-centred (best
            # approximation when no detector output is available).
            side = min(w, h)
            cx, cy = w // 2, h // 2
            half = side // 2
            box = (cx - half, cy - half, cx + half, cy + half)

        x1, y1, x2, y2 = box
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_rgb = cv2.resize(crop_rgb, CROP_SIZE, interpolation=cv2.INTER_AREA)
        frames_rgb.append(crop_rgb)

    cap.release()

    if not frames_rgb:
        raise RuntimeError(f"No frames decoded from {video_path}")

    frames_arr = np.stack(frames_rgb).astype(np.uint8)   # (N, 224, 224, 3)
    np.savez_compressed(str(out_frames), frames=frames_arr)

    # --- Phase 2: audio — reuse existing WAV if present, else extract fresh.
    if force or not out_audio.exists():
        _extract_audio(video_path, out_audio)

    return True


def _extract_audio(video_path: Path, out_wav: Path, target_sr: int = 16000):
    """Mono 16 kHz PCM WAV; falls back to 1-second silence on failure."""
    try:
        waveform, sr = torchaudio.load(str(video_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        torchaudio.save(str(out_wav), waveform, target_sr)
    except Exception as e:
        silence = torch.zeros(1, target_sr)
        torchaudio.save(str(out_wav), silence, target_sr)
        print(f"  [warn] audio extraction failed for {video_path.name} ({e}); wrote silence")


# ---------------------------------------------------------------------------
# Manifest-driven entry point
# ---------------------------------------------------------------------------

def _resolve_source(row: dict, dolos_dir: Path) -> Path:
    """
    The manifest records absolute paths baked on another machine
    (``/home/asdf/lieglass-dsc/...``). Prefer ``dolos_dir/{sample_id}.mp4``;
    fall back to the path-corrected manifest entry if that file is missing.
    """
    sid = row["sample_id"]
    candidate = dolos_dir / f"{sid}.mp4"
    if candidate.exists():
        return candidate

    raw = Path(row.get("video_path", ""))
    corrected = Path(str(raw).replace("/home/asdf/", str(PROJECT_ROOT) + "/"))
    return corrected


def main():
    parser = argparse.ArgumentParser(
        description="Full-frame preprocessing from DOLOS originals (3× face box, no blacking)"
    )
    parser.add_argument("--manifest",    default="Data/manifest_dolos.csv")
    parser.add_argument("--dolos_dir",   default="Data/DOLOS-all/DOLOS",
                        help="Directory containing the ORIGINAL DOLOS videos.")
    parser.add_argument("--feature_dir", default="features")
    parser.add_argument("--yolo_model",  default=str(YOLO_MODEL_PATH))
    parser.add_argument("--workers",     type=int,
                        default=max(1, (os.cpu_count() or 4) // 2),
                        help="Parallel worker processes.")
    parser.add_argument("--force",       action="store_true",
                        help="Re-process even if outputs exist.")
    parser.add_argument("--sample_id",   default=None,
                        help="Process a single sample (for debugging).")
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / args.manifest
    dolos_dir     = PROJECT_ROOT / args.dolos_dir
    feature_dir   = PROJECT_ROOT / args.feature_dir
    feature_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if args.sample_id:
        rows = [r for r in rows if r["sample_id"] == args.sample_id]
        if not rows:
            print(f"Sample not found: {args.sample_id}")
            return

    pending, skipped, missing = [], 0, []
    for row in rows:
        sid = row["sample_id"]
        src = _resolve_source(row, dolos_dir)
        if not src.exists():
            missing.append(sid)
            continue

        sample_dir = feature_dir / sid
        sample_dir.mkdir(parents=True, exist_ok=True)

        needs_work = args.force or not (
            (sample_dir / "frames_full.npz").exists()
            and (sample_dir / "audio.wav").exists()
        )
        if not needs_work:
            skipped += 1
            continue

        pending.append((sid, str(src), str(sample_dir), args.force))

    if missing:
        print(f"WARNING: {len(missing)} samples not found under {dolos_dir}:")
        for s in missing[:5]:
            print(f"  {s}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")

    print(f"{skipped} already done, {len(pending)} to process with {args.workers} worker(s).")
    if not pending:
        print("Nothing to do.")
        return

    errors = []
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(args.yolo_model,),
    ) as pool:
        for sid, ok, err in tqdm(
            pool.imap_unordered(_worker_task, pending),
            total=len(pending),
            desc="preprocess_fullframe",
            unit="video",
        ):
            if not ok:
                errors.append((sid, err))
                tqdm.write(f"  [ERROR] {sid}:\n{err}")

    ok = len(pending) - len(errors)
    print(f"\nDone. {ok}/{len(pending)} succeeded ({skipped} already existed).")
    if errors:
        err_log = PROJECT_ROOT / "preprocess_fullframe_errors.txt"
        with open(err_log, "w") as f:
            for sid, err in errors:
                f.write(f"{sid}\t{err}\n")
        print(f"Error log: {err_log}")


if __name__ == "__main__":
    main()
