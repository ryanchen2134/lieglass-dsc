"""
Step 1: Video Preprocessing — Active Speaker Detection & Face Cropping.

For each video:
1. Phase 1 (Analyze): YOLO + ByteTrack to find the active speaker (most frequent track ID).
2. Phase 2 (Write): Crop face to 224×224; write black frame when speaker not detected.
3. Build frame_mask: 1 = speaker detected, 0 = blacked out.
4. Extract audio as 16kHz mono WAV.

Input:  original video from Data/DOLOS-all/DOLOS/{sample_id}.mp4
Output (in features/{sample_id}/):
    video.mp4       - 224×224 face-cropped video
    frame_mask.npy  - BoolArray (n_frames,)
    audio.wav       - 16kHz mono WAV
"""

import os
import multiprocessing
import numpy as np
import cv2
import torchaudio
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import supervision as sv
from moviepy import VideoFileClip

CROP_SIZE = (224, 224)
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # lieglass-dsc/
YOLO_MODEL_PATH = PROJECT_ROOT / "Data" / "Resizer" / "yolov8n-face.pt"

# Per-process YOLO model (loaded once per worker, not shared across processes)
_worker_yolo: YOLO | None = None


def _init_worker(yolo_model_path: str):
    """Pool initializer: load YOLO once per worker process."""
    global _worker_yolo
    _worker_yolo = YOLO(yolo_model_path)


def _worker_task(args: tuple) -> tuple[str, bool, str]:
    """
    Top-level picklable function called by each pool worker.
    Returns (sample_id, success, error_msg).
    """
    sample_id, video_path_str, output_dir_str = args
    try:
        process_video(Path(video_path_str), Path(output_dir_str), _worker_yolo)
        return sample_id, True, ""
    except Exception:
        import traceback
        return sample_id, False, traceback.format_exc()


def process_video(video_path: Path, output_dir: Path, yolo_model: YOLO) -> bool:
    """
    Process a single video. Returns True on success, False on failure.
    Skips if all outputs already exist.
    """
    out_video = output_dir / "video.mp4"
    out_mask = output_dir / "frame_mask.npy"
    out_audio = output_dir / "audio.wav"

    if out_video.exists() and out_mask.exists() and out_audio.exists():
        return True  # already done

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- PHASE 1: Find active speaker ---
    tracker = sv.ByteTrack()
    id_counts = defaultdict(int)
    frame_detections = []

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo_model.predict(source=frame, conf=0.5, save=False, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        frame_detections.append(detections)
        if detections.tracker_id is not None:
            for track_id in detections.tracker_id:
                id_counts[track_id] += 1

    cap.release()

    if not id_counts:
        # No faces found; write all-black video and all-zero mask
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        frame_mask = np.zeros(total_frames, dtype=bool)
        np.save(str(out_mask), frame_mask)
        # Write black video
        _write_black_video(str(video_path), str(out_video), fps, CROP_SIZE, total_frames)
        _extract_audio(str(video_path), str(out_audio))
        return True

    target_id = max(id_counts, key=id_counts.get)

    # --- PHASE 2: Write cropped video + build frame_mask ---
    cap = cv2.VideoCapture(str(video_path))
    temp_path = str(output_dir / "_temp_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_path, fourcc, fps, CROP_SIZE)

    frame_mask = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        face_img = None
        if frame_idx < len(frame_detections):
            dets = frame_detections[frame_idx]
            if dets.tracker_id is not None and target_id in dets.tracker_id:
                match_idx = list(dets.tracker_id).index(target_id)
                x1, y1, x2, y2 = dets.xyxy[match_idx].astype(int)
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    try:
                        face_img = cv2.resize(frame[y1:y2, x1:x2], CROP_SIZE)
                    except Exception:
                        face_img = None

        if face_img is not None:
            writer.write(face_img)
            frame_mask.append(True)
        else:
            writer.write(np.zeros((CROP_SIZE[1], CROP_SIZE[0], 3), dtype=np.uint8))
            frame_mask.append(False)

        frame_idx += 1

    cap.release()
    writer.release()

    frame_mask_arr = np.array(frame_mask, dtype=bool)
    np.save(str(out_mask), frame_mask_arr)

    # --- PHASE 3: Mux audio ---
    # Use a per-video temp audio path to avoid collisions between parallel workers
    temp_audio = str(output_dir / "_temp_audio.m4a")
    try:
        proc_clip = VideoFileClip(temp_path)
        orig_clip = VideoFileClip(str(video_path))
        if orig_clip.audio is not None:
            final_clip = proc_clip.with_audio(orig_clip.audio)
            final_clip.write_videofile(
                str(out_video), codec="libx264", audio_codec="aac",
                temp_audiofile=temp_audio, logger=None,
            )
            final_clip.close()
        else:
            proc_clip.write_videofile(str(out_video), codec="libx264", logger=None)
        proc_clip.close()
        orig_clip.close()
    except Exception as e:
        # Fallback: just rename temp
        import shutil
        shutil.move(temp_path, str(out_video))
        print(f"    [warn] Audio mux failed ({e}); video saved without original audio")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

    # --- PHASE 4: Extract audio as WAV ---
    _extract_audio(str(video_path), str(out_audio))

    return True


def _write_black_video(video_path: str, out_path: str, fps: float, size: tuple, n_frames: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, size)
    black = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(black)
    writer.release()


def _extract_audio(video_path: str, out_wav: str, target_sr: int = 16000):
    try:
        waveform, sr = torchaudio.load(video_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # mono
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        torchaudio.save(out_wav, waveform, target_sr)
    except Exception as e:
        # If torchaudio fails (e.g., no audio stream), write silence
        import torch
        silence = torch.zeros(1, target_sr)  # 1 second of silence
        torchaudio.save(out_wav, silence, target_sr)
        print(f"    [warn] Audio extraction failed ({e}); wrote silence")


def main():
    import argparse
    import csv
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Step 1: Video preprocessing (224×224 crop + frame_mask + audio)")
    parser.add_argument("--manifest", default="Data/manifest_dolos.csv")
    parser.add_argument("--feature_dir", default="features")
    parser.add_argument("--yolo_model", default=str(YOLO_MODEL_PATH))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                        help="Number of parallel worker processes")
    parser.add_argument("--sample_id", default=None, help="Process a single sample only")
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / args.manifest
    feature_dir = PROJECT_ROOT / args.feature_dir

    with open(manifest_path, newline="") as f:
        all_samples = list(csv.DictReader(f))

    if args.sample_id:
        all_samples = [s for s in all_samples if s["sample_id"] == args.sample_id]
        if not all_samples:
            print(f"Sample not found: {args.sample_id}")
            return

    # Filter out already-completed samples before dispatching to workers
    pending = []
    skipped = 0
    for row in all_samples:
        sample_dir = feature_dir / row["sample_id"]
        if (sample_dir / "video.mp4").exists() and \
           (sample_dir / "frame_mask.npy").exists() and \
           (sample_dir / "audio.wav").exists():
            skipped += 1
            continue
        sample_dir.mkdir(parents=True, exist_ok=True)
        pending.append((
            row["sample_id"],
            row["video_path"],
            str(feature_dir / row["sample_id"]),
        ))

    print(f"{skipped} already done, {len(pending)} to process with {args.workers} workers.")

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
        for sample_id, success, err_msg in tqdm(
            pool.imap_unordered(_worker_task, pending),
            total=len(pending),
            desc="video_prep",
            unit="video",
        ):
            if not success:
                errors.append((sample_id, err_msg))
                tqdm.write(f"  [ERROR] {sample_id}:\n{err_msg}")

    print(f"\nDone. {len(pending) - len(errors)}/{len(pending)} succeeded"
          f" ({skipped} already existed).")
    if errors:
        for sid, _ in errors:
            print(f"  FAILED: {sid}")


if __name__ == "__main__":
    main()
