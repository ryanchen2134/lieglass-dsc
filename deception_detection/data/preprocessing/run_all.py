"""
Master Preprocessing Script — Run Steps 1-4 for all samples in the manifest.

Usage:
    cd /path/to/lieglass-dsc
    python -m deception_detection.data.preprocessing.run_all \
        --manifest Data/manifest_dolos.csv \
        --feature_dir features \
        --yolo_model Data/Resizer/yolov8n-face.pt \
        --whisper_model base \
        --steps 1234 \
        --device cuda

Steps:
    1 = video_prep   (YOLO crop, frame_mask, audio.wav)
    2 = whisper      (text.pt)
    3 = mfcc         (mfcc.pt)
    4 = landmarks    (landmarks.pt)

Checkpointing: each step is skipped if its output already exists.
Errors per sample are logged and do not abort the run.
"""

import argparse
import csv
import sys
import traceback
import torch
from pathlib import Path
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="Data/manifest_dolos.csv")
    p.add_argument("--feature_dir", default="features")
    p.add_argument("--yolo_model", default="Data/Resizer/yolov8n-face.pt")
    p.add_argument("--whisper_model", default="base", help="Whisper model size: tiny/base/small")
    p.add_argument("--steps", default="1234", help="Which steps to run (e.g. '234' to skip step 1)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--label_txt", action="store_true", default=True,
                   help="Also write label.txt per sample")
    return p.parse_args()


def load_manifest(path: str):
    samples = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    return samples


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[3]  # project root (lieglass-dsc/)
    feature_dir = root / args.feature_dir
    feature_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = root / args.manifest
    samples = load_manifest(str(manifest_path))
    print(f"Loaded {len(samples)} samples from {manifest_path}")

    steps = set(args.steps)

    # --- Lazy-load heavy models once ---
    yolo_model = None
    whisper_model = None
    landmark_model = None

    if "1" in steps:
        from ultralytics import YOLO
        yolo_path = root / args.yolo_model
        print(f"Loading YOLO from {yolo_path}...")
        yolo_model = YOLO(str(yolo_path))

    if "2" in steps:
        import whisper_timestamped as whisper_ts
        print(f"Loading Whisper ({args.whisper_model})...")
        whisper_model = whisper_ts.load_model(args.whisper_model, device=args.device)

    if "4" in steps:
        from .landmark_extract import _load_facexformer
        print("Loading FaceXFormer...")
        landmark_model = _load_facexformer(args.device)

    # --- Per-sample loop ---
    errors = []

    for row in tqdm(samples, desc="Preprocessing", unit="sample"):
        sample_id = row["sample_id"]
        label = int(row["label"])
        video_path = Path(row["video_path"])
        sample_dir = feature_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Write label.txt
        if args.label_txt:
            label_file = sample_dir / "label.txt"
            if not label_file.exists():
                label_file.write_text(str(label))

        try:
            # Step 1: video_prep
            if "1" in steps:
                from .video_prep import process_video
                process_video(video_path, sample_dir, yolo_model)

            # Step 2: whisper
            if "2" in steps:
                from .whisper_extract import extract_text
                audio_path = sample_dir / "audio.wav"
                extract_text(audio_path, sample_dir, whisper_model=whisper_model)

            # Step 3: mfcc
            if "3" in steps:
                from .mfcc_extract import extract_mfcc
                audio_path = sample_dir / "audio.wav"
                extract_mfcc(audio_path, sample_dir)

            # Step 4: landmarks
            if "4" in steps:
                from .landmark_extract import extract_landmarks
                extract_landmarks(sample_dir, sample_dir, landmark_model, args.device)

        except Exception as e:
            errors.append((sample_id, str(e)))
            tqdm.write(f"  [ERROR] {sample_id}: {e}")
            tqdm.write(traceback.format_exc())
            continue

    print(f"\nDone. {len(samples) - len(errors)}/{len(samples)} succeeded.")
    if errors:
        print(f"\nFailed samples ({len(errors)}):")
        for sid, err in errors:
            print(f"  {sid}: {err}")
        # Write error log
        err_log = root / "preprocessing_errors.txt"
        with open(err_log, "w") as f:
            for sid, err in errors:
                f.write(f"{sid}\t{err}\n")
        print(f"Error log written to: {err_log}")


if __name__ == "__main__":
    main()
