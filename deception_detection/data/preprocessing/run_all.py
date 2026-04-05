"""
Master Preprocessing Script — Step 1 only: YOLO face crop + audio extraction.

The bimodal model (Wav2Vec2 + CNN+ViT) reads audio.wav and video.mp4 directly
from each sample's feature directory.  No further feature extraction is needed.

Usage:
    cd /path/to/lieglass-dsc
    python -m deception_detection.data.preprocessing.run_all \
        --manifest Data/manifest_dolos.csv \
        --feature_dir features \
        --yolo_model Data/Resizer/yolov8n-face.pt \
        --device cuda

Checkpointing: step 1 is skipped if video.mp4 + audio.wav already exist.
Errors per sample are logged and do not abort the run.
"""

import argparse
import csv
import traceback
import torch
from pathlib import Path
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",    default="Data/manifest_dolos.csv")
    p.add_argument("--feature_dir", default="features")
    p.add_argument("--yolo_model",  default="Data/Resizer/yolov8n-face.pt")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--label_txt",   action="store_true", default=True)
    return p.parse_args()


def load_manifest(path: str):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[3]
    feature_dir = root / args.feature_dir
    feature_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = root / args.manifest
    samples = load_manifest(str(manifest_path))
    print(f"Loaded {len(samples)} samples from {manifest_path}")

    from ultralytics import YOLO
    yolo_path = root / args.yolo_model
    print(f"Loading YOLO from {yolo_path}...")
    yolo_model = YOLO(str(yolo_path))

    errors = []

    for row in tqdm(samples, desc="Preprocessing", unit="sample"):
        sample_id  = row["sample_id"]
        label      = int(row["label"])
        video_path = Path(row["video_path"])
        sample_dir = feature_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        if args.label_txt:
            label_file = sample_dir / "label.txt"
            if not label_file.exists():
                label_file.write_text(str(label))

        try:
            from .video_prep import process_video
            process_video(video_path, sample_dir, yolo_model)
        except Exception as e:
            errors.append((sample_id, str(e)))
            tqdm.write(f"  [ERROR] {sample_id}: {e}")
            tqdm.write(traceback.format_exc())

    print(f"\nDone. {len(samples) - len(errors)}/{len(samples)} succeeded.")
    if errors:
        print(f"\nFailed samples ({len(errors)}):")
        for sid, err in errors:
            print(f"  {sid}: {err}")
        err_log = root / "preprocessing_errors.txt"
        with open(err_log, "w") as f:
            for sid, err in errors:
                f.write(f"{sid}\t{err}\n")
        print(f"Error log: {err_log}")


if __name__ == "__main__":
    main()
