"""
Step 4: Facial Landmark Extraction using FaceXFormer.

For each processed video (features/{sample_id}/video.mp4) + frame_mask.npy:
1. Load FaceXFormer model.
2. Per frame: run FaceXFormer (if mask==1) or output zeros (if mask==0).
3. Per-video z-score normalize valid frames.
4. Compute delta and delta-delta via np.gradient; zero out blacked-out frames and neighbors.
5. Stack → (T_l, 136, 3).
6. Save landmarks.pt: {landmarks, timestamps, frame_mask}.

FaceXFormer model download:
    hf_hub_download(repo_id="kartiknarayan/facexformer", filename="ckpts/model.pt")
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from huggingface_hub import hf_hub_download


N_LANDMARKS = 68
N_COORDS = 2        # x, y
N_STACK = 3         # static, delta, delta-delta
INPUT_SIZE = 224    # FaceXFormer expected input


def _load_facexformer(device: str = "cuda"):
    """Download and load FaceXFormer model."""
    model_path = hf_hub_download(
        repo_id="kartiknarayan/facexformer",
        filename="ckpts/model.pt",
    )
    # FaceXFormer uses a custom architecture — import from the hub or local copy
    # The model file contains the full model state_dict; we need the FaceXFormer class.
    # The repo provides a model.py; we import it after downloading.
    # As a robust fallback, we use the timm/torchvision approach with the provided checkpoint.
    # NOTE: The FaceXFormer repo (github.com/pranavjadhav001/facexformer) must be cloned
    # or pip-installed. Here we assume it is available as `facexformer` package.
    try:
        print("Loading FaceXFormer model..."    )
        from .facexformer import FaceXFormer as FXF
        print("Initializing FaceXFormer model...")
        model = FXF()
        print(f"Loading FaceXFormer weights from {model_path}...")
        state = torch.load(model_path, map_location=device)
        print(f"Loaded FaceXFormer weights from {model_path}")
        # Handle both raw state_dict and wrapped checkpoint
        if isinstance(state, dict) and "state_dict_backbone" in state:
            state = state["state_dict_backbone"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]
        print("Loading state dict into FaceXFormer model...")
        model.load_state_dict(state, strict=False)
    except ImportError:
        raise ImportError(
            "FaceXFormer package not found. Install from: "
            "pip install git+https://github.com/Kartik-3004/facexformer.git"
        )
    model.to(device).eval()
    return model


import torchvision.transforms as T

_frame_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((INPUT_SIZE, INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_sample(sample_dir: Path):
    """
    CPU phase: read video frames and apply transforms.
    Returns (valid_indices, tensors, n_frames, fps, frame_mask) ready for GPU inference.
    """
    frame_mask = np.load(str(sample_dir / "frame_mask.npy"))
    cap = cv2.VideoCapture(str(sample_dir / "video.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(frame_mask))
    frame_mask = frame_mask[:n_frames]

    all_frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        all_frames.append(frame if ret else None)
    cap.release()

    valid_indices = [i for i in range(n_frames) if frame_mask[i] and all_frames[i] is not None]
    if valid_indices:
        tensors = torch.stack([
            _frame_transform(cv2.cvtColor(all_frames[i], cv2.COLOR_BGR2RGB))
            for i in valid_indices
        ])
    else:
        tensors = torch.zeros(0, 3, INPUT_SIZE, INPUT_SIZE)

    return valid_indices, tensors, n_frames, fps, frame_mask.copy()


def _run_facexformer_tensors(model, tensors: torch.Tensor, device: str, batch_size: int = 32) -> np.ndarray:
    """
    GPU phase: run FaceXFormer on pre-transformed tensors.
    Returns (N, 68, 2) array of [0,1]-normalised (x,y) landmark coordinates.
    """
    if tensors.shape[0] == 0:
        return np.zeros((0, 68, 2), dtype=np.float32)
    results = []
    for start in range(0, len(tensors), batch_size):
        batch = tensors[start:start + batch_size].to(device)
        n = batch.shape[0]
        tasks = torch.ones(n, dtype=torch.long, device=device)
        labels = torch.zeros(n, dtype=torch.long, device=device)
        with torch.no_grad():
            lm_out, *_ = model(batch, labels, tasks)
        lm = lm_out.view(n, 68, 2).cpu()
        lm = ((lm + 1) * INPUT_SIZE - 1) / (2 * INPUT_SIZE)
        results.append(lm.clamp(0.0, 1.0).numpy())
    return np.concatenate(results, axis=0).astype(np.float32)



def _save_landmarks(
    valid_indices: list, lm_batch: np.ndarray,
    n_frames: int, fps: float, frame_mask: np.ndarray,
    output_dir: Path,
):
    """Post-process and save landmarks.pt from GPU inference results."""
    raw_landmarks = np.zeros((n_frames, N_LANDMARKS * N_COORDS), dtype=np.float32)
    for idx, lm in zip(valid_indices, lm_batch):
        raw_landmarks[idx] = lm.reshape(-1)

    raw_landmarks_01 = raw_landmarks.copy()

    valid = frame_mask.astype(bool)
    if valid.sum() > 1:
        valid_lm = raw_landmarks[valid]
        mean = valid_lm.mean(axis=0)
        std = valid_lm.std(axis=0) + 1e-8
        raw_landmarks[valid] = (valid_lm - mean) / std
        raw_landmarks[~valid] = 0.0

    delta1 = np.gradient(raw_landmarks, axis=0)
    delta2 = np.gradient(delta1, axis=0)

    invalid = ~valid
    neighbor_mask = invalid.copy()
    neighbor_mask[1:] |= invalid[:-1]
    neighbor_mask[:-1] |= invalid[1:]
    delta1[neighbor_mask] = 0.0
    delta2[neighbor_mask] = 0.0
    raw_landmarks[invalid] = 0.0

    landmarks = np.stack([raw_landmarks, delta1, delta2], axis=-1)
    timestamps = np.array([i / fps for i in range(n_frames)], dtype=np.float32)

    torch.save(
        {
            "landmarks": torch.tensor(landmarks, dtype=torch.float32),
            "landmarks_raw": torch.tensor(raw_landmarks_01, dtype=torch.float32),
            "timestamps": torch.tensor(timestamps, dtype=torch.float32),
            "frame_mask": torch.tensor(frame_mask, dtype=torch.bool),
        },
        str(output_dir / "landmarks.pt"),
    )


def extract_landmarks(video_dir: Path, output_dir: Path, model, device: str, batch_size: int = 32) -> bool:
    """
    Convenience wrapper: load → infer → save in one call.
    Returns True on success.
    """
    if (output_dir / "landmarks.pt").exists():
        return True
    if not (video_dir / "video.mp4").exists():
        raise FileNotFoundError(f"video.mp4 not found: {video_dir}")
    if not (video_dir / "frame_mask.npy").exists():
        raise FileNotFoundError(f"frame_mask.npy not found: {video_dir}")

    valid_indices, tensors, n_frames, fps, frame_mask = _load_sample(video_dir)
    lm_batch = _run_facexformer_tensors(model, tensors, device, batch_size)
    _save_landmarks(valid_indices, lm_batch, n_frames, fps, frame_mask, output_dir)
    return True


def main():
    import argparse
    import csv
    import traceback
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(description="Step 4: FaceXFormer landmark extraction → landmarks.pt")
    parser.add_argument("--manifest", default="Data/manifest_dolos.csv")
    parser.add_argument("--feature_dir", default="features")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32, help="Frames per GPU batch")
    parser.add_argument("--prefetch", type=int, default=2,
                        help="Number of samples to prefetch on CPU while GPU runs inference")
    parser.add_argument("--sample_id", default=None, help="Process a single sample only")
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / args.manifest
    feature_dir = PROJECT_ROOT / args.feature_dir

    with open(manifest_path, newline="") as f:
        samples = list(csv.DictReader(f))

    if args.sample_id:
        samples = [s for s in samples if s["sample_id"] == args.sample_id]
        if not samples:
            print(f"Sample not found: {args.sample_id}")
            return

    # Filter already-done
    pending = [s for s in samples if not (feature_dir / s["sample_id"] / "landmarks.pt").exists()]
    print(f"Samples: {len(samples)} total, {len(samples) - len(pending)} already done, {len(pending)} pending")
    if not pending:
        print("Nothing to do.")
        return

    print(f"Loading FaceXFormer on {args.device}...")
    lm_model = _load_facexformer(args.device)

    errors = []

    import itertools

    # Pipeline: CPU workers prefetch+transform frames while GPU runs inference.
    # ThreadPoolExecutor is safe here because _load_sample only does I/O + CPU transforms.
    with ThreadPoolExecutor(max_workers=args.prefetch) as pool:
        queue = []  # list of (sample_id, sample_dir, Future)

        def submit(row):
            sid = row["sample_id"]
            sd = feature_dir / sid
            return sid, sd, pool.submit(_load_sample, sd)

        it = iter(pending)

        # Pre-fill queue using islice — does NOT exhaust the iterator
        for row in itertools.islice(it, args.prefetch):
            queue.append(submit(row))

        with tqdm(total=len(pending), desc="landmarks", unit="sample") as pbar:
            # Remaining samples + sentinel None to drain the last queued items
            for next_row in itertools.chain(it, [None]):
                sample_id, sample_dir, fut = queue.pop(0)

                # Keep queue full before blocking on fut.result()
                if next_row is not None:
                    queue.append(submit(next_row))

                try:
                    valid_indices, tensors, n_frames, fps, frame_mask = fut.result()
                    lm_batch = _run_facexformer_tensors(lm_model, tensors, args.device, args.batch_size)
                    _save_landmarks(valid_indices, lm_batch, n_frames, fps, frame_mask, sample_dir)
                except Exception as e:
                    errors.append((sample_id, str(e)))
                    tqdm.write(f"  [ERROR] {sample_id}: {e}")
                    tqdm.write(traceback.format_exc())

                pbar.update(1)

    print(f"\nDone. {len(pending) - len(errors)}/{len(pending)} succeeded.")
    if errors:
        for sid, err in errors:
            print(f"  {sid}: {err}")


if __name__ == "__main__":
    main()
