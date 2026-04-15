import csv
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path


def _sample_frames(
    video_path: Path,
    mask_path: Path | None,
    n: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Uniformly sample `n` frames from a video file.

    Returns:
        frames     FloatTensor (n, 3, 224, 224) — ImageNet-normalised RGB frames
        frame_mask BoolTensor  (n,)             — True = speaker visible (not black frame)

    frame_mask is derived from frame_mask.npy if available; otherwise all-True
    (assume every frame is valid).
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return torch.zeros(n, 3, 224, 224), torch.ones(n, dtype=torch.bool)

    indices = np.linspace(0, total - 1, n, dtype=int)

    # --- Frame mask sampled at same indices ---
    if mask_path is not None and mask_path.exists():
        full_mask = np.load(str(mask_path))          # (n_frames,) bool
        # Clip in case mask length differs slightly from frame count
        clipped = np.clip(indices, 0, len(full_mask) - 1)
        sampled_mask = torch.from_numpy(full_mask[clipped].copy())
    else:
        sampled_mask = torch.ones(n, dtype=torch.bool)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            frames.append(np.zeros((224, 224, 3), dtype=np.float32))
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - mean) / std
        frames.append(frame)

    cap.release()

    arr = np.stack(frames)               # (n, 224, 224, 3)
    arr = arr.transpose(0, 3, 1, 2)     # (n, 3, 224, 224)
    return torch.from_numpy(arr), sampled_mask


class DeceptionDataset(Dataset):
    """
    Loads audio.wav and video.mp4 from pre-processed feature directories.
    Each __getitem__ returns a dict with waveform, frames, label, and sample_id.
    """

    def __init__(self, manifest_csv: str, feature_dir: str, augment: bool = False):
        """
        Args:
            manifest_csv: CSV with columns [sample_id, label, dataset_source, ...].
            feature_dir:  Root dir containing per-sample feature folders.
            augment:      Whether to apply data augmentation (training only).
        """
        self.feature_dir = Path(feature_dir)
        self.augment = augment
        self.samples = []

        with open(manifest_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["sample_id"], int(row["label"])))

    def get_labels(self):
        """Return list of int labels (for WeightedRandomSampler)."""
        return [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id, label = self.samples[idx]
        feat_path = self.feature_dir / sample_id

        # --- Audio: raw 16 kHz waveform ---
        waveform, sr = torchaudio.load(feat_path / "audio.wav")
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.squeeze(0)  # (T,)

        # --- Visual: 64 uniformly sampled face frames + mask ---
        frames, frame_mask = _sample_frames(
            feat_path / "video.mp4",
            feat_path / "frame_mask.npy",
            n=64,
        )  # (64, 3, 224, 224), (64,)

        if self.augment:
            waveform = self._augment_waveform(waveform)
            frames   = self._augment_frames(frames)

        return {
            "waveform":   waveform,                                    # FloatTensor (T,)
            "frames":     frames,                                      # FloatTensor (64, 3, 224, 224)
            "frame_mask": frame_mask,                                  # BoolTensor  (64,)
            "label":      torch.tensor(label, dtype=torch.float32),   # scalar
            "sample_id":  sample_id,
        }

    def _augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Light additive noise augmentation for audio."""
        sigma = torch.empty(1).uniform_(0.0, 0.005).item()
        return waveform + torch.randn_like(waveform) * sigma

    def _augment_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Light pixel-noise augmentation for frames (after normalisation)."""
        sigma = torch.empty(1).uniform_(0.0, 0.02).item()
        return frames + torch.randn_like(frames) * sigma
