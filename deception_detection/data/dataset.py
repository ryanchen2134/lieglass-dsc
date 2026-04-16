import csv
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path

# ImageNet normalisation constants (float32, shape (3,) for broadcasting)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_frames(
    feat_path: Path,
    n: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load n uniformly-sampled face frames + validity mask for one sample.

    Fast path (preferred):
        Reads features/{id}/frames.npz produced by extract_frames.py.
        Pure numpy — no OpenCV, no video I/O → safe for num_workers > 0.

    Fallback:
        Decodes video.mp4 with OpenCV and loads frame_mask.npy.
        Used when frames.npz is not yet available.

    Returns:
        frames     FloatTensor (n, 3, 224, 224) — ImageNet-normalised RGB
        frame_mask BoolTensor  (n,)             — True = speaker visible
    """
    npz_path = feat_path / "frames.npz"

    if npz_path.exists():
        # --- Fast path: load pre-extracted numpy archive ---
        try:
            data   = np.load(str(npz_path))
            frames_all = data["frames"]   # (N, H, W, 3) uint8
            mask_all   = data["mask"]     # (N,) bool
        except (EOFError, ValueError, KeyError):
            # Corrupt / truncated npz — fall through to OpenCV or zeros
            pass
        else:
            N = len(frames_all)
            if N != n:
                idx    = np.linspace(0, N - 1, n, dtype=int)
                frames_all = frames_all[idx]
                mask_all   = mask_all[idx]
            frames = frames_all.astype(np.float32) / 255.0          # (n, H, W, 3)
            frames = (frames - _MEAN) / _STD                        # ImageNet normalise
            frames = np.ascontiguousarray(frames.transpose(0, 3, 1, 2))  # (n, 3, H, W)
            return torch.from_numpy(frames), torch.from_numpy(mask_all.copy())

    # --- Fallback: OpenCV video decode (or zeros if video is also missing) ---
    video_path = feat_path / "video.mp4"
    if video_path.exists() and video_path.stat().st_size > 200:
        return _sample_frames_opencv(video_path, feat_path / "frame_mask.npy", n)

    # No usable video source — return black frames with mask=False
    return torch.zeros(n, 3, 224, 224), torch.zeros(n, dtype=torch.bool)


def _sample_frames_opencv(
    video_path: Path,
    mask_path: Path | None,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decode video_path with OpenCV, uniformly sample n frames.
    NOTE: OpenCV is not fork-safe; keep num_workers=0 when using this path.
    """
    import cv2

    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return torch.zeros(n, 3, 224, 224), torch.ones(n, dtype=torch.bool)

    indices = np.linspace(0, total - 1, n, dtype=int)

    # Co-sample frame_mask.npy at the same frame indices
    if mask_path is not None and mask_path.exists():
        full_mask    = np.load(str(mask_path))
        clipped      = np.clip(indices, 0, len(full_mask) - 1)
        sampled_mask = torch.from_numpy(full_mask[clipped].copy())
    else:
        sampled_mask = torch.ones(n, dtype=torch.bool)

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
        frame = (frame - _MEAN) / _STD
        frames.append(frame)

    cap.release()

    arr = np.stack(frames)                                   # (n, 224, 224, 3)
    arr = np.ascontiguousarray(arr.transpose(0, 3, 1, 2))  # (n, 3, 224, 224)
    return torch.from_numpy(arr), sampled_mask


class DeceptionDataset(Dataset):
    """
    Loads audio.wav and face frames from pre-processed feature directories.

    Visual loading priority:
      1. frames.npz  (fast, no OpenCV — run extract_frames.py first)
      2. video.mp4   (OpenCV fallback — requires num_workers=0)

    Each __getitem__ returns a dict with:
      waveform    FloatTensor (T,)
      frames      FloatTensor (n_frames, 3, 224, 224)
      frame_mask  BoolTensor  (n_frames,)  True = speaker visible
      label       FloatTensor scalar
      sample_id   str
    """

    def __init__(self, manifest_csv: str, feature_dir: str, augment: bool = False, n_frames: int = 16):
        self.feature_dir = Path(feature_dir)
        self.augment     = augment
        self.n_frames    = n_frames
        self.samples     = []

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

        # --- Visual: n frames + validity mask ---
        frames, frame_mask = _load_frames(feat_path, n=self.n_frames)

        if self.augment:
            waveform   = self._augment_waveform(waveform)
            frames     = self._augment_frames(frames)

        return {
            "waveform":   waveform,
            "frames":     frames,
            "frame_mask": frame_mask,
            "label":      torch.tensor(label, dtype=torch.float32),
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
