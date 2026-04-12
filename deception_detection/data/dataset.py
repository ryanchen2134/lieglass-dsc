"""
DeceptionDataset v2 — loads raw audio waveforms and face frames from disk.

Per-sample feature directory layout (features/{sample_id}/):
  audio.wav    — 16 kHz mono WAV (speaker-cropped)
  video.mp4    — face-cropped, speaker-detected video
  text.pt      — dict{token_ids: LongTensor(n,), timestamps: FloatTensor(n,2)}

MFCC and landmark .pt files are no longer used.

Audio pipeline (per sample):
  1. Read audio.wav with soundfile (falls back to scipy.io.wavfile).
  2. Convert to mono if stereo.
  3. Resample to 16 kHz if needed (scipy.signal.resample).
  4. Pad or trim to TARGET_SAMPLES = 20480 (= 64 × 320 Wav2Vec2 downsample factor).
  → FloatTensor (20480,)

Visual pipeline (per sample):
  1. Open video.mp4 with cv2.VideoCapture.
  2. Sample 64 uniformly spaced frames (np.linspace).
  3. Resize each frame to 160×160, convert BGR→RGB.
  4. Stack → (64, 3, 160, 160) float32, normalise with ImageNet mean/std.
  → FloatTensor (64, 3, 160, 160)

Text pipeline (unchanged from v1):
  Load token_ids from text.pt.
  → LongTensor (n,)   (variable length; padded in collate_fn)
"""

import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

# Fixed audio sequence length: 64 Wav2Vec2 tokens × 320 samples/token
_TARGET_SAMPLES = 64 * 320  # 20480

# ImageNet normalisation constants
_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMG_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Number of uniformly sampled visual frames
_N_FRAMES = 64


def _load_waveform(audio_path: Path) -> torch.Tensor:
    """Load audio.wav → FloatTensor (20480,) at 16 kHz."""
    try:
        import soundfile as sf
        wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    except Exception:
        from scipy.io import wavfile
        sr, wav = wavfile.read(str(audio_path))
        wav = wav.astype(np.float32)
        # scipy returns int16 (−32768..32767) or float — normalise if int
        if wav.dtype != np.float32 or wav.max() > 1.1:
            wav = wav / 32768.0

    # Mono: average channels if needed
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    # Resample to 16 kHz if needed
    if sr != 16000:
        from scipy.signal import resample
        n_target = int(len(wav) * 16000 / sr)
        wav = resample(wav, n_target).astype(np.float32)

    # Pad or trim to fixed length
    if len(wav) >= _TARGET_SAMPLES:
        wav = wav[:_TARGET_SAMPLES]
    else:
        wav = np.pad(wav, (0, _TARGET_SAMPLES - len(wav)))

    return torch.from_numpy(wav)  # (20480,)


def _load_frames(video_path: Path) -> torch.Tensor:
    """
    Load 64 uniformly sampled frames from video.mp4.
    Returns FloatTensor (64, 3, 160, 160), ImageNet-normalised.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    n_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    indices = np.linspace(0, n_frames - 1, _N_FRAMES).astype(int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            frame = np.zeros((160, 160, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (160, 160), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()

    # Stack → (64, H, W, 3) → permute → (64, 3, H, W) → [0, 1] range
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0   # (64, 160, 160, 3)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)           # (64, 3, 160, 160)

    # ImageNet normalisation (broadcasting: (1,3,1,1))
    tensor = (tensor - _IMG_MEAN) / _IMG_STD
    return tensor  # (64, 3, 160, 160)


def _augment_frames(frames: torch.Tensor) -> torch.Tensor:
    """Light augmentation for training: additive Gaussian noise on pixel values."""
    sigma = torch.empty(1).uniform_(0.0, 0.01).item()
    return frames + torch.randn_like(frames) * sigma


class DeceptionDataset(Dataset):
    """
    Loads raw waveforms and face frames for the v2 PECL-adapted model.

    Each __getitem__ returns a dict with keys:
        waveform:          FloatTensor (20480,)
        frames:            FloatTensor (64, 3, 160, 160)
        text_token_ids:    LongTensor  (n,)
        label:             LongTensor  scalar (0 or 1)
        sample_id:         str
    """

    def __init__(self, manifest_csv: str, feature_dir: str, augment: bool = False):
        """
        Args:
            manifest_csv: Path to CSV with columns [sample_id, label, ...].
            feature_dir:  Root directory containing per-sample feature folders.
            augment:      Apply training augmentation (training set only).
        """
        self.feature_dir = Path(feature_dir)
        self.augment = augment
        self.samples = []  # list of (sample_id, label)

        with open(manifest_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["sample_id"], int(row["label"])))

    def get_labels(self) -> list:
        """Return list of int labels (used by WeightedRandomSampler and GroupKFold)."""
        return [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample_id, label = self.samples[idx]
        feat_path = self.feature_dir / sample_id

        # Audio
        waveform = _load_waveform(feat_path / "audio.wav")

        # Visual
        frames = _load_frames(feat_path / "video.mp4")
        if self.augment:
            frames = _augment_frames(frames)

        # Text (token IDs only — timestamps no longer used in v2)
        text_data = torch.load(feat_path / "text.pt", weights_only=True)
        token_ids = text_data["token_ids"]  # LongTensor (n,)

        return {
            "waveform":       waveform,
            "frames":         frames,
            "text_token_ids": token_ids,
            "label":          torch.tensor(label, dtype=torch.long),
            "sample_id":      sample_id,
        }
