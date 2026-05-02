"""
Dataset for the bimodal deception detection model.

Two frame sources are supported, in priority order:

  1. ``frames_full.npz`` — written by ``preprocess_fullframe.py``. Holds
     EVERY frame of the clip (variable N), uint8 RGB, already face-centred
     with a 3× YOLO box. No frame_mask — nothing is blacked out.

  2. ``frames.npz`` — legacy uniform-sampled archive (fixed N). Kept so old
     feature caches remain usable; we return the stored mask alongside.

Both sources are fork-safe (pure NumPy) so ``num_workers > 0`` is always safe.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


# Group-extraction patterns for leakage-safe cross-validation.
#   DOLOS sample IDs look like ``<group>_EP<num>_<...>`` where ``<group>``
#   identifies a contestant (e.g. ``AN_WILTY``, ``LS_WILTY``). All clips
#   from the same contestant — across every episode and every clip type —
#   must stay in the same fold.
#
#   RLT sample IDs look like ``<group>_Chunk<num>`` where ``<group>``
#   identifies a single trial (e.g. ``trial_truth_036``). All chunks of
#   the same trial must stay in the same fold.
_DOLOS_GROUP_RE = re.compile(r"^(.+?)_EP\d+")
_RLT_GROUP_RE = re.compile(r"^(.+?)_Chunk\d+")


def extract_group(sample_id: str) -> str:
    """Return the leakage-safe group key for a manifest sample ID."""
    m = _DOLOS_GROUP_RE.match(sample_id)
    if m:
        return m.group(1)
    m = _RLT_GROUP_RE.match(sample_id)
    if m:
        return m.group(1)
    # Fallback: treat the sample as its own singleton group so it remains
    # usable with group-aware splitters even if the naming is unfamiliar.
    return sample_id

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


def _load_audio(path: Path) -> tuple[torch.Tensor, int]:
    """Load a WAV via torchaudio (torchcodec backend), falling back to
    soundfile if the primary backend raises (e.g. torchcodec missing or
    incompatible FFmpeg). Returns (waveform[channels, time], sample_rate)."""
    try:
        return torchaudio.load(str(path))
    except Exception as e:
        try:
            import soundfile as sf
        except ImportError as imp_err:
            raise RuntimeError(
                f"torchaudio.load failed for {path} ({e}); install `soundfile` to enable fallback"
            ) from imp_err
        data, sr = sf.read(str(path), always_2d=True)  # (T, C)
        waveform = torch.from_numpy(data.T).float()    # (C, T)
        return waveform, sr


# ---------------------------------------------------------------------------
# Frame loaders
# ---------------------------------------------------------------------------

def _load_frames_full(
    npz_path: Path,
    max_frames: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load ALL frames from ``frames_full.npz``.

    Returns:
        frames     ByteTensor (N, 3, 224, 224)  uint8 RGB
        frame_mask BoolTensor (N,)              all True (no blacked frames)

    When ``max_frames`` is set and the clip is longer, we take a contiguous
    centre window. The task brief says "use all frames", so we only trim
    when forced by the VRAM cap.
    """
    data = np.load(str(npz_path))
    frames_hwc = data["frames"]                # (N, 224, 224, 3) uint8
    if frames_hwc.shape[-1] == 3:
        frames_hwc = frames_hwc[..., :1]
    N = len(frames_hwc)

    if max_frames is not None and N > max_frames:
        # Centre crop along the temporal axis — preserves continuity.
        start = (N - max_frames) // 2
        frames_hwc = frames_hwc[start:start + max_frames]
        N = max_frames

    frames_chw = np.ascontiguousarray(frames_hwc.transpose(0, 3, 1, 2))
    return torch.from_numpy(frames_chw), torch.ones(N, dtype=torch.bool)


def _load_frames_legacy(
    npz_path: Path,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load the legacy fixed-N archive (``frames.npz``) used by the old
    uniform-sampling pipeline. Resamples to ``n`` if the archive has a
    different count.
    """
    data = np.load(str(npz_path))
    frames_hwc = data["frames"]                # (N, H, W, 3) uint8
    if frames_hwc.shape[-1] == 3:
        frames_hwc = frames_hwc[..., :1]

    mask_all   = data["mask"]                  # (N,) bool
    N = len(frames_hwc)
    if N != n:
        idx = np.linspace(0, N - 1, n, dtype=int)
        frames_hwc = frames_hwc[idx]
        mask_all   = mask_all[idx]
    if frames_hwc.ndim == 4: # (N, H, W, C)
        frames_chw = np.ascontiguousarray(frames_hwc.transpose(0, 3, 1, 2))
    else: # (N, H, W) -> (N, 1, H, W)
        frames_chw = frames_hwc[:, np.newaxis, :, :]
    return torch.from_numpy(frames_chw), torch.from_numpy(mask_all.copy())


def _load_frames(
    feat_path: Path,
    max_frames: int | None,
    legacy_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the best frame source available for one sample."""
    full = feat_path / "frames_full.npz"
    if full.exists():
        try:
            return _load_frames_full(full, max_frames)
        except (EOFError, ValueError, KeyError):
            pass  # fall through to legacy / zeros

    legacy = feat_path / "frames.npz"
    if legacy.exists():
        try:
            return _load_frames_legacy(legacy, legacy_n)
        except (EOFError, ValueError, KeyError):
            pass

    # Nothing on disk — emit a single black frame so the model still runs.
    return (
        torch.zeros(1, 1, 224, 224, dtype=torch.uint8), # 1 channel
        torch.zeros(1, dtype=torch.bool),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DeceptionDataset(Dataset):
    """
    Returns a dict per item:
        waveform    FloatTensor (T,)
        frames      ByteTensor  (N, 3, 224, 224)   — variable N
        frame_mask  BoolTensor  (N,)
        label       FloatTensor scalar
        sample_id   str

    Frames are uint8; GPU-side normalisation in ``FusionModel.forward`` keeps
    the CPU→GPU transfer 4× smaller than pre-normalised float32.
    """

    def __init__(
        self,
        manifest_csv: str,
        feature_dir: str,
        augment: bool = False,
        max_frames: int | None = 400,
        legacy_n_frames: int = 16,
    ):
        self.feature_dir     = Path(feature_dir)
        self.augment         = augment
        self.max_frames      = max_frames
        self.legacy_n_frames = legacy_n_frames
        self.samples: list[tuple[str, int]] = []

        missing_count = 0
        with open(manifest_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = row["sample_id"]
                # Check if the mandatory audio file exists before adding to list
                check_path = self.feature_dir / sample_id / "audio.wav"
                
                if check_path.exists():
                    self.samples.append((sample_id, int(row["label"])))
                else:
                    missing_count += 1
        
        if missing_count > 0:
            print(f"NOTE: Skipped {missing_count} samples because audio.wav was missing in {feature_dir}")

    def get_labels(self) -> list[int]:
        return [label for _, label in self.samples]

    def get_groups(self) -> list[str]:
        """Group key per sample for leakage-safe (group-aware) CV splits."""
        return [extract_group(sid) for sid, _ in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample_id, label = self.samples[idx]
        feat_path = self.feature_dir / sample_id

        # --- Audio ---
        waveform, sr = _load_audio(feat_path / "audio.wav")
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.squeeze(0)  # (T,)

        # --- Frames ---
        frames, frame_mask = _load_frames(
            feat_path,
            max_frames=self.max_frames,
            legacy_n=self.legacy_n_frames,
        )

        if self.augment:
            waveform = self._augment_waveform(waveform)
            # Pixel-noise augmentation runs on GPU after uint8 → float
            # normalisation (see FusionModel.forward).

        if frames.shape[1] == 3:
            # Weighted average converts (N, 3, H, W) -> (N, 1, H, W)
            w = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
            frames = (frames.float() * w).sum(dim=1, keepdim=True).to(torch.uint8)

        return {
            "waveform":   waveform,
            "frames":     frames,
            "frame_mask": frame_mask,
            "label":      torch.tensor(label, dtype=torch.float32),
            "sample_id":  sample_id,
        }

    @staticmethod
    def _augment_waveform(waveform: torch.Tensor) -> torch.Tensor:
        sigma = torch.empty(1).uniform_(0.0, 0.005).item()
        return waveform + torch.randn_like(waveform) * sigma
