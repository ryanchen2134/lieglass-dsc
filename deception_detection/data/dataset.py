import csv
import torch
from torch.utils.data import Dataset
from pathlib import Path


class DeceptionDataset(Dataset):
    """
    Loads pre-extracted features from disk.
    Each __getitem__ returns a dict with all modalities + metadata.
    """

    def __init__(self, manifest_csv: str, feature_dir: str, augment: bool = False):
        """
        Args:
            manifest_csv: Path to CSV with columns [sample_id, label, dataset_source, ...].
            feature_dir:  Root directory containing per-sample feature folders.
            augment:      Whether to apply data augmentation (training only).
        """
        self.feature_dir = Path(feature_dir)
        self.augment = augment
        self.samples = []  # list of (sample_id, label)

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

        text_data = torch.load(feat_path / "text.pt", weights_only=True)
        mfcc_data = torch.load(feat_path / "mfcc.pt", weights_only=True)
        land_data = torch.load(feat_path / "landmarks.pt", weights_only=True)

        if self.augment:
            mfcc_data = self._augment_mfcc(mfcc_data)
            land_data = self._augment_landmarks(land_data)

        return {
            "text_token_ids": text_data["token_ids"],           # LongTensor (n,)
            "text_timestamps": text_data["timestamps"],         # FloatTensor (n, 2)
            "mfcc": mfcc_data["mfcc"],                          # FloatTensor (T_m, 13, 3)
            "mfcc_timestamps": mfcc_data["timestamps"],         # FloatTensor (T_m,)
            "landmarks": land_data["landmarks"],                # FloatTensor (T_l, 136, 3)
            "landmark_timestamps": land_data["timestamps"],     # FloatTensor (T_l,)
            "frame_mask": land_data["frame_mask"],              # BoolTensor (T_l,)
            "label": torch.tensor(label, dtype=torch.float32), # scalar
            "sample_id": sample_id,
        }

    def _augment_mfcc(self, mfcc_data: dict) -> dict:
        """
        Augmentations for MFCC:
        1. Additive Gaussian noise: sigma ~ U(0, 0.01)
        2. Temporal dropout: randomly zero ~5-10% of frames
        """
        mfcc = mfcc_data["mfcc"].clone()  # (T_m, 13, 3)
        T = mfcc.shape[0]

        # Additive noise
        sigma = torch.empty(1).uniform_(0, 0.01).item()
        mfcc = mfcc + torch.randn_like(mfcc) * sigma

        # Temporal dropout: zero out ~5-10% of frames
        drop_rate = torch.empty(1).uniform_(0.05, 0.10).item()
        drop_mask = torch.rand(T) < drop_rate
        mfcc[drop_mask] = 0.0

        return {**mfcc_data, "mfcc": mfcc}

    def _augment_landmarks(self, land_data: dict) -> dict:
        """
        Augmentations for landmarks:
        1. Additive Gaussian jitter on valid frames only: sigma ~ U(0, 0.005)
        """
        landmarks = land_data["landmarks"].clone()  # (T_l, 136, 3)
        frame_mask = land_data["frame_mask"]        # (T_l,) bool

        sigma = torch.empty(1).uniform_(0, 0.005).item()
        noise = torch.randn_like(landmarks) * sigma
        noise[~frame_mask] = 0.0  # only jitter valid frames
        landmarks = landmarks + noise

        return {**land_data, "landmarks": landmarks}
