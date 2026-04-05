"""
Dataset for the Real-life Deception Detection 2016 dataset.

Loads paired (transcript, gesture annotations) samples.
Labels: 0 = deceptive (lie), 1 = truthful.
"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .config import GESTURE_COLUMNS, RealLifeConfig


class RealLifeDataset(Dataset):
    """
    Loads transcripts and gesture annotation features from the Real-life
    Deception Detection 2016 dataset.

    Args:
        config:    RealLifeConfig with paths and text_max_length.
        root:      Project root directory. Paths in config are relative to this.
        split_ids: Optional list of sample IDs (video filenames) to include.
                   If None, all samples are loaded.
    """

    def __init__(self, config: RealLifeConfig, root: Path, split_ids=None):
        super().__init__()
        self.config = config
        self.root = root

        annotation_path = root / config.annotation_csv
        transcript_root = root / config.transcript_dir

        df = pd.read_csv(annotation_path)

        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)

        self.samples = []
        for _, row in df.iterrows():
            video_id = row["id"]  # e.g. "trial_lie_001.mp4"
            label_str = row["class"]  # "deceptive" or "truthful"
            label = 0 if label_str == "deceptive" else 1

            # Locate transcript file
            subdir = "Deceptive" if label == 0 else "Truthful"
            stem = Path(video_id).stem  # "trial_lie_001"
            transcript_path = transcript_root / subdir / f"{stem}.txt"
            if not transcript_path.exists():
                raise FileNotFoundError(
                    f"Transcript not found: {transcript_path}\n"
                    f"Expected for sample id='{video_id}' (label={label_str})."
                )

            annot_values = row[GESTURE_COLUMNS].values.astype("float32")
            transcript_text = transcript_path.read_text(encoding="utf-8").strip()

            if split_ids is None or video_id in split_ids:
                self.samples.append({
                    "sample_id": video_id,
                    "label": label,
                    "annotations": annot_values,
                    "transcript": transcript_text,
                })

    def get_labels(self) -> list:
        return [s["label"] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        encoding = self.tokenizer(
            s["transcript"],
            max_length=self.config.text_max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        token_ids = encoding["input_ids"].squeeze(0)       # (n,)
        attention_mask = encoding["attention_mask"].squeeze(0)  # (n,)

        return {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "annotations": torch.tensor(s["annotations"], dtype=torch.float32),
            "label": torch.tensor(s["label"], dtype=torch.float32),
            "sample_id": s["sample_id"],
        }
