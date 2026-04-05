"""
Collation for RealLifeDataset batches.

Pads token sequences to the longest in the batch; stacks everything else.
"""

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch: list) -> dict:
    """
    Args:
        batch: list of dicts from RealLifeDataset.__getitem__

    Returns dict with:
        token_ids      (B, n_max)  LongTensor, padded with tokenizer pad_id=0
        attention_mask (B, n_max)  LongTensor, 1=valid 0=pad
        annotations    (B, 39)     FloatTensor
        label          (B,)        FloatTensor
        sample_ids     list[str]
    """
    token_ids = pad_sequence(
        [item["token_ids"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    attention_mask = pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    annotations = torch.stack([item["annotations"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    sample_ids = [item["sample_id"] for item in batch]

    return {
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "annotations": annotations,
        "label": labels,
        "sample_ids": sample_ids,
    }
