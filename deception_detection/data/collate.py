"""
collate_fn v2 — simplified batch collation for fixed-size audio/visual inputs.

Audio (20480 samples) and visual (64×3×160×160) are fixed-size, so they just
need torch.stack. Only text token IDs are variable-length and still need
pad_sequence.

Returned batch dict:
    waveform:          FloatTensor (B, 20480)
    frames:            FloatTensor (B, 64, 3, 160, 160)
    text_token_ids:    LongTensor  (B, n_max)
    text_padding_mask: BoolTensor  (B, n_max)  — True = PAD position
    label:             LongTensor  (B,)
    sample_ids:        list[str]
"""

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch: list) -> dict:
    """
    Args:
        batch: list of dicts from DeceptionDataset.__getitem__.

    Returns:
        Batched tensors as described above.
    """
    B = len(batch)

    # ------------------------------------------------------------------ #
    # Fixed-size modalities — just stack
    # ------------------------------------------------------------------ #
    waveforms = torch.stack([item["waveform"] for item in batch])   # (B, 20480)
    frames    = torch.stack([item["frames"]   for item in batch])   # (B, 64, 3, 160, 160)
    labels    = torch.stack([item["label"]    for item in batch])   # (B,) LongTensor

    # ------------------------------------------------------------------ #
    # Text — variable length, needs padding
    # ------------------------------------------------------------------ #
    text_ids = [item["text_token_ids"] for item in batch]  # list of LongTensor (n_i,)

    # Guard against empty transcriptions (rare edge case)
    text_ids_safe = [
        t if t.shape[0] > 0 else torch.zeros(1, dtype=torch.long)
        for t in text_ids
    ]
    text_ids_padded = pad_sequence(
        text_ids_safe, batch_first=True, padding_value=0
    )  # (B, n_max)

    n_max = text_ids_padded.shape[1]
    text_padding_mask = torch.zeros(B, n_max, dtype=torch.bool)
    for i, t in enumerate(text_ids):
        if t.shape[0] == 0:
            text_padding_mask[i, :] = True    # all-pad for empty transcription
        else:
            text_padding_mask[i, t.shape[0]:] = True   # pad positions

    # ------------------------------------------------------------------ #
    # Sample IDs (not a tensor)
    # ------------------------------------------------------------------ #
    sample_ids = [item["sample_id"] for item in batch]

    return {
        "waveform":          waveforms,
        "frames":            frames,
        "text_token_ids":    text_ids_padded,
        "text_padding_mask": text_padding_mask,
        "label":             labels,
        "sample_ids":        sample_ids,
    }
