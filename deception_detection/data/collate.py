import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """
    Collate a list of dataset items into a padded batch.

    Returns dict with keys:
        waveform        FloatTensor (B, T_max)           — zero-padded waveforms
        waveform_mask   BoolTensor  (B, T_max)           — True = valid sample
        frames          FloatTensor (B, 64, 3, 224, 224)
        frame_mask      BoolTensor  (B, 64)              — True = speaker visible
        label           FloatTensor (B,)
        sample_ids      list of str
    """
    # --- Waveform (variable length) ---
    waves = [item["waveform"] for item in batch]
    wave_padded = pad_sequence(waves, batch_first=True, padding_value=0.0)  # (B, T_max)

    T_max = wave_padded.shape[1]
    B = len(batch)
    wave_mask = torch.zeros(B, T_max, dtype=torch.bool)
    for i, w in enumerate(waves):
        wave_mask[i, :w.shape[0]] = True  # True = valid

    # --- Frames (fixed 64 frames per sample) ---
    frames     = torch.stack([item["frames"]     for item in batch])  # (B, 64, 3, 224, 224)
    frame_mask = torch.stack([item["frame_mask"] for item in batch])  # (B, 64) bool

    # --- Labels & IDs ---
    labels     = torch.stack([item["label"] for item in batch])   # (B,)
    sample_ids = [item["sample_id"] for item in batch]

    return {
        "waveform":      wave_padded,
        "waveform_mask": wave_mask,
        "frames":        frames,
        "frame_mask":    frame_mask,
        "label":         labels,
        "sample_ids":    sample_ids,
    }
