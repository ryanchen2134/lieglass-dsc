"""
Variable-length collate for the full-frame pipeline.

Each item has a different number of frames (all frames of the clip are kept).
We pad to the batch maximum along the temporal axis and emit ``frame_mask``
telling the model which frames are real vs. padding.
"""

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """
    Returns dict with keys:
        waveform        FloatTensor (B, T_max)              zero-padded
        waveform_mask   BoolTensor  (B, T_max)              True = valid
        frames          ByteTensor  (B, N_max, 3, 224, 224) uint8
        frame_mask      BoolTensor  (B, N_max)              True = real frame
        label           FloatTensor (B,)
        sample_ids      list[str]
    """
    # --- Audio ---
    waves = [item["waveform"] for item in batch]
    wave_padded = pad_sequence(waves, batch_first=True, padding_value=0.0)
    T_max = wave_padded.shape[1]
    B = len(batch)
    wave_mask = torch.zeros(B, T_max, dtype=torch.bool)
    for i, w in enumerate(waves):
        wave_mask[i, :w.shape[0]] = True

    # --- Frames: pad temporal axis to batch max. ---
    frame_tensors = [item["frames"] for item in batch]            # list of (N_i, 3, 224, 224)
    mask_tensors  = [item["frame_mask"] for item in batch]        # list of (N_i,)
    N_max = max(t.shape[0] for t in frame_tensors)

    _, C, H, W = frame_tensors[0].shape
    frames_padded = torch.zeros(B, N_max, C, H, W, dtype=torch.uint8)
    frame_mask    = torch.zeros(B, N_max, dtype=torch.bool)
    for i, (f, m) in enumerate(zip(frame_tensors, mask_tensors)):
        n = f.shape[0]
        frames_padded[i, :n] = f
        frame_mask[i, :n]    = m

    labels     = torch.stack([item["label"] for item in batch])
    sample_ids = [item["sample_id"] for item in batch]

    return {
        "waveform":      wave_padded,
        "waveform_mask": wave_mask,
        "frames":        frames_padded,
        "frame_mask":    frame_mask,
        "label":         labels,
        "sample_ids":    sample_ids,
    }
