import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """
    Pad all modalities to max length in the batch. Return padding masks.

    Returns dict with keys:
        text_token_ids:          LongTensor  (B, n_max)
        text_timestamps:         FloatTensor (B, n_max, 2)
        text_padding_mask:       BoolTensor  (B, n_max)       — True = PAD
        mfcc:                    FloatTensor (B, T_m_max, 13, 3)
        mfcc_timestamps:         FloatTensor (B, T_m_max)
        mfcc_padding_mask:       BoolTensor  (B, T_m_max)     — True = PAD
        landmarks:               FloatTensor (B, T_l_max, 136, 3)
        landmark_timestamps:     FloatTensor (B, T_l_max)
        landmark_padding_mask:   BoolTensor  (B, T_l_max)     — True = PAD
        frame_mask:              BoolTensor  (B, T_l_max)     — True = VALID speaker frame
        label:                   FloatTensor (B,)
        sample_ids:              list of str
    """
    B = len(batch)

    # --- Text ---
    text_ids = [item["text_token_ids"] for item in batch]       # list of (n_i,) LongTensor
    text_ts  = [item["text_timestamps"] for item in batch]      # list of (n_i, 2) FloatTensor

    # Handle empty text sequences
    text_ids_padded = pad_sequence(
        [t if t.shape[0] > 0 else torch.zeros(1, dtype=torch.long) for t in text_ids],
        batch_first=True, padding_value=0,
    )  # (B, n_max)
    text_ts_padded = pad_sequence(
        [t if t.shape[0] > 0 else torch.zeros(1, 2) for t in text_ts],
        batch_first=True, padding_value=0.0,
    )  # (B, n_max, 2)

    n_max = text_ids_padded.shape[1]
    text_padding_mask = torch.zeros(B, n_max, dtype=torch.bool)
    for i, t in enumerate(text_ids):
        real_len = max(t.shape[0], 1)  # at least 1 (we added a dummy token for empty)
        if t.shape[0] == 0:
            text_padding_mask[i, :] = True  # all pad
        else:
            text_padding_mask[i, real_len:] = True

    # --- MFCC ---
    mfcc_list = [item["mfcc"] for item in batch]          # list of (T_m_i, 13, 3)
    mfcc_ts_list = [item["mfcc_timestamps"] for item in batch]  # list of (T_m_i,)

    mfcc_padded = pad_sequence(mfcc_list, batch_first=True, padding_value=0.0)      # (B, T_m_max, 13, 3)
    mfcc_ts_padded = pad_sequence(mfcc_ts_list, batch_first=True, padding_value=0.0) # (B, T_m_max)

    T_m_max = mfcc_padded.shape[1]
    mfcc_padding_mask = torch.zeros(B, T_m_max, dtype=torch.bool)
    for i, t in enumerate(mfcc_list):
        mfcc_padding_mask[i, t.shape[0]:] = True

    # --- Landmarks ---
    land_list    = [item["landmarks"] for item in batch]          # list of (T_l_i, 136, 3)
    land_ts_list = [item["landmark_timestamps"] for item in batch] # list of (T_l_i,)
    fmask_list   = [item["frame_mask"] for item in batch]          # list of (T_l_i,) bool

    land_padded    = pad_sequence(land_list,    batch_first=True, padding_value=0.0) # (B, T_l_max, 136, 3)
    land_ts_padded = pad_sequence(land_ts_list, batch_first=True, padding_value=0.0) # (B, T_l_max)
    fmask_padded   = pad_sequence(fmask_list,   batch_first=True, padding_value=False) # (B, T_l_max)

    T_l_max = land_padded.shape[1]
    land_padding_mask = torch.zeros(B, T_l_max, dtype=torch.bool)
    for i, t in enumerate(land_list):
        land_padding_mask[i, t.shape[0]:] = True

    # --- Labels & IDs ---
    labels = torch.stack([item["label"] for item in batch])  # (B,)
    sample_ids = [item["sample_id"] for item in batch]

    return {
        "text_token_ids":        text_ids_padded,
        "text_timestamps":       text_ts_padded,
        "text_padding_mask":     text_padding_mask,
        "mfcc":                  mfcc_padded,
        "mfcc_timestamps":       mfcc_ts_padded,
        "mfcc_padding_mask":     mfcc_padding_mask,
        "landmarks":             land_padded,
        "landmark_timestamps":   land_ts_padded,
        "landmark_padding_mask": land_padding_mask,
        "frame_mask":            fmask_padded,
        "label":                 labels,
        "sample_ids":            sample_ids,
    }
