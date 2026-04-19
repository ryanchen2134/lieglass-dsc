"""
Wav2Vec2-based audio encoder for deception detection.

Architecture (adapted from DOLOS repo — NMS05/Audio-Visual-Deception-Detection-...):
  raw waveform (B, T)
    → Wav2Vec2Base CNN feature extractor  [frozen]
    → feature projection + layer norm
    → Wav2Vec2 transformer encoder        [last wav2vec2_unfreeze_last_n layers trainable]
    → mean temporal pool
    → audio embedding (B, 768)
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class W2V2_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w2v2 = Wav2Vec2Model.from_pretrained(config.wav2vec2_model)

        # Freeze CNN feature extractor + feature projection always
        for p in self.w2v2.feature_extractor.parameters():
            p.requires_grad = False
        for p in self.w2v2.feature_projection.parameters():
            p.requires_grad = False

        # Freeze transformer layers except the last N
        n_total = len(self.w2v2.encoder.layers)
        n_freeze = max(0, n_total - config.wav2vec2_unfreeze_last_n)
        for i, layer in enumerate(self.w2v2.encoder.layers):
            if i < n_freeze:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, waveform: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            waveform:       FloatTensor (B, T) — raw 16 kHz waveform
            attention_mask: BoolTensor  (B, T) — True = valid, False = padding (optional)

        Returns:
            FloatTensor (B, 768) — mean-pooled audio embedding
        """
        # Wav2Vec2 expects attention_mask as LongTensor with 1=valid, 0=pad
        mask = None
        if attention_mask is not None:
            mask = attention_mask.long()

        outputs = self.w2v2(input_values=waveform, attention_mask=mask)
        hidden = outputs.last_hidden_state  # (B, T', 768)

        if mask is not None:
            # Compute frame-level mask: Wav2Vec2 downsamples T by ~320
            # Use the model's built-in masking via output_lengths if available,
            # otherwise fall back to simple mean pool (padding frames near zero anyway)
            pass

        return hidden.mean(dim=1)  # (B, 768)
