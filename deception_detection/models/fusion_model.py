"""
Cross-modal fusion model: Wav2Vec2 audio + CNN+temporal-transformer visual,
fused with multi-stage bidirectional cross-attention (DOLOS PAVF).

Pipeline:
  audio: waveform -> W2V2 (UT-Adapter) -> stages [F_1, F_mid, F_end] of (B, T', 768)
  visual: frames  -> CNN + temporal Transformer (UT-Adapter)
                                       -> stages [F_1, F_mid, F_end] of (B, N, d_visual)
  -> MultiStageFusion (bidirectional cross-attn per stage; weighted-sum aggregate)
  -> classifier head -> logit (B,)
"""

import torch
import torch.nn as nn

from .audio_model import W2V2_Model
from .cross_fusion import MultiStageFusion
from .visual_model import ViT_Model


class FusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if len(config.audio_fusion_layers) != len(config.visual_fusion_layers):
            raise ValueError("audio_fusion_layers and visual_fusion_layers must have equal length.")

        self.audio_model = W2V2_Model(config)
        self.visual_model = ViT_Model(config)

        self.fusion = MultiStageFusion(
            n_stages=len(config.audio_fusion_layers),
            d_audio=config.d_audio,
            d_visual=config.d_visual,
            d_fused=config.d_fused,
            n_heads=config.fusion_n_heads,
            dropout=config.fusion_dropout,
            aggregator=config.fusion_aggregator,
        )

        self.head = nn.Sequential(
            nn.Linear(config.d_fused, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        waveform = batch["waveform"]
        frames = batch["frames"]
        waveform_mask = batch.get("waveform_mask")
        frame_mask = batch.get("frame_mask")

        # uint8 -> float in [0, 1]; per-backbone mean/std lives inside the
        # visual encoder (CLIP / ArcFace use different normalization).
        if frames.dtype == torch.uint8:
            frames = frames.float().div(255.0)

        audio_stages, a_mask = self.audio_model.forward_multistage(waveform, waveform_mask)
        visual_stages, v_mask = self.visual_model.forward_multistage(frames, frame_mask)

        fused = self.fusion(audio_stages, visual_stages, a_mask=a_mask, v_mask=v_mask)
        logits = self.head(fused).squeeze(-1)
        return logits
