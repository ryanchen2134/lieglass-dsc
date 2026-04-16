"""
Cross-modal fusion model combining Wav2Vec2 audio and CNN+ViT visual streams.

Architecture (adapted from DOLOS repo — NMS05/Audio-Visual-Deception-Detection-...):
  audio_emb (B, 768) + visual_emb (B, 768)
    → CrossFusionModule: project → correlation → residual concat → (B, 512)
    → classifier head → logit (B,)

Usage:
    model = FusionModel(config)
    logit = model(batch)   # batch["waveform"] (B,T), batch["frames"] (B,64,3,224,224)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio_model import W2V2_Model
from .visual_model import ViT_Model


class CrossFusionModule(nn.Module):
    """
    Gated audio-visual fusion.

    Projects both embeddings to d_cross, then uses a learned gate to blend
    them before projecting to d_out.  No batch-level operations — stable
    with small batch sizes.
    """

    def __init__(self, d_audio: int = 768, d_visual: int = 768, d_cross: int = 256, d_out: int = 512):
        super().__init__()
        self.audio_proj  = nn.Linear(d_audio,  d_cross)
        self.visual_proj = nn.Linear(d_visual, d_cross)
        # Gate: takes both projections → scalar per feature
        self.gate = nn.Linear(2 * d_cross, d_cross)
        self.out  = nn.Linear(2 * d_cross, d_out)

    def forward(self, audio: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        a = self.audio_proj(audio)    # (B, d_cross)
        v = self.visual_proj(visual)  # (B, d_cross)

        # Soft gate: how much audio vs visual to use per feature
        g = torch.sigmoid(self.gate(torch.cat([a, v], dim=-1)))  # (B, d_cross)
        a_gated = g * a
        v_gated = (1 - g) * v

        fused = torch.cat([a_gated, v_gated], dim=-1)  # (B, 2*d_cross)
        return self.out(fused)                          # (B, d_out)


class FusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.audio_model  = W2V2_Model(config)
        self.visual_model = ViT_Model(config)

        self.fusion = CrossFusionModule(
            d_audio=config.d_audio,
            d_visual=config.d_visual,
            d_cross=config.d_cross,
            d_out=config.d_fused,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(config.d_fused),
            nn.Linear(config.d_fused, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 1),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict with keys:
                waveform        FloatTensor (B, T)
                waveform_mask   BoolTensor  (B, T)   True=valid  (optional)
                frames          FloatTensor (B, n_frames, 3, 224, 224)
                frame_mask      BoolTensor  (B, n_frames)  True=speaker visible (optional)

        Returns:
            logits FloatTensor (B,) — raw (pre-sigmoid) logits
        """
        waveform = batch["waveform"]
        frames   = batch["frames"]
        waveform_mask = batch.get("waveform_mask")
        frame_mask    = batch.get("frame_mask")      # (B, n_frames) bool or None

        audio_emb  = self.audio_model(waveform, waveform_mask)   # (B, 768)
        visual_emb = self.visual_model(frames, frame_mask)        # (B, 768)

        fused  = self.fusion(audio_emb, visual_emb)              # (B, 512)
        logits = self.head(fused).squeeze(-1)                    # (B,)
        return logits
