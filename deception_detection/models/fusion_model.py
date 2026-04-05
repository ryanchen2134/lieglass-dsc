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
    Plug-in Audio-Visual Fusion (PAVF) from DOLOS.

    Projects both embeddings to d_cross, computes batch-level cross-modal
    attention (correlation matrix), applies residual fusion, and projects
    to d_out.
    """

    def __init__(self, d_audio: int = 768, d_visual: int = 768, d_cross: int = 256, d_out: int = 512):
        super().__init__()
        self.audio_proj  = nn.Linear(d_audio,  d_cross)
        self.visual_proj = nn.Linear(d_visual, d_cross)
        self.scale = d_cross ** -0.5
        self.out = nn.Linear(2 * d_cross, d_out)

    def forward(self, audio: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio:  FloatTensor (B, d_audio)
            visual: FloatTensor (B, d_visual)

        Returns:
            FloatTensor (B, d_out)
        """
        a = self.audio_proj(audio)    # (B, d_cross)
        v = self.visual_proj(visual)  # (B, d_cross)

        # Batch-level cross-modal correlation (B, B)
        corr = F.softmax(a @ v.t() * self.scale, dim=-1)   # (B, B)
        corr_t = F.softmax(v @ a.t() * self.scale, dim=-1) # (B, B)

        # Cross-attended features with residual
        a_fused = a + corr   @ v  # (B, d_cross)
        v_fused = v + corr_t @ a  # (B, d_cross)

        fused = torch.cat([a_fused, v_fused], dim=-1)  # (B, 2*d_cross)
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
                waveform_mask   BoolTensor  (B, T)  True=valid  (optional)
                frames          FloatTensor (B, n_frames, 3, 224, 224)

        Returns:
            logits FloatTensor (B,) — raw (pre-sigmoid) logits
        """
        waveform = batch["waveform"]
        frames   = batch["frames"]
        waveform_mask = batch.get("waveform_mask")

        audio_emb  = self.audio_model(waveform, waveform_mask)  # (B, 768)
        visual_emb = self.visual_model(frames)                   # (B, 768)

        fused  = self.fusion(audio_emb, visual_emb)              # (B, 512)
        logits = self.head(fused).squeeze(-1)                    # (B,)
        return logits
