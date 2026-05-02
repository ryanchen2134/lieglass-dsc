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
    def __init__(self, audio_dim=768, visual_dim=768, fusion_dim=512):
        super().__init__()
        # Cross-Attention: Audio queries Visual
        self.cross_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=8, batch_first=True)
        
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        
        # Residual fusion layer
        self.fc_fuse = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

    def forward(self, a, v):
        # a, v: (B, 768) -> (B, 512)
        a_proj = self.audio_proj(a).unsqueeze(1) # (B, 1, 512)
        v_proj = self.visual_proj(v).unsqueeze(1) # (B, 1, 512)
        
        # Audio "looks at" Video
        # Query=Audio, Key=Video, Value=Video
        attn_out, _ = self.cross_attn(query=a_proj, key=v_proj, value=v_proj)
        
        # Combine the original audio with the visually-informed features
        combined = torch.cat([a_proj.squeeze(1), attn_out.squeeze(1)], dim=-1)
        return self.fc_fuse(combined)


class FusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.audio_model  = W2V2_Model(config)
        self.visual_model = ViT_Model(config)

        self.fusion = CrossFusionModule(
            audio_dim=config.d_audio,
            visual_dim=config.d_visual,
            fusion_dim=config.d_fused
        )

        # Deeper head for final reasoning
        self.head = nn.Sequential(
            nn.Linear(config.d_fused, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Buffers for Grayscale (ImageNet-style mean/std for 1-channel)
        self.register_buffer("_img_mean", torch.tensor([0.5]).view(1, 1, 1, 1, 1)) 
        self.register_buffer("_img_std",  torch.tensor([0.25]).view(1, 1, 1, 1, 1))

    def forward(self, batch: dict) -> torch.Tensor:
        waveform = batch["waveform"]
        frames   = batch["frames"] # (B, N, C, H, W)
        waveform_mask = batch.get("waveform_mask")
        frame_mask    = batch.get("frame_mask")

        # 1. Grayscale Slicing (Ensure 1-channel)
        if frames.shape[2] == 3:
            # Optimal slicing: [Batch, Frames, 1, Height, Width]
            frames = frames[:, :, 0:1, :, :] 

        # 2. Preprocessing
        if frames.dtype == torch.uint8:
            frames = frames.float().div(255.0)
            
        frames = (frames - self._img_mean) / self._img_std

        # 3. Backbone Passes
        audio_emb  = self.audio_model(waveform, waveform_mask)
        visual_emb = self.visual_model(frames, frame_mask)

        # 4. Fusion & Logits
        fused  = self.fusion(audio_emb, visual_emb)
        logits = self.head(fused).squeeze(-1)
        return logits