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
        # Project both to the same size for the gate
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        
        # The Gate: Takes both modalities and outputs a score between 0 and 1
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, a, v):
        # a: (B, 768), v: (B, 768)
        a_proj = torch.tanh(self.audio_proj(a))
        v_proj = torch.tanh(self.visual_proj(v))
        
        # Calculate gate weight
        gate_input = torch.cat([a_proj, v_proj], dim=-1)
        g = self.gate(gate_input) # Weight for audio vs visual
        
        # Gated fusion: g decides how much audio to keep vs visual
        # If g is 0.8, it's 80% audio and 20% visual
        h = g * a_proj + (1 - g) * v_proj
        
        return self.output_layer(h)


class FusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.audio_model  = W2V2_Model(config)
        self.visual_model = ViT_Model(config)

        self.fusion = CrossFusionModule(
            audio_dim=config.d_audio,   # Changed from d_audio
            visual_dim=config.d_visual, # Changed from d_visual
            fusion_dim=config.d_fused   # Changed from d_out or whatever was there
        )

        self.head = nn.Sequential(
            nn.LayerNorm(config.d_fused),
            nn.Linear(config.d_fused, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 1),
        )

        # ImageNet normalisation constants, registered as buffers so they
        # move to the correct device with the model (incl. DataParallel replicas).
        self.register_buffer("_img_mean", torch.tensor([0.45]).view(1, 1, 1, 1, 1)) 
        self.register_buffer("_img_std",  torch.tensor([0.22]).view(1, 1, 1, 1, 1))

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict with keys:
                waveform        FloatTensor (B, T)
                waveform_mask   BoolTensor  (B, T)   True=valid  (optional)
                frames          ByteTensor  (B, N, 3, 224, 224)  uint8 RGB (N varies per batch)
                frame_mask      BoolTensor  (B, N)   True=real frame (False=padding, optional)

        Returns:
            logits FloatTensor (B,) — raw (pre-sigmoid) logits
        """
        waveform = batch["waveform"]
        frames   = batch["frames"]
        waveform_mask = batch.get("waveform_mask")
        frame_mask    = batch.get("frame_mask")

        # 1. FORCE 1-CHANNEL IMMEDIATELY
        if frames.shape[2] == 3:
            # Slicing is better than repeating/averaging for speed
            frames = frames[:, :, :1, :, :] 

        # 2. CONVERT TO FLOAT
        if frames.dtype == torch.uint8:
            frames = frames.float().div(255.0)
            
        # 3. NORMALIZE (using the 1-channel buffers you updated)
        frames = (frames - self._img_mean) / self._img_std

        # 4. REMOVE THE REPEAT BLOCK
        # Delete the lines: 
        # if frames.shape[2] == 1: 
        #     frames = frames.repeat(...)

        audio_emb  = self.audio_model(waveform, waveform_mask)
        visual_emb = self.visual_model(frames, frame_mask)

        fused  = self.fusion(audio_emb, visual_emb)
        logits = self.head(fused).squeeze(-1)
        return logits