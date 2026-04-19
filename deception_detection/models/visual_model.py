"""
CNN + temporal-transformer visual encoder (variable-length, full-frame pipeline).

Pipeline:
  face frames (B, N, 3, 224, 224)    — N varies per batch
    → CNN_Face per frame                          → (B*N, 256)
    → Linear(256 → d_visual)
    → sinusoidal positional encoding              (arbitrary length)
    → temporal Transformer encoder                ``vit_n_layers`` layers
         (self-attention padding mask from ``frame_mask``)
    → LayerNorm
    → masked mean pool over valid frames          → (B, d_visual)

Design notes
------------

The original DOLOS model stacked a fixed-N (=64) token sequence through the
last 4 layers of ViT-B/16 with a *learnable* positional embedding. That is
incompatible with variable-length inputs (the new full-frame pipeline keeps
every frame, so N varies per clip), and ViT-B/16's patch-level pretraining
transfers poorly to temporal token ordering anyway.

We instead use a native ``nn.TransformerEncoder`` on top of the same CNN_Face
trunk:

  * **Variable length** — sinusoidal positional encoding extends to any N;
    attention masks exclude padded frames.
  * **Faster** — PyTorch's FlashAttention kernels are used automatically;
    no HuggingFace wrapper and no frozen pretrained parameters sitting in
    VRAM unused.
  * **Same encoder *type*** — a multi-head self-attention Transformer over
    projected CNN tokens, matching the DOLOS design at the layer level.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from transformers import ViTModel, ViTConfig


# ---------------------------------------------------------------------------
# Per-frame CNN
# ---------------------------------------------------------------------------

def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class CNN_Face(nn.Module):
    """3-stage CNN (64/128/256) → 256-D global feature per frame."""

    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            _conv_block(1, 64),
            _conv_block(64, 64),
            nn.MaxPool2d(2),        # 224 → 112
        )
        self.stage2 = nn.Sequential(
            _conv_block(64, 128),
            _conv_block(128, 128),
            nn.MaxPool2d(2),        # 112 → 56
        )
        self.stage3 = nn.Sequential(
            _conv_block(128, 256),
            _conv_block(256, 256),
            nn.AdaptiveAvgPool2d(1),  # → (256, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x.view(x.size(0), -1)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

def _sinusoidal_pe(seq_len: int, d_model: int, device, dtype) -> torch.Tensor:
    """(1, seq_len, d_model) sinusoidal PE — supports arbitrary length."""
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * -(math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0).to(dtype)


# ---------------------------------------------------------------------------
# Visual model
# ---------------------------------------------------------------------------

class ViT_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 1. Load Config and Force 1-Channel
        v_config = ViTConfig.from_pretrained(config.vit_model)
        v_config.num_channels = 1
        
        # 2. Load Pretrained ViT
        self.vit = ViTModel.from_pretrained(
            config.vit_model, 
            config=v_config, 
            ignore_mismatched_sizes=True
        )

        # 3. Temporal Processor (The Upgrade)
        # We use 2 layers of Bi-directional GRU to capture complex patterns
        self.temporal_processor = nn.GRU(
            input_size=768,      # Standard ViT-base hidden size
            hidden_size=384,     # 384 * 2 (bidirectional) = 768 total output
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

    def forward(self, x, mask=None):
        # x shape: (Batch, Num_Frames, 1, 224, 224)
        batch_size, num_frames, C, H, W = x.shape
        
        # Flatten frames to process through ViT: (B*N, 1, 224, 224)
        x_flat = x.view(-1, C, H, W) 
        
        # 1. Spatial Feature Extraction
        outputs = self.vit(x_flat)
        # Reshape back to (Batch, Num_Frames, 768)
        frame_features = outputs.pooler_output.view(batch_size, num_frames, -1)

        # 2. Temporal Feature Extraction
        # gru_out shape: (Batch, Num_Frames, 768)
        gru_out, _ = self.temporal_processor(frame_features)

        # 3. Context-Aware Pooling
        # We average the GRU outputs. Because the GRU is bidirectional, 
        # every frame's vector now contains info from the frames before AND after it.
        if mask is not None:
            valid = mask.to(gru_out.dtype).unsqueeze(-1) # (B, N, 1)
            visual_summary = (gru_out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            visual_summary = gru_out.mean(dim=1)

        return visual_summary # Final shape: (Batch, 768)