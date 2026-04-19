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
            _conv_block(3, 64),
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
    """
    Name kept as ``ViT_Model`` for import compatibility; internally this is
    now a temporal Transformer encoder, not HuggingFace ViT.
    """

    def __init__(self, config):
        super().__init__()
        self.cnn_chunk_size = config.cnn_chunk_size
        self.d_model = config.d_visual

        self.cnn  = CNN_Face()
        self.proj = nn.Linear(256, self.d_model)

        # Head count — 8 divides 768 evenly and keeps per-head dim at 96.
        n_heads = getattr(config, "vit_n_heads", 8)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=4 * self.d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,                # pre-LN = more stable training
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.vit_n_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(self.d_model)

    # --------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------
    def forward(
        self,
        frames: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            frames:     FloatTensor (B, N, 3, 224, 224) — pre-normalised.
            frame_mask: BoolTensor  (B, N) — True for real frames, False for
                        padding. Padded positions are excluded from attention
                        and from the final mean pool.

        Returns:
            FloatTensor (B, d_visual).
        """
        B, N, C, H, W = frames.shape

        # ---- Per-frame CNN — chunked + gradient-checkpointed during training.
        # Rationale: 300+ frames per clip would otherwise retain huge
        # stage-1 activations for backward. Checkpointing trades ~30% extra
        # compute for a >10× cut in peak CNN VRAM.
        flat = frames.view(B * N, C, H, W)
        chunks = flat.split(self.cnn_chunk_size)
        if self.training:
            x = torch.cat(
                [grad_checkpoint(self.cnn, ch, use_reentrant=False) for ch in chunks]
            )
        else:
            x = torch.cat([self.cnn(ch) for ch in chunks])           # (B*N, 256)
        x = x.view(B, N, 256)

        # ---- Projection + sinusoidal positional encoding.
        x = self.proj(x)                                             # (B, N, d)
        x = x + _sinusoidal_pe(N, self.d_model, x.device, x.dtype)

        # ---- Temporal Transformer with padding mask.
        # src_key_padding_mask: True at padded positions (PyTorch convention).
        key_padding_mask = None
        if frame_mask is not None:
            key_padding_mask = ~frame_mask                           # (B, N)

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)                                             # (B, N, d)

        # ---- Masked mean pool.
        if frame_mask is not None:
            valid = frame_mask.to(x.dtype).unsqueeze(-1)             # (B, N, 1)
            n_valid = valid.sum(dim=1).clamp(min=1.0)
            return (x * valid).sum(dim=1) / n_valid
        return x.mean(dim=1)
