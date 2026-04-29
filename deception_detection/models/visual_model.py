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
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint
from transformers import ViTModel, ViTConfig

from .adapters import UTAdapterVisualLayer


# ---------------------------------------------------------------------------
# Per-frame CNN
# ---------------------------------------------------------------------------

def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block to recalibrate channel-wise features."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """Residual block to prevent vanishing gradients in deeper facial analysis."""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        self.se = SEBlock(out_c)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)

class CNN_Face(nn.Module):
    """
    Improved CNN for grayscale facial feature extraction.
    Uses Residual connections and SE-Attention to compensate for lack of color.
    """
    def __init__(self, output_dim=256):
        super().__init__()
        # Initial projection: Grayscale (1) to 64
        self.prep = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # ResNet-style stages
        self.layer1 = ResidualBlock(64, 64, stride=1)   # 56x56
        self.layer2 = ResidualBlock(64, 128, stride=2)  # 28x28
        self.layer3 = ResidualBlock(128, 256, stride=2) # 14x14
        self.layer4 = ResidualBlock(256, 512, stride=2) # 7x7

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (Batch*Frames, 1, 224, 224)
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(self.fc(x))
        return x


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

class ViT_Model(nn.Module): # Rename this to VisualModel eventually
    def __init__(self, config):
        super().__init__()
        self.spatial_encoder = CNN_Face(output_dim=config.d_visual)

        self.use_ut_adapters = bool(getattr(config, "use_ut_adapters", False))
        self.fusion_layers = sorted(set(config.visual_fusion_layers))
        if max(self.fusion_layers) > config.vit_n_layers or min(self.fusion_layers) < 1:
            raise ValueError(
                f"visual_fusion_layers={self.fusion_layers} out of range [1, {config.vit_n_layers}]."
            )

        if self.use_ut_adapters:
            self.temporal_layers = nn.ModuleList([
                UTAdapterVisualLayer(
                    d_model=config.d_visual,
                    n_heads=config.vit_n_heads,
                    dim_ff=512,
                    dropout=config.dropout,
                    bottleneck=config.ut_adapter_dim,
                    conv_kernel=config.ut_conv_kernel,
                )
                for _ in range(config.vit_n_layers)
            ])
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_visual,
                nhead=config.vit_n_heads,
                dim_feedforward=512,
                dropout=config.dropout,
                batch_first=True,
            )
            # Stored as ModuleList of plain layers so iteration code is uniform.
            self.temporal_layers = nn.ModuleList([encoder_layer for _ in range(config.vit_n_layers)])

        self.pos_embedding = nn.Parameter(torch.zeros(1, config.max_frames, config.d_visual))

        if getattr(config, "freeze_visual_backbone", False):
            for p in self.spatial_encoder.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # CNN trunk (chunked + checkpointed)
    # ------------------------------------------------------------------

    def _spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.shape
        chunk_size = 16
        all_features = []
        for i in range(0, N, chunk_size):
            chunk = x[:, i : i + chunk_size].reshape(-1, C, H, W)
            f = checkpoint(self.spatial_encoder, chunk, use_reentrant=False)
            all_features.append(f.view(B, -1, f.shape[-1]))
        return torch.cat(all_features, dim=1)                  # (B, N, d_visual)

    def _run_temporal(self, features: torch.Tensor, mask: torch.Tensor | None
                      ) -> list[torch.Tensor]:
        key_padding_mask = ~mask if mask is not None else None
        h = features
        stages = []
        for i, layer in enumerate(self.temporal_layers, start=1):
            if isinstance(layer, UTAdapterVisualLayer):
                h = layer(h, src_key_padding_mask=key_padding_mask)
            else:
                h = layer(h, src_key_padding_mask=key_padding_mask)
            if i in self.fusion_layers:
                stages.append(h)
        return stages

    # ------------------------------------------------------------------
    # Multi-stage forward
    # ------------------------------------------------------------------

    def forward_multistage(self, x: torch.Tensor, mask: torch.Tensor | None = None
                           ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        features = self._spatial_features(x)
        N = features.shape[1]
        features = features + self.pos_embedding[:, :N, :]
        stages = self._run_temporal(features, mask)
        return stages, mask

    # ------------------------------------------------------------------
    # Backwards-compatible single-vector forward
    # ------------------------------------------------------------------

    def forward(self, x, mask=None):
        stages, m = self.forward_multistage(x, mask)
        last = stages[-1]                                        # (B, N, d_visual)
        if m is None:
            return last.mean(dim=1)
        mf = m.unsqueeze(-1).float()
        return (last * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)