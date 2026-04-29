"""
Pretrained-backbone visual encoder + adapter-tuned temporal Transformer.

Pipeline:
  grayscale frames (B, N, 1, 224, 224)
    -> spatial backbone (frozen, RGB-pretrained):
         "clip"    : CLIPVisionModel  (768-d)
         "arcface" : facenet-pytorch InceptionResnetV1 vggface2  (512-d)
    -> Linear projection to d_visual
    -> sinusoidal position addition
    -> temporal Transformer (UTAdapterVisualLayer ×N)
    -> intermediate hidden states at config.visual_fusion_layers

The 1-channel grayscale input is replicated to 3 channels inside the encoder so
RGB-pretrained backbones consume it directly — the input pipeline is
deliberately grayscale to match real-life AR-glasses operating conditions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint

from .adapters import UTAdapterVisualLayer


# ---------------------------------------------------------------------------
# Pretrained spatial backbones
# ---------------------------------------------------------------------------


class CLIPVisionEncoder(nn.Module):
    """Frozen Hugging Face CLIP vision tower returning per-frame pooled embeddings."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        from transformers import CLIPVisionModel
        self.model = CLIPVisionModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.output_dim = self.model.config.hidden_size           # 768 for ViT-B/32
        # CLIP image normalization (matches openai/clip processor)
        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        # Frozen pretrained backbone always stays in eval mode (LayerNorms,
        # dropout, etc.) regardless of the parent module's training flag.
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*N, 1 or 3, H, W) float in [0, 1]
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        x = (x - self.mean) / self.std
        return self.model(pixel_values=x).pooler_output           # (B*N, output_dim)


class ArcFaceVisionEncoder(nn.Module):
    """Frozen face-pretrained ResNet trunk (InceptionResnetV1 / VGGFace2)."""

    def __init__(self):
        super().__init__()
        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError as e:
            raise RuntimeError(
                "visual_backbone='arcface' requires `facenet-pytorch`. "
                "Install with: pip install facenet-pytorch"
            ) from e
        self.model = InceptionResnetV1(pretrained="vggface2")
        for p in self.model.parameters():
            p.requires_grad = False
        self.output_dim = 512
        # Face-domain normalization to [-1, 1]
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*N, 1 or 3, H, W) float in [0, 1]
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        # vggface2 weights expect 160×160; resize once in the encoder.
        if x.shape[-1] != 160 or x.shape[-2] != 160:
            x = F.interpolate(x, size=160, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.model(x)                                      # (B*N, 512)


def _build_spatial_encoder(config) -> nn.Module:
    if config.visual_backbone == "clip":
        return CLIPVisionEncoder(config.visual_backbone_model)
    if config.visual_backbone == "arcface":
        return ArcFaceVisionEncoder()
    raise ValueError(f"Unknown visual_backbone: {config.visual_backbone!r}")


# ---------------------------------------------------------------------------
# Temporal positional encoding
# ---------------------------------------------------------------------------


def _sinusoidal_pe(seq_len: int, d_model: int, device, dtype) -> torch.Tensor:
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
    """Pretrained per-frame backbone + temporal Transformer with UT-Adapters."""

    def __init__(self, config):
        super().__init__()
        self.spatial_encoder = _build_spatial_encoder(config)
        self.spatial_proj = nn.Linear(self.spatial_encoder.output_dim, config.d_visual)

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
            self.temporal_layers = nn.ModuleList([encoder_layer for _ in range(config.vit_n_layers)])

        self.pos_embedding = nn.Parameter(torch.zeros(1, config.max_frames, config.d_visual))
        self.cnn_chunk_size = int(config.cnn_chunk_size)

    # ------------------------------------------------------------------
    # Per-frame backbone (chunked so peak activation stays bounded)
    # ------------------------------------------------------------------

    def _spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.shape
        chunk_size = self.cnn_chunk_size
        outs = []
        for i in range(0, N, chunk_size):
            chunk = x[:, i : i + chunk_size].reshape(-1, C, H, W)
            f = self.spatial_encoder(chunk)                       # (chunk*B, output_dim)
            outs.append(f.view(B, -1, f.shape[-1]))
        feats = torch.cat(outs, dim=1)                            # (B, N, output_dim)
        return self.spatial_proj(feats)                           # (B, N, d_visual)

    # ------------------------------------------------------------------
    # Temporal stack with multi-stage capture
    # ------------------------------------------------------------------

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

    def forward_multistage(self, x: torch.Tensor, mask: torch.Tensor | None = None
                           ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        features = self._spatial_features(x)
        N = features.shape[1]
        features = features + self.pos_embedding[:, :N, :]
        stages = self._run_temporal(features, mask)
        return stages, mask

    def forward(self, x, mask=None):
        stages, m = self.forward_multistage(x, mask)
        last = stages[-1]
        if m is None:
            return last.mean(dim=1)
        mf = m.unsqueeze(-1).float()
        return (last * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
