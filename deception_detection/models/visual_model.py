"""
CNN + ViT-B/16 visual encoder for deception detection.

Architecture (adapted from DOLOS repo — NMS05/Audio-Visual-Deception-Detection-...):
  face frames (B, n_frames, 3, 224, 224)
    → CNN_Face: lightweight 3-stage CNN per frame → (B*n_frames, 256)
    → Linear(256→768) + learnable positional embeddings
    → ViT-B/16 encoder layers (last vit_n_layers of the pretrained model)
                               [last vit_unfreeze_last_n layers trainable]
    → mean pool across frames
    → visual embedding (B, 768)
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from transformers import ViTModel


def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class CNN_Face(nn.Module):
    """
    Lightweight CNN that extracts a 256-D feature vector from a single 224×224 face frame.
    Three conv stages with max-pooling, ending in global average pooling.
    """

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
        """x: (N, 3, 224, 224)  →  (N, 256)"""
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x.view(x.size(0), -1)  # (N, 256)


class ViT_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_frames       = config.n_frames
        self.cnn_chunk_size = config.cnn_chunk_size  # cap peak CNN activation

        # Per-frame CNN (always trainable — randomly initialised)
        self.cnn = CNN_Face()

        # Project CNN output to ViT hidden dim
        self.proj = nn.Linear(256, 768)

        # Learnable positional embeddings for n_frames tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, config.n_frames, 768))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Load the last vit_n_layers encoder layers from ViT-B/16
        vit = ViTModel.from_pretrained(config.vit_model)
        all_layers = vit.encoder.layer                     # 12 layers total
        n_total = len(all_layers)
        start = max(0, n_total - config.vit_n_layers)
        self.vit_layers = nn.ModuleList(all_layers[start:])

        # Freeze all but the last vit_unfreeze_last_n of those layers
        n_freeze = max(0, len(self.vit_layers) - config.vit_unfreeze_last_n)
        for i, layer in enumerate(self.vit_layers):
            if i < n_freeze:
                for p in layer.parameters():
                    p.requires_grad = False

        self.norm = nn.LayerNorm(768)

        # Explicitly delete the full ViT to free memory
        del vit

    def forward(
        self,
        frames: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            frames:     FloatTensor (B, n_frames, 3, 224, 224)
            frame_mask: BoolTensor  (B, n_frames) — True = speaker visible (not black).
                        When provided, only valid frames contribute to the output
                        embedding (masked mean pooling).  If all frames in a sample
                        are black the fallback is a uniform mean over all tokens.

        Returns:
            FloatTensor (B, 768) — (masked) mean-pooled visual embedding
        """
        B, T, C, H, W = frames.shape

        # CNN feature extraction with gradient checkpointing.
        #
        # Problem: naively running self.cnn on all B*T frames at once fills the GPU
        # with intermediate activations for the entire backward pass (~1.6 GB for
        # B=8, T=64 at stage-1 alone). Chunking alone doesn't help backward: PyTorch
        # still retains all chunk activations for torch.cat's gradient.
        #
        # Fix: gradient checkpointing — forward activations are discarded after each
        # chunk's forward pass and recomputed on-the-fly during backward. This trades
        # a small amount of extra compute (~30%) for a large reduction in peak VRAM.
        # During eval (no grad needed) we skip checkpointing for speed.
        flat = frames.view(B * T, C, H, W)                          # (B*T, 3, 224, 224)
        chunks = flat.split(self.cnn_chunk_size)
        if self.training:
            x = torch.cat(
                [grad_checkpoint(self.cnn, chunk, use_reentrant=False)
                 for chunk in chunks]
            )
        else:
            x = torch.cat([self.cnn(chunk) for chunk in chunks])    # (B*T, 256)
        x = x.view(B, T, 256)                                       # (B, T, 256)

        # Project + positional embedding
        x = self.proj(x) + self.pos_embed  # (B, T, 768)

        # ViT encoder layers
        for layer in self.vit_layers:
            x = layer(x)[0]               # ViTLayer returns (hidden_states, ...)

        x = self.norm(x)                   # (B, T, 768)

        # --- Masked mean pooling ---
        # Black frames (frame_mask == False) are excluded from the average so they
        # don't dilute the visual embedding with zero-input CNN activations.
        if frame_mask is not None:
            valid = frame_mask.to(x.dtype).unsqueeze(-1)   # (B, T, 1)
            n_valid = valid.sum(dim=1)                     # (B, 1)
            # Fall back to uniform mean for samples where every frame is black.
            all_black = (n_valid == 0)                     # (B, 1)
            n_valid = n_valid.clamp(min=1.0)
            pooled = (x * valid).sum(dim=1) / n_valid      # (B, 768)
            if all_black.any():
                uniform = x.mean(dim=1)                    # (B, 768)
                pooled = torch.where(all_black, uniform, pooled)
            return pooled

        return x.mean(dim=1)               # (B, 768)
