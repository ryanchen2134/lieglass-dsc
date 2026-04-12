"""
CrossFusionModule — PECL-style bilinear cross-modal fusion.

Applied once per encoder layer pair (audio_i, visual_i) to produce a
64-dimensional fused representation at each of the 64 token positions.

Architecture:
  1. Project audio and visual from d_in (768) to d_proj (256).
  2. Compute bilinear correlation matrix:
         corr = audio_proj @ W @ visual_proj^T   →  (B, 64, 64)
     where W ∈ R^{d_proj × d_proj} is a learnable parameter.
  3. Bidirectional softmax attention:
         a2v = softmax(corr, dim=-1)   — each audio token attends to visual tokens
         v2a = softmax(corr, dim=-2)   — each visual token attends to audio tokens
  4. Weighted sums with residual connections:
         audio_fused  = audio_proj  + bmm(a2v, visual_proj)
         visual_fused = visual_proj + bmm(v2a^T, audio_proj)
  5. Bottleneck projection:
         out = Linear(d_proj, d_out)(audio_fused + visual_fused)  →  (B, 64, 64)

Input:  audio  (B, 64, 768),  visual (B, 64, 768)
Output: fused  (B, 64, 64)

After N layers, the caller concatenates fusion outputs along the last dim:
    (B, 64, 64*N) → mean-pool → (B, 64*N)  →  classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossFusionModule(nn.Module):
    """
    Bilinear cross-modal fusion between audio and visual hidden states.

    Args:
        d_in:   Input dimension from audio / visual encoders (768).
        d_proj: Intermediate projection dimension for bilinear op (256).
        d_out:  Output dimension per token position (64).
    """

    def __init__(self, d_in: int = 768, d_proj: int = 256, d_out: int = 64):
        super().__init__()
        self.proj_a     = nn.Linear(d_in, d_proj)
        self.proj_v     = nn.Linear(d_in, d_proj)
        # Learnable bilinear weight matrix
        self.W          = nn.Parameter(torch.randn(d_proj, d_proj) * 0.01)
        self.bottleneck = nn.Linear(d_proj, d_out)

    def forward(self, audio: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio:  (B, T, d_in)  — audio hidden states at layer i.
            visual: (B, T, d_in)  — visual hidden states at layer i.
                    T is fixed at 64 (both modalities share the same sequence length).

        Returns:
            fused: (B, T, d_out)  — fused representation.
        """
        a = self.proj_a(audio)    # (B, T, d_proj)
        v = self.proj_v(visual)   # (B, T, d_proj)

        # Bilinear correlation: (B, T, d_proj) @ (d_proj, d_proj) → (B, T, d_proj)
        # then (B, T, d_proj) @ (B, d_proj, T) → (B, T, T)
        corr = torch.bmm(a @ self.W, v.transpose(1, 2))   # (B, T, T)

        a2v = F.softmax(corr, dim=-1)          # audio attends to visual  (B, T, T)
        v2a = F.softmax(corr, dim=-2)          # visual attends to audio  (B, T, T)

        # Weighted sums + residual connections
        audio_fused  = a + torch.bmm(a2v, v)                    # (B, T, d_proj)
        visual_fused = v + torch.bmm(v2a.transpose(1, 2), a)    # (B, T, d_proj)

        # Merge and project to output dimension
        merged = audio_fused + visual_fused                      # (B, T, d_proj)
        return self.bottleneck(merged)                           # (B, T, d_out)
