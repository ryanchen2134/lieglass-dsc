"""
Bidirectional cross-modal fusion (DOLOS PAVF) and multi-stage aggregator.

Each ``BidirectionalCrossFusion`` block implements both A-V and V-A attention
on sequence-level audio/visual features, then masked-mean-pools each direction
and concatenates them through a fusion head. ``MultiStageFusion`` runs one such
block per (audio_stage, visual_stage) pair (F_1, F_mid, F_end) and aggregates
the resulting (B, d_fused) vectors via a softmaxed weighted sum.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(t: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return t.mean(dim=1)
    m = mask.unsqueeze(-1).to(t.dtype)
    return (t * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class BidirectionalCrossFusion(nn.Module):
    def __init__(self, d_audio: int, d_visual: int, d_fused: int,
                 n_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.audio_proj = nn.Linear(d_audio, d_fused)
        self.visual_proj = nn.Linear(d_visual, d_fused)
        self.av_attn = nn.MultiheadAttention(d_fused, n_heads, dropout=dropout, batch_first=True)
        self.va_attn = nn.MultiheadAttention(d_fused, n_heads, dropout=dropout, batch_first=True)
        self.fuse_head = nn.Sequential(
            nn.Linear(2 * d_fused, d_fused),
            nn.LayerNorm(d_fused),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, a: torch.Tensor, v: torch.Tensor,
                a_mask: torch.Tensor | None = None,
                v_mask: torch.Tensor | None = None) -> torch.Tensor:
        a_p = self.audio_proj(a)                             # (B, T_a, d_fused)
        v_p = self.visual_proj(v)                            # (B, T_v, d_fused)

        # MultiheadAttention key_padding_mask: True = masked-out (padding).
        v_kpm = (~v_mask) if v_mask is not None else None
        a_kpm = (~a_mask) if a_mask is not None else None

        av, _ = self.av_attn(a_p, v_p, v_p, key_padding_mask=v_kpm, need_weights=False)  # (B, T_a, d_fused)
        va, _ = self.va_attn(v_p, a_p, a_p, key_padding_mask=a_kpm, need_weights=False)  # (B, T_v, d_fused)

        av_pool = _masked_mean(av, a_mask)                   # (B, d_fused)
        va_pool = _masked_mean(va, v_mask)                   # (B, d_fused)

        return self.fuse_head(torch.cat([av_pool, va_pool], dim=-1))   # (B, d_fused)


class MultiStageFusion(nn.Module):
    def __init__(self, n_stages: int, d_audio: int, d_visual: int, d_fused: int,
                 n_heads: int = 8, dropout: float = 0.2, aggregator: str = "weighted_sum"):
        super().__init__()
        self.n_stages = n_stages
        self.aggregator = aggregator
        self.blocks = nn.ModuleList([
            BidirectionalCrossFusion(d_audio, d_visual, d_fused, n_heads=n_heads, dropout=dropout)
            for _ in range(n_stages)
        ])
        if aggregator == "weighted_sum":
            self.stage_weights = nn.Parameter(torch.zeros(n_stages))   # softmax -> uniform init

    def forward(self, audio_stages: list[torch.Tensor], visual_stages: list[torch.Tensor],
                a_mask: torch.Tensor | None = None,
                v_mask: torch.Tensor | None = None) -> torch.Tensor:
        if len(audio_stages) != self.n_stages or len(visual_stages) != self.n_stages:
            raise ValueError(
                f"Expected {self.n_stages} stages each; got "
                f"{len(audio_stages)} audio / {len(visual_stages)} visual."
            )
        outs = [
            block(a, v, a_mask=a_mask, v_mask=v_mask)
            for block, a, v in zip(self.blocks, audio_stages, visual_stages)
        ]
        stacked = torch.stack(outs, dim=0)                   # (S, B, d_fused)

        if self.aggregator == "sum":
            return stacked.sum(dim=0)
        if self.aggregator == "weighted_sum":
            w = F.softmax(self.stage_weights, dim=0).view(-1, 1, 1)
            return (stacked * w).sum(dim=0)
        raise ValueError(f"Unknown aggregator: {self.aggregator!r}")
