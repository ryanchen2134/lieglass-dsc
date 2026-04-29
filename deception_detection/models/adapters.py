"""
UT-Adapter primitives and per-modality wrapped encoder layers (DOLOS Section 4.1).

UT-Adapter:    U(X) = L_2(P(C(P(L_1(X; W_1)); W_C)); W_2)
  L_1: Linear(D -> 128)
  P:   Permutation between (B, T, 128) and (B, 128, T)
  C:   Conv1d(128, 128, kernel=3) along temporal dim
  L_2: Linear(128 -> D)
  No activation (per spec).

Audio layer (post-LN, eq. 6):
  X'' = AN( X  + LN(H(X))   + U(X)  )
  X'  = AN( X'' + LN(MLP(X'')) + U(X'') )

Visual layer (pre-LN, eqs. 4-5):
  X'' = AN( X  + H(LN(X))   + U(X)  )
  X'  = AN( X'' + MLP(LN(X'')) + U(X'') )
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UTAdapter(nn.Module):
    def __init__(self, d_model: int, bottleneck: int = 128, conv_kernel: int = 3):
        super().__init__()
        self.l1 = nn.Linear(d_model, bottleneck)
        self.conv = nn.Conv1d(bottleneck, bottleneck, kernel_size=conv_kernel, padding=conv_kernel // 2)
        self.l2 = nn.Linear(bottleneck, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        h = self.l1(x)                         # (B, T, 128)
        h = self.conv(h.transpose(1, 2))       # (B, 128, T)
        h = h.transpose(1, 2)                  # (B, T, 128)
        return self.l2(h)                      # (B, T, D)


class AdapterNorm(nn.LayerNorm):
    """Trainable LayerNorm; subclassed solely so it's identifiable by name."""
    pass


class UTAdapterAudioLayer(nn.Module):
    """Wraps a frozen HuggingFace ``Wav2Vec2EncoderLayer`` with parallel UT-Adapters."""

    def __init__(self, frozen_layer: nn.Module, d_model: int = 768, bottleneck: int = 128, conv_kernel: int = 3):
        super().__init__()
        self.attention = frozen_layer.attention
        self.dropout = frozen_layer.dropout
        self.layer_norm = frozen_layer.layer_norm
        self.feed_forward = frozen_layer.feed_forward
        self.final_layer_norm = frozen_layer.final_layer_norm

        self.u_attn = UTAdapter(d_model, bottleneck, conv_kernel)
        self.u_ff = UTAdapter(d_model, bottleneck, conv_kernel)
        self.an1 = AdapterNorm(d_model)
        self.an2 = AdapterNorm(d_model)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_out, _, _ = self.attention(hidden_states, attention_mask=attention_mask, output_attentions=False)
        attn_out = self.dropout(attn_out)
        x2 = self.an1(hidden_states + self.layer_norm(attn_out) + self.u_attn(hidden_states))

        ff_out = self.feed_forward(x2)
        x1 = self.an2(x2 + self.final_layer_norm(ff_out) + self.u_ff(x2))
        return (x1,)


class UTAdapterVisualLayer(nn.Module):
    """Pre-LN transformer encoder layer with parallel UT-Adapters on MHSA and MLP."""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int = 512,
                 dropout: float = 0.6, bottleneck: int = 128, conv_kernel: int = 3):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.u1 = UTAdapter(d_model, bottleneck, conv_kernel)
        self.u2 = UTAdapter(d_model, bottleneck, conv_kernel)
        self.an1 = AdapterNorm(d_model)
        self.an2 = AdapterNorm(d_model)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=src_key_padding_mask, need_weights=False)
        x2 = self.an1(x + attn_out + self.u1(x))
        x1 = self.an2(x2 + self.mlp(self.ln2(x2)) + self.u2(x2))
        return x1


def freeze_audio_backbone_keep_adapters(audio_module: nn.Module) -> None:
    """Freeze all params except UT-Adapter and AdapterNorm modules inside wrapped layers."""
    keep = ("u_attn", "u_ff", "an1", "an2")
    for name, p in audio_module.named_parameters():
        if any(k in name for k in keep):
            p.requires_grad = True
        else:
            p.requires_grad = False
