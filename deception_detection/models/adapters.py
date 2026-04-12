"""
EfficientConvPass adapter — PECL-style 1D-conv bottleneck adapter.

Inserted parallel to MHSA and FFN sublayers in frozen transformer layers.
Gradient flows only through the adapter; backbone weights stay frozen.

Architecture (per adapter, per sublayer):
    x  →  Linear(d_model, bottleneck)
       →  GELU
       →  Conv1d(bottleneck, bottleneck, kernel_size=3, padding=1)
       →  GELU
       →  Linear(bottleneck, d_model)
       →  + x   (residual)

Output is added to the backbone sublayer output before the residual:
    x_new = x + backbone_sublayer(LN(x)) + adapter(LN(x))
"""

import torch
import torch.nn as nn


class EfficientConvPass(nn.Module):
    """
    Parallel 1D-conv adapter for frozen transformer sublayers.

    Args:
        d_model:    Transformer hidden dimension (768 for Wav2Vec2-BASE / ViT-B/16).
        bottleneck: Internal bottleneck width (default 32, ~100K params per adapter).
    """

    def __init__(self, d_model: int = 768, bottleneck: int = 32):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.conv = nn.Conv1d(bottleneck, bottleneck, kernel_size=3, padding=1)
        self.up   = nn.Linear(bottleneck, d_model)
        self.act  = nn.GELU()

        # Initialise up-projection to zero so the adapter is a no-op at init.
        # This lets the frozen backbone run as-is at the start of training.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)  — pre-LayerNorm input to the sublayer
        Returns:
            (B, T, d_model)     — adapter output (added to backbone output externally)
        """
        h = self.act(self.down(x))          # (B, T, bottleneck)
        h = h.transpose(1, 2)              # (B, bottleneck, T)
        h = self.act(self.conv(h))         # (B, bottleneck, T)
        h = h.transpose(1, 2)             # (B, T, bottleneck)
        return self.up(h)                  # (B, T, d_model)  — no residual here;
                                           # caller adds this to backbone output + x
