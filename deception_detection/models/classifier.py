"""
SimpleClassificationHead — replaces the previous BiDirTransformerClassifier.

The old classifier had ~15M trainable parameters (2-layer BiDir transformer,
d_model=896, 4 heads, ff=3584) which massively overfit on ~1150 samples.

This replacement is a 2-layer MLP with ~10K parameters:
    Linear(d_in, 64) → GELU → Dropout(0.5) → Linear(64, 2)

Output is 2-class logits for CrossEntropyLoss (not sigmoid/BCE).
"""

import torch
import torch.nn as nn


class SimpleClassificationHead(nn.Module):
    """
    Lightweight 2-layer MLP classifier.

    Args:
        d_in:      Input dimension (n_fusion_layers * d_fusion_out + d_text_proj).
                   Default: 4*64 + 256 = 512.
        d_hidden:  Hidden dimension (default 64).
        dropout:   Dropout probability (default 0.5).
        n_classes: Number of output classes (default 2).
    """

    def __init__(
        self,
        d_in: int = 512,
        d_hidden: int = 64,
        dropout: float = 0.5,
        n_classes: int = 2,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, d_in)
        Returns:
            logits: (B, n_classes)  — raw scores for CrossEntropyLoss.
        """
        return self.head(x)
