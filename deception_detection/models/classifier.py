import torch
import torch.nn as nn


class BiDirTransformerClassifier(nn.Module):
    """
    Bidirectional transformer encoder with [CLS] token pooling and a dense head.

    Input:  fused multimodal sequence (B, n, d_fused)
    Output: logits (B,) — raw logit, apply sigmoid + BCE loss externally.
    """

    def __init__(
        self,
        d_fused: int,
        n_layers: int,
        n_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Learnable [CLS] token — initialized at unit scale to match concatenated sequence tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_fused))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_fused,
            nhead=n_heads,
            dim_feedforward=d_fused * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_fused),
            nn.Linear(d_fused, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, fused: torch.Tensor, padding_mask=None) -> torch.Tensor:
        """
        Args:
            fused:        (B, n, d_fused)
            padding_mask: (B, n) BoolTensor, True = PAD (optional)
        Returns:
            logits: (B,)
        """
        B = fused.shape[0]

        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, d_fused)
        x = torch.cat([cls, fused], dim=1)               # (B, n+1, d_fused)

        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=fused.device)
            full_mask = torch.cat([cls_mask, padding_mask], dim=1)  # (B, n+1)
        else:
            full_mask = None

        x = self.transformer(x, src_key_padding_mask=full_mask)

        cls_out = x[:, 0, :]                             # (B, d_fused)
        return self.head(cls_out).squeeze(-1)            # (B,)
