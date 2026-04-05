"""
Bimodal deception detection model for the Real-life dataset.

Modalities:
  - Text:        frozen all-MiniLM-L6-v2 → mean-pool → 384-dim
  - Annotations: 39 binary gesture features → MLP → 64-dim
Fusion: concatenate → MLP classifier → binary logit
"""

import torch
import torch.nn as nn

from ..models.encoders.text_encoder import TextEncoder
from .config import RealLifeConfig


class AnnotationEncoder(nn.Module):
    """
    Encodes 39 binary gesture annotation features into a fixed-size vector.

    Architecture:
        Linear(n_annot → d_hidden) → BatchNorm → GELU
        Linear(d_hidden → d_out)   → BatchNorm
    """

    def __init__(self, n_annot: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_annot, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
            nn.BatchNorm1d(d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor (B, n_annot)
        Returns:
            FloatTensor (B, d_out)
        """
        return self.net(x)


class ClassifierHead(nn.Module):
    """
    Two-layer MLP classifier operating on the fused representation.

    Architecture:
        LayerNorm → Linear(d_fused → d_hidden) → GELU → Dropout → Linear(d_hidden → 1)
    """

    def __init__(self, d_fused: int, d_hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_fused),
            nn.Linear(d_fused, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor (B, d_fused)
        Returns:
            FloatTensor (B,) — raw logits
        """
        return self.net(x).squeeze(-1)


class BimodalDeceptionModel(nn.Module):
    """
    Bimodal deception detector: text + gesture annotations → lie probability.

    Text branch: frozen HF transformer → masked mean-pool → 384-dim
    Annot branch: AnnotationEncoder → 64-dim
    Fusion: concat → ClassifierHead → logit
    """

    def __init__(self, config: RealLifeConfig):
        super().__init__()
        self.text_encoder = TextEncoder(model_name=config.text_model_name)
        self.annot_encoder = AnnotationEncoder(
            n_annot=config.n_annotations,
            d_hidden=config.d_annot_hidden,
            d_out=config.d_annot_out,
        )
        self.classifier = ClassifierHead(
            d_fused=config.d_fused,
            d_hidden=config.d_hidden,
            dropout=config.dropout,
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict with
                token_ids      (B, n)   LongTensor
                attention_mask (B, n)   LongTensor (1=valid, 0=pad)
                annotations    (B, 39)  FloatTensor

        Returns:
            logits (B,) FloatTensor
        """
        token_ids = batch["token_ids"]
        attention_mask = batch["attention_mask"]
        annotations = batch["annotations"]

        # Text: (B, n, 384) → masked mean-pool → (B, 384)
        token_emb = self.text_encoder(token_ids, attention_mask.bool())
        mask_f = attention_mask.unsqueeze(-1).float()  # (B, n, 1)
        text_emb = (token_emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)

        # Annotations: (B, 39) → (B, 64)
        annot_emb = self.annot_encoder(annotations)

        # Fuse and classify
        fused = torch.cat([text_emb, annot_emb], dim=-1)  # (B, 448)
        return self.classifier(fused)
