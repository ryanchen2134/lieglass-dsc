"""
MultimodalDeceptionModel v2 — PECL-Adapted 3-Tower Architecture.

Pipeline:
  1. Audio tower:   frozen Wav2Vec2-BASE + EfficientConvPass adapters
                    → N per-layer hidden states (B, 64, 768)
  2. Visual tower:  trainable CNN face extractor + frozen ViT-B/16 + adapters
                    → N per-layer hidden states (B, 64, 768)
  3. Text tower:    frozen all-MiniLM-L6-v2, mean-pooled → projected to (B, d_text_proj)
  4. CrossFusion:   N CrossFusionModules (one per layer pair) → (B, 64, 64*N)
                    → mean-pool over 64 positions → (B, 64*N)
  5. Classifier:    cat([av_pooled, text_proj]) → SimpleClassificationHead → (B, 2)

All backbone parameters are frozen; only adapters, CNN face extractor,
CrossFusionModules, text projection, and classifier are trained (~2-3M params).
"""

import torch
import torch.nn as nn

from .encoders.text_encoder import TextEncoder
from .encoders.audio_encoder import Wav2Vec2AudioEncoder
from .encoders.visual_encoder import ViTVisualEncoder
from .fusion.cross_fusion import CrossFusionModule
from .classifier import SimpleClassificationHead


class MultimodalDeceptionModel(nn.Module):
    """
    v2 3-Tower PECL-Adapted deception detection model.

    Args:
        config: ModelConfig dataclass (see config.py).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ------------------------------------------------------------------ #
        # Tower 1: Audio — Wav2Vec2-BASE + adapters
        # ------------------------------------------------------------------ #
        self.audio_encoder = Wav2Vec2AudioEncoder(
            model_name=config.wav2vec2_model,
            n_layers=config.wav2vec2_n_layers,
            bottleneck=config.adapter_bottleneck,
        )

        # ------------------------------------------------------------------ #
        # Tower 2: Visual — CNN face extractor + ViT-B/16 + adapters
        # ------------------------------------------------------------------ #
        self.visual_encoder = ViTVisualEncoder(
            model_name=config.vit_model,
            n_layers=config.vit_n_layers,
            bottleneck=config.adapter_bottleneck,
        )

        # ------------------------------------------------------------------ #
        # Tower 3: Text — frozen all-MiniLM-L6-v2 + mean-pool + linear proj
        # ------------------------------------------------------------------ #
        self.text_encoder = TextEncoder(config.text_model_name)
        # Project mean-pooled text from 384 → d_text_proj (256)
        self.text_projection = nn.Linear(config.d_text, config.d_text_proj)

        # ------------------------------------------------------------------ #
        # CrossFusionModules — one per encoder layer
        # ------------------------------------------------------------------ #
        self.cross_fusions = nn.ModuleList([
            CrossFusionModule(
                d_in=768,
                d_proj=config.d_fusion_proj,
                d_out=config.d_fusion_out,
            )
            for _ in range(config.n_fusion_layers)
        ])

        # ------------------------------------------------------------------ #
        # Classifier
        # d_in = n_fusion_layers * d_fusion_out + d_text_proj
        #       = 4 * 64 + 256 = 512
        # ------------------------------------------------------------------ #
        d_clf_in = config.n_fusion_layers * config.d_fusion_out + config.d_text_proj
        self.classifier = SimpleClassificationHead(
            d_in=d_clf_in,
            dropout=config.dropout,
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict from collate_fn with keys:
                waveform:          (B, 20480)           FloatTensor
                frames:            (B, 64, 3, 160, 160) FloatTensor
                text_token_ids:    (B, n_max)            LongTensor
                text_padding_mask: (B, n_max)            BoolTensor, True=PAD
                label:             (B,)                  LongTensor
        Returns:
            logits: (B, 2)   — 2-class logits for CrossEntropyLoss.
        """
        # ------------------------------------------------------------------ #
        # 1. Audio tower
        # ------------------------------------------------------------------ #
        audio_hiddens, _ = self.audio_encoder(batch["waveform"])
        # audio_hiddens: list of n_layers tensors, each (B, 64, 768)

        # ------------------------------------------------------------------ #
        # 2. Visual tower
        # ------------------------------------------------------------------ #
        visual_hiddens, _ = self.visual_encoder(batch["frames"])
        # visual_hiddens: list of n_layers tensors, each (B, 64, 768)

        # ------------------------------------------------------------------ #
        # 3. Layer-wise CrossFusion
        # ------------------------------------------------------------------ #
        fusion_outputs = []
        for i, cross_fn in enumerate(self.cross_fusions):
            fused_i = cross_fn(audio_hiddens[i], visual_hiddens[i])  # (B, 64, 64)
            fusion_outputs.append(fused_i)

        # Concatenate across layers and mean-pool over 64 token positions
        av_fused  = torch.cat(fusion_outputs, dim=-1)  # (B, 64, 64*N)
        av_pooled = av_fused.mean(dim=1)               # (B, 64*N = 256)

        # ------------------------------------------------------------------ #
        # 4. Text tower — mean-pool over valid tokens → project
        # ------------------------------------------------------------------ #
        text_emb = self.text_encoder(
            batch["text_token_ids"],
            attention_mask=~batch["text_padding_mask"],  # HF expects True=valid
        )  # (B, n_max, 384)

        # Mean-pool over valid (non-padded) positions
        valid_mask = ~batch["text_padding_mask"]                  # (B, n_max), True=valid
        valid_float = valid_mask.float().unsqueeze(-1)            # (B, n_max, 1)
        text_pooled = (text_emb * valid_float).sum(dim=1) / (    # (B, 384)
            valid_float.sum(dim=1).clamp(min=1e-6)
        )
        text_proj = self.text_projection(text_pooled)             # (B, 256)

        # ------------------------------------------------------------------ #
        # 5. Classify
        # ------------------------------------------------------------------ #
        combined = torch.cat([av_pooled, text_proj], dim=-1)      # (B, 512)
        return self.classifier(combined)                          # (B, 2)
