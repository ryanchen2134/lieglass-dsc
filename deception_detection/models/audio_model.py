"""
Wav2Vec2-based audio encoder with parallel UT-Adapters and multi-stage hidden-state extraction.

Pipeline:
  raw waveform (B, T)
    -> Wav2Vec2 CNN feature extractor   [frozen]
    -> feature projection + layer norm  [frozen]
    -> Wav2Vec2 transformer encoder     [each layer wrapped with UT-Adapters; backbone frozen]
    -> intermediate hidden states at config.audio_fusion_layers
    -> (list of (B, T', 768), frame attention mask (B, T'))
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

from .adapters import UTAdapterAudioLayer, freeze_audio_backbone_keep_adapters


class W2V2_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w2v2 = Wav2Vec2Model.from_pretrained(config.wav2vec2_model)
        self.use_ut_adapters = bool(getattr(config, "use_ut_adapters", False))
        self.fusion_layers = sorted(set(config.audio_fusion_layers))

        n_total = len(self.w2v2.encoder.layers)
        if max(self.fusion_layers) > n_total or min(self.fusion_layers) < 1:
            raise ValueError(
                f"audio_fusion_layers={self.fusion_layers} out of range [1, {n_total}]."
            )

        if self.use_ut_adapters:
            wrapped = nn.ModuleList([
                UTAdapterAudioLayer(
                    layer,
                    d_model=config.d_audio,
                    bottleneck=config.ut_adapter_dim,
                    conv_kernel=config.ut_conv_kernel,
                )
                for layer in self.w2v2.encoder.layers
            ])
            self.w2v2.encoder.layers = wrapped
            freeze_audio_backbone_keep_adapters(self)
        else:
            # Legacy freezing path: feature extractor + projection always frozen;
            # transformer layers frozen except last N.
            for p in self.w2v2.feature_extractor.parameters():
                p.requires_grad = False
            for p in self.w2v2.feature_projection.parameters():
                p.requires_grad = False
            n_freeze = max(0, n_total - config.wav2vec2_unfreeze_last_n)
            for i, layer in enumerate(self.w2v2.encoder.layers):
                if i < n_freeze:
                    for p in layer.parameters():
                        p.requires_grad = False

    # ------------------------------------------------------------------
    # Multi-stage forward
    # ------------------------------------------------------------------

    def forward_multistage(self, waveform: torch.Tensor, attention_mask: torch.Tensor | None = None
                           ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """
        Returns:
            stages: list of (B, T', 768) hidden states at fusion_layers (1-indexed).
            frame_mask: (B, T') bool mask of valid frames, or None.
        """
        mask_long = attention_mask.long() if attention_mask is not None else None

        outputs = self.w2v2(
            input_values=waveform,
            attention_mask=mask_long,
            output_hidden_states=True,
            return_dict=True,
        )
        # outputs.hidden_states is a tuple of length (n_layers + 1):
        #   [0] is the input embedding (post pos_conv + layer_norm + dropout)
        #   [i] for i >= 1 is the output of encoder layer i (1-indexed).
        all_hidden = outputs.hidden_states
        stages = [all_hidden[i] for i in self.fusion_layers]

        frame_mask = None
        if attention_mask is not None:
            T_prime = all_hidden[0].shape[1]
            frame_mask = self.w2v2._get_feature_vector_attention_mask(T_prime, mask_long).bool()

        return stages, frame_mask

    # ------------------------------------------------------------------
    # Backwards-compatible single-vector forward
    # ------------------------------------------------------------------

    def forward(self, waveform: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        stages, frame_mask = self.forward_multistage(waveform, attention_mask)
        last = stages[-1]                                  # (B, T', 768)
        if frame_mask is None:
            return last.mean(dim=1)
        m = frame_mask.unsqueeze(-1).float()
        return (last * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
