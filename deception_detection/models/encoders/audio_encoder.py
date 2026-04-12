"""
Wav2Vec2AudioEncoder — frozen Wav2Vec2-BASE backbone + EfficientConvPass adapters.

Architecture (PECL audio tower):
  1. Frozen CNN feature extractor  (Wav2Vec2 feature_extractor)
  2. Frozen feature projection     (linear + LayerNorm)
  3. Frozen positional conv embed  (pos_conv_embed)
  4. Frozen encoder LayerNorm
  5. N frozen Wav2Vec2EncoderLayer, each augmented with two parallel
     EfficientConvPass adapters (one for the attention sublayer, one for the FFN).

Input:  raw waveform (B, 20480)  — 16 kHz, padded/trimmed to 64×320 samples.
Output: layer_hiddens (list of N tensors, each (B, 64, 768))
        final_hidden  (B, 64, 768)

Only the adapter parameters are trainable (~50 K per adapter × 2 adapters × N layers).
"""

import torch
import torch.nn as nn
from ..adapters import EfficientConvPass


class Wav2Vec2AudioEncoder(nn.Module):
    """
    Frozen Wav2Vec2-BASE with parallel EfficientConvPass adapters.

    Args:
        model_name:  HuggingFace model ID (default "facebook/wav2vec2-base").
        n_layers:    Number of encoder layers to use (default 4).
        d_model:     Transformer hidden dim (768 for wav2vec2-base, do not change).
        bottleneck:  Adapter bottleneck width (default 32).
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        n_layers: int = 4,
        d_model: int = 768,
        bottleneck: int = 32,
    ):
        super().__init__()
        self.n_layers = n_layers

        # ------------------------------------------------------------------ #
        # Load pretrained Wav2Vec2 and extract the components we need.
        # Wav2Vec2ForCTC / Wav2Vec2Model both expose .wav2vec2; we use the
        # base model directly so we get clean hidden states.
        # ------------------------------------------------------------------ #
        from transformers import Wav2Vec2Model
        base = Wav2Vec2Model.from_pretrained(model_name)

        # Backbone components (all frozen)
        self.feature_extractor  = base.feature_extractor       # CNN
        self.feature_projection = base.feature_projection      # Linear + LN
        self.pos_conv_embed     = base.encoder.pos_conv_embed   # positional conv
        self.encoder_layer_norm = base.encoder.layer_norm       # final LN before transformer layers
        self.encoder_layers     = nn.ModuleList(
            base.encoder.layers[:n_layers]
        )

        # Freeze every backbone parameter
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        for p in self.feature_projection.parameters():
            p.requires_grad = False
        for p in self.pos_conv_embed.parameters():
            p.requires_grad = False
        for p in self.encoder_layer_norm.parameters():
            p.requires_grad = False
        for layer in self.encoder_layers:
            for p in layer.parameters():
                p.requires_grad = False

        # ------------------------------------------------------------------ #
        # Trainable EfficientConvPass adapters — one pair per layer.
        # attn_adapters[i] runs parallel to layer i's self-attention sublayer.
        # ffn_adapters[i]  runs parallel to layer i's feed-forward sublayer.
        # ------------------------------------------------------------------ #
        self.attn_adapters = nn.ModuleList(
            [EfficientConvPass(d_model, bottleneck) for _ in range(n_layers)]
        )
        self.ffn_adapters = nn.ModuleList(
            [EfficientConvPass(d_model, bottleneck) for _ in range(n_layers)]
        )

    # ------------------------------------------------------------------ #
    # Keep frozen backbone modules in eval mode regardless of model.train()
    # ------------------------------------------------------------------ #
    def train(self, mode: bool = True):
        super().train(mode)
        self.feature_extractor.eval()
        self.feature_projection.eval()
        self.pos_conv_embed.eval()
        self.encoder_layer_norm.eval()
        for layer in self.encoder_layers:
            layer.eval()
        return self

    # ------------------------------------------------------------------ #
    # Forward helpers — decomposed layer forward with parallel adapters
    # ------------------------------------------------------------------ #
    def _adapted_layer_forward(
        self,
        x: torch.Tensor,
        layer,
        attn_adapter: EfficientConvPass,
        ffn_adapter: EfficientConvPass,
    ) -> torch.Tensor:
        """
        Runs one Wav2Vec2EncoderLayer with parallel adapters.

        HuggingFace Wav2Vec2EncoderLayer structure (pre-LN):
            residual = x
            x = layer.layer_norm(x)
            x, _ = layer.attention(x)
            x = layer.dropout(x)
            x = residual + x              ← first residual

            x = x + layer.feed_forward(layer.final_layer_norm(x))  ← second residual

        With adapters added *parallel* to each sublayer output:
            # Attention sublayer
            x_norm = layer.layer_norm(x)
            attn_out = layer.attention(x_norm)[0] via dropout
            x = x + attn_out + attn_adapter(x_norm)

            # FFN sublayer
            x_norm2 = layer.final_layer_norm(x)
            x = x + layer.feed_forward(x_norm2) + ffn_adapter(x_norm2)
        """
        # --- Attention sublayer ---
        # Wav2Vec2Attention returns (attn_output, attn_weights, past_key_value)
        residual = x
        x_norm = layer.layer_norm(x)
        with torch.no_grad():
            attn_out = layer.attention(
                x_norm,
                attention_mask=None,
                output_attentions=False,
            )[0]                               # take first element of tuple
            attn_out = layer.dropout(attn_out)
        x = residual + attn_out + attn_adapter(x_norm)

        # --- Feed-forward sublayer ---
        x_norm2 = layer.final_layer_norm(x)
        with torch.no_grad():
            ffn_out = layer.feed_forward(x_norm2)
        x = x + ffn_out + ffn_adapter(x_norm2)

        return x

    def forward(self, waveform: torch.Tensor):
        """
        Args:
            waveform: (B, 20480)  — mono 16 kHz, padded/trimmed to 64*320 samples.

        Returns:
            layer_hiddens: list of n_layers tensors, each (B, 64, 768).
            final_hidden:  (B, 64, 768)  — same as layer_hiddens[-1].
        """
        with torch.no_grad():
            # CNN feature extraction: (B, 20480) → (B, T_feat, 512)
            # T_feat ≈ 64 given the 20480-sample input.
            extract_features = self.feature_extractor(waveform).transpose(1, 2)
            # (B, T_feat, 512) after transpose (HF returns B×512×T)

            # Feature projection: 512 → 768, with layer norm
            hidden_states, _ = self.feature_projection(extract_features)
            # (B, T_feat, 768)

            # Positional convolutional embedding
            position_embeddings = self.pos_conv_embed(hidden_states)
            hidden_states = hidden_states + position_embeddings
            hidden_states = self.encoder_layer_norm(hidden_states)
            # (B, 64, 768)

        # Layer-wise forward with adapters
        layer_hiddens = []
        x = hidden_states
        for i in range(self.n_layers):
            x = self._adapted_layer_forward(
                x,
                self.encoder_layers[i],
                self.attn_adapters[i],
                self.ffn_adapters[i],
            )
            layer_hiddens.append(x)

        return layer_hiddens, x  # list of (B,64,768), final (B,64,768)
