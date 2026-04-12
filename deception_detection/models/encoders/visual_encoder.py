"""
ViTVisualEncoder — CNN face extractor + frozen ViT-B/16 backbone + EfficientConvPass adapters.

Architecture (PECL visual tower):
  1. Trainable CNN face extractor (3-block residual CNN → 256-dim per frame)
  2. Trainable Linear(256, 768) + ReLU projection
  3. Trainable learned positional embedding (1, 64, 768)
  4. N frozen ViTLayer blocks, each augmented with two parallel
     EfficientConvPass adapters (one for the attention sublayer, one for the FFN).

Input:  face frames (B, 64, 3, 160, 160) — 64 uniformly sampled, ImageNet-normalised.
Output: layer_hiddens (list of N tensors, each (B, 64, 768))
        final_hidden  (B, 64, 768)

The CNN face extractor and projection are fully trainable.
The ViT backbone layers are frozen; only the adapter parameters are trained.
"""

import torch
import torch.nn as nn
from ..adapters import EfficientConvPass


# --------------------------------------------------------------------------- #
# CNN face extractor                                                           #
# --------------------------------------------------------------------------- #

class _ResBlock(nn.Module):
    """Residual block: Conv→BN→GELU→Conv→BN + skip projection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.skip(x))


class CNNFaceExtractor(nn.Module):
    """
    3-block residual CNN: (B, 3, 160, 160) → (B, 256).

    Matches PECL's cnn_face design: 3 → 64 → 128 → 256, spatial dims
    halved at each block, then global average pooling to a single vector.
    """

    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            _ResBlock(3,   64,  stride=2),   # → (B, 64,  80, 80)
            _ResBlock(64,  128, stride=2),   # → (B, 128, 40, 40)
            _ResBlock(128, 256, stride=2),   # → (B, 256, 20, 20)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B_frames, 3, H, W)
        Returns:
            (B_frames, 256)
        """
        x = self.blocks(x)          # (B_frames, 256, H', W')
        x = self.pool(x)            # (B_frames, 256, 1, 1)
        return x.flatten(1)         # (B_frames, 256)


# --------------------------------------------------------------------------- #
# Full visual encoder                                                          #
# --------------------------------------------------------------------------- #

class ViTVisualEncoder(nn.Module):
    """
    Frozen ViT-B/16 (first N layers) with trainable CNN face extractor and adapters.

    Args:
        model_name:  HuggingFace ViT model ID (default "google/vit-base-patch16-224").
        n_layers:    Number of ViT encoder layers to use (default 4).
        n_frames:    Fixed number of input frames (default 64).
        d_model:     ViT hidden dim (768 for vit-base, do not change).
        bottleneck:  Adapter bottleneck width (default 32).
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        n_layers: int = 4,
        n_frames: int = 64,
        d_model: int = 768,
        bottleneck: int = 32,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_frames = n_frames

        # ------------------------------------------------------------------ #
        # Trainable CNN face extractor + projection
        # ------------------------------------------------------------------ #
        self.cnn_face = CNNFaceExtractor()                         # 3→256
        self.frame_proj = nn.Sequential(
            nn.Linear(256, d_model),
            nn.ReLU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, n_frames, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ------------------------------------------------------------------ #
        # Load pretrained ViT and extract transformer layers.
        # We only use the attention+FFN layers, NOT the patch embedding —
        # our CNN face extractor serves as the per-frame feature extractor.
        # ------------------------------------------------------------------ #
        from transformers import ViTModel
        base = ViTModel.from_pretrained(model_name)

        self.vit_layers = nn.ModuleList(
            base.encoder.layer[:n_layers]
        )

        # Freeze all ViT backbone params
        for layer in self.vit_layers:
            for p in layer.parameters():
                p.requires_grad = False

        # ------------------------------------------------------------------ #
        # Trainable EfficientConvPass adapters — one pair per ViT layer.
        # ------------------------------------------------------------------ #
        self.attn_adapters = nn.ModuleList(
            [EfficientConvPass(d_model, bottleneck) for _ in range(n_layers)]
        )
        self.ffn_adapters = nn.ModuleList(
            [EfficientConvPass(d_model, bottleneck) for _ in range(n_layers)]
        )

    # ------------------------------------------------------------------ #
    # Keep frozen ViT layers in eval mode.
    # ------------------------------------------------------------------ #
    def train(self, mode: bool = True):
        super().train(mode)
        for layer in self.vit_layers:
            layer.eval()
        return self

    # ------------------------------------------------------------------ #
    # Decomposed ViT layer forward with parallel adapters.
    # ------------------------------------------------------------------ #
    def _adapted_layer_forward(
        self,
        x: torch.Tensor,
        layer,
        attn_adapter: EfficientConvPass,
        ffn_adapter: EfficientConvPass,
    ) -> torch.Tensor:
        """
        Runs one HuggingFace ViTLayer with parallel adapters.

        HuggingFace ViTLayer structure (pre-LN):
            # Attention sublayer
            x_norm = layer.layernorm_before(x)
            attn_out = layer.attention(x_norm)[0]
            x = x + attn_out

            # FFN sublayer (ViTOutput includes its own residual)
            x_norm2 = layer.layernorm_after(x)
            intermediate = layer.intermediate(x_norm2)
            x = layer.output(intermediate, x)   ← output = dense+dropout+residual

        With parallel adapters:
            x_norm = layer.layernorm_before(x)
            attn_out = layer.attention(x_norm)[0]
            x = x + attn_out + attn_adapter(x_norm)

            x_norm2 = layer.layernorm_after(x)
            intermediate = layer.intermediate(x_norm2)
            ffn_out = layer.output.dense(intermediate) + dropout
            x = x + ffn_out + ffn_adapter(x_norm2)
        """
        # --- Attention sublayer ---
        # ViTAttention.forward returns (attention_output,) when output_attentions=False.
        x_norm = layer.layernorm_before(x)
        with torch.no_grad():
            attn_out = layer.attention(x_norm, output_attentions=False)[0]
        x = x + attn_out + attn_adapter(x_norm)

        # --- FFN sublayer ---
        # Decompose ViTOutput to bypass its built-in residual so we can add our adapter.
        x_norm2 = layer.layernorm_after(x)
        with torch.no_grad():
            intermediate = layer.intermediate(x_norm2)   # ViTIntermediate: dense+act
            ffn_out = layer.output.dense(intermediate)    # ViTOutput.dense
            ffn_out = layer.output.dropout(ffn_out)
        x = x + ffn_out + ffn_adapter(x_norm2)

        return x

    def forward(self, frames: torch.Tensor):
        """
        Args:
            frames: (B, 64, 3, 160, 160) — ImageNet-normalised face frames.

        Returns:
            layer_hiddens: list of n_layers tensors, each (B, 64, 768).
            final_hidden:  (B, 64, 768)  — same as layer_hiddens[-1].
        """
        B, T, C, H, W = frames.shape  # T == 64

        # CNN face extractor on all frames simultaneously
        frames_flat = frames.view(B * T, C, H, W)     # (B*64, 3, 160, 160)
        cnn_feat = self.cnn_face(frames_flat)          # (B*64, 256)
        cnn_feat = cnn_feat.view(B, T, -1)             # (B, 64, 256)

        # Project to d_model and add positional embedding
        x = self.frame_proj(cnn_feat)                  # (B, 64, 768)
        x = x + self.pos_embed                         # (B, 64, 768)

        # Layer-wise forward with adapters
        layer_hiddens = []
        for i in range(self.n_layers):
            x = self._adapted_layer_forward(
                x,
                self.vit_layers[i],
                self.attn_adapters[i],
                self.ffn_adapters[i],
            )
            layer_hiddens.append(x)

        return layer_hiddens, x  # list of (B,64,768), final (B,64,768)
