"""
ModelConfig v2 — PECL-Adapted 3-Tower architecture configuration.

Key changes from v1:
  - Replaced MFCC/landmark encoder params with Wav2Vec2 + ViT params.
  - Replaced cross-attention params with CrossFusion params.
  - Replaced BiDirTransformer classifier params with simple MLP params.
  - Switched from OneCycleLR to CosineAnnealingLR.
  - Switched from BCEWithLogitsLoss to CrossEntropyLoss (2-class).
"""

from dataclasses import dataclass, field
import torch


@dataclass
class ModelConfig:
    # ------------------------------------------------------------------ #
    # Audio encoder — Wav2Vec2-BASE + EfficientConvPass adapters
    # ------------------------------------------------------------------ #
    wav2vec2_model: str = "facebook/wav2vec2-base"
    wav2vec2_n_layers: int = 4          # number of encoder layers to use (PECL default)
    adapter_bottleneck: int = 32        # adapter internal width (~50K params per adapter)

    # ------------------------------------------------------------------ #
    # Visual encoder — CNN face extractor + ViT-B/16 + adapters
    # ------------------------------------------------------------------ #
    vit_model: str = "google/vit-base-patch16-224"
    vit_n_layers: int = 4               # same as wav2vec2_n_layers

    # ------------------------------------------------------------------ #
    # Text encoder — frozen all-MiniLM-L6-v2, mean-pooled
    # ------------------------------------------------------------------ #
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    d_text: int = 384                   # output dim of text encoder (fixed)
    d_text_proj: int = 256              # projection dim for mean-pooled text

    # ------------------------------------------------------------------ #
    # CrossFusion — PECL-style bilinear cross-modal fusion
    # ------------------------------------------------------------------ #
    n_fusion_layers: int = 4            # one CrossFusionModule per encoder layer pair
    d_fusion_proj: int = 256            # bilinear intermediate dimension
    d_fusion_out: int = 64             # per-layer output dimension

    # ------------------------------------------------------------------ #
    # Classifier — SimpleClassificationHead
    # d_clf_in = n_fusion_layers * d_fusion_out + d_text_proj = 4*64+256 = 512
    # ------------------------------------------------------------------ #
    dropout: float = 0.5

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    batch_size: int = 16
    learning_rate: float = 1e-3         # fixed LR base for CosineAnnealingLR
    weight_decay: float = 1e-2
    max_epochs: int = 200
    patience: int = 50                  # early stopping on val AUC-ROC
    grad_clip: float = 1.0

    # ------------------------------------------------------------------ #
    # Cross-validation — GroupKFold on speaker ID (identity-aware)
    # ------------------------------------------------------------------ #
    n_folds: int = 8
    seed: int = 42

    # ------------------------------------------------------------------ #
    # Paths (overridable via CLI in train.py)
    # ------------------------------------------------------------------ #
    feature_dir: str = "features"
    manifest_csv: str = "Data/manifest_dolos.csv"
    checkpoint_dir: str = "checkpoints"

    # ------------------------------------------------------------------ #
    # Device
    # ------------------------------------------------------------------ #
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
