from dataclasses import dataclass, field
import torch


@dataclass
class ModelConfig:
    # Text encoder
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    d_text: int = 384               # output dim of the text encoder (fixed by model choice)

    # MFCC encoder
    n_mfcc_coefficients: int = 13
    d_mfcc: int = 256               # output dim of MFCC encoder

    # Landmark encoder
    n_landmarks: int = 68
    landmark_coords: int = 2        # x, y (2D landmarks)
    d_landmark: int = 256           # output dim of landmark encoder

    # Cross-attention
    cross_attn_heads: int = 4
    cross_attn_dropout: float = 0.1

    # Classifier
    d_fused: int = 384 + 256 + 256  # d_text + d_mfcc + d_landmark = 896
    n_clf_layers: int = 2           # reduced from 3 — was 29M params, too large for ~1150 samples
    clf_heads: int = 4              # reduced from 8
    clf_ff_mult: int = 4
    dropout: float = 0.4            # increased from 0.3 for stronger regularization

    # Training
    batch_size: int = 16
    learning_rate: float = 5*1e-4
    weight_decay: float = 1e-2
    max_epochs: int = 200 #100
    patience: int = 50 #15
    grad_clip: float = 1.0
    pos_weight: float = 2.33        # Placeholder; recomputed from actual class ratio per fold

    # Cross-validation
    n_folds: int = 8 #5
    seed: int = 42

    # Paths (set at runtime)
    feature_dir: str = "features"
    manifest_csv: str = "Data/manifest_dolos.csv"
    checkpoint_dir: str = "checkpoints"

    # Device (set at runtime)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
