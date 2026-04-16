from dataclasses import dataclass, field
import torch


@dataclass
class ModelConfig:
    # Wav2Vec2 audio encoder
    wav2vec2_model: str = "facebook/wav2vec2-base"
    d_audio: int = 768
    wav2vec2_unfreeze_last_n: int = 2   # unfreeze last N transformer layers

    # Visual encoder (CNN + ViT-B/16)
    vit_model: str = "google/vit-base-patch16-224"
    d_visual: int = 768
    n_frames: int = 16                  # uniformly sampled frames per video (16 = 4× less CNN memory than 64)
    vit_n_layers: int = 4               # number of ViT encoder layers to use
    vit_unfreeze_last_n: int = 2        # unfreeze last N of those layers
    cnn_chunk_size: int = 32            # frames processed by CNN at once; caps peak GPU activation

    # Cross-modal fusion
    d_cross: int = 256                  # projection dim inside CrossFusionModule
    d_fused: int = 512                  # output dim of CrossFusionModule
    dropout: float = 0.2

    # Training
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 1e-2
    max_epochs: int = 200
    patience: int = 30
    grad_clip: float = 1.0
    pos_weight: float = 2.33           # placeholder; recomputed per fold
    grad_accum_steps: int = 1          # gradient accumulation; effective_batch = batch_size × steps

    # Cross-validation
    n_folds: int = 8
    seed: int = 42

    # Paths
    feature_dir: str = "features"
    manifest_csv: str = "Data/manifest_dolos.csv"
    checkpoint_dir: str = "checkpoints"

    # DataLoader
    # num_workers=0 is the safe default: OpenCV VideoCapture inside forked/spawned
    # workers deadlocks on many Linux systems. Increase only if your OS + OpenCV
    # build are confirmed to be fork-safe (e.g. after testing with num_workers=2).
    num_workers: int = 0

    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
