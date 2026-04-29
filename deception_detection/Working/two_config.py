from dataclasses import dataclass, field
import torch


@dataclass
class ModelConfig:
    # --- Wav2Vec2 audio encoder ---
    wav2vec2_model: str = "facebook/wav2vec2-base"
    d_audio: int = 768
    wav2vec2_unfreeze_last_n: int = 0   # 0 = fully frozen; raise to fine-tune

    # --- Visual encoder (CNN_Face + temporal Transformer) ---
    # Note: ``vit_model`` is kept for backwards compatibility with serialized
    # configs but is no longer loaded — the temporal encoder is a native
    # ``nn.TransformerEncoder`` built from scratch (see visual_model.py).
    vit_model: str = "google/vit-base-patch16-224"
    d_visual: int = 768
    vit_n_layers: int = 4               # temporal Transformer depth
    vit_n_heads: int = 8                # multi-head attention heads
    vit_unfreeze_last_n: int = 4        # retained for compat; all temporal layers are trainable
    cnn_chunk_size: int = 32            # frames per CNN chunk (caps peak activation)
    in_channels: int = 1

    # Full-frame pipeline — every frame of the clip is used. ``max_frames``
    # is a safety cap: if a clip has more frames than this, a contiguous
    # centre window is taken. Set to ``None`` for no cap (watch VRAM).
    max_frames: int = 256
    legacy_n_frames: int = 16           # used only when falling back to frames.npz

    # --- Cross-modal fusion ---
    d_cross: int = 128
    d_fused: int = 256
    dropout: float = 0.5

    # --- Training ---
    batch_size: int = 8                 # smaller default — clips are longer now
    learning_rate: float = 6e-6
    weight_decay: float = 0.15
    max_epochs: int = 100
    patience: int = 12
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    pos_weight: float = 2.33            # placeholder; sampler balances classes
    grad_accum_steps: int = 8           # effective_batch = batch_size × steps

    # --- Cross-validation ---
    n_folds: int = 8
    seed: int = 2222

    # --- Paths ---
    feature_dir: str = "features"
    manifest_csv: str = "Data/manifest_dolos.csv"
    checkpoint_dir: str = "checkpoints"

    # --- DataLoader ---
    num_workers: int = 2
    prefetch_factor: int = 1

    # --- Device ---
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
