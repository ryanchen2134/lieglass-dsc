# Multimodal Deception Detection — Full Project Specification

## 1. Project Overview

A multimodal lie/truth binary classifier operating on video recordings. Three modalities are extracted from each video — text transcription, audio MFCCs, and facial landmark coordinates — encoded independently, temporally aligned to text-token resolution via cross-attention, concatenated, and classified by a bidirectional transformer.

**No contrastive learning. No projection heads. No InfoNCE loss.** The model is trained end-to-end with binary cross-entropy only.

---

## 2. Architecture Summary

```
Raw Video
  ├─ Audio Channel ─┬─ Whisper ──────────── Timestamped Tokens ── Text Encoder ──────── text_emb (B, n, d_text)
  │                 └─ Spectral Sub ─ MFCC ─ CMVN ────────────── MFCC Encoder ──────── mfcc_emb (B, T_m, d_mfcc)
  │
  └─ Visual Frames ── YOLO Crop ── FaceXFormer ── Normalize ── Δ/ΔΔ ── Landmark Encoder ── land_emb (B, T_l, d_land)

text_emb (queries) + mfcc_emb (K/V)  ──→ Cross-Attention ──→ mfcc_aligned  (B, n, d_mfcc)
text_emb (queries) + land_emb (K/V)  ──→ Cross-Attention ──→ land_aligned  (B, n, d_land)
                                                                 ↓ (with frame_mask for blacked-out frames)

Concat([text_emb, mfcc_aligned, land_aligned], dim=-1) ──→ fused (B, n, d_text + d_mfcc + d_land)
    ↓
Prepend [CLS] token ──→ (B, n+1, d_fused)
    ↓
Bidirectional Transformer Encoder (2–4 layers)
    ↓
[CLS] output vector ──→ LayerNorm ──→ Dropout ──→ Dense(d_fused, 1) ──→ Sigmoid ──→ ŷ ∈ [0,1]

Loss: BCE with pos_weight for class imbalance
```

---

## 3. Data

### 3.1 Datasets

- **Real-Life Trial**: courtroom trial clips labeled truth/lie. ~120 videos.
- **DOLOS**: interview-style deception dataset. ~900+ videos.
- **Box of Lies** (optional): game-show format, more casual. ~200 clips.
- Total: ~1200–2000 videos, average <60s each. This is a small dataset.

### 3.2 Raw Data Format

Each sample has:
- A video file (variable resolution, variable FPS, variable length).
- A CSV manifest: `filename,label` where label ∈ {0, 1} (0=lie, 1=truth). Confirm label semantics per dataset and unify.

### 3.3 Video Preprocessing Pipeline (Offline, Run Once)

This runs before training. Each video is processed into a standardized format and pre-extracted feature files.

#### Step 1: Active Speaker Detection and Face Cropping

```
Input:  raw video file
Output: processed video file (square, face-cropped, non-speaker frames blacked out)
        + frame_mask array (binary, per-frame: 1=valid, 0=blacked-out)
        + audio file (copied from original)
```

**Active speaker detection**: The face appearing in the most frames across the video is designated the active speaker. Use YOLO for face detection per frame. Track identity across frames using IoU overlap of bounding boxes (or a simple tracker like SORT).

**Face cropping and alignment**: Crop to YOLO bounding box, resize to a fixed square resolution (e.g., 224×224). **Do NOT orient/align the face rotationally** — head rotation may contain deception signal.

**Blacking out**: Frames where the active speaker's face is not detected → set all pixels to zero. Store a binary mask array `frame_mask[i] = 1 if speaker detected in frame i, else 0`.

**Output**: A new video file at fixed resolution + a `.npy` file for `frame_mask` + extracted audio as `.wav`.

#### Step 2: Whisper Transcription (Offline)

```
Input:  audio .wav file
Output: list of (token_id, token_text, t_start, t_end) tuples
```

Use `whisper-timestamped` or `stable-ts` for word-level timestamps. Then tokenize using the text encoder's tokenizer. Sub-word tokens from the same word share the parent word's `(t_start, t_end)`.

Save as a `.json` or `.pt` per video:
```json
{
  "token_ids": [101, 2054, 2003, ...],
  "timestamps": [[0.0, 0.24], [0.24, 0.51], ...]
}
```

#### Step 3: MFCC Extraction (Offline)

```
Input:  audio .wav file
Output: MFCC tensor (n_mfcc_frames, n_coefficients, 3)
        + mfcc_timestamps array (n_mfcc_frames,) — center time of each frame
```

Pipeline:
1. **Spectral subtraction**: Estimate noise from the first 0.5s of audio (or a silence-detected segment). Subtract estimated noise power spectrum from each frame. This partially normalizes recording conditions.
2. **MFCC computation**: Frame length = 25ms, frame stride = 10ms (matching phoneme resolution). Extract `n_coefficients` = 13 MFCCs per frame (standard). Include energy as the 0th coefficient.
3. **Delta and delta-delta**: Compute first and second time derivatives of the MFCC coefficients. Stack to get `(n_mfcc_frames, 13, 3)`.
4. **CMVN (Cepstral Mean and Variance Normalization)**: Normalize per-utterance (entire video). Subtract mean, divide by std for each coefficient across all frames in this video. This normalizes speaker and channel variability.

Compute `mfcc_timestamps[i]` = center time of frame `i` = `i * stride + frame_length / 2`.

Save as `.pt`: `{"mfcc": tensor, "timestamps": array}`.

#### Step 4: Facial Landmark Extraction (Offline)

```
Input:  processed video frames (cropped, square)
Output: landmark tensor (n_video_frames, n_landmarks * 2, 3)
        + landmark_timestamps array (n_video_frames,)
```

1. **FaceXFormer**: Run on each frame. Outputs 68 2D landmark coordinates (x, y) per frame → `(n_frames, 68, 2)`. Reshape to `(n_frames, 136)`.
   ```python
   from huggingface_hub import hf_hub_download
   hf_hub_download(repo_id="kartiknarayan/facexformer", filename="ckpts/model.pt", local_dir="./")
   ```
2. **Normalization**: Per-frame, normalize coordinates to [0, 1] relative to the face bounding box (already cropped, so this is relative to image dimensions). Then, per-video, z-score normalize each landmark coordinate across all frames (mean=0, std=1).
3. **Delta and delta-delta**: Compute first and second time derivatives along the frame axis. This captures facial movement velocity and acceleration (micro-expressions). Stack → `(n_frames, 136, 3)`.
4. **Timestamps**: `landmark_timestamps[i] = i / fps` where fps is the processed video's frame rate.
5. **Blacked-out frames**: For frames where `frame_mask[i] = 0`, set landmark values to zero (they are meaningless). The mask propagates to cross-attention later.

Save as `.pt`: `{"landmarks": tensor, "timestamps": array, "frame_mask": mask}`.

#### Step 5: Feature Directory Structure

After preprocessing, each video `sample_001` has:
```
features/
  sample_001/
    text.pt          # {token_ids: LongTensor(n,), timestamps: FloatTensor(n, 2)}
    mfcc.pt          # {mfcc: FloatTensor(T_m, 13, 3), timestamps: FloatTensor(T_m,)}
    landmarks.pt     # {landmarks: FloatTensor(T_l, 136, 3), timestamps: FloatTensor(T_l,), frame_mask: BoolTensor(T_l,)}
    label.txt        # "0" or "1"
```

The manifest CSV maps `sample_id → label` and optionally `dataset_source` (for stratified splitting).

---

## 4. Dataset and DataLoader

### 4.1 Dataset Class

```python
# data/dataset.py

import torch
from torch.utils.data import Dataset
from pathlib import Path

class DeceptionDataset(Dataset):
    """
    Loads pre-extracted features from disk.
    Each __getitem__ returns a dict with all modalities + metadata.
    """

    def __init__(self, manifest_csv: str, feature_dir: str, augment: bool = False):
        """
        Args:
            manifest_csv: Path to CSV with columns [sample_id, label, dataset_source].
            feature_dir:  Root directory containing per-sample feature folders.
            augment:      Whether to apply data augmentation (training only).
        """
        # Parse manifest_csv into list of (sample_id, label) tuples.
        # Store self.samples, self.feature_dir, self.augment.
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id, label = self.samples[idx]
        feat_path = Path(self.feature_dir) / sample_id

        text_data = torch.load(feat_path / "text.pt")
        mfcc_data = torch.load(feat_path / "mfcc.pt")
        land_data = torch.load(feat_path / "landmarks.pt")

        # --- Augmentation (applied on-the-fly, training only) ---
        if self.augment:
            mfcc_data = self._augment_mfcc(mfcc_data)
            land_data = self._augment_landmarks(land_data)

        return {
            "text_token_ids": text_data["token_ids"],          # LongTensor (n,)
            "text_timestamps": text_data["timestamps"],        # FloatTensor (n, 2) — start, end per token
            "mfcc": mfcc_data["mfcc"],                         # FloatTensor (T_m, 13, 3)
            "mfcc_timestamps": mfcc_data["timestamps"],        # FloatTensor (T_m,)
            "landmarks": land_data["landmarks"],               # FloatTensor (T_l, 136, 3)
            "landmark_timestamps": land_data["timestamps"],    # FloatTensor (T_l,)
            "frame_mask": land_data["frame_mask"],             # BoolTensor (T_l,)
            "label": torch.tensor(label, dtype=torch.float32), # scalar
            "sample_id": sample_id,
        }

    def _augment_mfcc(self, mfcc_data):
        """
        Augmentations for MFCC (applied to mfcc_data["mfcc"] in-place or copy):
        1. Additive Gaussian noise: mfcc += N(0, sigma) where sigma ~ U(0, 0.01)
        2. Temporal dropout: randomly zero out ~5-10% of frames (entire frame set to 0).
           This simulates brief audio dropouts.
        DO NOT apply time warping or frequency masking that distorts temporal structure.
        Return modified mfcc_data dict.
        """
        pass

    def _augment_landmarks(self, land_data):
        """
        Augmentations for landmarks (applied to land_data["landmarks"]):
        1. Additive Gaussian jitter: landmarks += N(0, sigma) where sigma ~ U(0, 0.005)
           Only apply to frames where frame_mask == 1.
        DO NOT apply geometric transforms (rotation, scaling, flipping) — spatial
        arrangement and head orientation contain deception signal.
        DO NOT apply temporal manipulation — temporal dynamics are signal.
        Return modified land_data dict.
        """
        pass
```

### 4.2 Collate Function

Samples have variable sequence lengths across all three modalities. The collate function pads each modality to the max length in the batch and produces padding masks.

```python
# data/collate.py

import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Pad all modalities to max length in batch. Return padding masks.

    Returns dict with keys:
        text_token_ids:     LongTensor  (B, n_max)
        text_timestamps:    FloatTensor (B, n_max, 2)
        text_padding_mask:  BoolTensor  (B, n_max)        — True = PAD position
        mfcc:               FloatTensor (B, T_m_max, 13, 3)
        mfcc_timestamps:    FloatTensor (B, T_m_max)
        mfcc_padding_mask:  BoolTensor  (B, T_m_max)      — True = PAD position
        landmarks:          FloatTensor (B, T_l_max, 136, 3)
        landmark_timestamps:FloatTensor (B, T_l_max)
        landmark_padding_mask: BoolTensor (B, T_l_max)    — True = PAD position
        frame_mask:         BoolTensor  (B, T_l_max)      — True = VALID speaker frame
        label:              FloatTensor (B,)
        sample_ids:         list of str
    """
    # For each key:
    # 1. Collect from all samples in batch.
    # 2. Determine max length along the sequence dim (dim 0 of each tensor).
    # 3. Pad with zeros to max length.
    # 4. Create padding mask: True where padded, False where real data.
    #
    # For frame_mask: pad with False (padded positions are not valid frames).
    #
    # IMPORTANT: The text_padding_mask will later be extended by 1 position
    # for the [CLS] token in the classifier. That happens in the model forward,
    # not here.
    #
    # Use torch.nn.utils.rnn.pad_sequence where convenient.
    # pad_sequence expects list of (L_i, ...) tensors and pads dim 0.
    pass
```

### 4.3 Sampler for Class Imbalance

```python
# data/sampler.py

from torch.utils.data import WeightedRandomSampler

def make_weighted_sampler(labels):
    """
    Args:
        labels: list of int (0 or 1) for each sample in dataset.
    Returns:
        WeightedRandomSampler that oversamples the minority class so each epoch
        draws approximately 50/50 class balance.
    """
    class_counts = [labels.count(0), labels.count(1)]
    sample_weights = [1.0 / class_counts[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )
```

### 4.4 DataLoader Assembly

```python
# In train.py or a data utility:

from torch.utils.data import DataLoader

dataset = DeceptionDataset(manifest_csv, feature_dir, augment=True)
sampler = make_weighted_sampler(dataset.get_labels())  # dataset should expose labels list
loader = DataLoader(
    dataset,
    batch_size=config.batch_size,   # 16-32 (small dataset, keep it reasonable)
    sampler=sampler,                # oversamples minority class
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
    drop_last=True,                 # avoid batch-size-1 edge cases
)
```

---

## 5. Model Components

### 5.1 Config

```python
# config.py

from dataclasses import dataclass

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

    # Positional encoding
    pe_dim: int = 64                # dimension of sinusoidal timestamp PE
                                    # Must be <= min(d_text, d_mfcc, d_landmark)
                                    # Added to embeddings, so d_text etc. must be >= pe_dim
                                    # Actually: PE is added, so pe_dim must equal d_query and d_kv respectively.
                                    # Better approach: PE is a separate additive vector of same dim as embedding.
                                    # See section 5.3 for implementation.

    # Classifier
    d_fused: int = 384 + 256 + 256  # d_text + d_mfcc + d_landmark = 896
    n_clf_layers: int = 3           # transformer encoder layers
    clf_heads: int = 8              # attention heads in classifier
    clf_ff_mult: int = 4            # feedforward multiplier
    dropout: float = 0.3            # classifier dropout

    # Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 200
    max_epochs: int = 100
    patience: int = 15              # early stopping
    grad_clip: float = 1.0
    pos_weight: float = 2.33        # BCE pos_weight — compute from actual class ratio

    # Cross-validation
    n_folds: int = 5
    seed: int = 42
```

### 5.2 Text Encoder (Frozen)

```python
# models/encoders/text_encoder.py

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class TextEncoder(nn.Module):
    """
    Wraps a pretrained sentence transformer to produce per-token contextual embeddings.

    IMPORTANT: SentenceTransformer by default produces a single sentence vector.
    We need per-token outputs. Access the underlying transformer model's
    token-level hidden states.

    The internal model is typically:
        SentenceTransformer → model[0] (Transformer) → model[1] (Pooling)
    We skip the pooling layer and use the token-level outputs from model[0].

    Alternative: use HuggingFace transformers directly:
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        outputs = model(input_ids=token_ids, attention_mask=mask)
        token_embeddings = outputs.last_hidden_state  # (B, n, 384)

    This is probably cleaner. Use the HuggingFace approach.

    All parameters are frozen. No gradients.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, token_ids, attention_mask):
        """
        Args:
            token_ids:      LongTensor (B, n)
            attention_mask:  BoolTensor (B, n) — False where padded
        Returns:
            FloatTensor (B, n, d_text=384) — per-token embeddings
        """
        outputs = self.model(input_ids=token_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


# NOTE ON TOKENIZER ALIGNMENT:
# Whisper produces words. The sentence transformer uses its own tokenizer
# which may sub-word tokenize differently. Two approaches:
#
# Option A (simpler): Use Whisper's transcription as raw text.
#   Tokenize with the sentence transformer's tokenizer at feature-extraction time.
#   Each sub-word token inherits the parent word's (t_start, t_end).
#   Store these token_ids and timestamps in text.pt.
#
# Option B: Use Whisper's own encoder hidden states instead of a separate text model.
#   This avoids the tokenizer mismatch entirely. Whisper encoder outputs
#   are (n_audio_frames, d_whisper). You'd need a projection layer.
#   Worth experimenting with later.
#
# START WITH OPTION A.
```

### 5.3 Continuous Sinusoidal Positional Encoding

```python
# models/positional_encoding.py

import torch
import torch.nn as nn
import math

class ContinuousPositionalEncoding(nn.Module):
    """
    Produces sinusoidal positional encodings from real-valued timestamps.

    Unlike standard transformer PE which uses integer positions,
    this takes arbitrary float timestamps (in seconds) and produces
    a d_model-dimensional encoding vector per timestamp.

    PE(t, 2i)   = sin(t / omega_i)
    PE(t, 2i+1) = cos(t / omega_i)

    where omega_i = 10000^(2i / d_model), following the original
    Transformer paper but with continuous t instead of integer pos.

    This is ADDED to the embeddings (not concatenated), so d_pe must equal
    d_model of the embeddings it's added to.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # Precompute the inverse frequencies
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)  # (d_model // 2,)

    def forward(self, timestamps):
        """
        Args:
            timestamps: FloatTensor of any shape (...), values in seconds.
        Returns:
            FloatTensor (..., d_model) — positional encoding vectors.
        """
        # timestamps: (...) → (..., 1)
        t = timestamps.unsqueeze(-1)  # (..., 1)
        # inv_freq: (d_model // 2,) → broadcast to (..., d_model // 2)
        sinusoid = t * self.inv_freq  # (..., d_model // 2)
        pe = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)  # (..., d_model)
        return pe


# USAGE IN CROSS-ATTENTION:
# For text queries (timestamps are intervals):
#   text_midpoints = (text_timestamps[:, :, 0] + text_timestamps[:, :, 1]) / 2  # (B, n)
#   text_pe = pe_encoder_text(text_midpoints)  # (B, n, d_text)
#   text_emb_with_pe = text_emb + text_pe
#
# For MFCC keys (timestamps are scalars — frame centers):
#   mfcc_pe = pe_encoder_mfcc(mfcc_timestamps)  # (B, T_m, d_mfcc)
#   mfcc_emb_with_pe = mfcc_emb + mfcc_pe
#
# IMPORTANT: Each modality gets its OWN PE instance with d_model matching
# that modality's embedding dimension. The cross-attention computes
# Q·K^T where Q has text PE and K has modality PE, so temporal proximity
# naturally increases attention weight.
```

### 5.4 MFCC Encoder

```python
# models/encoders/mfcc_encoder.py

import torch
import torch.nn as nn

class MFCCEncoder(nn.Module):
    """
    Encodes raw MFCC features (with delta and delta-delta) into a learned
    representation suitable for cross-attention alignment.

    Input:  (B, T_m, n_coefficients, 3)  — 3 = [static, delta, delta-delta]
    Output: (B, T_m, d_out)

    Architecture: Flatten per-frame features, then 1D conv stack along time axis
    to learn local temporal patterns (prosodic contours, intonation shifts).
    Residual connections for gradient flow.
    """

    def __init__(self, n_coefficients: int = 13, d_out: int = 256):
        super().__init__()
        d_in = n_coefficients * 3  # 39

        self.input_proj = nn.Linear(d_in, d_out)

        # 1D conv stack: operates on time axis, learns local patterns
        # kernel_size=5 covers ~50ms at 10ms stride — about 2 phonemes
        self.conv_stack = nn.Sequential(
            nn.Conv1d(d_out, d_out, kernel_size=5, padding=2, groups=1),
            nn.BatchNorm1d(d_out),
            nn.GELU(),
            nn.Conv1d(d_out, d_out, kernel_size=5, padding=2, groups=1),
            nn.BatchNorm1d(d_out),
            nn.GELU(),
            nn.Conv1d(d_out, d_out, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm1d(d_out),
            nn.GELU(),
        )
        # Residual projection if dims don't match (they do here, but
        # include for safety if d_in != d_out in future changes)
        self.residual = nn.Identity()

    def forward(self, mfcc, padding_mask=None):
        """
        Args:
            mfcc: (B, T_m, n_coefficients, 3)
            padding_mask: (B, T_m) BoolTensor, True = padded (optional)
        Returns:
            (B, T_m, d_out)
        """
        B, T, D, C = mfcc.shape
        x = mfcc.reshape(B, T, D * C)         # (B, T, 39)
        x = self.input_proj(x)                 # (B, T, d_out)

        residual = self.residual(x)

        x = x.permute(0, 2, 1)                # (B, d_out, T) for Conv1d
        x = self.conv_stack(x)
        x = x.permute(0, 2, 1)                # (B, T, d_out)

        x = x + residual                       # residual connection
        return x
```

### 5.5 Landmark Encoder

```python
# models/encoders/landmark_encoder.py

import torch
import torch.nn as nn

class LandmarkEncoder(nn.Module):
    """
    Encodes facial landmark coordinates (with delta and delta-delta) into a
    learned representation.

    Input:  (B, T_l, n_landmarks * coords, 3)  — 3 = [static, delta, delta-delta]
            e.g., (B, T_l, 136, 3) for 68 landmarks × 2 coords
    Output: (B, T_l, d_out)

    Architecture:
    1. Per-frame MLP: abstracts raw coordinates into facial behavior features
       (e.g., eyebrow raise magnitude, lip compression, gaze shift patterns).
    2. Temporal 1D conv: captures movement patterns across frames.
    """

    def __init__(self, n_landmarks: int = 68, coords: int = 2, d_out: int = 256):
        super().__init__()
        d_in = n_landmarks * coords * 3  # 68 * 2 * 3 = 408

        # Per-frame MLP
        self.frame_mlp = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, d_out),
            nn.GELU(),
        )

        # Temporal context via 1D conv
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_out, d_out, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_out),
            nn.GELU(),
            nn.Conv1d(d_out, d_out, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_out),
            nn.GELU(),
        )

        self.residual = nn.Identity()

    def forward(self, landmarks, padding_mask=None):
        """
        Args:
            landmarks: (B, T_l, n_landmarks * coords, 3)
            padding_mask: (B, T_l) BoolTensor, True = padded (optional)
        Returns:
            (B, T_l, d_out)
        """
        B, T, S, C = landmarks.shape
        x = landmarks.reshape(B, T, S * C)     # (B, T, 408)
        x = self.frame_mlp(x)                   # (B, T, d_out)

        residual = self.residual(x)

        x = x.permute(0, 2, 1)                  # (B, d_out, T)
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1)                  # (B, T, d_out)

        x = x + residual
        return x
```

### 5.6 Cross-Attention Modality Aligner

```python
# models/cross_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.positional_encoding import ContinuousPositionalEncoding

class ModalityAligner(nn.Module):
    """
    Cross-attention module that compresses a variable-length modality sequence
    (MFCC or landmarks) down to n vectors aligned with text tokens.

    Text embeddings are QUERIES. Modality embeddings are KEYS and VALUES.
    Output has shape (B, n, d_kv) — one vector per text token, in the
    modality's feature space.

    Continuous sinusoidal positional encodings based on real-valued timestamps
    are added to both queries and keys so the attention naturally learns
    temporal correspondence.

    For landmarks: a frame_mask is applied so blacked-out frames are never
    attended to (attention weight = 0, no gradient flow).
    """

    def __init__(self, d_query: int, d_kv: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_kv // n_heads
        assert d_kv % n_heads == 0, f"d_kv ({d_kv}) must be divisible by n_heads ({n_heads})"

        # Project queries from text space to modality space
        self.q_proj = nn.Linear(d_query, d_kv)
        self.k_proj = nn.Linear(d_kv, d_kv)
        self.v_proj = nn.Linear(d_kv, d_kv)
        self.out_proj = nn.Linear(d_kv, d_kv)

        # Separate PE for query and key (different d_model if d_query != d_kv)
        # PE is added BEFORE projection for queries (in text space)
        # and BEFORE projection for keys (in modality space)
        self.pe_query = ContinuousPositionalEncoding(d_query)
        self.pe_key = ContinuousPositionalEncoding(d_kv)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(self, text_emb, text_timestamps, modality_emb, modality_timestamps,
                modality_mask=None, text_padding_mask=None, modality_padding_mask=None):
        """
        Args:
            text_emb:            (B, n, d_query) — text encoder output
            text_timestamps:     (B, n, 2) — start, end per token
            modality_emb:        (B, T, d_kv) — MFCC or landmark encoder output
            modality_timestamps: (B, T) — center time per frame
            modality_mask:       (B, T) BoolTensor, True = VALID frame (optional)
                                 Used for blacked-out landmark frames.
                                 None for MFCC (all frames valid unless padded).
            text_padding_mask:    (B, n) BoolTensor, True = PAD (optional)
            modality_padding_mask:(B, T) BoolTensor, True = PAD (optional)

        Returns:
            (B, n, d_kv) — aligned modality representation, one per text token
        """
        B, n, _ = text_emb.shape
        T = modality_emb.shape[1]

        # Add continuous positional encoding
        text_midpoints = (text_timestamps[:, :, 0] + text_timestamps[:, :, 1]) / 2.0  # (B, n)
        text_with_pe = text_emb + self.pe_query(text_midpoints)       # (B, n, d_query)
        mod_with_pe = modality_emb + self.pe_key(modality_timestamps)  # (B, T, d_kv)

        # Project to Q, K, V
        Q = self.q_proj(text_with_pe)   # (B, n, d_kv)
        K = self.k_proj(mod_with_pe)    # (B, T, d_kv)
        V = self.v_proj(modality_emb)   # (B, T, d_kv) — NOTE: V from raw, not PE-added

        # Reshape for multi-head attention
        Q = Q.view(B, n, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, n, d_head)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, T, d_head)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, T, d_head)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, h, n, T)

        # Apply masks to keys:
        # 1. Padding mask: padded modality positions should not be attended to
        # 2. Frame mask: blacked-out landmark frames should not be attended to
        # Combine into a single mask: position is invalid if padded OR blacked-out
        key_mask = torch.zeros(B, T, dtype=torch.bool, device=attn_scores.device)
        if modality_padding_mask is not None:
            key_mask = key_mask | modality_padding_mask       # True = invalid
        if modality_mask is not None:
            key_mask = key_mask | (~modality_mask)            # frame_mask True=valid, invert

        if key_mask.any():
            # Expand to (B, 1, 1, T) for broadcasting with (B, h, n, T)
            attn_scores = attn_scores.masked_fill(key_mask[:, None, None, :], float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, h, n, T)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        out = torch.matmul(attn_weights, V)  # (B, h, n, d_head)
        out = out.transpose(1, 2).contiguous().view(B, n, -1)  # (B, n, d_kv)
        out = self.out_proj(out)

        return out
```

### 5.7 Classifier Head

```python
# models/classifier.py

import torch
import torch.nn as nn
import math

class BiDirTransformerClassifier(nn.Module):
    """
    Bidirectional transformer encoder followed by [CLS] token pooling
    and a dense classification head.

    Input:  fused multimodal sequence (B, n, d_fused) where d_fused = d_text + d_mfcc + d_landmark
    Output: logits (B,) — raw logit for binary classification (sigmoid applied in loss)
    """

    def __init__(self, d_fused: int, n_layers: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.3):
        super().__init__()

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_fused) * 0.02)

        # Standard transformer encoder (bidirectional by default — no causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_fused,
            nhead=n_heads,
            dim_feedforward=d_fused * ff_mult,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,          # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_fused),
            nn.Dropout(dropout),
            nn.Linear(d_fused, 1),    # single logit for binary classification
        )

    def forward(self, fused, padding_mask=None):
        """
        Args:
            fused:         (B, n, d_fused) — concatenated multimodal embeddings
            padding_mask:  (B, n) BoolTensor, True = PAD position

        Returns:
            logits: (B,) — raw logit, apply sigmoid + BCE loss externally
        """
        B = fused.shape[0]

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, d_fused)
        x = torch.cat([cls, fused], dim=1)                 # (B, n+1, d_fused)

        # Extend padding mask for [CLS] (always valid)
        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=fused.device)
            full_mask = torch.cat([cls_mask, padding_mask], dim=1)  # (B, n+1)
        else:
            full_mask = None

        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=full_mask)

        # Extract [CLS] output
        cls_out = x[:, 0, :]                               # (B, d_fused)
        logits = self.head(cls_out).squeeze(-1)             # (B,)

        return logits
```

### 5.8 Full Model

```python
# models/full_model.py

import torch
import torch.nn as nn
from models.encoders.text_encoder import TextEncoder
from models.encoders.mfcc_encoder import MFCCEncoder
from models.encoders.landmark_encoder import LandmarkEncoder
from models.cross_attention import ModalityAligner
from models.classifier import BiDirTransformerClassifier

class MultimodalDeceptionModel(nn.Module):
    """
    Full multimodal deception detection model.

    Pipeline:
    1. Encode each modality independently.
    2. Align MFCC and landmarks to text-token resolution via cross-attention.
    3. Concatenate aligned representations along feature dimension.
    4. Classify with bidirectional transformer + [CLS] pooling + dense head.

    Loss: Binary cross-entropy with logits (pos_weight for class imbalance).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # --- Encoders ---
        self.text_encoder = TextEncoder(config.text_model_name)
        # Frozen — no gradients. Already handled inside TextEncoder.

        self.mfcc_encoder = MFCCEncoder(
            n_coefficients=config.n_mfcc_coefficients,
            d_out=config.d_mfcc
        )
        self.landmark_encoder = LandmarkEncoder(
            n_landmarks=config.n_landmarks,
            coords=config.landmark_coords,
            d_out=config.d_landmark
        )

        # --- Cross-Attention Aligners ---
        self.mfcc_aligner = ModalityAligner(
            d_query=config.d_text,
            d_kv=config.d_mfcc,
            n_heads=config.cross_attn_heads,
            dropout=config.cross_attn_dropout,
        )
        self.landmark_aligner = ModalityAligner(
            d_query=config.d_text,
            d_kv=config.d_landmark,
            n_heads=config.cross_attn_heads,
            dropout=config.cross_attn_dropout,
        )

        # --- Classifier ---
        self.classifier = BiDirTransformerClassifier(
            d_fused=config.d_text + config.d_mfcc + config.d_landmark,  # 896
            n_layers=config.n_clf_layers,
            n_heads=config.clf_heads,
            ff_mult=config.clf_ff_mult,
            dropout=config.dropout,
        )

    def forward(self, batch):
        """
        Args:
            batch: dict from collate_fn with all modality tensors and masks.

        Returns:
            logits: (B,) — raw logits for BCE loss
        """
        # 1. Text encoding (frozen, no grad)
        text_emb = self.text_encoder(
            batch["text_token_ids"],
            attention_mask=~batch["text_padding_mask"]  # HF expects True=valid
        )  # (B, n, d_text)

        # 2. MFCC encoding
        mfcc_emb = self.mfcc_encoder(
            batch["mfcc"],
            padding_mask=batch["mfcc_padding_mask"]
        )  # (B, T_m, d_mfcc)

        # 3. Landmark encoding
        land_emb = self.landmark_encoder(
            batch["landmarks"],
            padding_mask=batch["landmark_padding_mask"]
        )  # (B, T_l, d_landmark)

        # 4. Cross-attend MFCC to text resolution
        mfcc_aligned = self.mfcc_aligner(
            text_emb=text_emb,
            text_timestamps=batch["text_timestamps"],
            modality_emb=mfcc_emb,
            modality_timestamps=batch["mfcc_timestamps"],
            modality_mask=None,                           # all MFCC frames valid
            text_padding_mask=batch["text_padding_mask"],
            modality_padding_mask=batch["mfcc_padding_mask"],
        )  # (B, n, d_mfcc)

        # 5. Cross-attend landmarks to text resolution
        land_aligned = self.landmark_aligner(
            text_emb=text_emb,
            text_timestamps=batch["text_timestamps"],
            modality_emb=land_emb,
            modality_timestamps=batch["landmark_timestamps"],
            modality_mask=batch["frame_mask"],            # blacked-out frames masked
            text_padding_mask=batch["text_padding_mask"],
            modality_padding_mask=batch["landmark_padding_mask"],
        )  # (B, n, d_landmark)

        # 6. Concatenate along feature dimension (NOT addition)
        fused = torch.cat([text_emb, mfcc_aligned, land_aligned], dim=-1)
        # (B, n, d_text + d_mfcc + d_landmark) = (B, n, 896)

        # 7. Classify
        logits = self.classifier(
            fused,
            padding_mask=batch["text_padding_mask"]
        )  # (B,)

        return logits
```

---

## 6. Training

### 6.1 Training Loop

```python
# train.py

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np

def train_one_epoch(model, loader, optimizer, scheduler, pos_weight, device, grad_clip):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        logits = model(batch)  # (B,)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            batch["label"],
            pos_weight=pos_weight
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * logits.size(0)
        preds = torch.sigmoid(logits).detach().cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch["label"].cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, pos_weight, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        logits = model(batch)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            batch["label"],
            pos_weight=pos_weight
        )

        total_loss += loss.item() * logits.size(0)
        preds = torch.sigmoid(logits).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch["label"].cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    binary_preds = [1 if p > 0.5 else 0 for p in all_preds]

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, binary_preds),
        "f1": f1_score(all_labels, binary_preds),
        "auc_roc": roc_auc_score(all_labels, all_preds),
    }
    return metrics
```

### 6.2 Cross-Validation Harness

```python
def run_cross_validation(config, manifest_csv, feature_dir):
    """
    5-fold stratified cross-validation.
    Reports mean ± std of accuracy, F1, AUC-ROC.
    """
    dataset = DeceptionDataset(manifest_csv, feature_dir, augment=False)
    labels = dataset.get_labels()  # list of int
    sample_ids = list(range(len(dataset)))

    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(sample_ids, labels)):
        print(f"\n=== Fold {fold + 1}/{config.n_folds} ===")

        # Create fold-specific datasets
        train_dataset = DeceptionDataset(manifest_csv, feature_dir, augment=True)
        train_dataset.samples = [train_dataset.samples[i] for i in train_idx]
        val_dataset = DeceptionDataset(manifest_csv, feature_dir, augment=False)
        val_dataset.samples = [val_dataset.samples[i] for i in val_idx]

        # Compute class weight from training fold
        train_labels = [labels[i] for i in train_idx]
        n_neg = train_labels.count(0)
        n_pos = train_labels.count(1)
        pos_weight = torch.tensor([n_neg / n_pos], device=config.device)

        # Sampler for training
        train_sampler = make_weighted_sampler(train_labels)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                  sampler=train_sampler, collate_fn=collate_fn,
                                  num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                                shuffle=False, collate_fn=collate_fn,
                                num_workers=4, pin_memory=True)

        # Init model, optimizer, scheduler
        model = MultimodalDeceptionModel(config).to(config.device)
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        total_steps = config.max_epochs * len(train_loader)
        scheduler = OneCycleLR(optimizer, max_lr=config.learning_rate,
                               total_steps=total_steps, pct_start=0.1)

        # Training with early stopping
        best_val_auc = 0.0
        patience_counter = 0

        for epoch in range(config.max_epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler, pos_weight,
                config.device, config.grad_clip
            )
            val_metrics = evaluate(model, val_loader, pos_weight, config.device)

            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                  f"| val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} "
                  f"val_f1={val_metrics['f1']:.3f} val_auc={val_metrics['auc_roc']:.3f}")

            if val_metrics["auc_roc"] > best_val_auc:
                best_val_auc = val_metrics["auc_roc"]
                best_metrics = val_metrics
                patience_counter = 0
                # Save best model checkpoint for this fold
                torch.save(model.state_dict(), f"checkpoints/fold_{fold}_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        fold_metrics.append(best_metrics)

    # Aggregate results
    print("\n=== Cross-Validation Results ===")
    for metric in ["accuracy", "f1", "auc_roc"]:
        values = [m[metric] for m in fold_metrics]
        print(f"{metric}: {np.mean(values):.3f} ± {np.std(values):.3f}")

    return fold_metrics
```

---

## 7. File Structure

```
deception_detection/
├── config.py                           # ModelConfig dataclass (section 5.1)
├── data/
│   ├── __init__.py
│   ├── dataset.py                      # DeceptionDataset (section 4.1)
│   ├── collate.py                      # collate_fn (section 4.2)
│   ├── sampler.py                      # make_weighted_sampler (section 4.3)
│   └── preprocessing/
│       ├── __init__.py
│       ├── run_all.py                  # Master script: runs steps 1-4 for all videos
│       ├── video_prep.py              # Step 1: YOLO crop, active speaker, blacking out
│       ├── whisper_extract.py         # Step 2: whisper-timestamped → token_ids + timestamps
│       ├── mfcc_extract.py            # Step 3: spectral sub → MFCC → Δ/ΔΔ → CMVN → save .pt
│       └── landmark_extract.py        # Step 4: FaceXFormer → normalize → Δ/ΔΔ → save .pt
├── models/
│   ├── __init__.py
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── text_encoder.py            # Frozen HuggingFace encoder (section 5.2)
│   │   ├── mfcc_encoder.py            # 1D conv stack (section 5.4)
│   │   └── landmark_encoder.py        # MLP + temporal conv (section 5.5)
│   ├── positional_encoding.py         # ContinuousPositionalEncoding (section 5.3)
│   ├── cross_attention.py             # ModalityAligner (section 5.6)
│   ├── classifier.py                  # BiDirTransformerClassifier (section 5.7)
│   └── full_model.py                  # MultimodalDeceptionModel (section 5.8)
├── train.py                            # Training loop + CV harness (section 6)
├── evaluate.py                         # Standalone evaluation script
├── requirements.txt                    # See section 8
└── checkpoints/                        # Saved model weights per fold
```

---

## 8. Dependencies

```
# requirements.txt

torch>=2.0
torchaudio
torchvision
transformers
sentence-transformers
whisper-timestamped        # or stable-ts
librosa                    # MFCC extraction, spectral processing
numpy
scipy
scikit-learn               # StratifiedKFold, metrics
pandas                     # manifest CSV handling
ultralytics                # YOLO
huggingface_hub            # FaceXFormer download
opencv-python              # video frame extraction
tqdm
tensorboard                # optional: logging
```

---

## 9. Preprocessing Implementation Notes

### 9.1 video_prep.py

```python
"""
For each video:
1. Load video, extract all frames.
2. Run YOLO face detection on each frame. Store bounding boxes.
3. Track faces across frames using IoU-based matching.
   - For each detected face, maintain a track (list of frame indices).
   - Active speaker = track with the most frames.
4. For each frame:
   - If the active speaker's face is detected: crop to bounding box, resize to 224x224.
   - Else: output a 224x224 black frame. Set frame_mask[i] = 0.
5. Save:
   - processed_video.mp4 (or just save frames as a tensor .pt)
   - frame_mask.npy
   - audio.wav (extracted from original via ffmpeg/torchaudio)
"""
```

### 9.2 whisper_extract.py

```python
"""
For each audio.wav:
1. Run whisper-timestamped: get word-level (word, t_start, t_end) list.
2. Concatenate words into a single string.
3. Tokenize with the text encoder's tokenizer (AutoTokenizer from sentence-transformers/all-MiniLM-L6-v2).
4. Align sub-word tokens back to words:
   - Use tokenizer's word_ids() or offset_mapping to find which word each token belongs to.
   - Each token inherits the parent word's (t_start, t_end).
5. Save text.pt:
   {
     "token_ids": LongTensor(n,),
     "timestamps": FloatTensor(n, 2)
   }

EDGE CASES:
- [CLS] and [SEP] special tokens: exclude from the sequence or assign
  timestamps of (0.0, 0.0) and (end_time, end_time) respectively.
  Recommendation: exclude them. The text encoder's output includes them
  but we can slice them out after encoding.
- Empty transcription (silence): token_ids is empty. Handle in collate
  (the sample contributes only MFCC and landmark modalities, text is all-pad).
"""
```

### 9.3 mfcc_extract.py

```python
"""
For each audio.wav:
1. Load audio at 16kHz mono (librosa or torchaudio).
2. Spectral subtraction:
   - Estimate noise power spectrum from first 0.5s (or detect silence).
   - Subtract from each STFT frame's power spectrum.
   - Clip negative values to small epsilon.
   - Reconstruct cleaned signal via inverse STFT (or apply directly in MFCC pipeline).
   - NOTE: Alternatively, apply spectral subtraction within the MFCC pipeline
     by modifying the power spectrum before mel filterbank application.
3. MFCC computation (use librosa):
   - n_mfcc=13, n_fft=400 (25ms at 16kHz), hop_length=160 (10ms at 16kHz)
   - librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13, n_fft=400, hop_length=160)
   - Output: (13, T_m)
4. Delta and delta-delta:
   - librosa.feature.delta(mfcc, order=1)  → (13, T_m)
   - librosa.feature.delta(mfcc, order=2)  → (13, T_m)
   - Stack: (13, T_m, 3) → transpose to (T_m, 13, 3)
5. CMVN (per-utterance):
   - For each of the 39 features (13×3): subtract mean, divide by std across T_m.
6. Timestamps:
   - mfcc_timestamps[i] = (i * hop_length + n_fft / 2) / sr
   - This gives the center time of each MFCC frame in seconds.
7. Save mfcc.pt:
   {
     "mfcc": FloatTensor(T_m, 13, 3),
     "timestamps": FloatTensor(T_m,)
   }
"""
```

### 9.4 landmark_extract.py

```python
"""
For each processed video (cropped frames):
1. Load FaceXFormer model.
2. For each frame:
   - If frame_mask[i] == 0 (blacked out): output zeros (136,) for landmarks.
   - Else: run FaceXFormer, extract 68 landmark (x, y) coordinates → (136,).
3. Normalize per-video:
   - For valid frames only (frame_mask == 1):
     Per coordinate, compute mean and std across all valid frames.
     z-score normalize: (x - mean) / std.
   - Blacked-out frames remain zero.
4. Delta and delta-delta:
   - Compute np.gradient along frame axis for each coordinate.
   - First derivative (velocity of facial movement).
   - Second derivative (acceleration — micro-expression indicator).
   - Stack → (T_l, 136, 3).
   - For blacked-out frames: delta/delta-delta will be influenced by
     neighboring frames. Set delta/delta-delta to zero for blacked-out frames
     AND for frames immediately adjacent to blacked-out frames (boundary artifacts).
5. Timestamps:
   - landmark_timestamps[i] = i / fps
6. Save landmarks.pt:
   {
     "landmarks": FloatTensor(T_l, 136, 3),
     "timestamps": FloatTensor(T_l,),
     "frame_mask": BoolTensor(T_l,)
   }
"""
```

---

## 10. Key Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Fusion method | Concatenation (not addition) | Preserves modality-specific features; allows classifier to detect cross-modal mismatches (key deception signal) |
| Contrastive learning | Removed (for now) | Simplifies training; can be added back as auxiliary loss later |
| Temporal alignment | Cross-attention with continuous sinusoidal PE | Soft, differentiable selection; respects real-valued timestamps; no hard frame binning |
| Text encoder | Frozen `all-MiniLM-L6-v2` | Small, fast, good token-level embeddings; no need for generative capability |
| MFCC encoder | 1D conv stack with residual | Captures local temporal patterns (prosody); lightweight |
| Landmark encoder | Per-frame MLP + temporal conv | Abstracts raw coordinates to behavioral features; temporal conv for movement patterns |
| Blacked-out frame handling | Additive -inf mask in cross-attention | Zero attention weight → no gradient flow through invalid frames |
| Classification head | BiDir Transformer + [CLS] pooling | Learns cross-modal interactions; [CLS] provides clean aggregation point |
| Loss | BCE with pos_weight | Directly optimizes binary task; pos_weight handles class imbalance |
| Evaluation | 5-fold stratified CV | Small dataset requires robust evaluation; single split is unreliable |
| Augmentation constraints | No geometric/temporal transforms | Head orientation and temporal dynamics are deception signals |
| Face orientation | Not normalized | Deliberate — head rotation during speech is a potential deception indicator |
| Oversampling | WeightedRandomSampler | Combined with augmentation, minority class gets diverse repeated views |

---

## 11. Future Extensions (Post-Prototype)

- **Auxiliary CL loss**: Add back InfoNCE as a regularizer with lambda annealing (section from prior discussion). The architecture cleanly supports this by adding projection heads that branch off the aligned representations.
- **Whisper encoder hidden states**: Use Whisper's internal audio representations instead of (or alongside) the separate text encoder, avoiding tokenizer mismatch.
- **LoRA on text encoder**: If fine-tuning helps, use low-rank adaptation instead of full unfreezing.
- **Memory bank for CL**: MoCo-style momentum encoder to increase effective batch size for InfoNCE (critical given small dataset).
- **Modality dropout**: During training, randomly zero out entire modalities (e.g., 10% chance of dropping landmarks entirely). Forces the model to not over-rely on any single modality. Also makes inference more robust to missing data.
- **Attention visualization**: Extract and plot cross-attention weights to interpret which audio/visual frames the model attends to for each text token. Critical for interpretability in a deception detection context.
