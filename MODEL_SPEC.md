# Model Specification — Bimodal Deception Detection

Adapted from the DOLOS paper architecture
([NMS05/Audio-Visual-Deception-Detection-DOLOS-Dataset-and-Parameter-Efficient-Crossmodal-Learning](https://github.com/NMS05/Audio-Visual-Deception-Detection-DOLOS-Dataset-and-Parameter-Efficient-Crossmodal-Learning)).

---

## Dataset

| Property | Value |
|---|---|
| Source | DOLOS dataset |
| Manifest | `Data/manifest_dolos.csv` |
| Samples | 1 435 |
| Label 0 (deceptive) | 769 |
| Label 1 (truthful) | 666 |
| Class ratio | ~1.15 : 1 |
| Features root | `features/{sample_id}/` |
| Per-sample files | `audio.wav` (16 kHz mono), `video.mp4` (224×224 YOLO face-cropped) |

---

## Preprocessing

Single step: `video_prep.py` via `run_all.py`.

- YOLO face detection → crop to bounding box → resize to 224×224 → write `video.mp4`
- FFmpeg audio extraction → 16 kHz mono WAV → write `audio.wav`

No further feature extraction. The model reads raw waveforms and video frames directly.

---

## Architecture

```
audio.wav (16 kHz mono)
  └─ W2V2_Model ──────────────────────────────── audio_emb  (B, 768)

video.mp4 (224×224 face-cropped)
  └─ ViT_Model ───────────────────────────────── visual_emb (B, 768)

audio_emb + visual_emb
  └─ CrossFusionModule ───────────────────────── fused      (B, 512)
       └─ classifier head ─────────────────────── logit     (B,)
```

### Audio encoder — `W2V2_Model` (`models/audio_model.py`)

| Layer | Detail |
|---|---|
| Backbone | `facebook/wav2vec2-base` (HuggingFace) |
| CNN feature extractor | Frozen |
| Feature projection + layer norm | Frozen |
| Transformer encoder | 12 layers total; last 2 trainable |
| Pooling | Mean over time frames |
| Output | `(B, 768)` |

Input: raw waveform `(B, T)` at 16 kHz, optionally with padding mask `(B, T)`.

### Visual encoder — `ViT_Model` (`models/visual_model.py`)

| Layer | Detail |
|---|---|
| Frame sampling | 64 frames uniformly sampled per video |
| Per-frame CNN (`CNN_Face`) | 3-stage conv (64→128→256 channels), MaxPool×2, AdaptiveAvgPool → `(B*64, 256)` |
| Projection | `Linear(256 → 768)` |
| Positional embeddings | Learnable `(1, 64, 768)`, init trunc-normal σ=0.02 |
| ViT encoder | Last 4 layers of `google/vit-base-patch16-224`; last 2 trainable |
| Layer norm | `LayerNorm(768)` |
| Pooling | Mean over 64 frame tokens |
| Output | `(B, 768)` |

Input: `(B, 64, 3, 224, 224)` ImageNet-normalised frames.

`CNN_Face` detail:
```
Stage 1: Conv(3→64)×2 + BN + ReLU, MaxPool2d(2)   224→112
Stage 2: Conv(64→128)×2 + BN + ReLU, MaxPool2d(2)  112→56
Stage 3: Conv(128→256)×2 + BN + ReLU, AdaptiveAvgPool2d(1)  → (N, 256)
```

### Fusion — `CrossFusionModule` (`models/fusion_model.py`)

Plug-in Audio-Visual Fusion (PAVF) from DOLOS:

```
a = Linear(768 → 256)(audio_emb)   # (B, 256)
v = Linear(768 → 256)(visual_emb)  # (B, 256)

corr   = softmax(a @ v.T / √256)   # (B, B)  audio→visual attention
corr_T = softmax(v @ a.T / √256)   # (B, B)  visual→audio attention

a_fused = a + corr   @ v           # (B, 256)
v_fused = v + corr_T @ a           # (B, 256)

fused = Linear(512 → 512)(cat[a_fused, v_fused])  # (B, 512)
```

### Classifier head (`FusionModel`)

```
LayerNorm(512)
Linear(512 → 128) + GELU
Dropout(0.4)
Linear(128 → 1) → squeeze → logit (B,)
```

---

## Trainable parameters

| Component | Frozen | Trainable |
|---|---|---|
| Wav2Vec2 CNN extractor | Yes | — |
| Wav2Vec2 feature projection | Yes | — |
| Wav2Vec2 transformer layers 0–9 | Yes | — |
| Wav2Vec2 transformer layers 10–11 | — | Yes |
| CNN_Face (3-stage) | — | Yes |
| ViT projection + pos embeddings | — | Yes |
| ViT encoder layers 0–1 (of 4 used) | Yes | — |
| ViT encoder layers 2–3 (of 4 used) | — | Yes |
| CrossFusionModule | — | Yes |
| Classifier head | — | Yes |

---

## Training

| Hyperparameter | Value |
|---|---|
| Cross-validation | 8-fold stratified |
| Batch size | 8 |
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Weight decay | 1e-2 |
| LR schedule | OneCycleLR (10% warmup) |
| Max epochs | 200 |
| Early stopping patience | 50 epochs (on val AUC-ROC) |
| Gradient clipping | 1.0 |
| Class balancing | `WeightedRandomSampler` (50/50 per epoch) |
| Loss | `BCEWithLogitsLoss` (pos_weight=1.0; sampler handles balance) |

Evaluation metrics: AUC-ROC (primary), F1, accuracy.

### Data augmentation (training only)

- **Audio**: additive Gaussian noise, σ ~ Uniform(0, 0.005)
- **Frames**: additive pixel noise (post-normalisation), σ ~ Uniform(0, 0.02)

---

## Inference

```python
from deception_detection.config import ModelConfig
from deception_detection.models import FusionModel

config = ModelConfig()
model = FusionModel(config)
model.load_state_dict(torch.load("checkpoints/fold_0_best.pt"))
model.eval()

logit = model(batch)           # batch: {waveform, frames, waveform_mask}
prob  = torch.sigmoid(logit)   # P(truthful)
```

Or use `evaluate.py`:
```bash
python -m deception_detection.evaluate \
    --checkpoint checkpoints/fold_0_best.pt \
    --manifest Data/manifest_dolos.csv \
    --feature_dir features
```

---

## File map

```
deception_detection/
  config.py                          ModelConfig dataclass
  train.py                           8-fold CV training loop
  evaluate.py                        standalone checkpoint evaluator
  models/
    audio_model.py                   W2V2_Model
    visual_model.py                  ViT_Model, CNN_Face
    fusion_model.py                  FusionModel, CrossFusionModule
  data/
    dataset.py                       DeceptionDataset (waveform + frames)
    collate.py                       collate_fn (pad waveforms, stack frames)
    sampler.py                       make_weighted_sampler
    preprocessing/
      video_prep.py                  YOLO face crop + audio extraction
      run_all.py                     batch preprocessor (step 1 only)
```
