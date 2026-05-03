"""
plot_confusion.py — Load fold checkpoints, run inference, plot confusion matrices
=================================================================================
Run from your project root:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python plot_confusion.py

Reads:  checkpoints/20260429_214636/fold_X_best.pt  (folds 0-4)
        checkpoints/20260429_214636/splits.json      (exact val indices)
        checkpoints/20260429_214636/config.json      (model config)
        features/<clip_id>/audio.wav
        features/<clip_id>/._frames.npz  (preferred, newest)
        features/<clip_id>/frames.npz    (fallback)
        features/<clip_id>/video.mp4     (fallback if no npz)

Saves:  output/plots/4_model_evaluation/confusion_matrix_combined.png
        output/plots/4_model_evaluation/per_fold/fold_X.png
        output/plots/4_model_evaluation/per_fold_metrics_summary.png
        output/cache/model_predictions.csv
"""

import matplotlib
matplotlib.use("Agg")

import os
import sys
import cv2
import json
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, f1_score, accuracy_score
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR     = os.path.join(ROOT, "checkpoints", "20260429_214636")
FEATURE_DIR  = os.path.join(ROOT, "features")
OUTPUT_PLOTS = os.path.join(ROOT, "output", "plots", "4_model_evaluation")
OUTPUT_FOLD  = os.path.join(OUTPUT_PLOTS, "per_fold")
OUTPUT_CACHE = os.path.join(ROOT, "output", "cache")

os.makedirs(OUTPUT_PLOTS, exist_ok=True)
os.makedirs(OUTPUT_FOLD,  exist_ok=True)
os.makedirs(OUTPUT_CACHE, exist_ok=True)

sys.path.insert(0, ROOT)

from deception_detection.config import ModelConfig
from deception_detection.models.fusion_model import FusionModel

sns.set_theme(style="darkgrid", font_scale=1.1)
CLASSES = ["Lie", "Truth"]   # 0=lie, 1=truth

# ── Load config from checkpoint — set ALL fields from training run ─────────────

print("\n━━ Loading config from checkpoint ━━")
with open(os.path.join(CKPT_DIR, "config.json")) as f:
    ckpt_cfg = json.load(f)

config = ModelConfig()

# Core architecture — must match checkpoint exactly
config.visual_backbone       = ckpt_cfg["visual_backbone"]          # "clip" or "arcface"
config.d_audio               = ckpt_cfg.get("d_audio",        768)
config.d_visual              = ckpt_cfg.get("d_visual",       768)
config.d_fused               = ckpt_cfg.get("d_fused",        256)
config.dropout               = ckpt_cfg.get("dropout",        0.6)
config.vit_n_layers          = ckpt_cfg.get("vit_n_layers",   4)
config.vit_n_heads           = ckpt_cfg.get("vit_n_heads",    8)
config.max_frames            = ckpt_cfg.get("max_frames",     256)
config.cnn_chunk_size        = ckpt_cfg.get("cnn_chunk_size", 32)
config.in_channels           = ckpt_cfg.get("in_channels",    1)

# Fusion architecture
config.audio_fusion_layers   = ckpt_cfg.get("audio_fusion_layers",  [4, 8, 12])
config.visual_fusion_layers  = ckpt_cfg.get("visual_fusion_layers", [1, 2, 4])
config.fusion_aggregator     = ckpt_cfg.get("fusion_aggregator",    "weighted_sum")
config.fusion_n_heads        = ckpt_cfg.get("fusion_n_heads",       8)
config.fusion_dropout        = ckpt_cfg.get("fusion_dropout",       0.2)

# UT adapters
config.use_ut_adapters       = ckpt_cfg.get("use_ut_adapters",  True)
config.ut_adapter_dim        = ckpt_cfg.get("ut_adapter_dim",   128)
config.ut_conv_kernel        = ckpt_cfg.get("ut_conv_kernel",   3)

# CV params
config.n_folds               = ckpt_cfg.get("n_folds", 5)
config.seed                  = ckpt_cfg.get("seed",    1919)

print(f"  visual_backbone      : {config.visual_backbone}")
print(f"  max_frames           : {config.max_frames}")
print(f"  audio_fusion_layers  : {config.audio_fusion_layers}")
print(f"  visual_fusion_layers : {config.visual_fusion_layers}")
print(f"  fusion_aggregator    : {config.fusion_aggregator}")
print(f"  n_folds / seed       : {config.n_folds} / {config.seed}")

# ── Device ────────────────────────────────────────────────────────────────────

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"\nRunning inference on: {device}")
torch.set_grad_enabled(False)

# ── Load splits.json ──────────────────────────────────────────────────────────

print("\n━━ Loading splits.json ━━")
with open(os.path.join(CKPT_DIR, "splits.json")) as f:
    splits = json.load(f)

print(f"  Folds: {len(splits)}")
for s in splits:
    print(f"  Fold {s['fold']}: train={len(s['train'])}  val={len(s['val'])}")

# ── Frames loader — tries newest file first ───────────────────────────────────

def load_frames(clip_dir, max_frames):
    """
    Load frames preferring ._frames.npz (newest) → frames.npz → video.mp4 fallback.
    Returns uint8 tensor (N, 1, H, W) or None.
    """
    # Try ._frames.npz first (newest commit)
    for npz_name in ["._frames.npz", "frames.npz"]:
        npz_path = os.path.join(clip_dir, npz_name)
        if os.path.exists(npz_path):
            try:
                data = np.load(npz_path)
                key  = list(data.keys())[0]
                frames_np = data[key]  # (N, H, W, 3) channels last

                # Convert (N, H, W, 3) → (N, 1, H, W) grayscale
                if frames_np.ndim == 4 and frames_np.shape[-1] == 3:
                    frames_np = (0.299 * frames_np[..., 0] +
                                 0.587 * frames_np[..., 1] +
                                 0.114 * frames_np[..., 2]).astype(np.uint8)
                    frames_np = frames_np[:, np.newaxis, :, :]  # (N, 1, H, W)
                elif frames_np.ndim == 3:
                    frames_np = frames_np[:, np.newaxis, :, :]  # (N, 1, H, W)
                frames = torch.from_numpy(frames_np).to(torch.uint8)
                if max_frames and frames.shape[0] > max_frames:
                    start  = (frames.shape[0] - max_frames) // 2
                    frames = frames[start:start + max_frames]
                return frames
            except Exception:
                continue  # try next file

    # Fallback: extract from video.mp4
    video_path = os.path.join(clip_dir, "video.mp4")
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    frame_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (224, 224))
        frame_list.append(gray)
    cap.release()
    if len(frame_list) == 0:
        return None
    frames_np = np.stack(frame_list)[:, np.newaxis, :, :]
    frames = torch.from_numpy(frames_np).to(torch.uint8)
    if max_frames and frames.shape[0] > max_frames:
        start  = (frames.shape[0] - max_frames) // 2
        frames = frames[start:start + max_frames]
    return frames

# ── Dataset ───────────────────────────────────────────────────────────────────

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, samples, feature_dir, max_frames):
        self.samples     = samples
        self.feature_dir = feature_dir
        self.max_frames  = max_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item     = self.samples[idx]
        clip_id  = item["sample_id"]
        label    = int(item["label"])
        clip_dir = os.path.join(self.feature_dir, clip_id)

        try:
            # Audio
            audio_path   = os.path.join(clip_dir, "audio.wav")
            waveform, sr = torchaudio.load(audio_path)
            waveform     = waveform.mean(0)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)

            # Frames
            frames = load_frames(clip_dir, self.max_frames)
            if frames is None:
                return None

            return {
                "waveform":      waveform,
                "frames":        frames,
                "label":         label,
                "clip_id":       clip_id,
                "waveform_mask": None,
                "frame_mask":    None,
            }
        except Exception as e:
            print(f"  ⚠ Skipped {clip_id}: {e}")
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    max_wav    = max(b["waveform"].shape[0] for b in batch)
    max_frames = max(b["frames"].shape[0]   for b in batch)

    waveforms, frames, labels, clip_ids = [], [], [], []
    wav_masks, frame_masks = [], []

    for b in batch:
        w = b["waveform"]
        f = b["frames"]

        pad_w = max_wav - w.shape[0]
        waveforms.append(F.pad(w, (0, pad_w)))
        wav_masks.append(torch.cat([torch.ones(w.shape[0]),
                                    torch.zeros(pad_w)]).bool())

        pad_f = max_frames - f.shape[0]
        frames.append(torch.cat([f, torch.zeros(pad_f, *f.shape[1:],
                                                dtype=f.dtype)]) if pad_f > 0 else f)
        frame_masks.append(torch.cat([torch.ones(f.shape[0]),
                                       torch.zeros(pad_f)]).bool())

        labels.append(b["label"])
        clip_ids.append(b["clip_id"])

    return {
        "waveform":      torch.stack(waveforms),
        "frames":        torch.stack(frames),
        "waveform_mask": torch.stack(wav_masks),
        "frame_mask":    torch.stack(frame_masks),
        "label":         torch.tensor(labels, dtype=torch.long),
        "clip_id":       clip_ids,
    }

# ── Plot helper ───────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, y_proba, save_path):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    f1  = f1_score(y_true, y_pred, zero_division=0)

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(cm_pct, annot=False,
                cmap=sns.light_palette("#D92323", as_cmap=True),
                xticklabels=CLASSES, yticklabels=CLASSES,
                linewidths=2, linecolor="white", ax=ax,
                cbar_kws={"label": "% of actual class"},
                vmin=0, vmax=100)

    cell_centers = [0.25, 0.75]
    for i in range(2):
        for j in range(2):
            tc     = "white" if cm_pct[i, j] > 55 else "#1a1a2e"
            x_frac = cell_centers[j]
            y_frac = 1 - cell_centers[i]
            ax.text(x_frac, y_frac + 0.06, f"{cm[i, j]}",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=45, fontweight="bold", color=tc)
            ax.text(x_frac, y_frac - 0.06, f"({cm_pct[i, j]:.1f}%)",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=22, fontweight="bold", color=tc, alpha=0.9)

    ax.set_title(f"Acc = {acc:.3f},  AUC = {auc:.3f},  F1 = {f1:.3f}",
                 fontsize=15, fontweight="bold", pad=16)
    ax.set_xlabel("Predicted Label", fontsize=14, labelpad=10)
    ax.set_ylabel("True Label", fontsize=14, labelpad=10)
    ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✓ Saved → {save_path}")
    return acc, auc, f1

# ── Key remapping — fixes CLIP path mismatch between transformers versions ────

def remap_state_dict(state_dict):
    """
    Remap checkpoint keys to match current transformers/model version.
    Handles CLIP vision_model wrapper path change:
      old: visual_model.spatial_encoder.model.vision_model.X
      new: visual_model.spatial_encoder.model.X
    """
    new_sd = {}
    for k, v in state_dict.items():
        new_k = k.replace(
            "visual_model.spatial_encoder.model.vision_model.",
            "visual_model.spatial_encoder.model."
        )
        new_sd[new_k] = v
    return new_sd

# ── Build model once ──────────────────────────────────────────────────────────

print("\n━━ Building model (once) ━━")
model = FusionModel(config).to(device)
model.eval()
print(f"  Total params     : {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable params : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ── Inference loop ────────────────────────────────────────────────────────────

print("\n━━ Running inference on each fold's validation set ━━")

all_y_true   = []
all_y_pred   = []
all_y_proba  = []
all_clip_ids = []
fold_results = {}

for split in splits:
    fold_id   = split["fold"]
    val_items = split["val"]

    ckpt_path = os.path.join(CKPT_DIR, f"fold_{fold_id}_best.pt")

    if not os.path.exists(ckpt_path):
        print(f"  ⚠ Fold {fold_id}: checkpoint missing, skipping")
        continue
    if os.path.getsize(ckpt_path) < 10_000:
        print(f"  ⚠ Fold {fold_id}: LFS pointer not downloaded, skipping")
        continue

    print(f"\n  Fold {fold_id}: {len(val_items)} val samples")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict",
                 checkpoint.get("state_dict", checkpoint)) \
                 if isinstance(checkpoint, dict) else checkpoint
    state_dict = remap_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)
    del checkpoint, state_dict

    dataset = InferenceDataset(val_items, FEATURE_DIR, config.max_frames)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    fold_true, fold_proba = [], []

    for batch in loader:
        if batch is None:
            continue
        batch_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
            if k != "clip_id"
        }
        logits   = model(batch_device)
        probs    = torch.sigmoid(logits).cpu().numpy()
        labels_b = batch["label"].numpy()

        fold_proba.extend(probs.tolist())
        fold_true.extend(labels_b.tolist())
        all_clip_ids.extend(batch["clip_id"])

    if len(fold_true) == 0:
        print(f"  ⚠ Fold {fold_id}: no predictions, skipping")
        continue

    fold_pred = (np.array(fold_proba) > 0.5).astype(int)
    acc = accuracy_score(fold_true, fold_pred)
    auc = roc_auc_score(fold_true, fold_proba)
    f1  = f1_score(fold_true, fold_pred, zero_division=0)
    print(f"    acc={acc:.3f}  auc={auc:.3f}  f1={f1:.3f}  "
          f"({len(fold_true)}/{len(val_items)} evaluated)")

    per_fold_path = os.path.join(OUTPUT_FOLD, f"fold_{fold_id}.png")
    plot_confusion_matrix(fold_true, fold_pred, fold_proba, per_fold_path)

    fold_results[fold_id] = (fold_true, fold_pred, fold_proba)
    all_y_true.extend(fold_true)
    all_y_pred.extend(fold_pred)
    all_y_proba.extend(fold_proba)

# ── Aggregate ─────────────────────────────────────────────────────────────────

all_y_true  = np.array(all_y_true)
all_y_pred  = np.array(all_y_pred)
all_y_proba = np.array(all_y_proba)

if len(all_y_true) == 0:
    print("❌ No predictions collected — check checkpoints and features folders")
    exit()

print(f"\n{'━'*55}")
print(f"  Folds evaluated  : {list(fold_results.keys())}")
print(f"  Total clips      : {len(all_y_true)}")
print(f"  Overall accuracy : {accuracy_score(all_y_true, all_y_pred):.3f}")
print(f"  Overall AUC      : {roc_auc_score(all_y_true, all_y_proba):.3f}")
print(f"  Overall F1       : {f1_score(all_y_true, all_y_pred, zero_division=0):.3f}")
print(f"\n  Classification Report:")
print(classification_report(all_y_true, all_y_pred,
                             target_names=CLASSES, zero_division=0))

results_df = pd.DataFrame({
    "clip_id": all_clip_ids,
    "y_true":  all_y_true,
    "y_pred":  all_y_pred,
    "y_proba": all_y_proba,
})
preds_path = os.path.join(OUTPUT_CACHE, "model_predictions.csv")
results_df.to_csv(preds_path, index=False)
print(f"  ✓ Predictions saved → {preds_path}")

# ── Combined confusion matrix ─────────────────────────────────────────────────

print("\n━━ Plotting combined confusion matrix ━━")
out_combined = os.path.join(OUTPUT_PLOTS, "confusion_matrix_combined.png")
plot_confusion_matrix(all_y_true, all_y_pred, all_y_proba, out_combined)

# ── Per-fold metrics summary ──────────────────────────────────────────────────

print("\n━━ Plotting per-fold metrics summary ━━")

fold_ids  = sorted(fold_results.keys())
fold_accs = [accuracy_score(fold_results[f][0], fold_results[f][1]) for f in fold_ids]
fold_aucs = [roc_auc_score(fold_results[f][0],  fold_results[f][2]) for f in fold_ids]
fold_f1s  = [f1_score(fold_results[f][0], fold_results[f][1], zero_division=0)
             for f in fold_ids]

x     = np.arange(len(fold_ids))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle("Per-Fold Metrics Summary", fontsize=14, fontweight="bold")

for i, (vals, label, color) in enumerate([
    (fold_accs, "Accuracy", "#4C9BE8"),
    (fold_aucs, "AUC",      "#2ecc71"),
    (fold_f1s,  "F1 Score", "#E8714C"),
]):
    bars = ax.bar(x + i * width, vals, width, label=label,
                  color=color, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.2f}", ha="center", fontsize=7, fontweight="bold")

ax.axhline(0.54, color="gray", linestyle=":", linewidth=1.5,
           label="Human baseline (~0.54)", alpha=0.8)
ax.axhline(0.5,  color="white", linestyle="--", linewidth=1.0,
           alpha=0.4, label="Chance (0.50)")

ax.set_xticks(x + width)
ax.set_xticklabels([f"Fold {f}" for f in fold_ids], fontsize=10)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0.3, 1.0)
ax.legend(fontsize=9)
sns.despine()
plt.tight_layout()
out_summary = os.path.join(OUTPUT_PLOTS, "per_fold_metrics_summary.png")
plt.savefig(out_summary, dpi=150, bbox_inches="tight")
plt.close()
print(f"    ✓ Saved → {out_summary}")

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ All outputs saved to: {OUTPUT_PLOTS}
  confusion_matrix_combined.png
  per_fold/fold_X.png
  per_fold_metrics_summary.png
  {preds_path}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")