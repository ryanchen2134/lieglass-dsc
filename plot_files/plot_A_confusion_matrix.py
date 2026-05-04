"""
plot_A_confusion_matrix.py — Confusion matrices with LieGlass dark theme
=========================================================================
Run from plot_files/:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python plot_A_confusion_matrix.py

On first run: performs full inference and saves output/cache/model_predictions.csv
On subsequent runs: loads from cache and skips inference entirely.
Delete output/cache/model_predictions.csv to force re-inference.

Saves:  output/plots/A_confusion/A_confusion_matrix_combined.png
        output/plots/A_confusion/per_fold/fold_X.png
        output/plots/A_confusion/A_per_fold_metrics_summary.png
        output/cache/model_predictions.csv
"""

import matplotlib
matplotlib.use("Agg")

import os, sys, cv2, json
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

# ── Theme ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import apply_theme, COLORS, ROOT, CKPT_DIR, CACHE_DIR
apply_theme()

sys.path.insert(0, ROOT)
from deception_detection.config import ModelConfig
from deception_detection.models.fusion_model import FusionModel

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURE_DIR  = os.path.join(ROOT, "features")
OUTPUT_PLOTS = os.path.join(ROOT, "output", "plots", "A_confusion")
OUTPUT_FOLD  = os.path.join(OUTPUT_PLOTS, "per_fold")
preds_path   = os.path.join(CACHE_DIR, "model_predictions.csv")

os.makedirs(OUTPUT_PLOTS, exist_ok=True)
os.makedirs(OUTPUT_FOLD,  exist_ok=True)
os.makedirs(CACHE_DIR,    exist_ok=True)

CLASSES = ["Lie", "Truth"]   # 0=lie, 1=truth

# ── Load config ───────────────────────────────────────────────────────────────

print("\n━━ Loading config ━━")
with open(os.path.join(CKPT_DIR, "config.json")) as f:
    ckpt_cfg = json.load(f)

config = ModelConfig()
config.visual_backbone      = ckpt_cfg["visual_backbone"]
config.d_audio              = ckpt_cfg.get("d_audio",             768)
config.d_visual             = ckpt_cfg.get("d_visual",            384)
config.d_fused              = ckpt_cfg.get("d_fused",             256)
config.dropout              = ckpt_cfg.get("dropout",             0.4)
config.vit_n_layers         = ckpt_cfg.get("vit_n_layers",        3)
config.vit_n_heads          = ckpt_cfg.get("vit_n_heads",         8)
config.max_frames           = ckpt_cfg.get("max_frames",          256)
config.cnn_chunk_size       = ckpt_cfg.get("cnn_chunk_size",      64)
config.in_channels          = ckpt_cfg.get("in_channels",         1)
config.audio_fusion_layers  = ckpt_cfg.get("audio_fusion_layers", [4, 8, 12])
config.visual_fusion_layers = ckpt_cfg.get("visual_fusion_layers",[1, 2, 3])
config.fusion_aggregator    = ckpt_cfg.get("fusion_aggregator",   "weighted_sum")
config.fusion_n_heads       = ckpt_cfg.get("fusion_n_heads",      8)
config.fusion_dropout       = ckpt_cfg.get("fusion_dropout",      0.3)
config.use_ut_adapters      = ckpt_cfg.get("use_ut_adapters",     True)
config.ut_adapter_dim       = ckpt_cfg.get("ut_adapter_dim",      128)
config.ut_conv_kernel       = ckpt_cfg.get("ut_conv_kernel",      3)
config.n_folds              = ckpt_cfg.get("n_folds",             5)
config.seed                 = ckpt_cfg.get("seed",                1919)

print(f"  visual_backbone : {config.visual_backbone}")
print(f"  n_folds / seed  : {config.n_folds} / {config.seed}")

# ── Load splits ───────────────────────────────────────────────────────────────

with open(os.path.join(CKPT_DIR, "splits.json")) as f:
    splits = json.load(f)

# ── Plot helper ───────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, y_proba, save_path):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    f1  = f1_score(y_true, y_pred, zero_division=0)

    fig, ax = plt.subplots(figsize=(10, 9))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    sns.heatmap(cm_pct, annot=False,
                cmap=sns.light_palette(COLORS["red"], as_cmap=True),
                xticklabels=CLASSES, yticklabels=CLASSES,
                linewidths=0, linecolor=COLORS["bg"], ax=ax,
                cbar_kws={"label": "% of actual class"},
                vmin=0, vmax=100)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks([0.5, 1.5], minor=False)
    ax.set_yticks([0.5, 1.5], minor=False)
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    
    ax.grid(False)
    for _, spine in ax.spines.items():
        spine.set_visible(False)

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(COLORS["gray_light"])
    cbar.ax.tick_params(colors=COLORS["gray"])

    cell_centers = [0.25, 0.75]
    for i in range(2):
        for j in range(2):
            tc     = COLORS["white"] if cm_pct[i, j] > 55 else COLORS["bg"]
            x_frac = cell_centers[j]
            y_frac = 1 - cell_centers[i]
            ax.text(x_frac, y_frac + 0.06, f"{cm[i, j]}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=45, fontweight="bold", color=tc)
            ax.text(x_frac, y_frac - 0.06, f"({cm_pct[i, j]:.1f}%)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=22, fontweight="bold", color=tc, alpha=0.9)

    ax.set_title(f"Acc = {acc:.3f},  AUC = {auc:.3f},  F1 = {f1:.3f}",
                 fontsize=15, fontweight="bold", pad=16, color=COLORS["white"])
    ax.set_xlabel("Predicted Label", fontsize=14, labelpad=10,
                  color=COLORS["gray_light"])
    ax.set_ylabel("True Label", fontsize=14, labelpad=10,
                  color=COLORS["gray_light"])
    ax.tick_params(labelsize=14, colors=COLORS["gray_light"])
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor=COLORS["bg"])
    plt.close(fig)
    print(f"    ✓ Saved → {save_path}")
    return acc, auc, f1

# ── Check cache ───────────────────────────────────────────────────────────────

fold_results = {}

if os.path.exists(preds_path):
    print(f"\n━━ Loading cached predictions ━━")
    results_df   = pd.read_csv(preds_path)
    all_clip_ids = results_df["clip_id"].tolist()
    all_y_true   = results_df["y_true"].values
    all_y_pred   = results_df["y_pred"].values
    all_y_proba  = results_df["y_proba"].values

    # Rebuild fold_results from cache
    for split in splits:
        fold_id = split["fold"]
        val_ids = {s["sample_id"] for s in split["val"]}
        mask    = results_df["clip_id"].isin(val_ids)
        if mask.sum() == 0:
            continue
        fold_df = results_df[mask]
        fold_results[fold_id] = (
            fold_df["y_true"].values,
            fold_df["y_pred"].values,
            fold_df["y_proba"].values,
        )

    print(f"  ✓ {len(results_df)} predictions loaded from cache")
    print(f"  ✓ Folds found: {list(fold_results.keys())}")
    print(f"  ℹ  Delete {preds_path} to force re-inference")

else:
    # ── Full inference ────────────────────────────────────────────────────────

    print("\n━━ No cache found — running full inference ━━")

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")
    torch.set_grad_enabled(False)

    # Frames loader
    def load_frames(clip_dir, max_frames):
        for npz_name in ["._frames.npz", "frames.npz"]:
            npz_path = os.path.join(clip_dir, npz_name)
            if os.path.exists(npz_path):
                try:
                    data      = np.load(npz_path)
                    frames_np = data[list(data.keys())[0]]
                    if frames_np.ndim == 4 and frames_np.shape[-1] == 3:
                        frames_np = (0.299 * frames_np[..., 0] +
                                     0.587 * frames_np[..., 1] +
                                     0.114 * frames_np[..., 2]).astype(np.uint8)
                        frames_np = frames_np[:, np.newaxis, :, :]
                    elif frames_np.ndim == 3:
                        frames_np = frames_np[:, np.newaxis, :, :]
                    frames = torch.from_numpy(frames_np).to(torch.uint8)
                    if max_frames and frames.shape[0] > max_frames:
                        start  = (frames.shape[0] - max_frames) // 2
                        frames = frames[start:start + max_frames]
                    return frames
                except Exception:
                    continue
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
        if not frame_list:
            return None
        frames = torch.from_numpy(
            np.stack(frame_list)[:, np.newaxis, :, :]
        ).to(torch.uint8)
        if max_frames and frames.shape[0] > max_frames:
            start  = (frames.shape[0] - max_frames) // 2
            frames = frames[start:start + max_frames]
        return frames

    # Dataset
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
                waveform, sr = torchaudio.load(os.path.join(clip_dir, "audio.wav"))
                waveform = waveform.mean(0)
                if sr != 16000:
                    waveform = torchaudio.functional.resample(waveform, sr, 16000)
                frames = load_frames(clip_dir, self.max_frames)
                if frames is None:
                    return None
                return {"waveform": waveform, "frames": frames, "label": label,
                        "clip_id": clip_id, "waveform_mask": None, "frame_mask": None}
            except Exception as e:
                print(f"  ⚠ Skipped {clip_id}: {e}")
                return None

    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        max_wav    = max(b["waveform"].shape[0] for b in batch)
        max_frames = max(b["frames"].shape[0]   for b in batch)
        waveforms, frames, labels, clip_ids = [], [], [], []
        wav_masks, frame_masks = [], []
        for b in batch:
            w, f  = b["waveform"], b["frames"]
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
        return {"waveform": torch.stack(waveforms), "frames": torch.stack(frames),
                "waveform_mask": torch.stack(wav_masks),
                "frame_mask": torch.stack(frame_masks),
                "label": torch.tensor(labels, dtype=torch.long),
                "clip_id": clip_ids}

    def remap_state_dict(sd):
        return {k.replace("visual_model.spatial_encoder.model.vision_model.",
                           "visual_model.spatial_encoder.model."): v
                for k, v in sd.items()}

    # Build model
    print("\n━━ Building model ━━")
    model = FusionModel(config).to(device)
    model.eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,} total")

    # Inference loop
    print("\n━━ Running inference ━━")
    all_y_true, all_y_pred, all_y_proba, all_clip_ids = [], [], [], []

    for split in splits:
        fold_id   = split["fold"]
        val_items = split["val"]
        ckpt_path = os.path.join(CKPT_DIR, f"fold_{fold_id}_best.pt")

        if not os.path.exists(ckpt_path):
            print(f"  ⚠ Fold {fold_id}: checkpoint missing")
            continue
        if os.path.getsize(ckpt_path) < 10_000:
            print(f"  ⚠ Fold {fold_id}: LFS pointer, skipping")
            continue

        print(f"\n  Fold {fold_id}: {len(val_items)} samples")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd   = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) \
               if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(remap_state_dict(sd), strict=True)
        del ckpt, sd

        loader = torch.utils.data.DataLoader(
            InferenceDataset(val_items, FEATURE_DIR, config.max_frames),
            batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0
        )

        fold_true, fold_proba = [], []
        for batch in loader:
            if batch is None:
                continue
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items() if k != "clip_id"}
            probs    = torch.sigmoid(model(batch_dev)).cpu().numpy()
            labels_b = batch["label"].numpy()
            fold_proba.extend(probs.tolist())
            fold_true.extend(labels_b.tolist())
            all_clip_ids.extend(batch["clip_id"])

        if not fold_true:
            continue

        fold_pred = (np.array(fold_proba) > 0.5).astype(int)
        acc = accuracy_score(fold_true, fold_pred)
        auc = roc_auc_score(fold_true, fold_proba)
        f1  = f1_score(fold_true, fold_pred, zero_division=0)
        print(f"    acc={acc:.3f}  auc={auc:.3f}  f1={f1:.3f}  "
              f"({len(fold_true)}/{len(val_items)} evaluated)")

        fold_results[fold_id] = (fold_true, fold_pred, fold_proba)
        all_y_true.extend(fold_true)
        all_y_pred.extend(fold_pred)
        all_y_proba.extend(fold_proba)

    all_y_true  = np.array(all_y_true)
    all_y_pred  = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)

    if len(all_y_true) == 0:
        print("❌ No predictions collected")
        exit()

    # Save cache
    pd.DataFrame({"clip_id": all_clip_ids, "y_true": all_y_true,
                  "y_pred": all_y_pred, "y_proba": all_y_proba}
                 ).to_csv(preds_path, index=False)
    print(f"\n  ✓ Predictions cached → {preds_path}")

# ── Summary ───────────────────────────────────────────────────────────────────

all_y_true  = np.array(all_y_true)  if not isinstance(all_y_true,  np.ndarray) else all_y_true
all_y_pred  = np.array(all_y_pred)  if not isinstance(all_y_pred,  np.ndarray) else all_y_pred
all_y_proba = np.array(all_y_proba) if not isinstance(all_y_proba, np.ndarray) else all_y_proba

print(f"\n{'━'*50}")
print(f"  Folds evaluated  : {list(fold_results.keys())}")
print(f"  Total clips      : {len(all_y_true)}")
print(f"  Overall accuracy : {accuracy_score(all_y_true, all_y_pred):.3f}")
print(f"  Overall AUC      : {roc_auc_score(all_y_true, all_y_proba):.3f}")
print(f"  Overall F1       : {f1_score(all_y_true, all_y_pred, zero_division=0):.3f}")
print(f"\n  Classification Report:")
print(classification_report(all_y_true, all_y_pred,
                             target_names=CLASSES, zero_division=0))

# ── Combined confusion matrix ─────────────────────────────────────────────────

print("\n━━ Combined confusion matrix ━━")
out_combined = os.path.join(OUTPUT_PLOTS, "A_confusion_matrix_combined.png")
plot_confusion_matrix(all_y_true, all_y_pred, all_y_proba, out_combined)

# ── Per-fold metrics summary ──────────────────────────────────────────────────

print("\n━━ Per-fold metrics summary ━━")
fold_ids  = sorted(fold_results.keys())
fold_accs = [accuracy_score(fold_results[f][0], fold_results[f][1]) for f in fold_ids]
fold_aucs = [roc_auc_score(fold_results[f][0],  fold_results[f][2]) for f in fold_ids]
fold_f1s  = [f1_score(fold_results[f][0], fold_results[f][1], zero_division=0)
             for f in fold_ids]

x, width = np.arange(len(fold_ids)), 0.25
fig, ax  = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor(COLORS["bg"])
ax.set_facecolor(COLORS["panel"])
fig.suptitle("Per-Fold Metrics Summary", fontsize=14, fontweight="bold",
             color=COLORS["white"])

for i, (vals, label, color) in enumerate([
    (fold_accs, "Accuracy", COLORS["truth"]),
    (fold_aucs, "AUC",      COLORS["red"]),
    (fold_f1s,  "F1 Score", COLORS["gray"]),
]):
    bars = ax.bar(x + i * width, vals, width, label=label,
                  color=color, alpha=0.85, edgecolor=COLORS["bg"])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.2f}", ha="center", fontsize=7,
                fontweight="bold", color=COLORS["white"])

ax.axhline(0.54, color=COLORS["gray"], linestyle="--",
           linewidth=1.5, alpha=0.7, label="Human baseline (0.54)")
ax.axhline(0.50, color=COLORS["grid"], linestyle=":",
           linewidth=1.0, alpha=0.5, label="Chance (0.50)")
ax.set_xticks(x + width)
ax.set_xticklabels([f"Fold {f}" for f in fold_ids], fontsize=10)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0.3, 1.0)
ax.legend(fontsize=9)

out_summary = os.path.join(OUTPUT_PLOTS, "A_per_fold_metrics_summary.png")
fig.savefig(out_summary, dpi=150, bbox_inches="tight",
            facecolor=COLORS["bg"], edgecolor=COLORS["bg"])
plt.close(fig)
print(f"    ✓ Saved → {out_summary}")

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ All outputs saved to: {OUTPUT_PLOTS}
  A_confusion_matrix_combined.png
  per_fold/fold_X.png
  A_per_fold_metrics_summary.png
  Cache: {preds_path}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")