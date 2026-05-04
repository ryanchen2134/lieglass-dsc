"""
plot_I_ablation.py — Real ablation: audio-only, visual-only, full fusion
=========================================================================
Run from plot_files/:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python plot_I_ablation.py

Runs inference 3 times on fold 0 val set:
  1. Full fusion   — normal forward pass
  2. Audio only    — frames zeroed out
  3. Visual only   — waveform zeroed out

Saves: output/plots/I_ablation.png
       output/cache/ablation_results.json  (cache)
"""

import matplotlib
matplotlib.use("Agg")

import os, sys, json
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import apply_theme, save_plot, COLORS, ROOT, CKPT_DIR, CACHE_DIR
apply_theme()

sys.path.insert(0, ROOT)
from deception_detection.config import ModelConfig
from deception_detection.models.fusion_model import FusionModel

FEATURE_DIR = os.path.join(ROOT, "features")
CACHE_FILE  = os.path.join(CACHE_DIR, "ablation_results.json")

# ── Load config ───────────────────────────────────────────────────────────────

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

# ── Check cache ───────────────────────────────────────────────────────────────

if os.path.exists(CACHE_FILE):
    print(f"\n━━ Loading cached ablation results ━━")
    with open(CACHE_FILE) as f:
        results = json.load(f)
    print(f"  ✓ Loaded from {CACHE_FILE}")
    print(f"  ℹ  Delete to force re-run")

else:
    print(f"\n━━ No cache — running ablation inference ━━")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")
    torch.set_grad_enabled(False)

    with open(os.path.join(CKPT_DIR, "splits.json")) as f:
        splits = json.load(f)
    val_items = splits[0]["val"]
    print(f"  Val samples (fold 0): {len(val_items)}")

    model = FusionModel(config).to(device)
    model.eval()
    ckpt  = torch.load(os.path.join(CKPT_DIR, "fold_0_best.pt"),
                       map_location=device, weights_only=False)
    sd    = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) \
            if isinstance(ckpt, dict) else ckpt
    new_sd = {k.replace("visual_model.spatial_encoder.model.vision_model.",
                         "visual_model.spatial_encoder.model."): v
              for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=True)
    del ckpt, sd
    print("  Model loaded.")

    # Load all clips into memory once so we only read files once
    print("\n  Loading clips...")
    clip_list = []
    labels    = []

    for item in val_items:
        clip_id  = item["sample_id"]
        label    = int(item["label"])
        clip_dir = os.path.join(FEATURE_DIR, clip_id)
        try:
            waveform, sr = torchaudio.load(os.path.join(clip_dir, "audio.wav"))
            waveform = waveform.mean(0)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)

            frames_np = np.load(os.path.join(clip_dir, "frames.npz"))["frames"]
            if frames_np.ndim == 4 and frames_np.shape[-1] == 3:
                frames_np = (0.299 * frames_np[..., 0] +
                             0.587 * frames_np[..., 1] +
                             0.114 * frames_np[..., 2]).astype(np.uint8)
            frames_np = frames_np[:, np.newaxis, :, :]
            frames = torch.from_numpy(frames_np).to(torch.uint8)
            if config.max_frames and frames.shape[0] > config.max_frames:
                start  = (frames.shape[0] - config.max_frames) // 2
                frames = frames[start:start + config.max_frames]

            clip_list.append((waveform, frames))
            labels.append(label)
        except Exception as e:
            print(f"    ⚠ Skipped {clip_id}: {e}")

    print(f"  Loaded {len(clip_list)}/{len(val_items)} clips")

    def run_inference(clip_list, labels, mode):
        """mode: 'full' | 'audio_only' | 'visual_only'"""
        all_proba = []
        for waveform, frames in clip_list:
            w = waveform.unsqueeze(0).to(device)
            f = frames.unsqueeze(0).to(device)
            if mode == "audio_only":
                f = torch.zeros_like(f)
            elif mode == "visual_only":
                w = torch.zeros_like(w)
            batch_dev = {"waveform": w, "frames": f,
                         "waveform_mask": None, "frame_mask": None}
            prob = torch.sigmoid(model(batch_dev)).item()
            all_proba.append(prob)
        return round(roc_auc_score(labels, all_proba), 4)

    print("\n  Running full fusion...")
    auc_full   = run_inference(clip_list, labels, "full")
    print(f"    AUC = {auc_full:.4f}")

    print("  Running audio only (frames zeroed)...")
    auc_audio  = run_inference(clip_list, labels, "audio_only")
    print(f"    AUC = {auc_audio:.4f}")

    print("  Running visual only (waveform zeroed)...")
    auc_visual = run_inference(clip_list, labels, "visual_only")
    print(f"    AUC = {auc_visual:.4f}")

    results = {
        "full_fusion": auc_full,
        "audio_only":  auc_audio,
        "visual_only": auc_visual,
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Cached → {CACHE_FILE}")

# ── Plot ──────────────────────────────────────────────────────────────────────

print(f"\n━━ Results ━━")
print(f"  Audio Only  : {results['audio_only']:.3f}")
print(f"  Visual Only : {results['visual_only']:.3f}")
print(f"  Full Fusion : {results['full_fusion']:.3f}")

labels_plot = ["Audio Only", "Visual Only", "Full Fusion"]
values      = [results["audio_only"], results["visual_only"], results["full_fusion"]]
colors      = [COLORS["gray"], COLORS["gray"], COLORS["red"]]

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor(COLORS["bg"])

bars = ax.bar(labels_plot, values, color=colors, width=0.45,
              edgecolor=COLORS["bg"], linewidth=1.5, alpha=0.9)

for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{v:.3f}", ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=COLORS["white"])

ax.axhline(0.54, color=COLORS["gray"], linestyle="--",
           linewidth=1.5, alpha=0.7, label="Human baseline (0.54)")
ax.axhline(0.50, color=COLORS["grid"], linestyle=":",
           linewidth=1.0, alpha=0.5, label="Chance (0.50)")

ax.set_ylim(0.3, 1.0)
ax.set_ylabel("AUC-ROC", fontsize=11)
ax.set_title("Fusion vs Single-Modality Performance", fontsize=13,
             fontweight="bold")
ax.legend(fontsize=9)

# Improvement annotation
best_single = max(results["audio_only"], results["visual_only"])
delta = results["full_fusion"] - best_single
ax.annotate(f"+{delta:.3f} vs best\nsingle modality",
            xy=(2, results["full_fusion"]),
            xytext=(1.35, results["full_fusion"] + 0.07),
            fontsize=9, color=COLORS["red"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.5))

plt.tight_layout()
save_plot(fig, "I_ablation.png")