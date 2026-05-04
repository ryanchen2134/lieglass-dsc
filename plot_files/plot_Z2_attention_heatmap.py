"""
plot_Z2_attention_heatmap.py — Concatenated timeline: YW_WILTY_EP46 lie1/2/3
=============================================================================
Run: PYTORCH_ENABLE_MPS_FALLBACK=1 python plot_Z2_attention_heatmap.py

Three consecutive deception clips from the same subject/episode,
with visual gaps between clips to show they are not continuous.
Time axis resets per-clip so gaps are clearly represented.

Saves: output/plots/Z2_attention_heatmap.png
"""

import matplotlib
matplotlib.use("Agg")

import os, sys, json
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deception_detection.config import ModelConfig
from deception_detection.models.fusion_model import FusionModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import apply_theme, COLORS

apply_theme()

# ── Config ────────────────────────────────────────────────────────────────────

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR    = os.path.join(ROOT, "checkpoints", "20260429_214636")
FEATURE_DIR = os.path.join(ROOT, "features")

CLIP_IDS = ["YW_WILTY_EP46_lie3"]

SUBJECT     = "YW  —  Would I Lie To You? EP46"
GAP_SECONDS = 0.8    # visual gap width inserted between clips on x-axis

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

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Device: {device}")
print(f"Clips : {CLIP_IDS}")

# ── Load model ────────────────────────────────────────────────────────────────

print("Loading model...")
model = FusionModel(config).to(device)
model.eval()
torch.set_grad_enabled(False)

ckpt = torch.load(os.path.join(CKPT_DIR, "fold_0_best.pt"),
                  map_location=device, weights_only=False)
sd   = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) \
       if isinstance(ckpt, dict) else ckpt
new_sd = {k.replace("visual_model.spatial_encoder.model.vision_model.",
                     "visual_model.spatial_encoder.model."): v
          for k, v in sd.items()}
model.load_state_dict(new_sd, strict=True)
print("Model loaded.")

# ── Patch to capture V→A attention ───────────────────────────────────────────

captured = {"va": []}

def patch_block(block, stage_idx):
    def patched_forward(a, v, a_mask=None, v_mask=None):
        a_p = block.audio_proj(a)
        v_p = block.visual_proj(v)
        v_kpm = (~v_mask) if v_mask is not None else None
        a_kpm = (~a_mask) if a_mask is not None else None
        av_out, _ = block.av_attn(a_p, v_p, v_p,
                                   key_padding_mask=v_kpm, need_weights=False)
        va_out, va_w = block.va_attn(v_p, a_p, a_p,
                                      key_padding_mask=a_kpm,
                                      need_weights=True,
                                      average_attn_weights=True)
        if va_w is not None:
            captured["va"].append((stage_idx, va_w.detach().cpu()))
        from deception_detection.models.cross_fusion import _masked_mean
        av_pool = _masked_mean(av_out, a_mask)
        va_pool = _masked_mean(va_out, v_mask)
        return block.fuse_head(torch.cat([av_pool, va_pool], dim=-1))
    block.forward = patched_forward

for i, block in enumerate(model.fusion.blocks):
    patch_block(block, i)

# ── Process each clip ─────────────────────────────────────────────────────────

N_POINTS     = 200
TARGET_STAGE = 1

# Per-clip data stored separately (not yet concatenated)
clip_data = []   # list of dicts per clip
clip_durations = []

for clip_id in CLIP_IDS:
    clip_dir = os.path.join(FEATURE_DIR, clip_id)
    print(f"\nProcessing {clip_id}...")

    waveform, sr = torchaudio.load(os.path.join(clip_dir, "audio.wav"))
    waveform = waveform.mean(0)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform_np   = waveform.numpy()
    clip_duration = len(waveform_np) / 16000

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

    captured["va"].clear()
    batch = {
        "waveform":      waveform.unsqueeze(0).to(device),
        "frames":        frames.unsqueeze(0).to(device),
        "waveform_mask": None,
        "frame_mask":    None,
    }
    logit = model(batch)
    prob  = torch.sigmoid(logit).item()
    print(f"  Confidence: {prob*100:.1f}%  Duration: {clip_duration:.1f}s")

    # RMS energy
    chunk = max(1, len(waveform_np) // N_POINTS)
    rms   = np.array([np.sqrt(np.mean(waveform_np[i:i+chunk]**2))
                      for i in range(0, len(waveform_np)-chunk, chunk)])
    rms   = uniform_filter1d(rms, size=6)

    # V→A attention
    va_w = next((w for s, w in captured["va"] if s == TARGET_STAGE),
                 captured["va"][0][1])
    va   = va_w[0].numpy()
    va_per_audio = va.mean(axis=0)
    va_per_audio = uniform_filter1d(va_per_audio, size=5)

    # Resample to N_POINTS
    rms_r = interp1d(np.linspace(0,1,len(rms)),         rms        )(np.linspace(0,1,N_POINTS))
    va_r  = interp1d(np.linspace(0,1,len(va_per_audio)), va_per_audio)(np.linspace(0,1,N_POINTS))

    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    rms_n = norm(rms_r)
    va_n  = norm(va_r)
    fus_n = norm(rms_n * va_n)

    # Per-clip local time axis (always starts at 0)
    t_local = np.linspace(0, clip_duration, N_POINTS)

    clip_data.append({
        "clip_id":  clip_id,
        "duration": clip_duration,
        "prob":     prob,
        "t_local":  t_local,   # 0 → clip_duration
        "rms":      rms_n,
        "va":       va_n,
        "fusion":   fus_n,
    })
    clip_durations.append(clip_duration)

# ── Build display time axis with gaps ─────────────────────────────────────────
# Each clip gets its own segment; gaps are filled with NaN so lines break.

all_t_display  = []
all_rms        = []
all_va         = []
all_fusion     = []
all_probs      = [d["prob"] for d in clip_data]

# Store [start, end] of each clip's display window for shading/labels
clip_windows   = []   # list of (t_start, t_end, label, prob)
running_t      = 0.0

for d in clip_data:
    t_start  = running_t
    t_end    = running_t + d["duration"]

    # Map local time to display time
    t_disp = np.linspace(t_start, t_end, N_POINTS)
    all_t_display.append(t_disp)
    all_rms.append(d["rms"])
    all_va.append(d["va"])
    all_fusion.append(d["fusion"])

    clip_windows.append((t_start, t_end, d["clip_id"], d["prob"]))

    # Insert NaN gap after each clip (except last)
    if d is not clip_data[-1]:
        gap_t = np.array([t_end, t_end + GAP_SECONDS])
        all_t_display.append(gap_t)
        all_rms.append(np.array([np.nan, np.nan]))
        all_va.append(np.array([np.nan, np.nan]))
        all_fusion.append(np.array([np.nan, np.nan]))
        running_t = t_end + GAP_SECONDS
    else:
        running_t = t_end

t_all      = np.concatenate(all_t_display)
rms_all    = np.concatenate(all_rms)
va_all     = np.concatenate(all_va)
fusion_all = np.concatenate(all_fusion)

threshold = 0.65
co_active = np.where(np.isnan(fusion_all), False, fusion_all > threshold)

total_signal_duration = sum(clip_durations)
mean_prob = np.mean(all_probs)

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
fig.patch.set_facecolor(COLORS["bg"])
fig.subplots_adjust(hspace=0.1)

clip_labels = " → ".join([c["clip_id"].split("_lie")[1] for c in clip_data])
title_str   = (f"Subject: {SUBJECT}  —  Clips: lie{clip_labels}  "
               f"|  Mean Confidence: {mean_prob*100:.1f}% Deception")

# ── Panel 1: Overlay ─────────────────────────────────────────────────────────
axes[0].set_title(title_str, fontsize=11, fontweight="bold",
                  color=COLORS["white"])

axes[0].plot(t_all, rms_all, color=COLORS["truth"], linewidth=2.0,
             label="Audio Energy", zorder=3, alpha=0.9)
axes[0].fill_between(t_all, rms_all, alpha=0.12, color=COLORS["truth"])

axes[0].plot(t_all, va_all, color=COLORS["red"], linewidth=2.0,
             label="Visual Attention to Audio",
             zorder=3, alpha=0.9)
axes[0].fill_between(t_all, va_all, alpha=0.12, color=COLORS["red"])

# Shade where both > 0.5
rms_safe = np.where(np.isnan(rms_all), 0, rms_all)
va_safe  = np.where(np.isnan(va_all),  0, va_all)
both_high = (rms_safe > 0.5) & (va_safe > 0.5) & ~np.isnan(rms_all)
axes[0].fill_between(t_all, 0, 1.1, where=both_high,
                     color="#00ff88", alpha=0.15,
                     label="Both elevated")

axes[0].set_ylabel("Normalised Signal", fontsize=10)
axes[0].set_ylim(-0.05, 1.35)
axes[0].legend(loc="lower left", fontsize=9)
axes[0].tick_params(labelbottom=False)

# ── Panel 2: Fusion ───────────────────────────────────────────────────────────
axes[1].plot(t_all, fusion_all, color=COLORS["white"], linewidth=2.0,
             label="Fusion Signal  (Audio × Visual)", zorder=3)
axes[1].fill_between(t_all, fusion_all, where=co_active,
                     color=COLORS["red"], alpha=0.55,
                     label=f"Deception signal  (>{threshold:.0%})")
axes[1].fill_between(t_all, np.where(np.isnan(fusion_all), 0, fusion_all),
                     where=~co_active & ~np.isnan(fusion_all),
                     color=COLORS["white"], alpha=0.05)
axes[1].axhline(threshold, color=COLORS["red"], linestyle="--",
                linewidth=1.2, alpha=0.7)

axes[1].set_ylabel("Fusion Signal", fontsize=10)
axes[1].set_xlabel("Time (seconds)", fontsize=11)
axes[1].set_ylim(-0.05, 1.35)
axes[1].legend(loc="lower left", fontsize=9)
axes[1].set_xlim(0, running_t)

# ── Gap shading + clip labels ─────────────────────────────────────────────────
gap_starts = []
for i, (t_start, t_end, clip_id, prob) in enumerate(clip_windows):
    lie_num = clip_id.split("_lie")[-1]
    mid     = (t_start + t_end) / 2

    for ax in axes:
        # Light alternating clip background
        ax.axvspan(t_start, t_end, alpha=0.04,
                   color=COLORS["truth"] if i % 2 == 0 else COLORS["red"],
                   zorder=0)

    # Clip label on panel 1
    axes[0].text(mid, 1.22, f"Lie {lie_num}",
                 ha="center", fontsize=9, fontweight="bold",
                 color=COLORS["white"])
    axes[0].text(mid, 1.13, f"{prob*100:.0f}% conf.",
                 ha="center", fontsize=8,
                 color=COLORS["red"], fontfamily="monospace")

    # Gap shading after clip (except last)
    if i < len(clip_windows) - 1:
        gap_s = t_end
        gap_e = clip_windows[i+1][0]
        for ax in axes:
            ax.axvspan(gap_s, gap_e,
                       facecolor=COLORS["bg"], edgecolor="none",
                       zorder=6, alpha=1.0)
            ax.axvspan(gap_s, gap_e,
                       facecolor="none",
                       hatch="///", edgecolor=COLORS["grid"],
                       linewidth=0, alpha=0.4, zorder=7)
        axes[0].text((gap_s + gap_e) / 2, 0.5, "···",
                     ha="center", va="center", fontsize=14,
                     color=COLORS["gray"], zorder=8)

# Custom x-axis tick labels showing per-clip local time
# Build tick positions and labels
ticks      = []
tick_labels = []
for t_start, t_end, _, _ in clip_windows:
    dur = t_end - t_start
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        ticks.append(t_start + frac * dur)
        tick_labels.append(f"{frac * dur:.1f}s")

axes[1].set_xticks(ticks)
axes[1].set_xticklabels(tick_labels, fontsize=7, rotation=0)



output_path = os.path.join(ROOT, "output", "plots", "Z2_attention_heatmap.png")
fig.savefig(output_path, dpi=150, bbox_inches="tight",
            facecolor=COLORS["bg"], edgecolor=COLORS["bg"])
plt.close(fig)
print(f"\n✓ Saved → {output_path}")
print(f"Clips: {[c['clip_id'] for c in clip_data]}")
print(f"Confidences: {[str(round(c['prob']*100, 1)) + '%' for c in clip_data]}")
print(f"Mean confidence: {mean_prob*100:.1f}%")