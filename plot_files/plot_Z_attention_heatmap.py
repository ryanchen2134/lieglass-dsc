"""
plot_Z_attention_heatmap.py — Real cross-attention weights from a high-confidence deception clip
================================================================================================
Run: PYTORCH_ENABLE_MPS_FALLBACK=1 python plot_Z_attention_heatmap.py

Loads clip YW_WILTY_EP46_lie3 (model confidence: 95.9% deception),
patches the fusion layer to capture attention weights,
and visualizes the bidirectional A→V and V→A attention matrices.

Saves: output/plots/Z_attention_heatmap.png
"""

import matplotlib
matplotlib.use("Agg")

import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deception_detection.config import ModelConfig
from deception_detection.models.fusion_model import FusionModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import apply_theme, save_plot, COLORS

apply_theme()

# ── Config ────────────────────────────────────────────────────────────────────

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "checkpoints", "20260429_214636")
CLIP_ID  = "YW_WILTY_EP46_lie3"
CLIP_DIR = os.path.join(ROOT, "features", CLIP_ID)

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
print(f"Clip:   {CLIP_ID}")

# ── Load model ────────────────────────────────────────────────────────────────

print("Loading model...")
model = FusionModel(config).to(device)
model.eval()
torch.set_grad_enabled(False)

ckpt = torch.load(os.path.join(CKPT_DIR, "fold_0_best.pt"),
                  map_location=device, weights_only=False)
sd   = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) \
       if isinstance(ckpt, dict) else ckpt

# Remap CLIP keys
new_sd = {}
for k, v in sd.items():
    new_sd[k.replace("visual_model.spatial_encoder.model.vision_model.",
                      "visual_model.spatial_encoder.model.")] = v
model.load_state_dict(new_sd, strict=True)
print("Model loaded.")

# ── Patch attention layers to capture weights ─────────────────────────────────

captured = {"av": [], "va": []}

def patch_block(block, stage_idx):
    orig_forward = block.forward

    def patched_forward(a, v, a_mask=None, v_mask=None):
        a_p = block.audio_proj(a)
        v_p = block.visual_proj(v)

        v_kpm = (~v_mask) if v_mask is not None else None
        a_kpm = (~a_mask) if a_mask is not None else None

        # Extract weights by setting need_weights=True
        av_out, av_weights = block.av_attn(
            a_p, v_p, v_p,
            key_padding_mask=v_kpm,
            need_weights=True,
            average_attn_weights=True   # average over heads → (B, T_a, T_v)
        )
        va_out, va_weights = block.va_attn(
            v_p, a_p, a_p,
            key_padding_mask=a_kpm,
            need_weights=True,
            average_attn_weights=True   # (B, T_v, T_a)
        )

        if av_weights is not None:
            captured["av"].append((stage_idx, av_weights.detach().cpu()))
        if va_weights is not None:
            captured["va"].append((stage_idx, va_weights.detach().cpu()))

        from deception_detection.models.cross_fusion import _masked_mean
        av_pool = _masked_mean(av_out, a_mask)
        va_pool = _masked_mean(va_out, v_mask)
        return block.fuse_head(torch.cat([av_pool, va_pool], dim=-1))

    block.forward = patched_forward

# Patch all fusion blocks
for i, block in enumerate(model.fusion.blocks):
    patch_block(block, i)

# ── Load clip ─────────────────────────────────────────────────────────────────

print("Loading clip...")

# Audio
waveform, sr = torchaudio.load(os.path.join(CLIP_DIR, "audio.wav"))
waveform = waveform.mean(0)
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)
waveform_np = waveform.numpy()

# Frames
frames_np = np.load(os.path.join(CLIP_DIR, "frames.npz"))["frames"]
# Convert (N, H, W, 3) → (N, 1, H, W) grayscale
if frames_np.ndim == 4 and frames_np.shape[-1] == 3:
    frames_np = (0.299 * frames_np[..., 0] +
                 0.587 * frames_np[..., 1] +
                 0.114 * frames_np[..., 2]).astype(np.uint8)
frames_np = frames_np[:, np.newaxis, :, :]  # (N, 1, H, W)

frames = torch.from_numpy(frames_np).to(torch.uint8)
if config.max_frames and frames.shape[0] > config.max_frames:
    start  = (frames.shape[0] - config.max_frames) // 2
    frames = frames[start:start + config.max_frames]

n_frames = frames.shape[0]
clip_duration = len(waveform_np) / 16000

print(f"  Frames: {n_frames}  |  Duration: {clip_duration:.1f}s")

# ── Run inference ─────────────────────────────────────────────────────────────

print("Running forward pass...")

batch = {
    "waveform":      waveform.unsqueeze(0).to(device),
    "frames":        frames.unsqueeze(0).to(device),
    "waveform_mask": None,
    "frame_mask":    None,
}

logit    = model(batch)
prob     = torch.sigmoid(logit).item()
print(f"  Model confidence: {prob:.3f} (deception)")

# Use the middle fusion stage (most informative — stage 1 of 3)
target_stage = 1
av_weights = next((w for s, w in captured["av"] if s == target_stage), captured["av"][0][1])
va_weights = next((w for s, w in captured["va"] if s == target_stage), captured["va"][0][1])

# Shape: (1, T_a, T_v) and (1, T_v, T_a) — squeeze batch dim
av = av_weights[0].numpy()  # (T_a, T_v)
va = va_weights[0].numpy()  # (T_v, T_a)

print(f"  A→V attention shape: {av.shape}")
print(f"  V→A attention shape: {va.shape}")

# ── Plot ──────────────────────────────────────────────────────────────────────

print("Plotting...")

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor(COLORS["bg"])

gs = gridspec.GridSpec(3, 2,
                        height_ratios=[0.8, 2.5, 2.5],
                        width_ratios=[1, 0.04],
                        hspace=0.45, wspace=0.08)

# ── Audio waveform (top) ──────────────────────────────────────────────────────
ax_audio = fig.add_subplot(gs[0, 0])
t_audio  = np.linspace(0, clip_duration, len(waveform_np))
# RMS energy envelope
chunk   = max(1, len(waveform_np) // 300)
rms     = np.array([np.sqrt(np.mean(waveform_np[i:i+chunk]**2))
                    for i in range(0, len(waveform_np)-chunk, chunk)])
t_rms   = np.linspace(0, clip_duration, len(rms))

ax_audio.plot(t_audio[::10], waveform_np[::10],
              color=COLORS["truth"], linewidth=0.6, alpha=0.5)
ax_audio.plot(t_rms, rms, color=COLORS["white"], linewidth=1.5,
              label="RMS energy")
ax_audio.fill_between(t_rms, rms, alpha=0.2, color=COLORS["truth"])
ax_audio.set_xlim(0, clip_duration)
ax_audio.set_ylabel("Amplitude", fontsize=9)
ax_audio.set_title(
    f"Clip: {CLIP_ID}   |   Model Confidence: {prob*100:.1f}% Deception   "
    f"|   Stage {target_stage+1}/3 Cross-Attention",
    fontsize=11, fontweight="bold", color=COLORS["white"]
)
ax_audio.tick_params(labelbottom=False)

# ── A→V heatmap ───────────────────────────────────────────────────────────────
ax_av   = fig.add_subplot(gs[1, 0])
ax_cbar = fig.add_subplot(gs[1, 1])

im_av = ax_av.imshow(av, aspect="auto", cmap="inferno",
                      interpolation="bilinear", origin="upper")
ax_av.set_ylabel("Audio Timestep →", fontsize=10)
ax_av.set_xlabel("Visual Frame →", fontsize=10)
ax_av.set_title("Audio → Visual Attention  (Where audio looks at video)",
                fontsize=10, color=COLORS["red"], loc="left")
fig.colorbar(im_av, cax=ax_cbar)
ax_cbar.tick_params(labelsize=8)

# Annotate peak attention
peak_a, peak_v = np.unravel_index(np.argmax(av), av.shape)
ax_av.scatter(peak_v, peak_a, s=120, c=COLORS["red"],
              edgecolors=COLORS["white"], linewidths=1.5, zorder=5)
ax_av.annotate("Peak\nattention", xy=(peak_v, peak_a),
               xytext=(peak_v + av.shape[1]*0.08, peak_a - av.shape[0]*0.1),
               fontsize=8, color=COLORS["white"],
               arrowprops=dict(arrowstyle="->", color=COLORS["white"], lw=1.0))

# ── V→A heatmap ───────────────────────────────────────────────────────────────
ax_va    = fig.add_subplot(gs[2, 0])
ax_cbar2 = fig.add_subplot(gs[2, 1])

im_va = ax_va.imshow(va, aspect="auto", cmap="inferno",
                      interpolation="bilinear", origin="upper")
ax_va.set_ylabel("Visual Frame →", fontsize=10)
ax_va.set_xlabel("Audio Timestep →", fontsize=10)
ax_va.set_title("Visual → Audio Attention  (Where video looks at audio)",
                fontsize=10, color=COLORS["truth"], loc="left")
fig.colorbar(im_va, cax=ax_cbar2)
ax_cbar2.tick_params(labelsize=8)

peak_v2, peak_a2 = np.unravel_index(np.argmax(va), va.shape)
ax_va.scatter(peak_a2, peak_v2, s=120, c=COLORS["truth"],
              edgecolors=COLORS["white"], linewidths=1.5, zorder=5)
ax_va.annotate("Peak\nattention", xy=(peak_a2, peak_v2),
               xytext=(peak_a2 + va.shape[1]*0.08, peak_v2 - va.shape[0]*0.1),
               fontsize=8, color=COLORS["white"],
               arrowprops=dict(arrowstyle="->", color=COLORS["white"], lw=1.0))

# Time axis labels for audio waveform aligned to A→V
n_audio_steps = av.shape[0]
tick_positions = np.linspace(0, n_audio_steps - 1, 6).astype(int)
tick_labels    = [f"{clip_duration * p / (n_audio_steps-1):.1f}s"
                  for p in tick_positions]
ax_av.set_xticks(np.linspace(0, av.shape[1]-1, 6).astype(int))
ax_va.set_xticks(np.linspace(0, va.shape[1]-1, 6).astype(int))
ax_va.set_xticklabels(
    [f"{clip_duration * p / max(va.shape[1]-1,1):.1f}s"
     for p in np.linspace(0, va.shape[1]-1, 6).astype(int)],
    fontsize=8
)

output_path = os.path.join(ROOT, "output", "plots", "Z_attention_heatmap.png")
fig.savefig(output_path, dpi=150, bbox_inches="tight",
            facecolor=COLORS["bg"], edgecolor=COLORS["bg"])
plt.close(fig)
print(f"\n  ✓ Saved → {output_path}")
print(f"\n  Interpretation:")
print(f"  A→V: Bright clusters = audio moments strongly attending to specific video frames")
print(f"  V→A: Bright clusters = video frames strongly attending to specific audio moments")
print(f"  Diagonal structure = synchronized co-attention (audio & video aligned in time)")
print(f"  Off-diagonal = cross-temporal attention (model looking across time boundaries)")