"""
plot_X_what_model_sees.py — Raw glasses frame vs model input with landmarks
Run: python plot_X_what_model_sees.py
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import apply_theme, save_plot, COLORS
apply_theme()

fig = plt.figure(figsize=(14, 6))
fig.suptitle("What Goes Into the Model", fontsize=14, fontweight="bold")

# ── Left: simulated raw grayscale face frame ──────────────────────────────────
ax1 = fig.add_subplot(1, 3, 1)
ax1.set_facecolor(COLORS["bg"])

face = np.zeros((224, 224), dtype=np.float32)
Y, X = np.ogrid[:224, :224]
# Skin
skin = ((X-112)**2/70**2 + (Y-110)**2/90**2) < 1
face[skin] = 0.65
# Hair
hair = ((X-112)**2/70**2 + (Y-50)**2/50**2) < 1
face[hair] = 0.2
# Eyes
for ex, ey in [(82, 90), (142, 90)]:
    e = ((X-ex)**2 + (Y-ey)**2) < 144
    face[e] = 0.1
    ep = ((X-ex)**2 + (Y-ey)**2) < 36
    face[ep] = 0.05
# Nose
face[120:140, 108:116] = 0.45
# Mouth
face[148:155, 90:134] = 0.2
# Neck
face[195:224, 85:139] = 0.55
face = np.clip(face + np.random.RandomState(5).randn(224,224)*0.03, 0, 1)

ax1.imshow(face, cmap="gray", vmin=0, vmax=1)
ax1.set_title("Glasses Camera\n(Grayscale Stream)", fontsize=10,
               fontweight="bold", color=COLORS["gray_light"])
ax1.set_xticks([]); ax1.set_yticks([])
for spine in ax1.spines.values():
    spine.set_edgecolor(COLORS["grid"])

# ── Middle: same frame with landmarks overlay ─────────────────────────────────
ax2 = fig.add_subplot(1, 3, 2)
ax2.set_facecolor(COLORS["bg"])
ax2.imshow(face, cmap="gray", vmin=0, vmax=1, alpha=0.6)

# Landmark points (approximate face landmarks)
landmarks = [
    # Jawline
    *[(int(112 - 65*np.sin(a)), int(180 - 40*np.cos(a)))
      for a in np.linspace(-np.pi/2, np.pi/2, 9)],
    # Left eye
    *[(int(82 + 14*np.cos(a)), int(90 + 8*np.sin(a)))
      for a in np.linspace(0, 2*np.pi, 6)],
    # Right eye
    *[(int(142 + 14*np.cos(a)), int(90 + 8*np.sin(a)))
      for a in np.linspace(0, 2*np.pi, 6)],
    # Mouth
    *[(int(112 + 22*np.cos(a)), int(150 + 7*np.sin(a)))
      for a in np.linspace(0, 2*np.pi, 8)],
    # Nose bridge
    (112, 95), (112, 105), (112, 115), (105, 125), (119, 125),
    # Eyebrows
    *[(int(82 + 20*np.cos(a)), int(74 + 4*np.sin(a)))
      for a in np.linspace(-np.pi, 0, 5)],
    *[(int(142 + 20*np.cos(a)), int(74 + 4*np.sin(a)))
      for a in np.linspace(-np.pi, 0, 5)],
]

lx = [l[0] for l in landmarks]
ly = [l[1] for l in landmarks]
ax2.scatter(lx, ly, s=12, c=COLORS["red"], zorder=5, alpha=0.9)

# Audio waveform below
t_wave = np.linspace(0, 1, 200)
wave   = 0.3 * np.sin(20 * np.pi * t_wave) * np.exp(-2 * t_wave)
ax2_inset = ax2.inset_axes([0, -0.22, 1, 0.18])
ax2_inset.set_facecolor(COLORS["panel"])
ax2_inset.plot(t_wave, wave, color=COLORS["truth"], linewidth=1.2)
ax2_inset.axhline(0, color=COLORS["grid"], linewidth=0.8)
ax2_inset.set_xticks([]); ax2_inset.set_yticks([])
ax2_inset.set_xlabel("Audio waveform (16kHz)", fontsize=8,
                      color=COLORS["gray"])

ax2.set_title("Model Input\n(Landmarks + Waveform)", fontsize=10,
               fontweight="bold", color=COLORS["red"])
ax2.set_xticks([]); ax2.set_yticks([])
for spine in ax2.spines.values():
    spine.set_edgecolor(COLORS["red"])
    spine.set_linewidth(1.5)

# ── Right: output ─────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(1, 3, 3)
ax3.set_facecolor(COLORS["panel"])
ax3.set_xlim(0, 10); ax3.set_ylim(0, 10)
ax3.axis("off")
ax3.set_title("Model Output\n(HUD Display)", fontsize=10,
               fontweight="bold", color="#2ecc71")

from matplotlib.patches import FancyBboxPatch
hud = FancyBboxPatch((0.5, 0.5), 9, 9,
                      boxstyle="round,pad=0.3",
                      facecolor="#050510", edgecolor="#2ecc71",
                      linewidth=2)
ax3.add_patch(hud)

ax3.text(5, 8.5, "LIE SCORE", ha="center", fontsize=10,
         color="#2ecc71", fontfamily="monospace")

# Score bar
bar = FancyBboxPatch((1, 7.2), 8, 0.7,
                      boxstyle="round,pad=0.05",
                      facecolor=COLORS["grid"], edgecolor="none")
ax3.add_patch(bar)
bar_f = FancyBboxPatch((1, 7.2), 5.6, 0.7,
                        boxstyle="round,pad=0.05",
                        facecolor=COLORS["red"], edgecolor="none")
ax3.add_patch(bar_f)
ax3.text(5, 6.7, "70%  —  HIGH", ha="center", fontsize=10,
         color=COLORS["red"], fontweight="bold")

ax3.text(5, 5.8, "⚠ Contradiction", ha="center", fontsize=9,
         color=COLORS["red"], fontweight="bold")
ax3.text(5, 5.2, "\"I was home all evening\"\nvs. \"stopped by office\"",
         ha="center", fontsize=8, color=COLORS["gray_light"])

ax3.text(5, 3.8, "💡 Suggestion", ha="center", fontsize=9,
         color="#2ecc71", fontweight="bold")
ax3.text(5, 3.0, "\"What time did you\nleave the office?\"",
         ha="center", fontsize=8.5, color=COLORS["white"])

for spine in ax3.spines.values():
    spine.set_edgecolor("#2ecc71")

plt.tight_layout()
save_plot(fig, "X_what_model_sees.png")
