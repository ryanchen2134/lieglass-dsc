"""
plot_P_model_pipeline.py — Full model architecture flow diagram
Run: python plot_P_model_pipeline.py
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import apply_theme, save_plot, COLORS
apply_theme()

fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.axis("off")

def box(ax, x, y, w, h, label, sublabel, color, text_color=None):
    tc = text_color or COLORS["white"]
    rect = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.15",
                           facecolor=color, edgecolor=COLORS["grid"],
                           linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h*0.62, label, ha="center", va="center",
            fontsize=10, fontweight="bold", color=tc, zorder=3)
    ax.text(x + w/2, y + h*0.28, sublabel, ha="center", va="center",
            fontsize=8, color=COLORS["gray_light"], zorder=3)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=COLORS["gray"],
                                lw=1.8), zorder=4)

# Input
box(ax, 0.3, 2.5, 2.0, 2.0, "AR Glasses", "XReal One\n+ Eye Camera",
    COLORS["panel"])

# Split arrow
arrow(ax, 2.3, 4.0, 3.2, 5.2)
arrow(ax, 2.3, 3.0, 3.2, 1.8)

# Audio stream
box(ax, 3.2, 4.6, 2.2, 1.4, "Audio", "Raw Waveform\n16kHz mono",
    "#1a2a3a")
arrow(ax, 5.4, 5.3, 6.3, 5.3)
box(ax, 6.3, 4.6, 2.5, 1.4, "Wav2Vec2", "Speech encoder\n(frozen)",
    "#1a2a3a")
arrow(ax, 8.8, 5.3, 9.7, 5.3)
box(ax, 9.7, 4.6, 2.2, 1.4, "Audio\nFeatures", "768-d embedding\nper frame",
    "#1a2a3a")

# Visual stream
box(ax, 3.2, 1.0, 2.2, 1.4, "Video", "Grayscale frames\n224×224",
    "#2a1a1a")
arrow(ax, 5.4, 1.7, 6.3, 1.7)
box(ax, 6.3, 1.0, 2.5, 1.4, "ArcFace", "Face encoder\n(frozen)",
    "#2a1a1a")
arrow(ax, 8.8, 1.7, 9.7, 1.7)
box(ax, 9.7, 1.0, 2.2, 1.4, "Visual\nFeatures", "384-d embedding\nper frame",
    "#2a1a1a")

# Stream labels
ax.text(3.2, 6.3, "AUDIO STREAM", fontsize=8, color=COLORS["truth"],
        fontweight="bold", fontfamily="monospace")
ax.text(3.2, 0.6, "VISUAL STREAM", fontsize=8, color=COLORS["red"],
        fontweight="bold", fontfamily="monospace")

# Cross fusion
arrow(ax, 11.9, 5.3, 12.8, 3.9)
arrow(ax, 11.9, 1.7, 12.8, 3.1)
box(ax, 12.8, 2.5, 2.2, 2.0, "Cross-\nFusion", "Multi-stage\nbidirectional\ncross-attention",
    COLORS["red"])

# Output
arrow(ax, 15.0, 3.5, 15.6, 3.5)
box(ax, 15.6, 2.8, 0.3, 1.4, "", "", COLORS["red"])
ax.text(15.75, 3.5, "Lie\nScore", ha="center", va="center",
        fontsize=9, fontweight="bold", color=COLORS["white"], zorder=5)

ax.set_title("LieGlass Model Architecture", pad=16)
plt.tight_layout()
save_plot(fig, "P_model_pipeline.png")
