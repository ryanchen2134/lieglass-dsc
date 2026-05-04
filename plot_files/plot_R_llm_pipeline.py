"""
plot_R_llm_pipeline.py — Three-layer pipeline: Model + Whisper + Claude
Run: python plot_R_llm_pipeline.py
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import apply_theme, save_plot, COLORS
apply_theme()

fig, ax = plt.subplots(figsize=(15, 6))
ax.set_xlim(0, 15)
ax.set_ylim(0, 6)
ax.axis("off")
ax.set_title("Three-Layer Real-Time Pipeline", pad=16)

def box(ax, x, y, w, h, title, body, color, title_color=None):
    tc = title_color or COLORS["white"]
    rect = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.2",
                           facecolor=color, edgecolor=COLORS["grid"],
                           linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h*0.7, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color=tc, zorder=3)
    ax.text(x + w/2, y + h*0.3, body, ha="center", va="center",
            fontsize=8.5, color=COLORS["gray_light"], zorder=3)

def arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=COLORS["red"],
                                lw=2.0), zorder=4)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.18, label, ha="center", fontsize=8,
                color=COLORS["gray"], fontstyle="italic")

# Input
box(ax, 0.3, 2.2, 2.0, 1.6, "AR Glasses", "Audio + Video\ncaptured live",
    COLORS["panel"])

# Whisper
arrow(ax, 2.3, 3.0, 3.5, 4.5, "audio")
box(ax, 3.5, 3.8, 2.8, 1.4, "OpenAI Whisper", "Speech → Transcript\nin real time",
    "#1a2a1a")

# Model
arrow(ax, 2.3, 2.8, 3.5, 1.8, "video")
box(ax, 3.5, 0.8, 2.8, 1.4, "LieGlass Model", "Wav2Vec2 + ArcFace\n+ Cross-Fusion",
    "#2a1a1a")

# Claude
arrow(ax, 6.3, 4.5, 7.8, 3.8, "transcript")
arrow(ax, 6.3, 1.5, 7.8, 2.8, "lie score")

box(ax, 7.8, 2.3, 3.2, 2.0, "Claude Opus",
    "Contradiction detection\nFact checking\nProbing suggestions",
    COLORS["red"])

# HUD output
arrow(ax, 11.0, 3.3, 12.5, 3.3, "real-time")
box(ax, 12.5, 2.3, 2.2, 2.0, "HUD Display",
    "Lie score\nSuggestions\nTranscript",
    COLORS["panel"])

# Timing labels
ax.text(4.9, 0.4, "~real time", ha="center", fontsize=8,
        color=COLORS["red"], fontfamily="monospace")
ax.text(4.9, 3.4, "~real time", ha="center", fontsize=8,
        color=COLORS["red"], fontfamily="monospace")

plt.tight_layout()
save_plot(fig, "R_llm_pipeline.png")
