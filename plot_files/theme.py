"""
theme.py — Shared LieGlass visual theme for all plots
======================================================
Import this at the top of every plot script:
    from theme import apply_theme, COLORS, CKPT_DIR, ROOT, OUTPUT_DIR
"""

import os
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Project paths ─────────────────────────────────────────────────────────────
# plot_files/ is one level below project root
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR   = os.path.join(ROOT, "checkpoints", "20260429_214636")
OUTPUT_DIR = os.path.join(ROOT, "output", "plots")
CACHE_DIR  = os.path.join(ROOT, "output", "cache")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)

# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    "bg":          "#0a0a0a",   # slide background
    "panel":       "#111111",   # slightly lighter panel
    "grid":        "#1e1e1e",   # subtle gridlines
    "red":         "#D92323",   # primary accent — matches confusion matrix
    "red_light":   "#FF6B6B",   # lighter red for secondary elements
    "white":       "#FFFFFF",   # primary text
    "gray":        "#888888",   # secondary text / muted elements
    "gray_light":  "#CCCCCC",   # axis labels
    "truth":       "#4C9BE8",   # blue — truth class
    "lie":         "#D92323",   # red — lie class
}

# ── Apply theme globally ───────────────────────────────────────────────────────

def apply_theme():
    """Call once at the top of each script to apply the LieGlass dark theme."""
    mpl.rcParams.update({
        # Figure
        "figure.facecolor":     COLORS["bg"],
        "figure.edgecolor":     COLORS["bg"],
        "figure.dpi":           150,

        # Axes
        "axes.facecolor":       COLORS["panel"],
        "axes.edgecolor":       COLORS["grid"],
        "axes.labelcolor":      COLORS["gray_light"],
        "axes.titlecolor":      COLORS["white"],
        "axes.titlesize":       14,
        "axes.titleweight":     "bold",
        "axes.titlepad":        14,
        "axes.labelsize":       11,
        "axes.labelpad":        8,
        "axes.grid":            True,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.spines.left":     True,
        "axes.spines.bottom":   True,

        # Grid
        "grid.color":           COLORS["grid"],
        "grid.linewidth":       0.8,
        "grid.alpha":           1.0,

        # Ticks
        "xtick.color":          COLORS["gray"],
        "ytick.color":          COLORS["gray"],
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,

        # Legend
        "legend.facecolor":     COLORS["panel"],
        "legend.edgecolor":     COLORS["grid"],
        "legend.labelcolor":    COLORS["white"],
        "legend.fontsize":      10,
        "legend.framealpha":    1.0,

        # Lines
        "lines.linewidth":      2.0,
        "lines.antialiased":    True,

        # Text
        "text.color":           COLORS["white"],
        "font.family":          ["DejaVu Sans", "sans-serif"],

        # Save
        "savefig.facecolor":    COLORS["bg"],
        "savefig.edgecolor":    COLORS["bg"],
        "savefig.bbox":         "tight",
        "savefig.dpi":          150,
    })

def save_plot(fig, filename):
    """Save plot to output/plots/ with consistent settings."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  ✓ Saved → {path}")
    return path
