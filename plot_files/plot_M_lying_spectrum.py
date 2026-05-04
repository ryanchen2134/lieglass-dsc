"""
plot_M_lying_spectrum.py — Deception spectrum from complete lie to technically true
Run: python plot_M_lying_spectrum.py
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import apply_theme, save_plot, COLORS
apply_theme()

fig, ax = plt.subplots(figsize=(14, 5))

# Gradient spectrum bar
gradient = np.linspace(0, 1, 300).reshape(1, -1)
ax.imshow(gradient, aspect="auto", extent=[0, 10, 0.6, 1.0],
          cmap="RdYlGn", alpha=0.85, zorder=1)

# Spectrum labels
ax.text(0.15, 0.8, "Complete\nLie", ha="center", va="center",
        fontsize=11, fontweight="bold", color=COLORS["white"],
        transform=ax.transData)
ax.text(5.0, 0.8, "Technically\nTrue", ha="center", va="center",
        fontsize=11, fontweight="bold", color=COLORS["bg"],
        transform=ax.transData)
ax.text(9.85, 0.8, "Complete\nTruth", ha="center", va="center",
        fontsize=11, fontweight="bold", color=COLORS["bg"],
        transform=ax.transData)

# Example markers
examples = [
    (1.5,  "\"I was never there\"",          "Fabricated alibi"),
    (4.0,  "\"Driven by generative AI\"",     "Misleading framing"),
    (5.5,  "\"Hands-free access\"",           "Omits requirements"),
    (7.5,  "\"Universal solution\"",          "Overstated scope"),
    (9.2,  "\"Uses a language model\"",       "Literally accurate"),
]

for x, quote, label in examples:
    ax.annotate("", xy=(x, 0.6), xytext=(x, 0.35),
                arrowprops=dict(arrowstyle="->", color=COLORS["white"],
                                lw=1.5), zorder=3)
    ax.text(x, 0.28, quote, ha="center", va="top",
            fontsize=8.5, color=COLORS["white"],
            fontweight="bold", wrap=True,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["panel"],
                      edgecolor=COLORS["grid"], linewidth=1))
    ax.text(x, 0.12, label, ha="center", va="top",
            fontsize=7.5, color=COLORS["gray"], fontstyle="italic")

ax.set_xlim(0, 10)
ax.set_ylim(0, 1.1)
ax.axis("off")
ax.set_title("Deception Isn't Binary — It's a Spectrum", pad=16)

plt.tight_layout()
save_plot(fig, "M_lying_spectrum.png")
