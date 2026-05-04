"""
plot_Y_product_roadmap.py — Limitations reframed as a growth roadmap
Run: python plot_Y_product_roadmap.py
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import apply_theme, save_plot, COLORS
apply_theme()

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis("off")
ax.set_title("LieGlass — What's Next", pad=16)

tracks = [
    ("Near Term",  "#2ecc71", 0.6, [
        "Improved prompt engineering",
        "Better LLM probing templates",
        "Offline mode (no internet)",
        "Whisper fine-tuning",
    ]),
    ("Mid Term",   "#ffaa00", 4.9, [
        "Linguistic pattern analysis",
        "Proprietary deception dataset",
        "Per-speaker calibration",
        "Mobile app companion",
    ]),
    ("Long Term",  COLORS["red"], 9.2, [
        "Miniaturized onboard hardware",
        "Multi-language support",
        "Law enforcement integration",
        "Clinical / HR applications",
    ]),
]

# Timeline line
ax.axhline(0.8, xmin=0.04, xmax=0.96,
           color=COLORS["grid"], linewidth=2, zorder=1)

for label, color, x, items in tracks:
    # Node on timeline
    ax.scatter(x + 1.9, 0.8, s=200, color=color, zorder=3,
               edgecolors=COLORS["white"], linewidths=1.5)

    # Card
    rect = FancyBboxPatch((x, 1.2), 3.8, 4.4,
                           boxstyle="round,pad=0.2",
                           facecolor=COLORS["panel"],
                           edgecolor=color, linewidth=2)
    ax.add_patch(rect)

    # Connector
    ax.plot([x + 1.9, x + 1.9], [0.8, 1.2],
            color=color, linewidth=1.5, linestyle="--")

    # Label
    ax.text(x + 1.9, 5.3, label, ha="center", fontsize=12,
            fontweight="bold", color=color)

    # Items
    for i, item in enumerate(items):
        ax.text(x + 0.3, 4.6 - i * 0.75, f"→  {item}",
                fontsize=9, color=COLORS["gray_light"])

# Current marker
ax.text(0.5, 0.5, "NOW", fontsize=8, color=COLORS["gray"],
        fontfamily="monospace")

plt.tight_layout()
save_plot(fig, "Y_product_roadmap.png")
