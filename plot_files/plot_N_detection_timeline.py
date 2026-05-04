"""
plot_N_detection_timeline.py — Lie detection technology evolution timeline
Run: python plot_N_detection_timeline.py
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import apply_theme, save_plot, COLORS
apply_theme()

fig, ax = plt.subplots(figsize=(14, 6))

milestones = [
    (1,  "1920s",      "Polygraph",             "~54%\naccuracy",   COLORS["gray"]),
    (3,  "1970s",      "Voice Stress\nAnalysis", "~55%\naccuracy",  COLORS["gray"]),
    (5,  "2000s",      "Facial Action\nCoding",  "~58%\naccuracy",  COLORS["gray"]),
    (7,  "2010s",      "Unimodal\nML Models",    "~65%\naccuracy",  COLORS["gray"]),
    (9,  "2024",       "LieGlass\n(Ours)",        "76.5%\naccuracy", COLORS["red"]),
]

# Timeline bar
ax.axhline(0.5, xmin=0.05, xmax=0.95, color=COLORS["grid"],
           linewidth=2, zorder=1)

for x, year, label, acc, color in milestones:
    # Node
    ax.scatter(x, 0.5, s=200, color=color, zorder=3,
               edgecolors=COLORS["white"], linewidths=1.5)

    # Year above
    ax.text(x, 0.62, year, ha="center", fontsize=9,
            color=COLORS["gray"], fontfamily="monospace")

    # Label below
    ax.text(x, 0.36, label, ha="center", va="top",
            fontsize=10, fontweight="bold",
            color=COLORS["white"] if color == COLORS["red"] else COLORS["gray_light"])

    # Accuracy
    ax.text(x, 0.18, acc, ha="center", va="top",
            fontsize=9, color=color, fontweight="bold",
            fontfamily="monospace")

# Highlight LieGlass
ax.annotate("", xy=(9, 0.5), xytext=(7.5, 0.5),
            arrowprops=dict(arrowstyle="fancy", color=COLORS["red"],
                            lw=1.5, mutation_scale=20))

ax.set_xlim(0, 10.5)
ax.set_ylim(0, 0.85)
ax.axis("off")
ax.set_title("A Century of Lie Detection — LieGlass Leads the Field", pad=16)

plt.tight_layout()
save_plot(fig, "N_detection_timeline.png")
