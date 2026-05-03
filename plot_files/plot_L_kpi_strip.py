"""
plot_L_kpi_strip.py — KPI headline summary card
Run: python plot_L_kpi_strip.py
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as patches
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import apply_theme, save_plot, COLORS
apply_theme()

kpis = [
    ("Mean AUC",      "0.856",  "±0.017"),
    ("Accuracy",      "76.5%",  "±2.7%"),
    ("F1 Score",      "0.757",  "±0.031"),
    ("Lie Recall",    "82.3%",  "best fold"),
    ("vs. Human",     "+22.5pp","54% → 76.5%"),
]

fig, ax = plt.subplots(figsize=(14, 3.5))
ax.set_xlim(0, len(kpis))
ax.set_ylim(0, 1)
ax.axis("off")

card_w = 0.85
gap    = (1 - card_w) / 2

for i, (label, value, sub) in enumerate(kpis):
    x = i + gap
    # Card background
    rect = patches.FancyBboxPatch(
        (x, 0.08), card_w, 0.84,
        boxstyle="round,pad=0.02",
        facecolor=COLORS["panel"],
        edgecolor=COLORS["red"] if i == len(kpis)-1 else COLORS["grid"],
        linewidth=2 if i == len(kpis)-1 else 1
    )
    ax.add_patch(rect)

    # Value
    ax.text(x + card_w / 2, 0.62, value,
            ha="center", va="center",
            fontsize=22, fontweight="bold",
            color=COLORS["red"] if i == len(kpis)-1 else COLORS["white"])

    # Label
    ax.text(x + card_w / 2, 0.35, label,
            ha="center", va="center",
            fontsize=10, color=COLORS["gray_light"])

    # Sub
    ax.text(x + card_w / 2, 0.17, sub,
            ha="center", va="center",
            fontsize=8, color=COLORS["gray"],
            fontfamily="monospace")

fig.suptitle("LieGlass — Model Results at a Glance",
             fontsize=13, fontweight="bold", color=COLORS["white"], y=1.02)
plt.tight_layout()
save_plot(fig, "L_kpi_strip.png")
