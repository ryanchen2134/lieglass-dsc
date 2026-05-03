"""
plot_E_human_vs_model.py — Human vs LieGlass accuracy comparison
Run: python plot_E_human_vs_model.py
"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import apply_theme, save_plot, COLORS
apply_theme()

labels = ["Chance", "Human\nBaseline", "LieGlass\n(Ours)"]
values = [0.50,      0.54,              0.765]
colors = [COLORS["grid"], COLORS["gray"], COLORS["red"]]
errors = [0.0,        0.04,              0.027]   # ±std where known

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(labels, values, color=colors, width=0.45,
              edgecolor=COLORS["bg"], linewidth=1.5,
              yerr=errors, capsize=6,
              error_kw={"ecolor": COLORS["gray_light"], "elinewidth": 1.5})

# Value labels
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{v*100:.1f}%", ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=COLORS["white"])

# Delta annotation
ax.annotate("", xy=(2, 0.765), xytext=(1, 0.54),
            arrowprops=dict(arrowstyle="->", color=COLORS["red"],
                            lw=2.0, connectionstyle="arc3,rad=0.2"))
ax.text(1.72, 0.655, "+22.5pp", fontsize=11, color=COLORS["red"],
        fontweight="bold", fontfamily="monospace")

ax.set_ylim(0, 1.0)
ax.set_ylabel("Accuracy")
ax.set_title("LieGlass vs Human Lie Detection")
ax.axhline(0.5, color=COLORS["grid"], linestyle=":", linewidth=1.0, alpha=0.5)
plt.tight_layout()
save_plot(fig, "E_human_vs_model.png")
