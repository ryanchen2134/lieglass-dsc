"""
plot_O_dataset_composition.py — Dataset split and class balance donuts
Run: python plot_O_dataset_composition.py
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import apply_theme, save_plot, COLORS
apply_theme()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Training Dataset — 2,051 Video Clips", fontsize=15, fontweight="bold")

# ── Donut 1: Dataset source ───────────────────────────────────────────────────
sizes1  = [1435, 120, 496]
labels1 = ["DOLOS\n(Game Show)", "Real Life\nTrials", "Would I\nLie To You?"]
colors1 = [COLORS["red"], COLORS["truth"], COLORS["gray"]]

wedges1, _ = axes[0].pie(sizes1, colors=colors1, startangle=90,
                          wedgeprops=dict(width=0.55, edgecolor=COLORS["bg"],
                                          linewidth=3))
axes[0].set_title("Dataset Sources", fontsize=12, fontweight="bold", pad=12)

for i, (wedge, label, size) in enumerate(zip(wedges1, labels1, sizes1)):
    angle = (wedge.theta1 + wedge.theta2) / 2
    import numpy as np
    x = 0.75 * np.cos(np.radians(angle))
    y = 0.75 * np.sin(np.radians(angle))
    axes[0].text(x, y, f"{label}\n{size}", ha="center", va="center",
                 fontsize=9, fontweight="bold", color=COLORS["white"])

axes[0].text(0, 0, "2,051\nclips", ha="center", va="center",
             fontsize=14, fontweight="bold", color=COLORS["white"])

# ── Donut 2: Class balance ────────────────────────────────────────────────────
sizes2  = [1130, 921]
labels2 = ["Deceptive", "Truthful"]
colors2 = [COLORS["red"], COLORS["truth"]]

wedges2, _ = axes[1].pie(sizes2, colors=colors2, startangle=90,
                          wedgeprops=dict(width=0.55, edgecolor=COLORS["bg"],
                                          linewidth=3))
axes[1].set_title("Class Distribution", fontsize=12, fontweight="bold", pad=12)

import numpy as np
for wedge, label, size in zip(wedges2, labels2, sizes2):
    angle = (wedge.theta1 + wedge.theta2) / 2
    x = 0.75 * np.cos(np.radians(angle))
    y = 0.75 * np.sin(np.radians(angle))
    pct = size / sum(sizes2) * 100
    axes[1].text(x, y, f"{label}\n{pct:.0f}%", ha="center", va="center",
                 fontsize=10, fontweight="bold", color=COLORS["white"])

axes[1].text(0, 0, "55 / 45\nsplit", ha="center", va="center",
             fontsize=12, fontweight="bold", color=COLORS["white"])

plt.tight_layout()
save_plot(fig, "O_dataset_composition.png")
