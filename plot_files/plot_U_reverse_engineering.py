"""
plot_U_reverse_engineering.py — Scrambled → decoded → clean feed journey
Run: python plot_U_reverse_engineering.py
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import apply_theme, save_plot, COLORS
apply_theme()

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Reverse Engineering the Hidden Camera Feed", fontsize=14,
             fontweight="bold")

rng = np.random.RandomState(42)

# Frame 1: Scrambled
scrambled = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
axes[0].imshow(scrambled)
axes[0].set_title("Stage 1: Raw Developer Stream", fontsize=10,
                   fontweight="bold", color=COLORS["red"])
axes[0].set_xlabel("Completely scrambled —\nno usable signal", fontsize=9,
                    color=COLORS["gray"])
axes[0].set_xticks([]); axes[0].set_yticks([])
for spine in axes[0].spines.values():
    spine.set_edgecolor(COLORS["red"])
    spine.set_linewidth(2)

# Frame 2: Partially decoded
partial = rng.randint(0, 255, (224, 224), dtype=np.uint8)
# Add some face-like structure
for i in range(60, 160):
    for j in range(60, 160):
        partial[i, j] = int(partial[i, j] * 0.3 + 120)
axes[1].imshow(partial, cmap="gray")
axes[1].set_title("Stage 2: Manual Realignment", fontsize=10,
                   fontweight="bold", color="#ffaa00")
axes[1].set_xlabel("Structure emerging —\nmanual pixel correction", fontsize=9,
                    color=COLORS["gray"])
axes[1].set_xticks([]); axes[1].set_yticks([])
for spine in axes[1].spines.values():
    spine.set_edgecolor("#ffaa00")
    spine.set_linewidth(2)

# Frame 3: Clean grayscale
clean = np.zeros((224, 224), dtype=np.uint8)
# Simulate a face silhouette
Y, X = np.ogrid[:224, :224]
# Skin tone background
clean += 80
# Face oval
mask = ((X - 112)**2 / 60**2 + (Y - 100)**2 / 80**2) < 1
clean[mask] = 160
# Eyes
for ex, ey in [(85, 85), (139, 85)]:
    emask = ((X - ex)**2 + (Y - ey)**2) < 100
    clean[emask] = 30
# Mouth
for mx in range(90, 135):
    clean[125, mx] = 30
axes[2].imshow(clean, cmap="gray")
axes[2].set_title("Stage 3: Clean Grayscale Stream", fontsize=10,
                   fontweight="bold", color="#2ecc71")
axes[2].set_xlabel("Live grayscale feed —\nready for model inference", fontsize=9,
                    color=COLORS["gray"])
axes[2].set_xticks([]); axes[2].set_yticks([])
for spine in axes[2].spines.values():
    spine.set_edgecolor("#2ecc71")
    spine.set_linewidth(2)

# Arrows between frames
for ax_from, ax_to in [(axes[0], axes[1]), (axes[1], axes[2])]:
    fig.text(
        (ax_from.get_position().x1 + ax_to.get_position().x0) / 2,
        0.52, "→", ha="center", va="center",
        fontsize=28, color=COLORS["white"]
    )

plt.tight_layout()
save_plot(fig, "U_reverse_engineering.png")
