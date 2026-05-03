"""
plot_I_ablation.py — Per-modality ablation AUC bar chart
Run: python plot_I_ablation.py

NOTE: Values marked PLACEHOLDER should be replaced with real inference
results when available. Run inference with one modality zeroed:
  - Audio only:  zero out frames  (frames = torch.zeros_like(frames))
  - Visual only: zero out waveform (waveform = torch.zeros_like(waveform))
"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import apply_theme, save_plot, COLORS
apply_theme()

# Replace PLACEHOLDER values with real ablation results
conditions = [
    ("Audio Only",    0.68, True),   # PLACEHOLDER
    ("Visual Only",   0.63, True),   # PLACEHOLDER
    ("Full Fusion",   0.856, False),  # Real — from metrics.json mean AUC
]

labels    = [c[0] for c in conditions]
values    = [c[1] for c in conditions]
is_placeholder = [c[2] for c in conditions]
colors    = [COLORS["gray"] if p else COLORS["red"] for p in is_placeholder]
alphas    = [0.5 if p else 0.9 for p in is_placeholder]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(labels, values, color=colors, width=0.45,
              edgecolor=COLORS["bg"], linewidth=1.5)
for bar, color, alpha in zip(bars, colors, alphas):
    bar.set_alpha(alpha)

for bar, v, p in zip(bars, values, is_placeholder):
    label = f"{v:.3f}" + (" *" if p else "")
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            label, ha="center", fontsize=13, fontweight="bold",
            color=COLORS["white"])

ax.axhline(0.54, color=COLORS["gray"], linestyle="--",
           linewidth=1.5, alpha=0.7, label="Human baseline (0.54)")
ax.set_ylim(0.3, 1.0)
ax.set_ylabel("AUC-ROC")
ax.set_title("Fusion vs Single-Modality Performance")
ax.legend()
ax.text(0.98, 0.02, "* Placeholder — replace with real ablation results",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color=COLORS["gray"], fontstyle="italic")
plt.tight_layout()
save_plot(fig, "I_ablation.png")
