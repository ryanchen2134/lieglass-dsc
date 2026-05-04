"""
plot_Q_cross_fusion_benefit.py — Why cross-fusion catches what single modalities miss
Run: python plot_Q_cross_fusion_benefit.py
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import apply_theme, save_plot, COLORS
apply_theme()

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
fig.suptitle("Cross-Fusion Catches What Neither Modality Sees Alone", fontsize=14,
             fontweight="bold")

t = np.linspace(0, 10, 500)

# ── Audio signal (vocal stress) ───────────────────────────────────────────────
audio = 0.3 + 0.15 * np.sin(2 * t) + 0.1 * np.random.RandomState(1).randn(500)
# Spike at lie moment (t~6.5)
audio += 0.6 * np.exp(-((t - 6.5)**2) / 0.3)

axes[0].plot(t, audio, color=COLORS["truth"], linewidth=2)
axes[0].fill_between(t, audio, 0.3, alpha=0.15, color=COLORS["truth"])
axes[0].axvspan(6.0, 7.0, alpha=0.15, color=COLORS["red"], label="Lie moment")
axes[0].set_ylabel("Vocal Stress", fontsize=10)
axes[0].set_title("Audio Stream — Wav2Vec2", fontsize=10, color=COLORS["truth"],
                   loc="left")
axes[0].set_ylim(0, 1.5)
axes[0].legend(loc="upper left", fontsize=9)
axes[0].text(6.5, 1.3, "Spike detected", ha="center", fontsize=9,
             color=COLORS["red"], fontweight="bold")

# ── Visual signal (micro-expression) ─────────────────────────────────────────
visual = 0.4 + 0.1 * np.sin(1.5 * t + 1) + 0.08 * np.random.RandomState(2).randn(500)
# Spike slightly after audio
visual += 0.55 * np.exp(-((t - 6.8)**2) / 0.25)

axes[1].plot(t, visual, color=COLORS["red"], linewidth=2)
axes[1].fill_between(t, visual, 0.4, alpha=0.15, color=COLORS["red"])
axes[1].axvspan(6.0, 7.0, alpha=0.15, color=COLORS["red"])
axes[1].set_ylabel("Facial Motion", fontsize=10)
axes[1].set_title("Visual Stream — ArcFace", fontsize=10, color=COLORS["red"],
                   loc="left")
axes[1].set_ylim(0, 1.5)
axes[1].text(6.8, 1.3, "Spike detected", ha="center", fontsize=9,
             color=COLORS["red"], fontweight="bold")

# ── Fused confidence ──────────────────────────────────────────────────────────
fused = 1 / (1 + np.exp(-5 * (audio + visual - 1.5)))

axes[2].plot(t, fused, color=COLORS["white"], linewidth=2.5)
axes[2].fill_between(t, fused, 0, alpha=0.2, color=COLORS["red"])
axes[2].axhline(0.5, color=COLORS["gray"], linestyle="--",
                linewidth=1.2, alpha=0.7, label="Decision threshold")
axes[2].axvspan(6.0, 7.0, alpha=0.2, color=COLORS["red"])
axes[2].set_ylabel("Lie Score", fontsize=10)
axes[2].set_xlabel("Time (seconds)", fontsize=10)
axes[2].set_title("Cross-Fusion Output", fontsize=10, color=COLORS["white"],
                   loc="left")
axes[2].set_ylim(0, 1.05)
axes[2].legend(loc="upper left", fontsize=9)

# Annotation
axes[2].annotate("High confidence\ndeception detected",
                 xy=(6.65, 0.92), xytext=(4.5, 0.75),
                 fontsize=9, color=COLORS["red"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.5))

plt.tight_layout()
save_plot(fig, "Q_cross_fusion_benefit.png")
