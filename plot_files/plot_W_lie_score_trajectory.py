"""
plot_W_lie_score_trajectory.py — Live lie score over a single clip timeline
Run: python plot_W_lie_score_trajectory.py
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import apply_theme, save_plot, COLORS
apply_theme()

fig, ax = plt.subplots(figsize=(14, 5))

t    = np.linspace(0, 30, 1000)
rng  = np.random.RandomState(7)

# Base signal — mostly low
score = 0.2 + 0.08 * np.sin(0.8 * t) + 0.05 * rng.randn(1000)
score = np.clip(score, 0, 1)

# Two lie spikes
score += 0.55 * np.exp(-((t - 11)**2) / 1.2)
score += 0.70 * np.exp(-((t - 22)**2) / 0.9)
score  = np.clip(score, 0, 1)

# Smooth slightly
from scipy.ndimage import uniform_filter1d
score = uniform_filter1d(score, size=20)

# Color fill by score
ax.fill_between(t, score, 0.5, where=(score >= 0.5),
                color=COLORS["red"], alpha=0.35, label="Above threshold")
ax.fill_between(t, score, 0,   where=(score < 0.5),
                color=COLORS["truth"], alpha=0.15)
ax.plot(t, score, color=COLORS["white"], linewidth=2.0, zorder=3)

# Threshold line
ax.axhline(0.5, color=COLORS["gray"], linestyle="--",
           linewidth=1.5, alpha=0.8, label="Decision threshold (0.5)")

# Annotate lie moments
for tx, label in [(11, "Lie detected\n\"I was home\""), (22, "Lie detected\n\"Never met him\"")]:
    peak = score[np.argmin(np.abs(t - tx))]
    ax.scatter(tx, peak, s=150, color=COLORS["red"],
               edgecolors=COLORS["white"], linewidths=1.5, zorder=5)
    ax.annotate(label, xy=(tx, peak), xytext=(tx + 1.5, peak + 0.12),
                fontsize=9, color=COLORS["red"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.2))

# Conversation marker labels
events = [(5, "Subject\nintroduced"), (11, ""), (17, "Follow-up\nquestion"), (22, "")]
for ex, elabel in events:
    if elabel:
        ax.axvline(ex, color=COLORS["grid"], linewidth=1.0, alpha=0.6)
        ax.text(ex, -0.08, elabel, ha="center", fontsize=8,
                color=COLORS["gray"], va="top")

ax.set_xlim(0, 30)
ax.set_ylim(-0.05, 1.1)
ax.set_xlabel("Conversation Time (seconds)", fontsize=11)
ax.set_ylabel("Lie Score", fontsize=11)
ax.set_title("Real-Time Lie Score — Single Conversation Clip")
ax.legend(loc="upper left")

plt.tight_layout()
save_plot(fig, "W_lie_score_trajectory.png")
