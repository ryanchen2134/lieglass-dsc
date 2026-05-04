"""
plot_S_observe_vs_act.py — Competitors observe only; LieGlass acts in real time
Run: python plot_S_observe_vs_act.py
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import apply_theme, save_plot, COLORS
apply_theme()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Real-Time Intelligence Changes Everything", fontsize=14,
             fontweight="bold")

def draw_timeline(ax, title, title_color, steps, highlight_idx, note):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) + 1)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold",
                 color=title_color, pad=12)

    for i, (step, detail) in enumerate(reversed(steps)):
        y = i + 0.5
        is_hl = (len(steps) - 1 - i) == highlight_idx
        color = COLORS["red"] if is_hl else COLORS["panel"]
        edge  = COLORS["red"] if is_hl else COLORS["grid"]
        rect = FancyBboxPatch((0.5, y), 9, 0.7,
                               boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor=edge,
                               linewidth=2 if is_hl else 1, zorder=2)
        ax.add_patch(rect)
        ax.text(1.0, y + 0.35, step, va="center", fontsize=10,
                fontweight="bold" if is_hl else "normal",
                color=COLORS["white"])
        ax.text(9.0, y + 0.35, detail, va="center", ha="right",
                fontsize=8.5, color=COLORS["gray_light"])

    ax.text(5, len(steps) + 0.7, note, ha="center", fontsize=9,
            color=COLORS["gray"], fontstyle="italic")

# Competitors
draw_timeline(axes[0],
    "Traditional Tools", COLORS["gray"],
    [
        ("Conversation happens",       "Subject answers questions"),
        ("Recording analyzed",         "Audio/video processed"),
        ("Report generated",           "Hours or days later"),
        ("Results reviewed",           "After the fact"),
        ("❌  Conversation is over",   "Information arrives too late"),
    ],
    highlight_idx=4,
    note="You can't act on information you don't have yet"
)

# LieGlass
draw_timeline(axes[1],
    "LieGlass", COLORS["red"],
    [
        ("Conversation starts",        "Glasses capture A/V"),
        ("Model scores each moment",   "Live lie probability"),
        ("Whisper transcribes",        "Full transcript in real time"),
        ("Claude flags contradictions","Suggests follow-up questions"),
        ("✓  You act NOW",             "Expose deeper lies mid-conversation"),
    ],
    highlight_idx=4,
    note="Information is only useful when you can still use it"
)

plt.tight_layout()
save_plot(fig, "S_observe_vs_act.png")
