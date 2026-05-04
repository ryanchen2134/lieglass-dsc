"""
plot_V_why_glasses.py — Hardware comparison: phone vs screen vs earpiece vs glasses
Run: python plot_V_why_glasses.py
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
ax.set_title("Why AR Glasses? Every Alternative Falls Short.", pad=16)

options = [
    ("📱", "Phone",        ["Requires looking away", "Breaks eye contact", "Obvious to subject"],
     COLORS["gray"], False),
    ("🖥",  "Screen",       ["Tied to a desk", "No mobility", "Disrupts conversation flow"],
     COLORS["gray"], False),
    ("🎧", "Earpiece",     ["Audio feedback only", "No visual display", "Limited information"],
     COLORS["gray"], False),
    ("🕶",  "AR Glasses\n(LieGlass)", ["Eyes-forward always", "Hands-free operation",
                                       "Real-time HUD overlay"],
     COLORS["red"], True),
]

for i, (icon, name, pros, color, is_winner) in enumerate(options):
    x = i * 3.3 + 0.5
    edge = COLORS["red"] if is_winner else COLORS["grid"]
    lw   = 2.5 if is_winner else 1.0

    rect = FancyBboxPatch((x, 0.5), 2.8, 5.0,
                           boxstyle="round,pad=0.2",
                           facecolor=COLORS["panel"] if not is_winner else "#1a0a0a",
                           edgecolor=edge, linewidth=lw)
    ax.add_patch(rect)

    # Icon
    ax.text(x + 1.4, 4.8, icon, ha="center", va="center", fontsize=28)

    # Name
    ax.text(x + 1.4, 3.9, name, ha="center", va="center",
            fontsize=11, fontweight="bold",
            color=COLORS["red"] if is_winner else COLORS["white"])

    # Divider
    ax.axhline(3.6, xmin=(x + 0.1) / 14, xmax=(x + 2.7) / 14,
               color=edge, linewidth=0.8, alpha=0.5)

    # Pros/cons
    prefix = "✓" if is_winner else "✗"
    pcolor = "#2ecc71" if is_winner else COLORS["red"]
    for j, point in enumerate(pros):
        ax.text(x + 0.25, 3.1 - j * 0.75, f"{prefix}  {point}",
                va="center", fontsize=8.5,
                color=pcolor if is_winner else COLORS["gray_light"])

    if is_winner:
        ax.text(x + 1.4, 0.85, "BEST CHOICE",
                ha="center", fontsize=8, fontweight="bold",
                color=COLORS["red"], fontfamily="monospace")

plt.tight_layout()
save_plot(fig, "V_why_glasses.png")
