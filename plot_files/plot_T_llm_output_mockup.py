"""
plot_T_llm_output_mockup.py — Example Claude response during interrogation
Run: python plot_T_llm_output_mockup.py
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import apply_theme, save_plot, COLORS
apply_theme()

fig, ax = plt.subplots(figsize=(13, 7))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7)
ax.axis("off")
ax.set_title("Claude — Live Conversation Assistant", pad=16)

# Chat bubble background
chat_bg = FancyBboxPatch((0.4, 0.4), 12.2, 6.2,
                          boxstyle="round,pad=0.3",
                          facecolor=COLORS["panel"],
                          edgecolor=COLORS["grid"], linewidth=1.5)
ax.add_patch(chat_bg)

# Header
ax.text(0.9, 6.3, "Claude Opus", fontsize=11, fontweight="bold",
        color=COLORS["red"])
ax.text(0.9, 5.95, "Live Analysis  •  Real-time", fontsize=8.5,
        color=COLORS["gray"], fontfamily="monospace")
ax.axhline(5.8, xmin=0.06, xmax=0.94, color=COLORS["grid"], linewidth=1)

# Lie score bar
ax.text(0.9, 5.5, "Current Lie Score", fontsize=9, color=COLORS["gray_light"])
bar_bg = FancyBboxPatch((0.9, 5.1), 8.0, 0.3,
                         boxstyle="round,pad=0.05",
                         facecolor=COLORS["grid"], edgecolor="none")
ax.add_patch(bar_bg)
bar_fg = FancyBboxPatch((0.9, 5.1), 5.6, 0.3,
                         boxstyle="round,pad=0.05",
                         facecolor=COLORS["red"], edgecolor="none")
ax.add_patch(bar_fg)
ax.text(9.1, 5.25, "70%", fontsize=10, fontweight="bold", color=COLORS["red"],
        va="center")

# Contradiction flag
flag_box = FancyBboxPatch((0.9, 3.9), 11.2, 0.95,
                           boxstyle="round,pad=0.15",
                           facecolor="#2a0a0a", edgecolor=COLORS["red"],
                           linewidth=1.5)
ax.add_patch(flag_box)
ax.text(0.9 + 0.2, 4.6, "⚠  Contradiction Detected", fontsize=10,
        fontweight="bold", color=COLORS["red"])
ax.text(0.9 + 0.2, 4.2,
        "At 2:14, subject said \"I was home all evening\" but at 0:47\n"
        "mentioned \"stopping by the office after 6pm\"",
        fontsize=9, color=COLORS["white"])

# Suggestion
sug_box = FancyBboxPatch((0.9, 2.7), 11.2, 0.95,
                          boxstyle="round,pad=0.15",
                          facecolor="#0a1a0a", edgecolor="#2ecc71",
                          linewidth=1.5)
ax.add_patch(sug_box)
ax.text(0.9 + 0.2, 3.4, "💡  Suggested Follow-up", fontsize=10,
        fontweight="bold", color="#2ecc71")
ax.text(0.9 + 0.2, 3.0,
        "\"You mentioned stopping by the office — what time did you leave?\"",
        fontsize=9, color=COLORS["white"])

# Fact check
fc_box = FancyBboxPatch((0.9, 1.5), 11.2, 0.95,
                         boxstyle="round,pad=0.15",
                         facecolor="#0a0a2a", edgecolor=COLORS["truth"],
                         linewidth=1.5)
ax.add_patch(fc_box)
ax.text(0.9 + 0.2, 2.2, "🔍  Fact Check", fontsize=10,
        fontweight="bold", color=COLORS["truth"])
ax.text(0.9 + 0.2, 1.8,
        "Claim: \"The meeting was cancelled\" — No public record found. "
        "LinkedIn shows attendees checked in.",
        fontsize=9, color=COLORS["white"])

plt.tight_layout()
save_plot(fig, "T_llm_output_mockup.png")
