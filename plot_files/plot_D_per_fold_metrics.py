"""
plot_D_per_fold_metrics.py — Per-fold Accuracy, AUC, F1 bar chart
Run: python plot_D_per_fold_metrics.py
"""
import matplotlib; matplotlib.use("Agg")
import json, numpy as np, matplotlib.pyplot as plt, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import apply_theme, save_plot, COLORS, CKPT_DIR
apply_theme()

with open(os.path.join(CKPT_DIR, "metrics.json")) as f:
    metrics = json.load(f)

fold_ids = [m["fold"]     for m in metrics]
accs     = [m["accuracy"] for m in metrics]
aucs     = [m["auc_roc"]  for m in metrics]
f1s      = [m["f1"]       for m in metrics]

x, width = np.arange(len(fold_ids)), 0.25

fig, ax = plt.subplots(figsize=(12, 6))
for i, (vals, label, color) in enumerate([
    (accs, "Accuracy", COLORS["truth"]),
    (aucs, "AUC",      COLORS["red"]),
    (f1s,  "F1 Score", "#888888"),
]):
    bars = ax.bar(x + i * width, vals, width, label=label,
                  color=color, alpha=0.9, edgecolor=COLORS["bg"], linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{v:.2f}", ha="center", fontsize=8, fontweight="bold",
                color=COLORS["white"])

ax.axhline(0.54, color=COLORS["gray"], linestyle="--", linewidth=1.5,
           alpha=0.7, label="Human baseline (0.54)")
ax.axhline(0.50, color=COLORS["grid"], linestyle=":", linewidth=1.0,
           alpha=0.5, label="Chance (0.50)")
ax.set_xticks(x + width)
ax.set_xticklabels([f"Fold {f}" for f in fold_ids])
ax.set_ylabel("Score")
ax.set_ylim(0.3, 1.0)
ax.set_title("Model Performance Across Folds")
ax.legend(loc="lower right")
ax.text(0.98, 0.97,
        f"Mean AUC={np.mean(aucs):.3f}  Acc={np.mean(accs):.3f}  F1={np.mean(f1s):.3f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color=COLORS["gray"], fontfamily="monospace")
plt.tight_layout()
save_plot(fig, "D_per_fold_metrics.png")
