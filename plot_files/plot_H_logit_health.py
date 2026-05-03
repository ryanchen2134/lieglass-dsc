"""
plot_H_logit_health.py — Logit mean vs std scatter per fold
Run: python plot_H_logit_health.py
"""
import matplotlib; matplotlib.use("Agg")
import json, numpy as np, matplotlib.pyplot as plt, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme import apply_theme, save_plot, COLORS, CKPT_DIR
apply_theme()

with open(os.path.join(CKPT_DIR, "metrics.json")) as f:
    metrics = json.load(f)

fold_ids   = [m["fold"]       for m in metrics]
logit_mean = [m["logit_mean"] for m in metrics]
logit_std  = [m["logit_std"]  for m in metrics]
aucs       = [m["auc_roc"]    for m in metrics]

fig, ax = plt.subplots(figsize=(8, 6))

sc = ax.scatter(logit_mean, logit_std, c=aucs,
                cmap="RdYlGn", s=180, zorder=5,
                edgecolors=COLORS["white"], linewidths=1.2,
                vmin=0.6, vmax=1.0)

for i, fold_id in enumerate(fold_ids):
    ax.annotate(f"Fold {fold_id}",
                xy=(logit_mean[i], logit_std[i]),
                xytext=(8, 6), textcoords="offset points",
                fontsize=9, color=COLORS["gray_light"])

cbar = fig.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label("AUC-ROC", color=COLORS["gray_light"])
cbar.ax.yaxis.set_tick_params(color=COLORS["gray"])
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COLORS["gray_light"])

ax.axvline(0, color=COLORS["gray"], linestyle="--",
           linewidth=1.0, alpha=0.5, label="Zero mean (ideal)")
ax.set_xlabel("Logit Mean  (→ 0 = unbiased)")
ax.set_ylabel("Logit Std  (→ higher = more decisive)")
ax.set_title("Model Calibration per Fold")
ax.legend()
plt.tight_layout()
save_plot(fig, "H_logit_health.png")
