"""Figure S9: learning curve (tahini fraction, external test R² vs number of physical training samples).
Run: python -m revision.figS9_learning_curve -> revision/results/figures/FigureS9.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from revision.common import RESULTS

d = pd.read_csv(RESULTS / "r10" / "learning_curve.csv")
ORDER = ["PLS", "MLP", "KAN", "CNN", "RF", "SVM"]
LABEL = {"SVM": "SVR", "RF": "RF", "PLS": "PLS", "MLP": "MLP", "KAN": "KAN", "CNN": "CNN"}
COL = {"PLS": "#2a6fdb", "MLP": "#e8833a", "KAN": "#8e44ad", "CNN": "#1b9e77", "RF": "#7f7f7f", "SVM": "#c0392b"}
MK = {"PLS": "o", "MLP": "s", "KAN": "D", "CNN": "^", "RF": "v", "SVM": "P"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=300)
for ax, lo, models in zip(axes, (0.0, 0.90), (ORDER, ORDER[:4])):
    for m in models:
        g = d[d.model == m].sort_values("n_samples")
        ax.errorbar(g.n_samples, g["mean"], yerr=g["std"], color=COL[m], marker=MK[m], ms=5, lw=1.6,
                    capsize=2.5, label=LABEL[m] if ax is axes[0] else None)
    ax.set_xlabel("Physical samples in training")
    ax.set_xticks([10, 15, 20, 25, 30, 38])
    ax.set_ylim(lo, 1.0)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("External-test R² (tahini fraction)")
axes[0].set_title("(a) Full range", loc="left", fontsize=10)
axes[1].set_title("(b) Enlarged, R² ≥ 0.90 (four leading models)", loc="left", fontsize=10)
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1))
out = RESULTS / "figures" / "FigureS9.png"
fig.savefig(out, facecolor="white")
print(out)
