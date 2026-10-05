"""Figure S9: learning curve (tahini fraction, external test R² vs number of physical training samples).
Run: python -m revision.figS9_learning_curve -> revision/results/figures/FigureS9.{png,pdf}

Drawn with the publication style of paper1_rigor.figstyle -- the same fonts, the same colour for each
model as in every other figure, model names as in the text (SVR), and panel letters outside the frame
instead of titles inside it -- so that it matches the rest of the article and the Supplementary.
"""
import matplotlib.pyplot as plt
import pandas as pd

from paper1_rigor.figstyle import COL2, MODEL_COLOR, check_legend_clear, check_no_title, mlabel, panel, tidy
from revision.common import RESULTS

d = pd.read_csv(RESULTS / "r10" / "learning_curve.csv")
ORDER = ["PLS", "MLP", "KAN", "CNN", "RF", "SVM"]
MK = {"PLS": "o", "MLP": "s", "KAN": "D", "CNN": "^", "RF": "v", "SVM": "P"}

fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.7))
for k, (ax, lo, models) in enumerate(zip(axes, (0.0, 0.90), (ORDER, ORDER[:4]))):
    for m in models:
        g = d[d.model == m].sort_values("n_samples")
        ax.errorbar(g.n_samples, g["mean"], yerr=g["std"], color=MODEL_COLOR[m], marker=MK[m], ms=3.5,
                    lw=1.1, elinewidth=0.8, capsize=2, label=mlabel(m) if k == 0 else None)
    ax.set_xlabel("Physical samples in training")
    ax.set_xticks([10, 15, 20, 25, 30, 38])
    ax.set_ylim(lo, 1.0)
    tidy(ax)
    panel(ax, "ab"[k])
axes[0].set_ylabel("External-test $R^2$ (tahini fraction)")
axes[1].set_ylabel("External-test $R^2$ (enlarged)")
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.legend(*axes[0].get_legend_handles_labels(), ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.0))
check_no_title(fig)
check_legend_clear(fig)
out = RESULTS / "figures" / "FigureS9.png"
fig.savefig(out, facecolor="white")
fig.savefig(out.with_suffix(".pdf"))
print(out)
