"""Figures 6-8: robustness and generalisation, ablations, and stability.

Each figure is built from the CSV written by the corresponding experiment, so the figure and the
table in the manuscript are two renderings of one file rather than two independent computations.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper1_rigor.common import ROOT
from paper1_rigor.figstyle import (MODEL_COLOR, MODEL_ORDER, COL2, panel, save, tidy,
                                   check_no_title, ACCENT, NEUTRAL)

R = ROOT / "results_rigor"


# ------------------------------------------- Figure 6: robustness and generalisation
def fig_robustness():
    rob = pd.read_csv(R / "r07_robustness_raw.csv")
    gen = pd.read_csv(R / "r04_generalisation_raw.csv")

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.6))

    ax = tidy(axes[0])
    data = [rob.loc[rob.model == m, "R2"].to_numpy() for m in MODEL_ORDER]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", lw=1.0),
                    flierprops=dict(marker="o", ms=2, mfc="0.4", mec="none"))
    for p, m in zip(bp["boxes"], MODEL_ORDER):
        p.set_facecolor(MODEL_COLOR[m]); p.set_alpha(0.75); p.set_edgecolor("0.3"); p.set_lw(0.7)
    ax.set_xticklabels(MODEL_ORDER, rotation=45, ha="right")
    ax.set_ylabel("Hold-out $R^2$ (tahini %)")
    panel(ax, "a")

    for k, (ds, lab, letter) in enumerate([("tahini", "held-out tahini lot", "b"),
                                           ("mango", "held-out mango season", "c")]):
        ax = tidy(axes[k + 1])
        sub = gen[gen.dataset == ds]
        folds = sorted(sub.held_out.unique())
        x = np.arange(len(folds))
        w = 0.13
        for i, m in enumerate(MODEL_ORDER):
            v = [sub.loc[(sub.held_out == f) & (sub.model == m), "R2_mean"].iloc[0] for f in folds]
            ax.bar(x + (i - 2.5) * w, v, w, color=MODEL_COLOR[m], label=m if k == 0 else None)
        ax.axhline(0, color="0.4", lw=0.7)
        ax.set_xticks(x); ax.set_xticklabels(folds, rotation=0)
        ax.set_xlabel(lab)
        ax.set_ylabel("$R^2$ on the held-out group")
        panel(ax, letter)
    axes[1].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=6)

    fig.tight_layout()
    check_no_title(fig)
    save(fig, "Figure_06_robustness_generalisation")


# ---------------------------------------------------------------- Figure 7: ablations
def fig_ablation():
    ab = pd.read_csv(R / "r02_ablation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.4), sharey=False)
    for ax, ds, letter in [(axes[0], "tahini", "a"), (axes[1], "mango", "b")]:
        ax = tidy(ax)
        sub = ab[ab.dataset == ds].copy()
        base = sub.loc[sub.config == "published", "cv_R2"].iloc[0]
        sub = sub[sub.config != "published"]
        order = sub.iloc[::-1]
        y = np.arange(len(order))
        vals = order["cv_R2"].to_numpy()
        cols = [ACCENT if v >= base else "#DC2626" for v in vals]
        floor = min(0.0, np.percentile(vals, 5) - 0.05)
        clipped = np.clip(vals, floor, None)
        ax.barh(y, clipped, color=cols, alpha=0.85, height=0.7,
                xerr=order["cv_R2_sd"], error_kw=dict(lw=0.6, ecolor="0.35"))
        ax.axvline(base, color="0.25", ls="--", lw=1.0)
        ax.text(base, len(order) - 0.2, f"published\n({base:.3f})", fontsize=6, color="0.25",
                ha="center", va="bottom")
        for yi, v in zip(y, vals):
            if v < floor + 1e-9:
                ax.text(floor, yi, f" {v:.2f} ", fontsize=5.5, va="center", ha="left", color="white")
        ax.set_yticks(y); ax.set_yticklabels(order["config"], fontsize=6)
        ax.set_xlabel("Grouped five-fold cross-validated $R^2$")
        ax.set_xlim(floor, max(1.0, vals.max() + 0.05))
        panel(ax, letter)
    fig.tight_layout()
    check_no_title(fig)
    save(fig, "Figure_07_ablations")


# ------------------------------------------------------------- Figure 8: stability
def fig_stability():
    runs = pd.read_csv(R / "r03_stability_runs.csv")
    agr = pd.read_csv(R / "r03_stability_agreement.csv")

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.6))

    ax = tidy(axes[0])
    for i, ds in enumerate(["tahini", "mango"]):
        v = runs[(runs.dataset == ds) & (runs.source == "seed")]["symbolic_R2"].dropna().to_numpy()
        ax.scatter(np.full_like(v, i, dtype=float) + np.linspace(-0.12, 0.12, len(v)), v,
                   s=14, color=ACCENT, alpha=0.8, edgecolors="none")
        ax.hlines(v.mean(), i - 0.25, i + 0.25, color="0.2", lw=1.4)
        ax.text(i, v.mean(), f"  {v.mean():.3f}", fontsize=6.5, va="center", ha="left", color="0.2")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["tahini", "mango"])
    ax.set_ylabel("Symbolic-equation $R^2$\n(held-out, 10 seeds)")
    ax.set_xlim(-0.5, 1.7)
    panel(ax, "a")

    for k, (col, ylab, letter) in enumerate([("top3_overlap", "Mean pairwise top-3 overlap", "b"),
                                             ("spearman", "Mean pairwise Spearman $\\rho$", "c")]):
        ax = tidy(axes[k + 1])
        sources = ["seeds", "folds", "preprocessing"]
        x = np.arange(len(sources)); w = 0.36
        for i, (ds, c) in enumerate([("tahini", ACCENT), ("mango", NEUTRAL)]):
            v = [agr.loc[(agr.dataset == ds) & (agr.source == s), col].iloc[0] for s in sources]
            ax.bar(x + (i - 0.5) * w, v, w, color=c, label=ds)
        ax.set_xticks(x); ax.set_xticklabels(sources, rotation=20, ha="right")
        ax.set_ylabel(ylab); ax.set_ylim(0, 1.05)
        if k == 0:
            ax.legend(loc="lower right", fontsize=6.5)
        panel(ax, letter)

    fig.tight_layout()
    check_no_title(fig)
    save(fig, "Figure_08_stability")


def main():
    made = []
    for name, fn in [("robustness/generalisation", fig_robustness), ("ablations", fig_ablation),
                     ("stability", fig_stability)]:
        try:
            fn(); made.append(name)
        except FileNotFoundError as e:
            print(f"  skipped {name}: {e.filename} not written yet")
    print("built:", ", ".join(made) if made else "nothing")


if __name__ == "__main__":
    main()
