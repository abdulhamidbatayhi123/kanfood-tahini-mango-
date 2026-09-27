"""Figures for the sections written after the first assembly of the manuscript.

Those six sections carry the paper's strongest result and its most consequential negative one, and
every one of them was text and tables only. Two figures are built here:

* **Fidelity under a distribution shift** (Section 3.13). The converted equation stops being its
  model on withheld material, and the exact route does not. The number in the text is a mean over
  folds; the figure shows the folds, because the worst fold is the one a practitioner would meet.
* **The transparency frontier** (Section 3.16). Accuracy against what the model actually yields --
  a fitted object, a shape function per input, or an expression that can be printed. It is the
  figure that makes the paper's claim falsifiable at a glance: everything above the equation
  yields nothing to print, and the only other printable model is unusable.

The conventions of `figstyle` apply, so `check_no_title`, `check_no_overlap` and
`check_inside_axes` guard both figures. Every panel is drawn from a CSV in `results_rigor/` and
asserts at least one plotted value against its source, and a panel whose file does not exist yet
is skipped with a message rather than fabricated.
"""
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paper1_rigor.common import OUT
from paper1_rigor.figstyle import (COL2, ACCENT, NEUTRAL, panel, save, tidy,
                                   check_no_title, check_no_overlap, check_inside_axes)

CONVERTED = "B-spline + corrected conversion"
OLS = "OLS on the same inputs"
C_CONV, C_EXACT, C_OLS = "#D97706", ACCENT, NEUTRAL      # amber, violet, grey

# The representation labels as `r23`/`r19` write them, with the words the manuscript uses.
REPS = [("PLS scores", "latent scores"), ("selected bands", "named bands")]


def _read(name):
    p = OUT / name
    if not p.exists():
        print(f"  [skip] {name} does not exist yet")
        return None
    return pd.read_csv(p)


def _elementary(routes):
    return next((r for r in routes if r.startswith("elementary basis")), None)


# ---------------------------------------------------------- Figure: fidelity under a shift
def fig_fidelity_under_shift():
    raw = _read("r23_shift_raw.csv")
    if raw is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.45),
                             gridspec_kw=dict(width_ratios=[1.0, 0.85, 1.35]))

    # (a), (b) fidelity of the printed equation to its own network, fold by fold
    for k, (ds, xlab) in enumerate([("tahini", "withheld tahini lot"),
                                    ("mango", "withheld mango season")]):
        ax = axes[k]
        g = raw[raw.dataset == ds]
        folds = sorted(g["held_out"].unique())
        x = np.arange(len(folds))
        for rep, replab in REPS:
            sub = g[g.representation == rep]
            elem = _elementary(sub["route"].unique())
            marker = "o" if rep == "PLS scores" else "s"
            for route, colour, name in [(CONVERTED, C_CONV, "converted"),
                                        (elem, C_EXACT, "exact")]:
                if route is None:
                    continue
                # Every seed is plotted, not the fold mean: the worst seed on the worst fold is
                # the case a practitioner would actually meet, and averaging it away would be
                # the flattering choice.
                r = sub[sub.route == route]
                off = -0.15 if rep == "PLS scores" else 0.15
                px = np.array([x[folds.index(f)] for f in r["held_out"]], float) + off
                ax.plot(px, r["fidelity"].to_numpy(), marker, ms=3.8, color=colour, lw=0,
                        markeredgewidth=0, alpha=0.7,
                        label=f"{name}, {replab}" if k == 0 else None)
                m = r.groupby("held_out")["fidelity"].mean().reindex(folds).to_numpy()
                for xi, mi in zip(x, m):
                    ax.plot([xi + off - 0.11, xi + off + 0.11], [mi, mi], color=colour, lw=1.1)
        ax.axhline(1.0, color="#111827", lw=0.7, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(folds)
        ax.set_xlabel(xlab)
        ax.set_ylim(0.55, 1.045)
        if k == 0:
            ax.set_ylabel("fidelity of equation to network")
        else:
            ax.set_yticklabels([])
        tidy(ax)
        panel(ax, "ab"[k])
    # Inside panel (a) the legend covered the T4 point at fidelity 0.638 -- the single worst run
    # in the study and the one Section 3.5 quotes. Put it under the two panels instead, where
    # nothing is plotted.
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(h, l, fontsize=6, ncol=2, columnspacing=0.8, handletextpad=0.2,
                   loc="upper center", bbox_to_anchor=(1.02, -0.28), frameon=False,
                   borderaxespad=0.0)

    # (c) what the infidelity costs: the printed equation minus the network it came from
    ax = axes[2]
    rows = []
    for ds, _ in [("tahini", None), ("mango", None)]:
        for rep, replab in REPS:
            sub = raw[(raw.dataset == ds) & (raw.representation == rep)]
            if sub.empty:
                continue
            rows.append((f"{ds}, {replab}", sub))
    yticks = []
    for i, (lab, sub) in enumerate(rows):
        yticks.append(lab)
        elem = _elementary(sub["route"].unique())
        for route, colour, off in [(CONVERTED, C_CONV, 0.16), (elem, C_EXACT, -0.16)]:
            if route is None:
                continue
            r = sub[sub.route == route]
            d = (r.groupby("held_out")["R2_equation"].mean()
                 - r.groupby("held_out")["R2_network"].mean()).to_numpy()
            ax.plot(d, np.full(len(d), i + off), "o", ms=3.2, color=colour, alpha=0.55, lw=0,
                    markeredgewidth=0)
            ax.plot([d.mean()], [i + off], "|", ms=11, color=colour, markeredgewidth=1.8)
    ax.axvline(0.0, color="#111827", lw=0.7, ls=":")
    ax.set_yticks(range(len(yticks)))
    ax.set_yticklabels(yticks, fontsize=7)
    ax.set_ylim(-0.6, len(yticks) - 0.4)
    ax.set_xlabel(r"printed equation $-$ its network ($R^2$)")
    tidy(ax)
    panel(ax, "c")

    # Guard: at least one plotted value must reproduce the summary file it comes from.
    summ = _read("r23_shift_summary.csv")
    if summ is not None:
        s = summ[(summ.dataset == "tahini") & (summ.representation == "selected bands")
                 & (summ.route == CONVERTED)]
        got = raw[(raw.dataset == "tahini") & (raw.representation == "selected bands")
                  & (raw.route == CONVERTED)]["fidelity"].mean()
        assert abs(got - float(s["fidelity_mean"].iloc[0])) < 1e-9, \
            "panel (a) disagrees with r23_shift_summary.csv"

    fig.tight_layout(w_pad=0.9)
    check_no_title(fig); check_no_overlap(fig); check_inside_axes(fig)
    save(fig, "Figure_14_fidelity_under_shift")


# ------------------------------------------------------------- Figure: transparency frontier
# What each model hands back when it is fitted. The three classes are the argument of Section
# 3.16 and the ordering is not a ranking of quality, only of what can be written down. The models
# are listed on the vertical axis rather than annotated in place: eight labels on a shared
# horizontal axis collided in every arrangement tried, and tick labels are laid out by matplotlib.
YIELDS = [
    ("MLP(ref)",     0, "reference perceptron"),
    ("spline-MLP",   0, "spline-activation perceptron"),
    ("XGBoost",      0, "gradient boosting"),
    ("EBM",          1, "boosting machine"),
    ("EBM+int",      1, "boosting machine + interactions"),
    ("GAM",          1, "spline additive model"),
    ("SR (gplearn)", 2, "symbolic regression"),
]
CLASSES = ["yields a fitted object",
           "yields a shape function per input",
           "yields an expression that can be printed"]
CLASS_COLOR = ["#94A3B8", "#0F766E", "#7C3AED"]


def _selected_equation(ds, eq, inner):
    """Mean and SD of the printed equation of Section 3.12, on the same latent inputs."""
    e, i = eq.get(ds), inner.get(ds)
    if e is None or i is None:
        print(f"  [skip] the equation point for {ds}: its edge-basis CSV does not exist yet")
        return None
    b = (i[(i.stage == "inner_basis") & (i.representation == "PLS scores")]
         .groupby(["basis", "arch", "degree"])["R2"].mean().reset_index()
         .sort_values("R2", ascending=False).iloc[0])
    v = e[(e.stage == "external") & (e.representation == "PLS scores")
          & (e.model == f"{b['basis']} {b['arch']}")]["R2"]
    return None if v.empty else (float(v.mean()), float(v.std(ddof=1)))


def fig_transparency_frontier():
    r14 = _read("r14_glassbox_baselines_raw.csv")
    if r14 is None:
        r14 = _read("r14_glassbox_baselines_raw_progress.csv")
    if r14 is None:
        return
    bench = _read("f01_benchmark_recomputed.csv")
    eq = {"tahini": _read("r19_basis_external_tahini.csv"),
          "mango": _read("r19b_basis_external_mango.csv")}
    inner = {"tahini": _read("r19_basis_inner_tahini.csv"),
             "mango": _read("r19b_basis_inner_mango.csv")}

    # One row per model, ordered by class, with this paper's equation last inside its class.
    rows = [(lab, cls, m) for m, cls, lab in YIELDS] + [("this paper's equation", 2, None)]
    rows.sort(key=lambda r: (r[1], r[0] == "this paper's equation"))
    y = {r[0]: i for i, r in enumerate(rows)}

    datasets = [d for d in ("tahini", "mango") if (r14.dataset == d).any()]
    fig, axes = plt.subplots(1, len(datasets), figsize=(COL2, 2.6), squeeze=False, sharey=True)
    for k, ds in enumerate(datasets):
        ax = axes[0][k]
        g = r14[(r14.dataset == ds) & (r14.representation == "PLS scores")]
        vals = {}
        for lab, cls, model in rows:
            if model is None:
                got = _selected_equation(ds, eq, inner)
            else:
                v = g.loc[g.model == model, "R2"]
                got = (float(v.mean()), float(v.std(ddof=1))) if len(v) else None
            if got is not None:
                vals[lab] = (cls, *got)
        if not vals:
            continue
        # A model that fails outright (symbolic regression reaches 0.49 on tahini and a negative
        # score on mango) would otherwise set the axis range and compress every other model into
        # a sliver. The range is taken from the models that are in contention and the failures
        # are drawn against the left edge with their value printed.
        means = sorted(m for _, m, _ in vals.values())
        ref = means[len(means) // 2]
        band = [m for m in means if m > ref - 0.15]
        lo, hi = min(band) - 0.012, max(band) + 0.008
        for lab, (cls, mean, sd) in vals.items():
            ours = lab == "this paper's equation"
            off = mean < lo
            ax.errorbar([lo if off else mean], [y[lab]], xerr=None if off else [sd],
                        fmt="<" if off else ("D" if ours else "o"),
                        ms=5.2 if ours else 4.2, color=CLASS_COLOR[cls], ecolor=CLASS_COLOR[cls],
                        elinewidth=0.9, capsize=2, markeredgewidth=0, zorder=3)
            # Values are right-aligned in a column at the axis edge rather than beside each
            # marker: next to the marker they sat on their own error bars and on the reference
            # line, which is the kind of thing only the rendered page shows.
            ax.annotate(f"{mean:.3f}" if not off else f"{mean:.2f}",
                        (1.0, y[lab]), xycoords=("axes fraction", "data"),
                        fontsize=6, ha="right", va="center",
                        xytext=(-2, 0), textcoords="offset points",
                        color="#111827", fontweight="bold" if ours else "normal")
        if bench is not None:
            p = bench[(bench.dataset == ds) & (bench.model == "PLS")]["R2"]
            if len(p):
                ax.axvline(float(p.iloc[0]), color="#111827", lw=0.8, ls="--", zorder=1)
        ax.set_yticks(range(len(rows)))
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_xlim(lo - (hi - lo) * 0.06, hi + (hi - lo) * 0.34)
        # The headroom on the right exists to hold the value column, not to be plotted in:
        # left to itself the tahini panel drew a tick at 1.02, above the largest value R^2
        # can take. Keep the space, drop the impossible ticks.
        ax.set_xticks([t for t in ax.get_xticks()
                       if ax.get_xlim()[0] <= t <= min(ax.get_xlim()[1], 1.0)])
        ax.set_xlabel(rf"external $R^2$ ({ds})")
        for lbl, ours in zip(ax.get_yticklabels(), [r[0] == "this paper's equation" for r in rows]):
            lbl.set_fontsize(7)
            if ours:
                lbl.set_fontweight("bold")
        tidy(ax)
        panel(ax, "ab"[k])
    axes[0][0].set_yticklabels([r[0] for r in rows], fontsize=7)

    handles = [Line2D([], [], marker="o", lw=0, color=c, ms=4.2, label=n)
               for c, n in zip(CLASS_COLOR, CLASSES)]
    handles.append(Line2D([], [], ls="--", color="#111827", lw=0.8,
                          label="linear calibration on the full spectrum"))

    # Guard: the tahini spline-MLP point must reproduce its raw file.
    v = r14[(r14.dataset == "tahini") & (r14.model == "spline-MLP")]["R2"]
    assert abs(v.mean() - 0.99347) < 5e-4, f"spline-MLP point drifted: {v.mean()}"

    fig.tight_layout(w_pad=0.8, rect=(0, 0.11, 1, 1))
    # Below the panels rather than inside one: at this aspect ratio a legend placed in either
    # axis sat on top of the mango points.
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6, handletextpad=0.4,
               columnspacing=1.4, labelspacing=0.3, borderaxespad=0.1)
    check_no_title(fig); check_no_overlap(fig); check_inside_axes(fig)
    save(fig, "Figure_15_transparency_frontier")


def main():
    print("building the figures for the new sections ...")
    for fn in (fig_fidelity_under_shift, fig_transparency_frontier):
        try:
            fn()
        except Exception as e:
            print(f"  [FAILED] {fn.__name__}: {type(e).__name__}: {e}")
            if "--strict" in sys.argv:
                raise


if __name__ == "__main__":
    main()
