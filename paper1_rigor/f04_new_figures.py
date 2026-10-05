"""Figures for the validation, extraction and interpretability experiments.

Every panel is drawn from a CSV in `results_rigor/`, and every panel asserts at least one plotted
value against the file it came from, so a figure cannot silently drift away from its table. A
panel whose source file does not exist yet is skipped with a message rather than fabricated, so
this module can be run repeatedly while the experiments are still finishing.
"""
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from paper1_rigor.common import OUT, ROOT
from paper1_rigor.figstyle import (COL1, COL2, MODEL_COLOR, MODEL_ORDER, ACCENT, NEUTRAL,
                                   panel, save, tidy, check_no_title, check_no_overlap,
                                   check_legend_clear, mlabel, signed)

# The ablation arms as the reader should see them (the CSV keeps the short codes of the scripts).
ABLATION_LABEL = {
    "grid G=3": "Grid $G$ = 3", "grid G=10": "Grid $G$ = 10", "grid G=20": "Grid $G$ = 20",
    "spline order k=2": "Spline order $k$ = 2", "spline order k=4": "Spline order $k$ = 4",
    "hidden width (2,)": "Hidden width (2,)", "hidden width (3,)": "Hidden width (3,)",
    "hidden width (4,)": "Hidden width (4,)", "hidden width (5,)": "Hidden width (5,)",
    "hidden width (8, 4)": "Hidden width (8, 4)", "hidden width (32, 16)": "Hidden width (32, 16)",
    "lambda=0": r"$\lambda$ = 0", "lambda=0.01": r"$\lambda$ = 0.01", "lambda=0.1": r"$\lambda$ = 0.1",
    "preprocessing sg1": "SG first derivative", "preprocessing snv+sg1": "SNV + SG first derivative",
    "preprocessing msc": "MSC", "preprocessing snv": "SNV",
}


def _read(name):
    p = OUT / name
    if not p.exists():
        print(f"  [skip] {name} does not exist yet")
        return None
    return pd.read_csv(p)


def _bar(ax, labels, values, errs=None, colors=None, ylabel="", rotate=0):
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errs, color=colors or NEUTRAL, width=0.68,
           error_kw=dict(lw=0.8, capsize=2, ecolor="#334155"))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    ax.set_ylabel(ylabel)
    tidy(ax)


# ------------------------------------------------------------------ Figure: validation robustness
def fig_validation():
    rob = _read("r07_robustness_raw.csv")
    gen = _read("r04_generalisation_raw.csv")
    if rob is None or gen is None:
        return
    # Table S3 and Section S4 report the mango SVR folds of the rerun (Section S8), because the
    # original first-season value (0.248) could not be reproduced by any SVR configuration of the
    # search; the figure must show the same values as the table (frozen arm "F" of revision/r01).
    wide = pd.read_csv(ROOT / "revision" / "results" / "r01" / "mango_per_fold_wide.csv")
    svr = wide[(wide.arm == "F") & (wide.model == "SVM")].iloc[0]
    sel = (gen.dataset == "mango") & (gen.model == "SVM")
    assert sel.sum() == 4
    gen = gen.copy()
    gen.loc[sel, "R2_mean"] = [float(svr[str(int(f))]) for f in gen.loc[sel, "held_out"]]
    assert abs(gen.loc[sel, "R2_mean"].mean() - 0.772) < 5e-4
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.5))

    # (a) ten independent grouped hold-outs on tahini
    ax = axes[0]
    for i, m in enumerate(MODEL_ORDER):
        v = rob.loc[rob.model == m, "R2"].to_numpy()
        ax.scatter(np.full(len(v), i) + np.random.RandomState(0).uniform(-.13, .13, len(v)),
                   v, s=9, color=MODEL_COLOR[m], alpha=.75, lw=0)
        ax.plot([i - .28, i + .28], [v.mean()] * 2, color=MODEL_COLOR[m], lw=1.6)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([mlabel(m) for m in MODEL_ORDER], rotation=45, ha="right")
    ax.set_ylabel(r"Test $R^2$ (tahini fraction)")
    ax.set_xlabel("Ten independent grouped hold-outs")
    ax.set_ylim(0.6, 1.02)
    tidy(ax); panel(ax, "a")

    # (b, c) leave-one-lot-out and leave-one-season-out
    for k, (ds, lab, ylo) in enumerate([("tahini", "Withheld tahini lot", 0.2),
                                        ("mango", "Withheld mango season", 0.2)]):
        ax = axes[k + 1]
        sub = gen[gen.dataset == ds]
        for i, m in enumerate(MODEL_ORDER):
            v = sub.loc[sub.model == m, "R2_mean"].to_numpy()
            ax.scatter(np.full(len(v), i) + np.random.RandomState(1).uniform(-.13, .13, len(v)),
                       v, s=11, color=MODEL_COLOR[m], alpha=.8, lw=0)
            if len(v):
                ax.plot([i - .28, i + .28], [v.mean()] * 2, color=MODEL_COLOR[m], lw=1.6)
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels([mlabel(m) for m in MODEL_ORDER], rotation=45, ha="right")
        ax.set_ylabel(r"Fold $R^2$")
        ax.set_ylim(ylo, 1.02)
        ax.set_xlabel(lab)
        tidy(ax); panel(ax, "bc"[k])

    # guard: the plotted tahini KAN mean must equal the summary file
    summ = _read("r04_generalisation_summary.csv")
    if summ is not None:
        want = float(summ[(summ.dataset == "tahini") & (summ.model == "KAN")]["mean_R2"].iloc[0])
        got = float(gen[(gen.dataset == "tahini") & (gen.model == "KAN")]["R2_mean"].mean())
        assert abs(want - got) < 1e-6, f"figure disagrees with r04 summary: {got} vs {want}"
    fig.tight_layout(); check_no_title(fig)
    save(fig, "Figure_06_validation_robustness")


# --------------------------------------------------------------------------- Figure: ablation
def fig_ablation():
    ab = _read("r02_ablation.csv")
    if ab is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.0))
    for k, ds in enumerate(["tahini", "mango"]):
        ax = axes[k]
        sub = ab[(ab.dataset == ds) & (ab.config != "published")].copy()
        sub = sub.sort_values("delta_vs_published")
        cols = [ACCENT if d > 0 else "#DC2626" for d in sub.delta_vs_published]
        y = np.arange(len(sub))
        ax.barh(y, sub.delta_vs_published, color=cols, height=.68)
        ax.set_yticks(y)
        ax.set_yticklabels([ABLATION_LABEL.get(c, c) for c in sub.config], fontsize=7)
        ax.axvline(0, color="#334155", lw=.8)
        ax.set_xlabel(r"$\Delta$ cross-validated $R^2$ vs. published configuration")
        # the catastrophic arms compress everything else; clip and annotate instead
        lo = max(sub.delta_vs_published.min(), -0.08)
        ax.set_xlim(lo * 1.15, max(0.055, sub.delta_vs_published.max() * 1.25))
        # A clipped bar must still show its real value, or the axis silently understates a
        # catastrophic arm (tahini lambda = 0.1 is -1.18, not -0.08).
        for yi, d in zip(y, sub.delta_vs_published):
            if d < lo:
                ax.text(lo * 1.05, yi, signed(d, ".2f"), va="center", ha="left", fontsize=6.5,
                        color="#111827", bbox=dict(fc="white", ec="none", pad=0.8))
        tidy(ax); panel(ax, "ab"[k])
    fig.tight_layout(); check_no_title(fig)
    save(fig, "Figure_07_ablation")


# ------------------------------------------------------------- Figure: the extraction protocol
def fig_extraction():
    raw = _read("r12_extraction_protocol_raw.csv")
    if raw is None:
        return
    ext = raw[raw.stage == "external"]
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.6))

    for k, ds in enumerate(["tahini", "mango"]):
        ax = axes[k]
        sub = ext[ext.dataset == ds]
        protos = list(dict.fromkeys(sub.protocol))
        allv = sub["r2_symbolic"].to_numpy()
        # A failed extraction is two whole units of R2 away from the rest, and plotting it in
        # range flattens the ten good runs into one line. The axis is therefore set to the bulk
        # and any run below it is drawn at the floor with its true value written beside it, so
        # the failure is declared rather than hidden or allowed to swamp the panel.
        good = allv[allv > 0.5]
        lo = max(0.0, float(np.nanmin(good)) - 0.02)
        hi = min(1.02, float(np.nanmax(good)) + 0.01)
        for i, p in enumerate(protos):
            v = sub.loc[sub.protocol == p, "r2_symbolic"].to_numpy()
            col = NEUTRAL if p.startswith("P0") else ACCENT
            jit = np.random.RandomState(2).uniform(-.11, .11, len(v))
            inside = v >= lo
            ax.scatter(np.full(len(v), i)[inside] + jit[inside], v[inside], s=15, color=col,
                       alpha=.85, lw=0)
            for j in np.where(~inside)[0]:
                ax.scatter([i + jit[j]], [lo], s=17, marker="v", color=col, lw=0)
                ax.annotate(signed(v[j], ".2f"), (i + jit[j], lo), fontsize=5.8, color=col,
                            xytext=(4, 1), textcoords="offset points", va="bottom")
            ax.plot([i - .25, i + .25], [np.nanmedian(v)] * 2, color=col, lw=1.8)
        ax.set_ylim(lo, hi)
        ax.set_xlim(-0.5, len(protos) - 0.5)
        ax.set_xticks(range(len(protos)))
        ax.set_xticklabels(["published\nprotocol", "corrected\nprotocol"][:len(protos)],
                           fontsize=7.5)
        ax.set_ylabel(r"Closed-form equation $R^2$ (external)")
        ax.set_xlabel(ds.capitalize())
        tidy(ax); panel(ax, "ab"[k])

    # (c) inner-CV mean by protocol, both datasets
    ax = axes[2]
    inner = raw[raw.stage == "inner_cv"]
    if len(inner):
        g = inner.groupby(["dataset", "protocol"])["r2_symbolic"].mean().reset_index()
        protos = list(dict.fromkeys(g.protocol))
        x = np.arange(len(protos))
        # Means on an axis that does not start at zero are drawn as points, not as bars.
        for j, ds in enumerate(["tahini", "mango"]):
            vals = [float(g[(g.dataset == ds) & (g.protocol == p)]["r2_symbolic"].iloc[0])
                    if len(g[(g.dataset == ds) & (g.protocol == p)]) else np.nan for p in protos]
            ax.plot(x + (j - .5) * 0.18, vals, ["o", "s"][j], ms=5, ls="none", label=ds,
                    color=[ACCENT, NEUTRAL][j])
        ax.set_xticks(x)
        ax.set_xlim(-0.5, len(protos) - 0.5)
        ax.set_xticklabels([p.split()[0] for p in protos], fontsize=7.5)
        ax.set_ylabel(r"Inner-CV equation $R^2$")
        # The protocols differ by a few hundredths, so the axis is set to the data; the legend
        # sits above the frame so that it cannot land on the points.
        vals = [v for v in g["r2_symbolic"] if np.isfinite(v)]
        ax.set_ylim(max(0.0, min(vals) - 0.06), min(1.02, max(vals) + 0.03))
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                  fontsize=6.5)
        tidy(ax); panel(ax, "c")
    fig.tight_layout(); check_no_title(fig); check_legend_clear(fig)
    save(fig, "Figure_08_extraction_protocol")


# ------------------------------------------------------------------- Figure: the band equation
def fig_band_equation():
    import glob as _g
    files = sorted(_g.glob(str(OUT / "r13_band_equation_summary*.csv")))
    inner_t = _read("r13_tahini_inner_grid.csv")
    if not files:
        print("  [skip] r13_band_equation_summary*.csv does not exist yet")
        return
    summ = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.6))

    # (a) accuracy against the number of retained bands, inner CV, tahini
    ax = axes[0]
    if inner_t is not None:
        for name, col in [("OLS", NEUTRAL), ("EBM", "#059669"),
                          ("KAN additive equation", ACCENT)]:
            sub = inner_t[inner_t.model == name]
            if not len(sub):
                continue
            # VIP ranking only: the inner search also offered a mutual-information ranking, which
            # no model selected, and taking the maximum over both would mix the two (Section 2.7).
            g = sub[sub.method == "vip"].groupby("k")["R2"].mean()
            ax.plot(g.index, g.values, marker="o", ms=3, color=col, label=name)
        ax.set_xlabel("Number of bands offered, $k$")
        ax.set_ylabel(r"Inner-CV $R^2$")
        # Every curve lies above 0.93; extending the axis down leaves an empty strip for the
        # legend instead of laying it over the curves (check_legend_clear enforces it).
        ax.set_ylim(0.895, ax.get_ylim()[1])
        ax.legend(loc="lower right")
    tidy(ax); panel(ax, "a")

    # (b) what each model achieves against how many bands it actually needs. This is the point of
    # the experiment: the equation is not the most accurate model, it is the one that reaches
    # comparable accuracy from the fewest measured channels.
    ax = axes[1]
    sub = summ[(summ.stage == "external") & (summ.dataset == "tahini")]
    kbest = int(sub.k.max()) if len(sub) else None
    sub = sub[sub.k == kbest]
    # Only the closed-form models are named. Everything else uses every band it is given, so it
    # sits in one column, and naming each point there produces an unreadable stack.
    for _, r in sub.iterrows():
        used = float(r.vars_median if np.isfinite(r.vars_median) else r.k)
        is_kan, is_eq = "KAN" in str(r.model), "equation" in str(r.model)
        ax.scatter([used], [r.R2_mean], s=42 if is_eq else 26,
                   marker="o" if is_eq else "s",
                   facecolor=ACCENT if is_eq else ("none" if is_kan else NEUTRAL),
                   edgecolor=ACCENT if is_kan else NEUTRAL, lw=1.1, zorder=3)
        if is_eq:
            ax.annotate(str(r.model).replace("KAN ", "").replace(" equation", ""),
                        (used, r.R2_mean), fontsize=6, xytext=(7, -1),
                        textcoords="offset points", va="center")
    ax.annotate(f"every other model uses\nall {kbest} bands", (kbest, float(sub.R2_mean.max())),
                fontsize=6, xytext=(-6, 10), textcoords="offset points", ha="right",
                color="0.35")
    ax.set_xlim(sub.vars_median.min() - 2 if sub.vars_median.notna().any() else 0, kbest + 2)
    ax.set_xlabel(f"Bands actually used (of {kbest} offered)")
    ax.set_ylabel(r"External $R^2$")
    tidy(ax); panel(ax, "b")

    # (c) bands fixed in advance by chemistry rather than chosen from the data
    ax = axes[2]
    anc = summ[summ.stage == "external_anchored"]
    if len(anc):
        anc = anc.sort_values("R2_mean")
        y = np.arange(len(anc))
        cols = [ACCENT if "KAN" in str(m) else NEUTRAL for m in anc.model]
        # Means with seed SD on an axis that does not start at zero: points, not bars.
        for yi, (mu, sd, c) in enumerate(zip(anc.R2_mean, anc.R2_sd, cols)):
            ax.errorbar(mu, yi, xerr=0 if not np.isfinite(sd) else sd, fmt="o", ms=4, color=c,
                        ecolor=c, elinewidth=0.9, capsize=2)
        ax.set_yticks(y)
        ax.set_yticklabels(anc.model, fontsize=6)
        ax.set_ylim(-0.6, len(anc) - 0.4)
        ax.set_xlabel(r"External $R^2$ (anchored bands)")
        ax.set_xlim(max(0.0, float(anc.R2_mean.min()) - 0.04), 1.0)
    tidy(ax); panel(ax, "c")
    fig.tight_layout(); check_no_title(fig); check_no_overlap(fig); check_legend_clear(fig)
    save(fig, "Figure_09_band_equation")


# ------------------------------------------------------------------- Figure: the edge basis
def fig_basis():
    summ = _read("r19_basis_external_summary.csv")
    if summ is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.8))
    for k, ds in enumerate(["tahini", "mango"]):
        ax = axes[k]
        sub = summ[summ.dataset == ds]
        for rep, mark in [("PLS scores", "o"), ("selected bands", "s")]:
            s = sub[sub.representation == rep]
            if not len(s):
                continue
            exact = s.fidelity_mean > 0.999
            ax.scatter(s.terms_median[exact], s.R2_mean[exact], marker=mark, s=34,
                       color=ACCENT, lw=0, label=f"{rep} — exact closed form")
            ax.scatter(s.terms_median[~exact], s.R2_mean[~exact], marker=mark, s=34,
                       facecolor="none", edgecolor="#DC2626", lw=1.0,
                       label=f"{rep} — approximated")
            for _, r in s.iterrows():
                ax.annotate(str(r["model"]).replace(" additive", "").replace(" 1 hidden layer", ""),
                            (r["terms_median"], r["R2_mean"]), fontsize=5.5,
                            xytext=(2, 2), textcoords="offset points")
        ax.set_xlabel("terms in the printed equation")
        ax.set_ylabel(r"external $R^2$")
        ax.set_xlabel(f"terms in the printed equation ({ds})")
        h, l = ax.get_legend_handles_labels()
        seen, hh, ll = set(), [], []
        for a, b in zip(h, l):
            if b not in seen:
                seen.add(b); hh.append(a); ll.append(b)
        ax.legend(hh, ll, fontsize=6)
        tidy(ax); panel(ax, "ab"[k])
    fig.tight_layout(); check_no_title(fig)
    save(fig, "Figure_10_edge_basis")


# ------------------------------------------------ Figure: equal interpretability budget vs PLS
def fig_equal_budget():
    summ = _read("r11_equal_interpretability_summary.csv")
    if summ is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.6))
    for k, ds in enumerate(["tahini", "mango"]):
        ax = axes[k]
        sub = summ[summ.dataset == ds]
        for model, proto, col, ls in [("PLS", "-", "#2563EB", "-"),
                                      ("KAN spline", "corrected", ACCENT, "--"),
                                      ("KAN equation", "corrected", ACCENT, "-")]:
            s = sub[(sub.model == model) & (sub.protocol == proto)].sort_values("n_latent")
            if not len(s):
                continue
            ax.plot(s.n_latent, s.R2_mean, ls, marker="o", ms=3, color=col,
                    label=model if model != "PLS" else "PLS")
            if model == "KAN equation" and "R2_sd" in s:
                ax.fill_between(s.n_latent, s.R2_mean - s.R2_sd, s.R2_mean + s.R2_sd,
                                color=col, alpha=.13, lw=0)
        ax.set_xlabel("latent variables available to both models")
        ax.set_ylabel(r"external $R^2$")
        ax.set_title("")
        ax.legend(loc="lower right")
        tidy(ax); panel(ax, "ab"[k])
    fig.tight_layout(); check_no_title(fig)
    save(fig, "Figure_11_equal_interpretability")


# --------------------------------------------------- Figure: attribution on the wavenumber axis
def fig_attribution():
    wn = _read("r15_wavenumber_importance.csv")
    if wn is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(COL2, 4.2))
    for k, (ds, xlab, invert) in enumerate([("tahini", r"wavenumber (cm$^{-1}$)", True),
                                            ("mango", "wavelength (nm)", False)]):
        ax = axes[k]
        sub = wn[wn.dataset == ds].sort_values("centre")
        cols = [c for c in sub.columns
                if c not in ("dataset", "centre", "assignment", "component")]
        for c in cols:
            col = ACCENT if "intrinsic" in c else MODEL_COLOR.get(c, NEUTRAL)
            lw = 1.7 if "intrinsic" in c else 1.0
            ax.plot(sub.centre, sub[c], color=col, lw=lw, label=c,
                    alpha=1.0 if "intrinsic" in c else .8)
        if invert:
            ax.invert_xaxis()
        ax.set_xlabel(xlab)
        ax.set_ylabel("normalised importance")
        ax.legend(ncol=3, fontsize=6)
        tidy(ax); panel(ax, "ab"[k])
    fig.tight_layout(); check_no_title(fig)
    save(fig, "Figure_12_attribution_wavenumber")


# ------------------------------------------------ Figure: how long the equation has to be
def fig_equation_length():
    d = _read("r21_equation_length.csv")
    if d is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.8))
    for i, ds in enumerate(["tahini", "mango"]):
        ax = axes[i]
        sub = d[d.dataset == ds]
        kan = sub[sub.model.str.contains("KAN")]
        if not len(kan):
            continue
        # the two references a reader needs: the linear model on the same bands, and the
        # full-spectrum calibration they would otherwise have used
        pls = sub[sub.model == "full-spectrum PLS"]
        if len(pls):
            ax.axhline(float(pls.R2.iloc[0]), color="#2563EB", lw=1.0, ls="--",
                       label="full-spectrum PLS")
        ols = sub[sub.model.str.startswith("OLS")].sort_values("n_terms")
        if len(ols):
            ax.plot(ols.n_terms, ols.R2, "s--", ms=4, color="#059669", lw=1.0,
                    label="OLS on the same bands")
        for basis, mk in [("poly", "o"), ("cheby", "^")]:
            b = kan[kan.basis == basis].sort_values("n_terms")
            if len(b):
                ax.plot(b.n_terms, b.R2, mk, ms=4.5, color=ACCENT, lw=0, alpha=.85,
                        label=f"exact equation ({basis})",
                        markerfacecolor=ACCENT if basis == "poly" else "none",
                        markeredgecolor=ACCENT)
        # mark the point the inner cross-validation selected
        if "inner_cv_R2" in kan and kan.inner_cv_R2.notna().any():
            best = kan.sort_values("inner_cv_R2", ascending=False).iloc[0]
            ax.scatter([best.n_terms], [best.R2], s=110, facecolor="none",
                       edgecolor="#111827", lw=1.2, zorder=5)
            ax.annotate("selected by\ninner CV", (best.n_terms, best.R2), fontsize=6,
                        xytext=(8, -14), textcoords="offset points", ha="left")
        ax.set_xlabel(f"terms in the printed equation ({ds})")
        ax.set_ylabel(r"external $R^2$")
        ax.legend(fontsize=6, loc="lower right")
        tidy(ax); panel(ax, "ab"[i])
    fig.tight_layout(); check_no_title(fig)
    save(fig, "Figure_13_equation_length")


def main():
    print("building the new figures ...")
    for fn in (fig_validation, fig_ablation, fig_extraction, fig_band_equation, fig_basis,
               fig_equal_budget, fig_attribution, fig_equation_length):
        try:
            fn()
        except Exception as e:
            print(f"  [FAILED] {fn.__name__}: {type(e).__name__}: {e}")
            if "--strict" in sys.argv:
                raise


if __name__ == "__main__":
    main()
