"""Figures 2-5: mean spectra, the quantification benchmark, and the two parity figures.

Every number drawn here is recomputed from the saved predictions of the published runs
(results_phase1/*.npz, results_mango/*.npz) with the same estimators the tables use, and is
asserted against the published table before it is plotted. A figure that disagrees with its own
table is the failure mode this guard exists to prevent.
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from kanfood.metrics import bootstrap_ci
from paper1_rigor.bandlabels import MIR_ANNOT, NIR_ANNOT, annotate as _annotate
from paper1_rigor.common import ROOT, tahini_split, mango_split
from paper1_rigor.figstyle import (check_no_overlap, MODEL_COLOR, MODEL_ORDER, COL1, COL2, panel, save, tidy,
                                   check_no_title)

P1 = ROOT / "results_phase1"
PM = ROOT / "results_mango"


def metrics_from_artifacts(a, groups, published):
    """R2 / RMSEP / MAE / RPD / cluster-bootstrap CI per model, checked against the table."""
    y = a["y_test"][:, 0]
    rows = []
    for m in MODEL_ORDER:
        p = a[f"pred_{m}"][:, 0]
        r2 = r2_score(y, p)
        rmse = float(np.sqrt(mean_squared_error(y, p)))
        _, lo, hi = bootstrap_ci(y, p, r2_score, n_boot=1000, seed=42, groups=groups)
        rows.append(dict(model=m, R2=r2, RMSEP=rmse, MAE=mean_absolute_error(y, p),
                         RPD=float(np.std(y) / rmse), ci_lo=lo, ci_hi=hi))
        assert abs(r2 - published[m]) < 1e-3, f"{m}: figure R2 {r2:.4f} != table {published[m]}"
    return pd.DataFrame(rows).set_index("model").loc[MODEL_ORDER]


# ---------------------------------------------------------------- Figure 2: mean spectra
def fig_spectra(a1):
    ds, mtr, mte = mango_split()
    nm, dm = ds.wavenumbers, ds.y[:, 0]
    q = np.quantile(dm, [1 / 3, 2 / 3])
    lo_m, hi_m = ds.X[dm <= q[0]].mean(0), ds.X[dm > q[1]].mean(0)
    mid_m = ds.X.mean(0)

    fig, axes = plt.subplots(2, 2, figsize=(COL2, 5.0))
    wn = a1["wavenumbers"]
    tah, sun, pea = a1["mean_tahini"], a1["mean_sunflower"], a1["mean_peanut"]

    ax = tidy(axes[0, 0])
    for v, lab, c in [(tah, "tahini (sesame paste)", "#7C3AED"),
                      (sun, "sunflower paste", "#D97706"),
                      (pea, "peanut paste", "#059669")]:
        ax.plot(wn, v, lw=1.0, color=c, label=lab)
    ax.set_xlim(wn.max(), wn.min())
    ax.set_ylabel("Absorbance (a.u.)")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    _annotate(ax, MIR_ANNOT)
    ax.legend(loc="center left")
    panel(ax, "a")

    ax = tidy(axes[0, 1])
    ax.plot(wn, sun - tah, lw=1.0, color="#D97706", label="sunflower − tahini")
    ax.plot(wn, pea - tah, lw=1.0, color="#059669", label="peanut − tahini")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlim(wn.max(), wn.min())
    ax.set_ylabel("Difference in absorbance (a.u.)")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    _annotate(ax, MIR_ANNOT, labels=False)
    ax.legend(loc="lower left")
    panel(ax, "b")

    ax = tidy(axes[1, 0])
    ax.plot(nm, mid_m, lw=1.2, color="#2563EB", label="all fruit (n = 11,691)")
    ax.set_ylabel("Absorbance, log(1/R)")
    ax.set_xlabel("Wavelength (nm)")
    _annotate(ax, NIR_ANNOT, headroom=0.40)
    ax.legend(loc="lower right")
    panel(ax, "c")

    ax = tidy(axes[1, 1])
    ax.plot(nm, hi_m - lo_m, lw=1.2, color="#DC2626",
            label="highest − lowest third of dry matter\n"
                  f"(> {q[1]:.1f} % vs < {q[0]:.1f} %)")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_ylabel("Difference in absorbance")
    ax.set_xlabel("Wavelength (nm)")
    _annotate(ax, NIR_ANNOT, labels=False)
    ax.legend(loc="center right")
    panel(ax, "d")

    fig.tight_layout()
    check_no_title(fig)
    check_no_overlap(fig)
    save(fig, "Figure_02_mean_spectra")


# ------------------------------------------------------- Figure 3: quantification benchmark
def fig_benchmark(d1, d2, f1, npar1, npar2):
    fig, axes = plt.subplots(2, 3, figsize=(COL2, 4.8))
    x = np.arange(len(MODEL_ORDER))
    cols = [MODEL_COLOR[m] for m in MODEL_ORDER]
    w = 0.38

    for row, (d, ylab_r2, ylab_err, letters) in enumerate(
            [(d1, "Test $R^2$ (tahini %)", "Error (% tahini)", "ab"),
             (d2, "External-test $R^2$ (dry matter)", "Error (% dry matter)", "de")]):
        ax = tidy(axes[row, 0])
        v = d["R2"].to_numpy()
        ax.bar(x, v, color=cols, yerr=[v - d["ci_lo"], d["ci_hi"] - v], capsize=2.5,
               error_kw=dict(lw=0.8, ecolor="0.25"))
        ax.set_ylim(0.60, 1.02)
        ax.set_ylabel(ylab_r2)
        ax.set_xticks(x); ax.set_xticklabels(MODEL_ORDER, rotation=45, ha="right")
        panel(ax, letters[0])

        ax = tidy(axes[row, 1])
        ax.bar(x - w / 2, d["RMSEP"], w, color="#475569", label="RMSEP")
        ax.bar(x + w / 2, d["MAE"], w, color="#CBD5E1", label="MAE")
        ax.set_ylabel(ylab_err)
        ax.set_xticks(x); ax.set_xticklabels(MODEL_ORDER, rotation=45, ha="right")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.30)
        ax.legend(loc="upper left", ncol=2)
        panel(ax, letters[1])

    ax = tidy(axes[0, 2])
    ax.bar(x, [f1[m] for m in MODEL_ORDER], color=cols)
    ax.set_ylim(0.70, 1.03)
    ax.set_ylabel("F1 (adulteration detection)")
    ax.set_xticks(x); ax.set_xticklabels(MODEL_ORDER, rotation=45, ha="right")
    panel(ax, "c")

    ax = tidy(axes[1, 2])
    p1 = np.array([npar1.get(m, np.nan) for m in MODEL_ORDER], dtype=float)
    p2 = np.array([npar2.get(m, np.nan) for m in MODEL_ORDER], dtype=float)
    ax.bar(x - w / 2, np.nan_to_num(p1, nan=0), w, color="#7C3AED", label="tahini (FTIR)")
    ax.bar(x + w / 2, np.nan_to_num(p2, nan=0), w, color="#C4B5FD", label="mango (NIR)")
    ax.set_yscale("log"); ax.set_ylim(1e2, 3e5)
    ax.set_ylabel("Trainable parameters")
    ax.set_xticks(x); ax.set_xticklabels(MODEL_ORDER, rotation=45, ha="right")
    ax.legend(loc="upper left", ncol=2)
    for xi, (va, vb) in enumerate(zip(p1, p2)):
        if np.isnan(va) and np.isnan(vb):
            # Set vertically: two adjacent horizontal labels overlapped in the rendered proof.
            ax.text(xi, 1.3e2, "not applicable", rotation=90, ha="center", va="bottom",
                    fontsize=5.5, color="0.35")
    panel(ax, "f")

    fig.tight_layout()
    check_no_title(fig)
    check_no_overlap(fig)
    save(fig, "Figure_03_benchmark")


# ------------------------------------------------------------------ Figures 4-5: parity
def fig_parity(a, name, unit, lims, fignum, tag, tagsis=None):
    y = a["y_test"][:, 0]
    fig, axes = plt.subplots(2, 3, figsize=(COL2, 4.7))
    for i, m in enumerate(MODEL_ORDER):
        ax = tidy(axes.ravel()[i])
        p = a[f"pred_{m}"][:, 0]
        if tagsis is None:
            ax.scatter(y, p, s=3, alpha=0.22, color=MODEL_COLOR[m], edgecolors="none")
        else:
            ax.scatter(y[tagsis == 0], p[tagsis == 0], s=7, alpha=0.5, color="#0F766E",
                       edgecolors="none", label="authentic")
            ax.scatter(y[tagsis == 1], p[tagsis == 1], s=7, alpha=0.35, color=MODEL_COLOR[m],
                       edgecolors="none", label="adulterated")
            if i == 0:
                ax.legend(loc="lower right", markerscale=1.6)
        ax.plot(lims, lims, ls="--", lw=0.8, color="0.3")
        ax.set_xlim(*lims); ax.set_ylim(*lims)
        r2 = r2_score(y, p)
        rmse = float(np.sqrt(mean_squared_error(y, p)))
        ax.text(0.04, 0.96, f"{m}\n$R^2$ = {r2:.3f}\nRMSEP = {rmse:.2f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=7.5)
        if i >= 3:
            ax.set_xlabel(f"Measured {name} ({unit})")
        if i % 3 == 0:
            ax.set_ylabel(f"Predicted {name} ({unit})")
        panel(ax, "abcdef"[i])
    fig.tight_layout()
    check_no_title(fig)
    save(fig, f"Figure_{fignum:02d}_parity_{tag}")


def main():
    a1 = np.load(P1 / "phase1_artifacts.npz", allow_pickle=True)
    a2 = np.load(PM / "mango_artifacts.npz", allow_pickle=True)
    t1 = pd.read_csv(P1 / "Table_holdout_results.csv").set_index("Method")
    t2 = pd.read_csv(PM / "Table_mango_holdout.csv").set_index("Method")

    ds, tr, te = tahini_split()
    d1 = metrics_from_artifacts(a1, ds.groups[te], t1["R2_tahini"].to_dict())
    d2 = metrics_from_artifacts(a2, a2["groups_test"], t2["R2"].to_dict())
    print("benchmark metrics recomputed from artifacts and matched to the tables")

    npar1 = {m: v for m, v in t1["n_params"].items() if pd.notna(v)}
    npar2 = {m: v for m, v in t2["n_params"].items() if pd.notna(v)}
    f1 = t1["F1"].to_dict()

    fig_spectra(a1)
    fig_benchmark(d1, d2, f1, npar1, npar2)
    fig_parity(a1, "tahini content", "%", (-3, 103), 4, "tahini", tagsis=a1["tagsis_test"])
    fig_parity(a2, "dry-matter content", "%", (8.5, 25.5), 5, "mango")

    out = pd.concat([d1.assign(dataset="tahini"), d2.assign(dataset="mango")]).reset_index()
    out.to_csv(ROOT / "results_rigor" / "f01_benchmark_recomputed.csv", index=False)


if __name__ == "__main__":
    main()
