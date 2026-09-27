"""Figures 9-11: the KAN's learned response functions, the chemistry of its inputs, and the
agreement between its intrinsic importance and two post-hoc attributions.

The response curves and the equation are read off the *same* fitted network, at the same seed and
on the same split as the equation quoted in the manuscript, so the figure and the equation describe
one model rather than two.
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from kanfood.preprocess import Preprocessor
from kanfood.features import PLSFeatures
from kanfood.interpret import fit_kan
from paper1_rigor.bandlabels import MIR_ANNOT, NIR_ANNOT, annotate as band_annotate
from paper1_rigor.common import (TAHINI, MANGO, SEED, ROOT, tahini_split, mango_split)
from paper1_rigor.figstyle import (COL2, panel, save, tidy, check_no_title,
                                   check_no_overlap, ACCENT, NEUTRAL)

R = ROOT / "results_rigor"


def fit_equation_model(ds, tr, te, preprocess, nc):
    pp = Preprocessor(preprocess).fit(ds.X[tr])
    S_tr, S_te = pp.transform(ds.X[tr]), pp.transform(ds.X[te])
    pf = PLSFeatures(nc).fit(S_tr, ds.y[tr])
    Z_tr, Z_te = pf.transform(S_tr), pf.transform(S_te)
    sx, sy = MinMaxScaler((-1, 1)), StandardScaler()
    Zs, ys = sx.fit_transform(Z_tr), sy.fit_transform(ds.y[tr])
    m, _ = fit_kan(Zs, ys, width_hidden=(3,), steps=250, seed=SEED)
    return m, sx, sy, pf, Zs, ys, Z_te


def response_panel(ax, m, lin, i, g, ylab):
    base = np.zeros((len(g), lin.coef_.shape[0]))
    base[:, i] = g
    with torch.no_grad():
        out = m(torch.tensor(base, dtype=torch.float32)).numpy()[:, 0]
    kan_c = out - out[len(g) // 2]
    lin_c = lin.coef_[i] * g
    ax.plot(g, lin_c, color=NEUTRAL, ls="--", lw=1.1, label="linear (PLS) effect")
    ax.plot(g, kan_c, color=ACCENT, lw=1.5, label="KAN learned function")
    dev = float(np.max(np.abs(kan_c - lin_c)))
    ax.axhline(0, color="0.85", lw=0.6)
    # Clear space above the curves for the annotation; without it the variable label sits on the
    # dashed linear effect in the panels where that line reaches the top-left corner.
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + (y1 - y0) * 0.22)
    ax.text(0.03, 0.96, f"$x_{{{i+1}}}$   dev = {dev:.2f}", transform=ax.transAxes,
            va="top", ha="left", fontsize=6.5)
    ax.set_xlabel(f"$x_{{{i+1}}}$ (scaled)", fontsize=7)
    ax.set_ylabel(ylab, fontsize=6.5)
    ax.tick_params(labelsize=6.5)
    return dev


def fig_responses(tag, ds, tr, te, preprocess, nc, ylab, fignum, ncol=4):
    m, sx, sy, pf, Zs, ys, Z_te = fit_equation_model(ds, tr, te, preprocess, nc)
    lin = LinearRegression().fit(Zs, ys[:, 0])
    g = np.linspace(-1, 1, 160)
    nrow = int(np.ceil(nc / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(COL2, 1.55 * nrow))
    axes = np.atleast_1d(axes).ravel()
    devs = []
    for i in range(nc):
        devs.append(response_panel(tidy(axes[i]), m, lin, i, g, ylab))
        panel(axes[i], "abcdefghijklmnopqrstuvwx"[i], dx=-0.08, dy=1.00)
    for j in range(nc, len(axes)):
        axes[j].axis("off")
    # A legend inside the first panel sits on top of the dashed linear effect, whichever corner
    # it is placed in, because that line crosses the whole panel. One figure-level legend below
    # the panels keeps every panel clear.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout()
    check_no_title(fig)
    save(fig, f"Figure_{fignum:02d}_response_curves_{tag}")
    pd.DataFrame({"dataset": tag, "variable": [f"x{i+1}" for i in range(nc)],
                  "max_nonlinear_deviation": devs}).to_csv(
        R / f"f03_nonlinearity_{tag}.csv", index=False)
    return devs


def fig_chemistry(a1, ds_m, mtr):
    """Figure 11: what the equation's variables mean chemically, in both spectral regions."""
    fig, axes = plt.subplots(2, 1, figsize=(COL2, 4.6))

    ax = tidy(axes[0])
    wn, load = a1["wavenumbers"], a1["pls_loadings"]
    for k in range(3):
        ax.plot(wn, load[:, k], lw=1.0, label=f"$x_{{{k+1}}}$")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlim(wn.max(), wn.min())
    ax.set_xlabel("Wavenumber (cm$^{-1}$)"); ax.set_ylabel("PLS loading")
    band_annotate(ax, MIR_ANNOT, headroom=0.55)
    ax.legend(loc="lower left", ncol=3, fontsize=6.5)
    panel(ax, "a")

    ax = tidy(axes[1])
    pp = Preprocessor(MANGO["preprocess"]).fit(ds_m.X[mtr])
    pf = PLSFeatures(MANGO["kan_eq_nc"]).fit(pp.transform(ds_m.X[mtr]), ds_m.y[mtr])
    nm, loadm = ds_m.wavenumbers, pf.pls.x_loadings_
    for k in range(3):
        ax.plot(nm, loadm[:, k], lw=1.0, label=f"$x_{{{k+1}}}$")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("PLS loading (1st-derivative spectra)")
    band_annotate(ax, NIR_ANNOT, headroom=0.55)
    ax.legend(loc="lower left", ncol=3, fontsize=6.5)
    panel(ax, "b")

    fig.tight_layout()
    check_no_title(fig)
    check_no_overlap(fig)
    save(fig, "Figure_11_chemistry")


def fig_importance():
    """Figure 12: intrinsic KAN importance against permutation and SHAP attribution.

    Built from `r15` where it exists and from `r06` only as a fallback. Both ran the same
    comparison; `r15` is the one Table S12 is built from, and while the figure was drawn from
    `r06` the correlation printed in the panel differed from the table by two rank steps
    (tahini intrinsic-vs-SHAP 0.76 against 0.81) for no reason a reader could see.
    """
    try:
        _p = pd.read_csv(R / "r15_latent_importance.csv")
        per = (_p[_p.model == "KAN"]
               .dropna(subset=["KAN_intrinsic"])
               [["dataset", "component", "KAN_intrinsic", "permutation", "SHAP"]]
               .reset_index(drop=True))
        _c = pd.read_csv(R / "r15_shap_convergence.csv")
        _c = _c[(_c.model == "KAN") & (_c.n_bg == 100)]
        agr = pd.DataFrame({
            "dataset": _c.dataset,
            "rho_intrinsic_vs_SHAP": _c.rho_kanintrinsic_shap,
            "rho_intrinsic_vs_permutation": _c.rho_kanintrinsic_perm,
        }).set_index("dataset")
        if per.empty or agr.empty:
            raise FileNotFoundError("r15 files present but hold no KAN rows")
    except FileNotFoundError:
        print("  Figure 12: falling back to r06; Table S12 is built from r15")
        per = pd.read_csv(R / "r06_importance_per_component.csv")
        agr = pd.read_csv(R / "r06_importance_agreement.csv").set_index("dataset")
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.5))
    for ax, ds, letter in [(axes[0], "tahini", "a"), (axes[1], "mango", "b")]:
        ax = tidy(ax)
        sub = per[per.dataset == ds]
        x = np.arange(len(sub)); w = 0.27
        ax.bar(x - w, sub["KAN_intrinsic"], w, color=ACCENT, label="KAN intrinsic")
        ax.bar(x, sub["permutation"], w, color="#94A3B8", label="permutation")
        ax.bar(x + w, sub["SHAP"], w, color="#0F766E", label="SHAP")
        # With mango's twenty-four latent variables every label is drawn, they run together as
        # "x9x10x11..." and the axis is unreadable; label every other one past twelve.
        step = 1 if len(sub) <= 12 else 2
        ax.set_xticks(x[::step])
        ax.set_xticklabels([f"$x_{{{i+1}}}$" for i in range(0, len(sub), step)], fontsize=6.5)
        ax.set_ylabel("Relative importance")
        # Room above the tallest bar, so that neither it nor the legend meets the frame.
        tallest = float(sub[["KAN_intrinsic", "permutation", "SHAP"]].to_numpy().max())
        ax.set_ylim(0, tallest * 1.30)
        r = agr.loc[ds]
        ax.text(0.97, 0.95,
                f"Spearman $\\rho$\nintrinsic–SHAP {r['rho_intrinsic_vs_SHAP']:.2f}\n"
                f"intrinsic–perm. {r['rho_intrinsic_vs_permutation']:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.2)
        if letter == "a":
            ax.legend(loc="upper left", fontsize=6.2)
        panel(ax, letter)
    fig.tight_layout()
    check_no_title(fig)
    save(fig, "Figure_12_importance_agreement")


def main():
    ds, tr, te = tahini_split()
    fig_responses("tahini", ds, tr, te, TAHINI["preprocess"], TAHINI["kan_nc"],
                  "Δ standardised tahini %", 9, ncol=4)
    dm, mtr, mte = mango_split()
    fig_responses("mango", dm, mtr, mte, MANGO["preprocess"], MANGO["kan_eq_nc"],
                  "Δ standardised dry matter", 10, ncol=4)
    a1 = np.load(ROOT / "results_phase1" / "phase1_artifacts.npz", allow_pickle=True)
    fig_chemistry(a1, dm, mtr)
    try:
        fig_importance()
    except FileNotFoundError as e:
        print(f"  skipped importance figure: {e.filename} not written yet")


if __name__ == "__main__":
    main()
