"""The display equation: a calibration written in named absorbance bands, exactly.

This produces the expression the manuscript prints. The chain is closed end to end — bands are
selected inside the training partition, the network is fitted on those channels alone, the trained
network is converted to sympy (which for an elementary basis is a change of notation, not a fit),
the sympy is converted to LaTeX and then to OMML through Word's own transform, and the OMML is read
back and every coefficient checked. Nothing is transcribed at any point.

Two numbers are recorded next to the equation because a reader needs both: its R2 on the external
test set, and its FIDELITY -- the R2 of the printed expression against the predictions of the
network it came from, obtained by compiling the printed expression and running the held-out rows
through it. For an elementary basis fidelity should be 1 to within floating point; anything less is
the coefficient pruning, and is reported rather than assumed.
"""
import argparse
import sys

import numpy as np
import pandas as pd
import sympy
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

from kanfood.metrics import normalize_to_100
from kanfood.preprocess import Preprocessor
from paper1_rigor import basis_kan as BK
from paper1_rigor.common import (TAHINI, MANGO, SEED, ROOT, tahini_split, mango_split, save)
# write_wrapped, not write: a band equation runs to tens of terms and a single-line display
# is silently clipped at the right page margin (see the equations of Section 3.10).
from paper1_rigor.equation_omml import write_wrapped as write_omml
from paper1_rigor.r13_band_equation import score_vector, select

OUT = ROOT / "submission_foods" / "equations"
OUT.mkdir(parents=True, exist_ok=True)
DATASETS = {
    "tahini": dict(splitter=tahini_split, cfg=TAHINI, min_sep=15.0, normalise=True,
                   unit="cm^{-1}", lhs=r"\hat{y}_{\mathrm{tahini}}"),
    "mango": dict(splitter=mango_split, cfg=MANGO, min_sep=9.0, normalise=False,
                  unit="nm", lhs=r"\hat{y}_{\mathrm{DM}}"),
}


def band_names(axis, idx, unit):
    """A_3005 for a wavenumber, A_970 for a wavelength -- the label a reader can act on."""
    return [f"A_{{{axis[i]:.0f}}}" for i in idx]


def build(label, basis, degree, k, seed, lamb, prune, arch, number):
    d = DATASETS[label]
    ds, tr, te = d["splitter"]()
    pp = Preprocessor(d["cfg"]["preprocess"]).fit(ds.X[tr])
    S_tr, S_te = pp.transform(ds.X[tr]), pp.transform(ds.X[te])
    idx = select(score_vector(S_tr, ds.y[tr], "vip"), ds.wavenumbers, k, d["min_sep"])
    names = band_names(ds.wavenumbers, idx, d["unit"])
    A_tr, A_te, y_tr, y_te = S_tr[:, idx], S_te[:, idx], ds.y[tr], ds.y[te, 0]

    sx, sy = MinMaxScaler((-1, 1)).fit(A_tr), StandardScaler().fit(y_tr)
    Xtr, Xte = sx.transform(A_tr), np.clip(sx.transform(A_te), -1, 1)
    cls = BK.AdditiveKAN if arch == "additive" else BK.TwoLayerKAN
    kw = {} if arch == "additive" else dict(hidden=3)
    m = cls(len(idx), basis=basis, degree=degree, n_out=y_tr.shape[1], **kw)
    BK.fit(m, Xtr, sy.transform(y_tr), lamb=lamb, prune_frac=prune, seed=seed)

    pred_scaled = BK.predict(m, Xte)
    p = np.asarray(sy.inverse_transform(pred_scaled)).reshape(len(Xte), -1)
    pred = (normalize_to_100(p) if d["normalise"] else p)[:, 0]
    r2 = float(r2_score(y_te, pred))

    printed = BK.equation(m, var_names=names)
    ev = BK.evaluate(printed, Xte, var_names=names)
    fidelity = float(r2_score(pred_scaled[:, 0], ev))
    retained = [names[i - 1] for i in BK.retained(m)]

    expr, parts = printed
    latex_lines = [f"{sympy.latex(h)} = {sympy.latex(rhs)}" for h, rhs in parts]
    latex_lines.append(f"{d['lhs']} \\approx {sympy.latex(expr)}")
    name = f"eq{number}_{label}_bands_{basis}"
    write_omml(" \\quad ".join(latex_lines) if parts else latex_lines[-1], name)

    (OUT / f"{name}.txt").write_text(
        f"{label}: closed-form calibration in named absorbance bands\n"
        f"basis {basis} (degree {degree}), {arch}, {k} bands selected by PLS-VIP inside the "
        f"training partition, seed {seed}\n"
        f"external R2 = {r2:.4f}\n"
        f"fidelity of the printed equation to the trained network = {fidelity:.6f}\n"
        f"terms = {BK.n_terms(printed)}; bands retained after pruning = "
        f"{len(retained)} of {k}: {', '.join(retained)}\n"
        f"the scaled inputs are the min-max transform of the absorbances, fitted on the "
        f"training partition\n\n{BK.as_text(printed)}\n", encoding="utf-8")

    print(f"  {label:7s} {basis:8s} {arch:14s} k={k} seed={seed}  external R2 {r2:.4f}  "
          f"fidelity {fidelity:.6f}  {BK.n_terms(printed)} terms  "
          f"{len(retained)}/{k} bands retained")
    return dict(dataset=label, basis=basis, degree=degree, arch=arch, k=k, seed=seed,
                lamb=lamb, prune=prune, R2=r2, fidelity=fidelity,
                n_terms=BK.n_terms(printed), bands_retained=len(retained),
                bands=";".join(names), retained=";".join(retained),
                equation=BK.as_text(printed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tahini", choices=list(DATASETS))
    ap.add_argument("--basis", default="poly", choices=["cheby", "poly", "rbf", "fourier"])
    ap.add_argument("--degree", type=int, default=6)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--lamb", type=float, default=1e-4)
    ap.add_argument("--prune", type=float, default=0.05)
    ap.add_argument("--arch", default="additive", choices=["additive", "hidden"])
    ap.add_argument("--number", type=int, default=3)
    a = ap.parse_args()
    row = build(a.dataset, a.basis, a.degree, a.k, a.seed, a.lamb, a.prune, a.arch, a.number)
    save(pd.DataFrame([row]), f"band_equation_displayed_{a.dataset}_{a.basis}.csv")
    print()
    print((OUT / f"eq{a.number}_{a.dataset}_bands_{a.basis}.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
