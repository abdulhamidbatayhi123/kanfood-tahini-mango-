"""Does the edge basis have to be a B-spline? An ablation in which the equation is exact.

The question
------------
Everything the manuscript claims rests on converting a trained network into a closed-form
equation. In the published pipeline that conversion is `auto_symbolic`, which replaces each
learned B-spline edge with the best-fitting member of a small library of elementary functions.
`r12` shows most of the accuracy lost at that step is procedural, but a residual loss is
unavoidable in principle: a cubic B-spline is not an elementary function, so the conversion is
always an approximation, and `r12` also shows it occasionally fails outright on a particular
seed.

If the edge basis is elementary to begin with, there is nothing to approximate. A Chebyshev,
Fourier or Gaussian expansion is a finite sum of elementary terms, so the trained network *is* a
closed-form expression: printing it is a change of notation, not a fit. This experiment asks
whether that costs accuracy, and what it costs in equation length.

What is compared
----------------
* Bases: Chebyshev polynomials, a Fourier series, Gaussian radial basis functions, raw powers,
  and B-splines. The B-spline arm is the control -- it is trained identically but has no
  elementary closed form, which is precisely the property under test.
* The published pykan chain (B-spline + `auto_symbolic`, corrected protocol) is carried through
  as the reference the paper currently reports, via the shared `equation.extract`.
* Two depths: the additive network (a generalised additive model in the chosen basis, one
  univariate function per input) and one Kolmogorov-Arnold hidden layer.
* Two input representations: the PLS latent scores the main pipeline uses, and individually
  selected absorbance bands, so that the exact-equation result can be read either as a latent
  calibration or as an equation in named wavenumbers.

For every arm we report accuracy on held-out data, the number of terms in the printed
expression, and -- the number that matters here -- FIDELITY: the R2 of the printed equation
against the predictions of the network it came from, evaluated by lambdifying the printed
expression and running the held-out rows through it. For an elementary basis that number should
be 1 to within floating point; anything less is the coefficient pruning, and is reported rather
than assumed.

Selection is by grouped inner cross-validation on the training partition. The external test set
is used once, at the end.
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
import sympy
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from kanfood.metrics import normalize_to_100
from kanfood.preprocess import Preprocessor
from kanfood.split import stratified_group_kfold
from paper1_rigor import basis_kan as BK
from paper1_rigor.common import (seed_list, ckpt, TAHINI, MANGO, SEED, tahini_split, mango_split, project, save)
from paper1_rigor.equation import extract
from paper1_rigor.r13_band_equation import score_vector, select

warnings.filterwarnings("ignore")

INNER_FOLDS = 3
FINAL_SEEDS = [42, 7, 2024, 100, 13]
# Raised on the command line for the *Foods* revision (reviewer 2, point 8). `seed_list(n)` is a
# prefix of this list, so a larger budget extends the published result rather than redrawing it.
TAG = ""
BASES = ["cheby", "fourier", "rbf", "poly", "bspline"]
ARCHS = [("additive", BK.AdditiveKAN, {}), ("1 hidden layer", BK.TwoLayerKAN, dict(hidden=3))]
DEGREES = [4, 6, 8]
REGULARISERS = [(1e-5, 0.0), (1e-4, 0.05), (1e-4, 0.15), (1e-3, 0.05), (1e-3, 0.15)]
BAND_K = 8


def _post(p, normalise):
    p = np.asarray(p).reshape(len(p), -1)
    return (normalize_to_100(p) if normalise else p)[:, 0]


def fit_score(A_tr, y_tr, A_te, y_te1, basis, cls, kw, degree, lamb, prune, seed, normalise,
              want_equation=False, var_names=None):
    """One arm. Inputs are scaled to [-1, 1], which is the domain every basis is defined on."""
    sx, sy = MinMaxScaler((-1, 1)).fit(A_tr), StandardScaler().fit(y_tr)
    Xtr, Xte = sx.transform(A_tr), np.clip(sx.transform(A_te), -1, 1)
    # Tahini's target is the three-component composition, not a scalar: the network must emit all
    # three so that the sum-to-100 normalisation the rest of the pipeline applies still works.
    # The printed equation is for the first output, the tahini fraction, which is what is reported.
    m = cls(A_tr.shape[1], basis=basis, degree=degree, n_out=y_tr.shape[1], **kw)
    # Budget in epochs, so a large dataset is not silently undertrained (see basis_kan.fit).
    BK.fit(m, Xtr, sy.transform(y_tr), lamb=lamb, prune_frac=prune, seed=seed)
    pred_scaled = BK.predict(m, Xte)
    pred = _post(sy.inverse_transform(pred_scaled), normalise)
    out = dict(basis=basis, degree=degree, lamb=lamb, prune=prune, seed=seed,
               R2=float(r2_score(y_te1, pred)),
               n_params=int(sum(p.numel() for p in m.parameters())),
               n_vars=len(BK.retained(m)))
    out["fidelity"], out["n_terms"], out["equation"] = np.nan, np.nan, ""
    if want_equation and basis != "bspline":
        names = var_names or [f"x_{i+1}" for i in range(A_tr.shape[1])]
        printed = BK.equation(m, var_names=names)
        try:
            ev = BK.evaluate(printed, Xte, var_names=names)
            out["fidelity"] = float(r2_score(pred_scaled[:, 0], ev))
        except Exception:
            out["fidelity"] = float("nan")
        out["n_terms"] = BK.n_terms(printed)
        out["equation"] = BK.as_text(printed)[:4000]
    return out


def inner(ds, tr, cfg, rep, normalise, arms):
    folds = stratified_group_kfold(ds.groups[tr], ds.tagsis[tr], n_splits=INNER_FOLDS, seed=SEED)
    rows = []
    for a, b in folds:
        A_a, A_b = rep(tr[a], tr[b])
        for basis, aname, cls, kw, degree, lamb, prune in arms:
            r = fit_score(A_a, ds.y[tr[a]], A_b, ds.y[tr[b], 0], basis, cls, kw, degree,
                          lamb, prune, SEED, normalise)
            r.update(arch=aname)
            rows.append(r)
    return pd.DataFrame(rows)


def run(label, splitter, cfg, nc_key, normalise, min_sep):
    ds, tr, te = splitter()
    y_te1 = ds.y[te, 0]
    nc = cfg[nc_key]
    rows, ext = [], []
    print(f"\n=== {label}: edge-basis ablation ===", flush=True)

    # ---- representation 1: the PLS scores the published pipeline uses -------------------
    def rep_pls(i_a, i_b):
        A_a, A_b, *_ = project(ds, i_a, i_b, cfg["preprocess"], nc)
        return A_a, A_b

    # ---- representation 2: individually selected absorbance bands -----------------------
    def make_rep_band(k):
        def rep(i_a, i_b):
            pp = Preprocessor(cfg["preprocess"]).fit(ds.X[i_a])
            S_a, S_b = pp.transform(ds.X[i_a]), pp.transform(ds.X[i_b])
            idx = select(score_vector(S_a, ds.y[i_a], "vip"), ds.wavenumbers, k, min_sep)
            return S_a[:, idx], S_b[:, idx]
        return rep

    for rep_name, rep in [("PLS scores", rep_pls), ("selected bands", make_rep_band(BAND_K))]:
        print(f"  -- {rep_name}: basis x depth x degree, inner grouped CV --", flush=True)
        arms = [(b, an, cls, kw, d, 1e-4, 0.05)
                for b in BASES for an, cls, kw in ARCHS for d in DEGREES]
        t0 = time.time()
        df = inner(ds, tr, cfg, rep, normalise, arms)
        df["dataset"], df["representation"], df["stage"] = label, rep_name, "inner_basis"
        rows.append(df)
        agg = (df.groupby(["basis", "arch", "degree"])["R2"].mean().reset_index()
                 .sort_values("R2", ascending=False))
        print(agg.head(8).round(4).to_string(index=False), flush=True)
        print(f"     ({time.time() - t0:.0f}s)", flush=True)
        best = agg.iloc[0]

        print("  -- regularisation at the selected basis/depth/degree --", flush=True)
        cls, kw = next((c, k) for a, c, k in ARCHS if a == best["arch"])
        arms2 = [(best["basis"], best["arch"], cls, kw, int(best["degree"]), lam, pr)
                 for lam, pr in REGULARISERS]
        df2 = inner(ds, tr, cfg, rep, normalise, arms2)
        df2["dataset"], df2["representation"], df2["stage"] = label, rep_name, "inner_reg"
        rows.append(df2)
        agg2 = df2.groupby(["lamb", "prune"])["R2"].mean().reset_index().sort_values(
            "R2", ascending=False)
        print(agg2.round(4).to_string(index=False), flush=True)
        lam, pr = float(agg2.iloc[0]["lamb"]), float(agg2.iloc[0]["prune"])

        print("  -- external test --", flush=True)
        A_tr, A_te = rep(tr, te)
        names = None
        if rep_name == "selected bands":
            pp = Preprocessor(cfg["preprocess"]).fit(ds.X[tr])
            idx = select(score_vector(pp.transform(ds.X[tr]), ds.y[tr], "vip"),
                         ds.wavenumbers, BAND_K, min_sep)
            names = [f"A_{ds.wavenumbers[i]:.0f}" for i in idx]
        # every elementary basis at the selected depth/degree, plus the B-spline control
        for basis in BASES:
            for seed in FINAL_SEEDS:
                r = fit_score(A_tr, ds.y[tr], A_te, y_te1, basis, cls, kw, int(best["degree"]),
                              lam, pr, seed, normalise, want_equation=True, var_names=names)
                r.update(dataset=label, representation=rep_name, arch=best["arch"],
                         stage="external", model=f"{basis} {best['arch']}")
                ext.append(r)
            sub = [e for e in ext if e["model"] == f"{basis} {best['arch']}"
                   and e["representation"] == rep_name]
            print(f"    {basis:9s} R2 {np.mean([e['R2'] for e in sub]):.4f}  "
                  f"fidelity {np.nanmean([e['fidelity'] for e in sub]):.6f}  "
                  f"{np.nanmedian([e['n_terms'] for e in sub]):.0f} terms", flush=True)

        # the published chain on the same inputs, as the reference point
        for seed in FINAL_SEEDS:
            r = extract(A_tr, ds.y[tr], A_te, y_te1, normalise=normalise, seed=seed,
                        width=(3,), steps=400, prune_steps=100, refit_steps=200,
                        full_batch_symbolic=True)
            r.pop("model", None)
            ext.append(dict(dataset=label, representation=rep_name, stage="external",
                            model="pykan B-spline + auto_symbolic", basis="bspline+symbolic",
                            arch="(3,)", degree=np.nan, lamb=np.nan, prune=np.nan, seed=seed,
                            R2=r["r2_symbolic"], R2_network=r["r2_spline"],
                            fidelity=r["r2_fidelity"],
                            n_terms=r["n_terms"], n_vars=r["n_vars"],
                            n_params=r["n_params"], equation=r.get("equation", "")[:4000]))
        sub = [e for e in ext if e["model"] == "pykan B-spline + auto_symbolic"
               and e["representation"] == rep_name]
        print(f"    {'pykan+symbolic':9s} R2 {np.nanmean([e['R2'] for e in sub]):.4f}  "
              f"fidelity {np.nanmean([e['fidelity'] for e in sub]):.6f}  "
              f"{np.nanmedian([e['n_terms'] for e in sub]):.0f} terms", flush=True)
        ckpt(pd.concat(rows, ignore_index=True), f"r19_basis_inner_{label}_partial{TAG}.csv")
        ckpt(ext, f"r19_basis_external_{label}_partial{TAG}.csv")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(ext)


def main():
    global FINAL_SEEDS, TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(FINAL_SEEDS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--datasets", default="tahini,mango")
    a = ap.parse_args()
    FINAL_SEEDS = seed_list(a.seeds)
    TAG = ("_" + a.tag) if a.tag else ""
    want = [d for d in a.datasets.split(",") if d]
    print(f"r19: {len(FINAL_SEEDS)} seeds {FINAL_SEEDS}, tag={TAG!r}, datasets={want}", flush=True)
    t0 = time.time()
    parts_i, parts_e = [], []
    if "tahini" in want:
        i1, e1 = run("tahini", tahini_split, TAHINI, "kan_nc", True, 15.0)
        save(i1, f"r19_basis_inner_tahini{TAG}.csv")
        save(e1, f"r19_basis_external_tahini{TAG}.csv")
        parts_i.append(i1); parts_e.append(e1)
    if "mango" in want:
        i2, e2 = run("mango", mango_split, MANGO, "kan_eq_nc", False, 9.0)
        save(i2, f"r19_basis_inner_mango{TAG}.csv")
        save(e2, f"r19_basis_external_mango{TAG}.csv")
        parts_i.append(i2); parts_e.append(e2)
    save(pd.concat(parts_i, ignore_index=True), f"r19_basis_inner{TAG}.csv")
    ext = pd.concat(parts_e, ignore_index=True)
    save(ext, f"r19_basis_external_raw{TAG}.csv")
    summ = (ext.groupby(["dataset", "representation", "model"])
            .agg(R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
                 fidelity_mean=("fidelity", "mean"), fidelity_min=("fidelity", "min"),
                 terms_median=("n_terms", "median"), vars_median=("n_vars", "median"),
                 params_median=("n_params", "median"))
            .reset_index())
    save(summ, f"r19_basis_external_summary{TAG}.csv")
    print("\n", summ.to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
