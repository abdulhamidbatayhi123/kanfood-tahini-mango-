"""How short can the printed equation be before it stops being accurate?

An exact closed form in named bands is only useful if a person can read it. Two knobs set its
length: how many bands are retained, and what degree of expansion each band's function is given.
A degree-6 polynomial on eight bands is exact and 49 terms long — correct, and nobody will use it.

This traces the frontier. For each (bands, basis, degree) the equation is scored TWICE: by grouped
cross-validation inside the training partition, which is what the displayed equation is chosen on,
and on the external test set, which is what is reported. Both are given for every point, so a
reader can see that the configuration was not picked by looking at the test set — the distinction
matters here because the frontier is shallow and choosing its maximum on the test set would be
worth about 0.005 R2 of self-deception.

Two controls are carried at every setting: ordinary least squares on exactly the same bands, which
says whether the non-linearity is doing any work at all, and the full-spectrum PLS calibration,
which is the accuracy a reader gives up by moving to a handful of channels.
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

from kanfood.metrics import normalize_to_100
from kanfood.preprocess import Preprocessor
from paper1_rigor import basis_kan as BK
from kanfood.split import stratified_group_kfold
from paper1_rigor.common import (TAHINI, MANGO, SEED, tahini_split, mango_split, save)
from paper1_rigor.r13_band_equation import score_vector, select

K_GRID = [4, 6, 8, 12]
INNER_FOLDS = 3
DEGREES = [2, 3, 4, 6]
BASES = ["poly", "cheby"]
SEEDS = [42, 7, 2024]


def _post(p, normalise):
    p = np.asarray(p).reshape(len(p), -1)
    return (normalize_to_100(p) if normalise else p)[:, 0]


def fit_score(A_tr, y_tr, A_te, y_te1, basis, deg, seed, normalise, names=None):
    """One equation at one setting. Returns its R2, its printed length and its fidelity."""
    sx, sy = MinMaxScaler((-1, 1)).fit(A_tr), StandardScaler().fit(y_tr)
    Xtr, Xte = sx.transform(A_tr), np.clip(sx.transform(A_te), -1, 1)
    m = BK.AdditiveKAN(A_tr.shape[1], basis=basis, degree=deg, n_out=y_tr.shape[1])
    BK.fit(m, Xtr, sy.transform(y_tr), lamb=1e-4, prune_frac=0.05, seed=seed)
    ps = BK.predict(m, Xte)
    r2 = float(r2_score(y_te1, _post(sy.inverse_transform(ps), normalise)))
    if names is None:
        return r2, np.nan, np.nan
    printed = BK.equation(m, var_names=names)
    try:
        fid = float(r2_score(ps[:, 0], BK.evaluate(printed, Xte, var_names=names)))
    except Exception:
        fid = np.nan
    return r2, BK.n_terms(printed), fid


def inner_score(ds, tr, cfg, idx, basis, deg, normalise):
    """Grouped cross-validation inside the training partition. This is what selects the
    configuration; the external test set is only reported."""
    folds = stratified_group_kfold(ds.groups[tr], ds.tagsis[tr], n_splits=INNER_FOLDS, seed=SEED)
    out = []
    for a, b in folds:
        pp = Preprocessor(cfg["preprocess"]).fit(ds.X[tr[a]])
        S_a, S_b = pp.transform(ds.X[tr[a]]), pp.transform(ds.X[tr[b]])
        r2, _, _ = fit_score(S_a[:, idx], ds.y[tr[a]], S_b[:, idx], ds.y[tr[b], 0], basis, deg,
                             SEED, normalise)
        out.append(r2)
    return float(np.mean(out))


def run(label, splitter, cfg, min_sep, normalise):
    ds, tr, te = splitter()
    pp = Preprocessor(cfg["preprocess"]).fit(ds.X[tr])
    S_tr, S_te = pp.transform(ds.X[tr]), pp.transform(ds.X[te])
    y_tr, y_te = ds.y[tr], ds.y[te, 0]
    score = score_vector(S_tr, y_tr, "vip")

    # the accuracy a reader gives up by leaving the full spectrum
    pls = PLSRegression(n_components=cfg["pls_nc"]).fit(S_tr, y_tr)
    pls_r2 = float(r2_score(y_te, _post(pls.predict(S_te), normalise)))
    print(f"\n=== {label}: equation length against accuracy "
          f"(full-spectrum PLS reference {pls_r2:.4f}) ===", flush=True)

    rows = []
    for k in K_GRID:
        idx = select(score, ds.wavenumbers, k, min_sep)
        names = [f"A_{{{ds.wavenumbers[i]:.0f}}}" for i in idx]
        A_tr, A_te = S_tr[:, idx], S_te[:, idx]
        ols = float(r2_score(y_te, _post(LinearRegression().fit(A_tr, y_tr).predict(A_te),
                                         normalise)))
        folds = stratified_group_kfold(ds.groups[tr], ds.tagsis[tr], n_splits=INNER_FOLDS,
                                       seed=SEED)
        ols_inner = []
        for a, b in folds:
            pf = Preprocessor(cfg["preprocess"]).fit(ds.X[tr[a]])
            Sa, Sb = pf.transform(ds.X[tr[a]]), pf.transform(ds.X[tr[b]])
            ols_inner.append(float(r2_score(
                ds.y[tr[b], 0],
                _post(LinearRegression().fit(Sa[:, idx], ds.y[tr[a]]).predict(Sb[:, idx]),
                      normalise))))
        rows.append(dict(dataset=label, model="OLS on the same bands", basis="-", k=k,
                         degree=np.nan, seed=SEED, inner_cv_R2=float(np.mean(ols_inner)),
                         R2=ols, n_terms=k + 1, fidelity=1.0, bands=";".join(names)))
        for basis in BASES:
            for deg in DEGREES:
                t0 = time.time()
                inner = inner_score(ds, tr, cfg, idx, basis, deg, normalise)
                res = [fit_score(A_tr, y_tr, A_te, y_te, basis, deg, s, normalise, names)
                       for s in SEEDS]
                r2s = [r[0] for r in res]
                terms = [r[1] for r in res]
                fids = [r[2] for r in res]
                rows.append(dict(dataset=label, model=f"additive KAN ({basis})", basis=basis,
                                 k=k, degree=deg, seed=np.nan,
                                 inner_cv_R2=inner,
                                 R2=float(np.mean(r2s)), R2_sd=float(np.std(r2s, ddof=1)),
                                 n_terms=float(np.median(terms)),
                                 fidelity=float(np.nanmean(fids)),
                                 bands=";".join(names)))
                print(f"  k={k:<3} {basis:6s} degree {deg}  inner-CV {inner:.4f}  "
                      f"external {np.mean(r2s):.4f} ± {np.std(r2s, ddof=1):.4f}   "
                      f"{np.median(terms):.0f} terms   fidelity {np.nanmean(fids):.6f}   "
                      f"(OLS on the same bands {ols:.4f})   ({time.time() - t0:.0f}s)", flush=True)
    rows.append(dict(dataset=label, model="full-spectrum PLS", basis="-", k=np.nan,
                     degree=np.nan, seed=SEED, R2=pls_r2, n_terms=np.nan, fidelity=1.0,
                     bands=""))
    return rows


def main():
    t0 = time.time()
    rows = run("tahini", tahini_split, TAHINI, 15.0, True)
    save(pd.DataFrame(rows), "r21_equation_length_tahini.csv")
    rows += run("mango", mango_split, MANGO, 9.0, False)
    df = pd.DataFrame(rows)
    save(df, "r21_equation_length.csv")
    kan = df[df.model.str.contains("KAN")]
    print("\n-- selected by inner CV, per dataset (the external column is reported, not used "
          "to choose) --")
    for ds_ in df.dataset.dropna().unique():
        sub = kan[kan.dataset == ds_].sort_values("inner_cv_R2", ascending=False)
        if not len(sub):
            continue
        b = sub.iloc[0]
        o = df[(df.dataset == ds_) & (df.model.str.startswith("OLS")) & (df.k == b.k)]
        pls_row = df[(df.dataset == ds_) & (df.model == "full-spectrum PLS")]
        print(f"  {ds_}: {b.basis} degree {int(b.degree)} on {int(b.k)} bands -> "
              f"inner-CV {b.inner_cv_R2:.4f}, external {b.R2:.4f} +- {b.R2_sd:.4f}, "
              f"{b.n_terms:.0f} terms, fidelity {b.fidelity:.6f}\n"
              f"      OLS on the same bands: external {o.R2.iloc[0]:.4f} "
              f"({int(o.n_terms.iloc[0])} coefficients)"
              f" | full-spectrum PLS: {pls_row.R2.iloc[0]:.4f}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
