"""R08 -- Reviewer 1 comment 12: are fruit condition, cultivar, temperature and region controlled?

The mango benchmark was built by its authors to CONTAIN these sources of variation rather than to
control them (four seasons, two regions, ten cultivars, hard-green and ripened fruit, and a recorded
temperature treatment label Low/Mid/High/No). What we can do is measure how much of the external-test
error each factor explains, for every model, on the published season-4 test:

  1. Composition of the shift: covariate distribution in calibration (seasons 1-3) vs test (season 4).
  2. Per-covariate-level bias and RMSE on season 4, per model (pred - reference).
  3. Variance decomposition of the season-4 residuals: a population-clustered OLS of residual on
     type + cultivar + temperature label + region, with type-II ANOVA and partial eta^2.
  4. Cultivars absent from calibration are flagged.

Predictions are the published benchmark's (results_mango/mango_artifacts.npz from kanfood.run_mango),
aligned to metadata by row order.   Run: python -m revision.r08_mango_covariates
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

from revision.common import RESULTS, REPO, load, primary_split

OUT = RESULTS / "r08"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    ds, meta = load("mango")
    tr, te = primary_split("mango", ds, meta)
    art = np.load(REPO / "results_mango" / "mango_artifacts.npz", allow_pickle=True)
    assert np.allclose(art["y_test"][:, 0], ds.y[te, 0])
    m_te = meta.iloc[te].reset_index(drop=True)
    y = ds.y[te, 0]

    # 1. covariate shift
    rows = []
    for c in ("type", "cultivar", "temp", "region"):
        a = meta.iloc[tr][c].value_counts(normalize=True)
        b = m_te[c].value_counts(normalize=True)
        for lvl in sorted(set(a.index) | set(b.index)):
            rows.append({"covariate": c, "level": lvl, "calibration_share": round(float(a.get(lvl, 0)), 4),
                         "test_share": round(float(b.get(lvl, 0)), 4),
                         "absent_from_calibration": lvl not in a.index})
    pd.DataFrame(rows).to_csv(OUT / "covariate_shift.csv", index=False)

    # 2 + 3. residual structure per model
    models = [k[5:] for k in art.files if k.startswith("pred_")]
    lvl_rows, anova_rows = [], []
    for m in models:
        r = art[f"pred_{m}"][:, 0] - y
        d = m_te.assign(res=r, y=y)
        for c in ("type", "cultivar", "temp", "region"):
            for lvl, g in d.groupby(c):
                lvl_rows.append({"model": m, "covariate": c, "level": lvl, "n_spectra": len(g),
                                 "n_populations": g.group.nunique(), "bias": float(g.res.mean()),
                                 "rmse": float(np.sqrt(np.mean(g.res ** 2))),
                                 "reference_DM_mean": float(g.y.mean())})
        # One factor at a time: in the season-4 test set some covariates have a single level and cultivar
        # is nested in type, so a joint model is not identifiable. Each factor gets a one-way OLS with a
        # naive F test and a Wald test clustered by population.
        for term in ("type", "cultivar", "temp", "region"):
            if d[term].nunique() < 2:
                anova_rows.append({"model": m, "term": f"C({term})", "df": 0.0, "partial_eta2": np.nan,
                                   "F": np.nan, "p_naive": np.nan, "p_clustered": np.nan,
                                   "note": "single level in test set"})
                continue
            f1 = f"res ~ C({term})"
            aov = sm.stats.anova_lm(smf.ols(f1, d).fit(), typ=2)
            ss_t, ss_res = aov.loc[f"C({term})", "sum_sq"], aov.loc["Residual", "sum_sq"]
            fit = smf.ols(f1, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.group)[0]})
            names = [n for n in fit.params.index if n.startswith(f"C({term})")]
            R = np.zeros((len(names), len(fit.params)))
            for i, n in enumerate(names):
                R[i, list(fit.params.index).index(n)] = 1
            w = fit.wald_test(R, scalar=True)
            anova_rows.append({"model": m, "term": f"C({term})", "df": float(aov.loc[f"C({term})", "df"]),
                               "partial_eta2": float(ss_t / (ss_t + ss_res)), "F": float(aov.loc[f"C({term})", "F"]),
                               "p_naive": float(aov.loc[f"C({term})", "PR(>F)"]),
                               "p_clustered": float(w.pvalue), "note": ""})
    pd.DataFrame(lvl_rows).to_csv(OUT / "residual_by_covariate.csv", index=False)
    pd.DataFrame(anova_rows).to_csv(OUT / "residual_anova.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(lvl_rows).query("model=='KAN'").round(3).to_string(index=False))
    print(pd.DataFrame(anova_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
