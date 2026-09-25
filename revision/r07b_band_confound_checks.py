"""R07b -- follow-up to R07: can the composition signal at the selected channels be a vapour or batch
confound, and how large is authentic lot-to-lot variation in adulterant-equivalent units?

  1. Within-lot dose-response. Each production lot carries its own five peanut and five sunflower
     blends plus its authentic sample, measured as one batch. If the channel's response to composition
     were a batch/session artefact it would not replicate in the same direction inside every lot.
     Reported: per lot and series, the Spearman sign and the least-squares slope of sample-mean SNV
     absorbance on adulterant %; and how many of the 5 lots agree in sign.
  2. Is the vapour state itself confounded with composition? Per-scan vapour index (high-pass residual
     SD over 3700-3900 cm-1, as in R07) averaged per sample, correlated with adulterant % per series.
  3. Lot-to-lot variation of AUTHENTIC tahini in adulterant-equivalent %: (range of authentic lot means)
     / |slope of the pooled dose-response|, per channel. This is the per-channel analogue of the
     out-of-fold detection limit of Table 4.
  4. Named-band calibration with and without the vapour-affected channels: OLS on the 16 published
     channels on the primary split (must reproduce Table 7's 0.9923), then with 1502/1487/1472 removed,
     and on the 8 Equation-(3) channels with and without 1502/1487.

Run: python -m revision.r07b_band_confound_checks
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from revision.common import RESULTS, load, primary_split
from revision.r07_band_artifact_controls import PUBLISHED_16, EQ3_8, WINDOWS, nearest
from kanfood.preprocess import Preprocessor
from kanfood.metrics import normalize_to_100

OUT = RESULTS / "r07"


def main():
    ds, meta = load("tahini")
    wn, X = ds.wavenumbers, ds.X
    g = meta.group.to_numpy()
    S = Preprocessor("snv").fit(X).transform(X)
    targets = sorted(set(PUBLISHED_16 + WINDOWS), reverse=True)
    tidx = dict(zip(targets, nearest(wn, targets)))
    sm = pd.DataFrame(S).groupby(g).mean()
    info = pd.DataFrame({"g": g, "lot": meta.lot, "pea": ds.y[:, 1], "sun": ds.y[:, 2]}).groupby("g").first().loc[sm.index]

    # 1. within-lot dose-response
    rows = []
    for t, k in tidx.items():
        for series, col, other in (("peanut", "pea", "sun"), ("sunflower", "sun", "pea")):
            signs, slopes = [], []
            for L in sorted(info.lot.unique()):
                sel = ((info.lot == L) & (info[other] == 0)).to_numpy()
                x, a = info[col].to_numpy()[sel], sm.iloc[:, k].to_numpy()[sel]
                if len(x) < 3 or np.ptp(x) == 0:
                    continue
                slopes.append(np.polyfit(x, a, 1)[0])
                signs.append(np.sign(stats.spearmanr(x, a).statistic))
            pooled_sel = (info[other] == 0).to_numpy()
            pooled = np.polyfit(info[col].to_numpy()[pooled_sel], sm.iloc[:, k].to_numpy()[pooled_sel], 1)[0]
            agree = int(np.sum(np.sign(slopes) == np.sign(pooled)))
            rows.append({"channel_cm-1": t, "series": series, "pooled_slope_per_pct": float(pooled),
                         "lots_agreeing_in_sign": f"{agree}/{len(slopes)}",
                         "per_lot_slopes": [float(f"{s:.3g}") for s in slopes]})
    within = pd.DataFrame(rows)
    within.to_csv(OUT / "within_lot_dose_response.csv", index=False)

    # 2. vapour index vs composition
    raw_res = X - pd.DataFrame(X).groupby(g).transform("mean").to_numpy()
    hp = raw_res - savgol_filter(raw_res, 11, 2, axis=1)
    vap = hp[:, (wn >= 3700) & (wn <= 3900)].std(1)
    vs = pd.Series(vap).groupby(g).mean().loc[sm.index]
    vrows = []
    for series, col, other in (("peanut", "pea", "sun"), ("sunflower", "sun", "pea")):
        sel = (info[other] == 0).to_numpy()
        r = stats.spearmanr(info[col].to_numpy()[sel], vs.to_numpy()[sel])
        vrows.append({"series": series, "n_samples": int(sel.sum()), "spearman_rho_vapour_vs_adulterant": round(float(r.statistic), 3),
                      "p": round(float(r.pvalue), 4)})
    lot_v = stats.kruskal(*[vs[info.lot == L].to_numpy() for L in sorted(info.lot.unique())])
    vrows.append({"series": "lot effect on vapour index (Kruskal-Wallis, all 55 samples)", "n_samples": 55,
                  "spearman_rho_vapour_vs_adulterant": round(float(lot_v.statistic), 3), "p": round(float(lot_v.pvalue), 4)})
    pd.DataFrame(vrows).to_csv(OUT / "vapour_index_vs_composition.csv", index=False)

    # 2b. does the composition effect survive adjustment for the vapour index (and for lot)?
    #     sample-mean absorbance ~ adulterant% + vapour index [+ lot fixed effects], per series.
    import statsmodels.formula.api as smf
    prow = []
    for t, k in tidx.items():
        for series, col, other in (("peanut", "pea", "sun"), ("sunflower", "sun", "pea")):
            sel = (info[other] == 0).to_numpy()
            d = pd.DataFrame({"a": sm.iloc[:, k].to_numpy()[sel], "c": info[col].to_numpy()[sel],
                              "v": vs.to_numpy()[sel], "lot": info.lot.to_numpy()[sel]})
            d["a"] = (d.a - d.a.mean()) / d.a.std()
            d["v"] = (d.v - d.v.mean()) / d.v.std()
            m0 = smf.ols("a ~ c", d).fit()
            m1 = smf.ols("a ~ c + v", d).fit()
            m2 = smf.ols("a ~ c + v + C(lot)", d).fit()
            prow.append({"channel_cm-1": t, "series": series,
                         "coef_c_alone": m0.params["c"], "coef_c_adj_vapour": m1.params["c"],
                         "p_c_adj_vapour": m1.pvalues["c"], "coef_vapour": m1.params["v"], "p_vapour": m1.pvalues["v"],
                         "coef_c_adj_vapour_lot": m2.params["c"], "p_c_adj_vapour_lot": m2.pvalues["c"],
                         "R2_c_alone": m0.rsquared, "R2_c_vapour_lot": m2.rsquared})
    pd.DataFrame(prow).to_csv(OUT / "composition_effect_adjusted_for_vapour_and_lot.csv", index=False)

    # 3. authentic lot range in adulterant-equivalent %
    auth = info[(info.pea == 0) & (info.sun == 0)]
    erows = []
    for t, k in tidx.items():
        lot_means = sm.loc[auth.index].iloc[:, k].to_numpy()
        for series in ("peanut", "sunflower"):
            s = within[(within["channel_cm-1"] == t) & (within.series == series)].pooled_slope_per_pct.iloc[0]
            erows.append({"channel_cm-1": t, "series": series,
                          "authentic_lot_range_snv": float(np.ptp(lot_means)),
                          "authentic_lot_sd_snv": float(np.std(lot_means, ddof=1)),
                          "slope_snv_per_pct": float(s),
                          "lot_range_in_adulterant_pct": float(np.ptp(lot_means) / abs(s)) if s else np.nan,
                          "3.3xlot_sd_in_adulterant_pct": float(3.3 * np.std(lot_means, ddof=1) / abs(s)) if s else np.nan})
    pd.DataFrame(erows).round(4).to_csv(OUT / "authentic_lot_range_equivalent.csv", index=False)

    # 4. named-band OLS with/without vapour-affected channels (primary split)
    tr, te = primary_split("tahini", ds, meta)
    brows = []

    def ols(chans, label):
        idx = nearest(wn, chans)
        m = LinearRegression().fit(S[tr][:, idx], ds.y[tr])
        p = normalize_to_100(m.predict(S[te][:, idx]))
        brows.append({"band_set": label, "n_channels": len(idx), "channels": chans,
                      "external_R2_tahini": round(float(r2_score(ds.y[te, 0], p[:, 0])), 4)})

    ols(PUBLISHED_16, "published 16 (Table 7 OLS = 0.9923)")
    ols([c for c in PUBLISHED_16 if c not in (1502, 1487, 1472)], "16 minus 1502/1487/1472")
    ols(EQ3_8, "Equation (3) 8")
    ols([c for c in EQ3_8 if c not in (1502, 1487)], "Equation (3) 8 minus 1502/1487")
    ols([c for c in PUBLISHED_16 if c not in (1502, 1487, 1472, 1447)], "16 minus 1440-1510 region")
    bands = pd.DataFrame(brows)
    bands.to_csv(OUT / "named_band_ols_vapour_ablation.csv", index=False)

    pd.set_option("display.width", 250)
    print(within.to_string(index=False))
    print(pd.DataFrame(vrows).to_string(index=False))
    print(pd.read_csv(OUT / "authentic_lot_range_equivalent.csv").to_string(index=False))
    print(bands.to_string(index=False))


if __name__ == "__main__":
    main()
