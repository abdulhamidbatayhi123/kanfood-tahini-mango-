"""R12 -- a revision question: when a few channels are selected by VIP, how does the overlapping nature of
vibrational bands affect the result, and how do we know the number of channels is sufficient?

Tahini, published primary split, SNV, linear calibration (OLS on the selected channels; a linear control is
used deliberately so that the curve measures the information in the channels, not a model's capacity):

  1. Sufficiency: grouped 5-fold inner CV on the training partition and the external test, for k = 2..40
     channels ranked by VIP (12 PLS components, 15 cm-1 minimum separation -- the published rule, which
     reproduces the published 16-channel set exactly, see R07) -- selection re-done inside each inner fold.
     Reference lines: full-spectrum PLS (1762 channels).
  2. Overlap: the same k channels as (a) single channels, (b) band-integrated features -- the mean absorbance
     over a +/-7.5 cm-1 window centred on each channel (= the minimum separation, so windows do not overlap).
     If single-channel selection were fragile to band overlap, integrating across each band would change
     the result materially.
Run: python -m revision.r12_band_count_overlap
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from revision.common import RESULTS, load, primary_split
from revision.r07_band_artifact_controls import select_vip
from kanfood.preprocess import Preprocessor
from kanfood.split import stratified_group_kfold
from kanfood.metrics import normalize_to_100

OUT = RESULTS / "r12"
OUT.mkdir(parents=True, exist_ok=True)
KS = (2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40)
HALF = 7.5


def features(S, wn, idx, integrated):
    if not integrated:
        return S[:, idx]
    return np.column_stack([S[:, (wn >= wn[i] - HALF) & (wn <= wn[i] + HALF)].mean(1) for i in idx])


def score(Str, ytr, Ste, yte, wn, k, integrated):
    idx, _ = select_vip(Str, ytr, wn, k=k)
    m = LinearRegression().fit(features(Str, wn, idx, integrated), ytr)
    p = normalize_to_100(m.predict(features(Ste, wn, idx, integrated)))
    return r2_score(yte[:, 0], p[:, 0])


def main():
    ds, meta = load("tahini")
    wn = ds.wavenumbers
    S = Preprocessor("snv").fit(ds.X).transform(ds.X)       # per-spectrum transform, no fitted state
    tr, te = primary_split("tahini", ds, meta)
    folds = stratified_group_kfold(ds.groups[tr], ds.tagsis[tr], 5, 42)
    rows = []
    for k in KS:
        for integ in (False, True):
            cv = [score(S[tr][a], ds.y[tr][a], S[tr][b], ds.y[tr][b], wn, k, integ) for a, b in folds]
            ext = score(S[tr], ds.y[tr], S[te], ds.y[te], wn, k, integ)
            rows.append({"k_channels": k, "features": "band-integrated (+/-7.5 cm-1)" if integ else "single channel",
                         "inner_cv_R2_mean": np.mean(cv), "inner_cv_R2_sd": np.std(cv), "external_R2": ext})
            print(rows[-1], flush=True)
    full = PLSRegression(12).fit(S[tr], ds.y[tr])
    full_ext = r2_score(ds.y[te, 0], normalize_to_100(full.predict(S[te]))[:, 0])
    rows.append({"k_channels": len(wn), "features": "full-spectrum PLS (12 LV)", "inner_cv_R2_mean": np.nan,
                 "inner_cv_R2_sd": np.nan, "external_R2": full_ext})
    pd.DataFrame(rows).to_csv(OUT / "band_count_overlap.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
