"""R07 -- Are the selected ("unassigned") tahini channels chemistry, or drift / baseline / noise?

a revision question and a revision question. Five tests, each with a stated expectation under the
artifact hypothesis, computed on the raw released spectra (no model in the loop except test 4):

  1. Dose-response.  Chemistry => the sample-mean (SNV) absorbance at the channel varies monotonically
     with the adulterant fraction along each 25-level dilution series. Noise/drift => no relation.
     Spearman rho per series, with a permutation p-value over the 25 levels (series order shuffled).
  2. Lot effect among AUTHENTIC samples.  Batch drift/baseline => the five authentic lots differ at the
     channel more than replicate scans of one lot do (one-way ANOVA F over lots, replicate scans as
     within-lot error). Chemistry of adulteration => no particular lot effect is required.
  3. Water-vapour co-variation.  Residual atmospheric water vapour adds sharp rotational lines in the
     1300-1900 cm-1 bending region AND the 3500-3900 cm-1 stretching region, varying from scan to scan
     with the purge state. If a channel's replicate-to-replicate variation is vapour, it co-varies
     with the vapour-only stretching region (3700-3900 cm-1, where a lipid paste itself is ~flat).
     We correlate within-sample replicate residuals (scan minus its sample mean, high-pass filtered).
  4. VIP selection: reproduce the published 16-channel selection, then (a) label-permutation null
     (group-level shuffle of the composition, 200 draws) -- how often would each selected channel be
     picked from pure noise in the labels; (b) stability across the five leave-one-lot-out training
     sets; (c) survival under baseline-removing preprocessing (first-derivative Savitzky-Golay).
  5. Signal-to-noise: between-sample SD of sample means over mean within-sample replicate SD.

Outputs revision/results/r07/*.csv. Run: python -m revision.r07_band_artifact_controls
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression

from revision.common import RESULTS, load, primary_split, leave_one_unit_out
from kanfood.preprocess import Preprocessor

OUT = RESULTS / "r07"
OUT.mkdir(parents=True, exist_ok=True)

PUBLISHED_16 = [3022, 3005, 2962, 1502, 1487, 1472, 1447, 1302, 1286, 1242, 1227, 1196, 1180, 1140, 989, 974]
EQ3_8 = [1502, 1487, 1447, 1286, 1242, 1180, 1140, 989]
WINDOWS = [1490, 1235]                      # the two windows leading every model's attribution (S11)
MIN_SEP = 15.0                               # cm-1, published minimum separation
RNG = np.random.RandomState(20260924)


def vip_scores(pls: PLSRegression) -> np.ndarray:
    """Variable importance in projection (Wold), multi-response form: SS explained per component
    summed over responses."""
    t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p, a = w.shape
    ss = np.sum(t ** 2, axis=0) * np.sum(q ** 2, axis=0)
    wn = w / np.linalg.norm(w, axis=0, keepdims=True)
    return np.sqrt(p * (wn ** 2 @ ss) / ss.sum())


def select_vip(X, Y, wn, k=16, n_comp=12, min_sep=MIN_SEP):
    pls = PLSRegression(n_components=n_comp).fit(X, Y)
    v = vip_scores(pls)
    chosen = []
    for i in np.argsort(v)[::-1]:
        if all(abs(wn[i] - wn[j]) >= min_sep for j in chosen):
            chosen.append(i)
        if len(chosen) == k:
            break
    return sorted(chosen, key=lambda i: -wn[i]), v


def nearest(wn, targets):
    return [int(np.argmin(np.abs(wn - t))) for t in targets]


def main():
    ds, meta = load("tahini")
    wn, X = ds.wavenumbers, ds.X
    g = meta.group.to_numpy()
    S = Preprocessor("snv").fit(X).transform(X)          # SNV is per-spectrum; no fitted state
    targets = sorted(set(PUBLISHED_16 + WINDOWS), reverse=True)
    tidx = dict(zip(targets, nearest(wn, targets)))

    # ---------------------------------------------------------------- 1 + 5: dose-response, SNR
    sm = pd.DataFrame(S).groupby(g).mean()
    comp = pd.DataFrame({"g": g, "pea": ds.y[:, 1], "sun": ds.y[:, 2], "lot": meta.lot}).groupby("g").first().loc[sm.index]
    within_sd = pd.DataFrame(S).groupby(g).std().mean(0).to_numpy()
    snr_all = sm.std(0).to_numpy() / within_sd
    rows = []
    for series, col, other in (("peanut", "pea", "sun"), ("sunflower", "sun", "pea")):
        sel = (comp[other] == 0).to_numpy()          # authentic lots + this adulterant's 25 blends
        level = comp[col].to_numpy()[sel]
        for t, k in tidx.items():
            a = sm.iloc[:, k].to_numpy()[sel]
            rho = stats.spearmanr(level, a).statistic
            null = np.array([stats.spearmanr(RNG.permutation(level), a).statistic for _ in range(2000)])
            p = (1 + np.sum(np.abs(null) >= abs(rho))) / (1 + len(null))
            rows.append({"channel_cm-1": t, "series": series, "n_samples": int(sel.sum()),
                         "spearman_rho": round(float(rho), 3), "perm_p": round(float(p), 4)})
    dose = pd.DataFrame(rows)
    # all-channel reference distribution of |rho| (how extreme are the selected channels?)
    ref = []
    for series, col, other in (("peanut", "pea", "sun"), ("sunflower", "sun", "pea")):
        sel = (comp[other] == 0).to_numpy()
        r = stats.spearmanr(comp[col].to_numpy()[sel][:, None], sm.to_numpy()[sel], axis=0).statistic
        r = np.asarray(r)[0, 1:] if np.ndim(r) == 2 else np.asarray(r)
        ref.append(np.abs(r))
    dose["abs_rho_pct_rank_all_channels"] = [
        round(float((ref[0 if s == "peanut" else 1] < abs(r)).mean() * 100), 1)
        for s, r in zip(dose.series, dose.spearman_rho)]
    dose.to_csv(OUT / "dose_response.csv", index=False)

    # ---------------------------------------------------------------- 2: lot effect (authentic)
    auth = (meta.adulterant == "none").to_numpy()
    lots = meta.lot.to_numpy()[auth]
    lot_rows = []
    Fall = np.array([stats.f_oneway(*[S[auth][lots == L, k] for L in np.unique(lots)]).statistic
                     for k in range(S.shape[1])])
    for t, k in tidx.items():
        groups_ = [S[auth][lots == L, k] for L in np.unique(lots)]
        F, p = stats.f_oneway(*groups_)
        lot_rows.append({"channel_cm-1": t, "anova_F_lots": round(float(F), 2), "p": float(p),
                         "F_pct_rank_all_channels": round(float((Fall < F).mean() * 100), 1),
                         "lot_mean_range_snv": round(float(np.ptp([x.mean() for x in groups_])), 5),
                         "replicate_sd_snv": round(float(np.mean([x.std(ddof=1) for x in groups_])), 5)})
    pd.DataFrame(lot_rows).to_csv(OUT / "authentic_lot_effect.csv", index=False)

    # ---------------------------------------------------------------- 3: water-vapour co-variation
    # replicate residuals on the RAW absorbance, high-pass (subtract 11-point SG smooth) to isolate
    # sharp line-like structure; vapour reference = mean |residual| over 3700-3900 cm-1 per scan.
    raw_res = X - pd.DataFrame(X).groupby(g).transform("mean").to_numpy()
    hp = raw_res - savgol_filter(raw_res, 11, 2, axis=1)
    vap = (wn >= 3700) & (wn <= 3900)
    vap_ref = hp[:, vap].std(1)
    vap_rows = []
    corr_all = np.array([stats.spearmanr(vap_ref, np.abs(hp[:, k])).statistic for k in range(X.shape[1])])
    for t, k in tidx.items():
        rho = stats.spearmanr(vap_ref, np.abs(hp[:, k])).statistic
        vap_rows.append({"channel_cm-1": t, "rho_with_vapour_region": round(float(rho), 3),
                         "pct_rank_all_channels": round(float((corr_all < rho).mean() * 100), 1)})
    # reference: the strongest vapour-affected channels in 1300-1900 (for context)
    bend = (wn >= 1300) & (wn <= 1900)
    top_bend = np.argsort(np.where(bend, corr_all, -np.inf))[::-1][:10]
    pd.DataFrame(vap_rows).to_csv(OUT / "water_vapour_covariation.csv", index=False)
    pd.DataFrame({"channel_cm-1": np.round(wn[top_bend], 1), "rho_with_vapour_region": np.round(corr_all[top_bend], 3)}
                 ).to_csv(OUT / "water_vapour_top_bending_channels.csv", index=False)

    # ---------------------------------------------------------------- 4: VIP selection controls
    tr, te = primary_split("tahini", ds, meta)
    sel_rows, recon = [], {}
    for nc in (8, 12):
        chosen, _ = select_vip(S[tr], ds.y[tr], wn, n_comp=nc)
        got = sorted(np.round(wn[chosen]).astype(int).tolist(), reverse=True)
        match = len(set(got) & set(PUBLISHED_16))
        recon[nc] = (chosen, got, match)
        sel_rows.append({"n_components": nc, "selected_cm-1": got, "matches_published_16": match})
    nc_best = max(recon, key=lambda c: recon[c][2])
    ref_idx = nearest(wn, PUBLISHED_16)

    def hits(chosen, tol=2):
        return np.array([any(abs(wn[i] - wn[j]) <= tol * 1.93 for i in chosen) for j in ref_idx])

    # (a) label-permutation null: shuffle compositions between physical samples
    uniq = np.unique(g[tr])
    ymap = {u: ds.y[tr][g[tr] == u][0] for u in uniq}
    null_hits = []
    for _ in range(200):
        perm = dict(zip(uniq, RNG.permutation(uniq)))
        Yp = np.array([ymap[perm[u]] for u in g[tr]])
        ch, _ = select_vip(S[tr], Yp, wn, n_comp=nc_best)
        null_hits.append(hits(ch))
    null_rate = np.mean(null_hits, axis=0)
    # (b) stability across leave-one-lot-out training sets
    lolo = []
    for u, ltr, lte in leave_one_unit_out(meta.lot.to_numpy()):
        ch, _ = select_vip(S[ltr], ds.y[ltr], wn, n_comp=nc_best)
        lolo.append(hits(ch))
    lolo_rate = np.mean(lolo, axis=0)
    # (c) baseline-removing preprocessing
    D = Preprocessor("snv+sg1").fit(X).transform(X)
    chD, _ = select_vip(D[tr], ds.y[tr], wn, n_comp=nc_best)
    derivD = hits(chD, tol=4)
    pd.DataFrame({"channel_cm-1": PUBLISHED_16, "in_equation_3": [c in EQ3_8 for c in PUBLISHED_16],
                  "selected_under_label_permutation_rate": np.round(null_rate, 3),
                  "selected_in_leave_one_lot_out_rate": np.round(lolo_rate, 2),
                  "selected_after_snv_plus_first_derivative": derivD,
                  "sample_SNR": [round(float(snr_all[i]), 2) for i in ref_idx]}
                 ).to_csv(OUT / "vip_selection_controls.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(OUT / "vip_reproduction.csv", index=False)
    print(pd.DataFrame(sel_rows).to_string(index=False))
    print(pd.read_csv(OUT / "vip_selection_controls.csv").to_string(index=False))
    print(dose.to_string(index=False))
    print(pd.DataFrame(lot_rows).to_string(index=False))
    print(pd.DataFrame(vap_rows).to_string(index=False))


if __name__ == "__main__":
    main()
