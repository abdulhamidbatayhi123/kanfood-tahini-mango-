"""R03 -- analyses derived from the out-of-fold predictions of R01 (leave-one-lot-out, tahini).

Because every lot carries its own interleaved set of adulteration levels (T1: 4/24/44/64/84 %,
T2: 8/28/..., T5: 20/.../100 %) and both adulterants, leave-one-lot-out places EVERY level of both
adulterants, including 4, 8 and 12 %, out of fold exactly once, and every authentic lot too. That is
what makes the following possible without any new model fit:

  (a) a revision question -- identification and quantification at LOW levels: per level and
      adulterant, identification accuracy (argmax of predicted peanut vs sunflower fraction),
      detection rate at the threshold, predicted adulterant % and its error.
  (b) a revision question -- authentic tahini: per lot, mean predicted adulterant (bias), and the
      false-positive rate at the detection threshold, per spectrum and per sample (replicate mean).
      The threshold is re-chosen inside each outer training set with the published rule
      (run_experiment._choose_threshold: PLS, 3-fold grouped CV, max F1), so it never sees the lot.
  (c) a revision question -- lot-nested (two-stage) cluster bootstrap: resample lots with
      replacement, then physical samples within each drawn lot, for R^2, the out-of-fold LOD and the
      paired difference between models. With five lots, the lot level is coarse; that is reported.
  (d) The out-of-fold LOD (3.3 sigma/S) from these leave-one-lot-out predictions, beside Table 4's
      three grouped five-fold partitions.

Run: python -m revision.r03_low_level_authentic_bootstrap [arm ...]   (default arms: F A B)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from revision.common import RESULTS, load, SpectralDataset, _subset, leave_one_unit_out
from kanfood.run_experiment import _choose_threshold

OUT = RESULTS / "r03"
OUT.mkdir(parents=True, exist_ok=True)
R01 = RESULTS / "r01"
RNG_SEED = 20260924
N_BOOT = 2000


def oof_predictions(store_path, arm):
    """Seed-averaged out-of-fold predictions per model: DataFrame rows x [model, tahini, peanut, sunflower]."""
    rows = [json.loads(l) for l in open(store_path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("stage") == "final" and r["arm"] == arm]
    out = {}
    for m in sorted({r["model"] for r in rows}):
        acc, cnt = {}, {}
        for r in (r for r in rows if r["model"] == m):
            for i, p in zip(r["rows"], r["pred"]):
                acc[i] = acc.get(i, 0) + np.asarray(p)
                cnt[i] = cnt.get(i, 0) + 1
        idx = np.array(sorted(acc))
        out[m] = (idx, np.vstack([acc[i] / cnt[i] for i in idx]))
    return out


def lod(adult_true, adult_pred, authentic):
    sigma = np.std(adult_pred[authentic])
    slope = np.polyfit(adult_true, adult_pred, 1)[0]
    return 3.3 * sigma / abs(slope), sigma, slope


def thresholds_per_lot(ds, meta):
    th = {}
    for u, tr, te in leave_one_unit_out(meta.lot.to_numpy()):
        th[u] = _choose_threshold(_subset(ds, tr), seed=42)
    return th


def analyse(arm, ds, meta, th):
    preds = oof_predictions(R01 / "tahini_store.jsonl", arm)
    if not preds:
        return None
    lot = meta.lot.to_numpy()
    adult_true = 100 - ds.y[:, 0]
    auth = (meta.adulterant == "none").to_numpy()
    kind = meta.adulterant.to_numpy()
    level_rows, auth_rows, summ = [], [], []
    for m, (idx, P) in preds.items():
        assert len(idx) == len(ds.y), f"{arm}/{m}: out-of-fold coverage incomplete ({len(idx)})"
        pred_adult = 100 - P[:, 0]
        thr = np.array([th[l] for l in lot])
        flagged = P[:, 0] < thr
        ident = np.where(P[:, 1] > P[:, 2], "peanut", "sunflower")
        # (a) per level x adulterant
        for (k, lv), sub in pd.DataFrame({"k": kind, "lv": np.round(adult_true).astype(int)}).groupby(["k", "lv"]):
            if k == "none":
                continue
            ii = sub.index.to_numpy()
            level_rows.append({"arm": arm, "model": m, "adulterant": k, "level_pct": lv, "lot": lot[ii][0],
                               "n_spectra": len(ii), "identification_acc": float(np.mean(ident[ii] == k)),
                               "detection_rate": float(np.mean(flagged[ii])),
                               "pred_adulterant_mean": float(pred_adult[ii].mean()),
                               "pred_adulterant_sd": float(pred_adult[ii].std(ddof=1)),
                               "abs_error_mean": float(np.abs(pred_adult[ii] - lv).mean())})
        # (b) authentic tahini, per lot
        for L in sorted(np.unique(lot[auth])):
            ii = np.where(auth & (lot == L))[0]
            smp = pred_adult[ii].mean()
            auth_rows.append({"arm": arm, "model": m, "lot": L, "n_spectra": len(ii), "threshold_tahini_pct": th[L],
                              "pred_adulterant_mean": float(smp), "pred_adulterant_sd": float(pred_adult[ii].std(ddof=1)),
                              "false_positive_rate_spectra": float(np.mean(flagged[ii])),
                              "sample_mean_flagged": bool((100 - smp) < th[L])})
        L0, sg, sl = lod(adult_true, pred_adult, auth)
        summ.append({"arm": arm, "model": m, "oof_R2_tahini": float(r2_score(ds.y[:, 0], P[:, 0])),
                     "oof_LOD_pct": float(L0), "sigma_blank": float(sg), "slope": float(sl),
                     "LOQ_pct": float(10 * sg / abs(sl)),
                     "authentic_bias_pct": float(pred_adult[auth].mean()),
                     "authentic_FPR_spectra": float(np.mean(flagged[auth])),
                     "identification_acc_all": float(np.mean(ident[~auth] == kind[~auth])),
                     "identification_acc_le12pct": float(np.mean(ident[(~auth) & (adult_true <= 12)] == kind[(~auth) & (adult_true <= 12)])),
                     "detection_rate_le12pct": float(np.mean(flagged[(~auth) & (adult_true <= 12)]))})
    return preds, pd.DataFrame(level_rows), pd.DataFrame(auth_rows), pd.DataFrame(summ)


def two_stage_bootstrap(arm, ds, meta, preds):
    """Resample lots, then physical samples within each drawn lot (Field & Welsh-style two-stage)."""
    rng = np.random.RandomState(RNG_SEED)
    lot = meta.lot.to_numpy()
    grp = meta.group.to_numpy()
    auth = (meta.adulterant == "none").to_numpy()
    adult_true = 100 - ds.y[:, 0]
    lots = np.unique(lot)
    samples_by_lot = {L: np.unique(grp[lot == L]) for L in lots}
    rows_by_sample = {s: np.where(grp == s)[0] for s in np.unique(grp)}
    models = sorted(preds)
    P = {m: preds[m][1] for m in models}
    draws = {m: {"R2": [], "LOD": []} for m in models}
    for b in range(N_BOOT):
        chosen_lots = rng.choice(lots, len(lots), replace=True)
        idx = []
        for L in chosen_lots:
            ss = samples_by_lot[L]
            for s in rng.choice(ss, len(ss), replace=True):
                idx.append(rows_by_sample[s])
        idx = np.concatenate(idx)
        a = auth[idx]
        for m in models:
            pa = 100 - P[m][idx, 0]
            draws[m]["R2"].append(r2_score(ds.y[idx, 0], P[m][idx, 0]))
            draws[m]["LOD"].append(lod(adult_true[idx], pa, a)[0] if a.sum() > 1 else np.nan)
    rows, pair_rows = [], []
    for m in models:
        for k in ("R2", "LOD"):
            v = np.array(draws[m][k])
            v = v[np.isfinite(v)]
            rows.append({"arm": arm, "model": m, "metric": k, "median": float(np.median(v)),
                         "ci_lo": float(np.percentile(v, 2.5)), "ci_hi": float(np.percentile(v, 97.5)),
                         "n_valid_draws": int(len(v))})
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            for k in ("R2", "LOD"):
                d = np.array(draws[m1][k]) - np.array(draws[m2][k])
                d = d[np.isfinite(d)]
                p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
                pair_rows.append({"arm": arm, "metric": k, "model_A": m1, "model_B": m2,
                                  "diff_median": float(np.median(d)), "ci_lo": float(np.percentile(d, 2.5)),
                                  "ci_hi": float(np.percentile(d, 97.5)), "p_boot": float(min(1.0, p))})
    pairs = pd.DataFrame(pair_rows)
    from kanfood.metrics import holm_bonferroni
    pairs["p_holm"] = np.nan
    for (a_, k), g in pairs.groupby(["arm", "metric"]):
        pairs.loc[g.index, "p_holm"] = holm_bonferroni(g.p_boot.tolist())
    return pd.DataFrame(rows), pairs


def main():
    arms = sys.argv[1:] or ["F", "A", "B"]
    ds, meta = load("tahini")
    th = thresholds_per_lot(ds, meta)
    pd.DataFrame({"lot": list(th), "threshold_tahini_pct": list(th.values())}).to_csv(OUT / "thresholds_per_lot.csv", index=False)
    all_lv, all_auth, all_sum, all_boot, all_pairs = [], [], [], [], []
    for arm in arms:
        res = analyse(arm, ds, meta, th)
        if res is None:
            print(f"arm {arm}: no R01 predictions yet")
            continue
        preds, lv, au, sm = res
        bs, pairs = two_stage_bootstrap(arm, ds, meta, preds)
        all_lv.append(lv); all_auth.append(au); all_sum.append(sm); all_boot.append(bs); all_pairs.append(pairs)
    if not all_sum:
        return
    pd.concat(all_lv).to_csv(OUT / "per_level.csv", index=False)
    pd.concat(all_auth).to_csv(OUT / "authentic_per_lot.csv", index=False)
    pd.concat(all_sum).to_csv(OUT / "summary.csv", index=False)
    pd.concat(all_boot).to_csv(OUT / "two_stage_bootstrap.csv", index=False)
    pd.concat(all_pairs).to_csv(OUT / "two_stage_bootstrap_pairs.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.concat(all_sum).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
