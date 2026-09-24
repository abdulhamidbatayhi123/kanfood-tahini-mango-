"""R05 -- Regenerate Table 4's out-of-fold detection limits exactly, then answer Reviewer 1 comments 2, 4
and 5 on the same predictions.

Table 4 pools out-of-fold predictions over three grouped five-fold partitions of all 55 samples. The
partition seeds are not stated in the article; they are recovered here: kanfood.split.
stratified_group_kfold with seeds 42, 43 and 44 reproduces the PLS per-repeat limits 10.31, 7.53 and
9.59 % exactly. Models at the published configurations, model seed 42.

Outputs:
  table4_reproduction.csv   per model: OOF R^2, pooled LOD, LOD per repeat (compare with Table 4)
  two_stage_bootstrap.csv   lot -> sample two-stage bootstrap of the pooled LOD and OOF R^2 (R1-5)
  pairs.csv                 paired two-stage-bootstrap differences between models, Holm (R1-2)
  authentic.csv             per lot: mean predicted adulterant on authentic tahini (bias), per repeat (R1-4)
  per_level.csv             per adulterant x level: identification, error, detection (R1-3)
Run: python -m revision.r05_oof_lod
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from revision.common import RESULTS, CFG, MODEL_NAMES, load, fit_predict, ResultStore, run_jobs, log_to, _subset
from revision.r01_nested_generalisation import published_configs
from kanfood.split import stratified_group_kfold
from kanfood.metrics import holm_bonferroni
from kanfood.run_experiment import _choose_threshold

OUT = RESULTS / "r05"
OUT.mkdir(parents=True, exist_ok=True)
PARTITION_SEEDS = (42, 43, 44)
N_BOOT = 2000
_D = {}


def _data():
    if not _D:
        ds, meta = load("tahini")
        _D.update(ds=ds, meta=meta)
    return _D


def job(j):
    d = _data()
    te = np.array(j["te"])
    pred, npar, secs = fit_predict(d["ds"], np.array(j["tr"]), te, j["model"], j["params"],
                                   CFG["tahini"]["preprocess"], CFG["tahini"]["n_pls"], seed=42)
    return {"key": j["key"], "model": j["model"], "partition": j["partition"], "fold": j["fold"],
            "rows": te.tolist(), "pred": np.round(pred, 5).tolist()}


def lod(adult_true, adult_pred, auth):
    s = np.std(adult_pred[auth])
    sl = np.polyfit(adult_true, adult_pred, 1)[0]
    return 3.3 * s / abs(sl), s, sl


def main():
    d = _data()
    ds, meta = d["ds"], d["meta"]
    pub = published_configs("tahini")
    store = ResultStore(OUT / "store.jsonl")
    log = log_to(OUT / "run.log")
    jobs = []
    for ps in PARTITION_SEEDS:
        for f, (tr, te) in enumerate(stratified_group_kfold(ds.groups, ds.tagsis, 5, ps)):
            for m in MODEL_NAMES:
                jobs.append({"key": f"{m}|{ps}|{f}", "model": m, "partition": ps, "fold": f, "params": pub[m],
                             "tr": tr.tolist(), "te": te.tolist()})
    run_jobs(job, jobs, store, log=log)

    df = store.frame()
    n = len(ds.y)
    auth = (meta.adulterant == "none").to_numpy()
    at = 100 - ds.y[:, 0]
    lot = meta.lot.to_numpy()
    grp = meta.group.to_numpy()
    kind = meta.adulterant.to_numpy()
    P = {}
    for (m, ps), g in df.groupby(["model", "partition"]):
        arr = np.full((n, 3), np.nan)
        for _, r in g.iterrows():
            arr[r["rows"]] = np.array(r["pred"])
        assert not np.isnan(arr).any()
        P[(m, ps)] = arr

    # Table 4 reproduction. "Pooled" = each spectrum's out-of-fold prediction averaged over the three
    # repeated partitions (this reproduces every Table 4 entry; stacking the repeats does not).
    AVG = {m: np.mean([P[(m, ps)] for ps in PARTITION_SEEDS], axis=0) for m in MODEL_NAMES}
    rows = []
    for m in MODEL_NAMES:
        L, s, sl = lod(at, 100 - AVG[m][:, 0], auth)
        per = [lod(at, 100 - P[(m, ps)][:, 0], auth)[0] for ps in PARTITION_SEEDS]
        rows.append({"model": m, "oof_R2": r2_score(ds.y[:, 0], AVG[m][:, 0]), "LOD_pooled": L,
                     "LOQ_pooled": 10 * s / abs(sl), "LOD_per_repeat": [round(float(x), 2) for x in per],
                     "authentic_bias_pp": float((100 - AVG[m][:, 0])[auth].mean())})
    t4 = pd.DataFrame(rows)
    t4.to_csv(OUT / "table4_reproduction.csv", index=False)

    # two-stage bootstrap (lots, then samples within lots), paired across models
    rng = np.random.RandomState(20260924)
    lots = np.unique(lot)
    samples_by_lot = {L_: np.unique(grp[lot == L_]) for L_ in lots}
    rows_by_sample = {s_: np.where(grp == s_)[0] for s_ in np.unique(grp)}
    draws = {m: {"LOD": [], "R2": []} for m in MODEL_NAMES}
    for b in range(N_BOOT):
        idx = np.concatenate([rows_by_sample[s_] for L_ in rng.choice(lots, len(lots))
                              for s_ in rng.choice(samples_by_lot[L_], len(samples_by_lot[L_]))])
        a = auth[idx]
        for m in MODEL_NAMES:
            st = AVG[m][idx]
            draws[m]["LOD"].append(lod(at[idx], 100 - st[:, 0], a)[0] if a.sum() > 2 else np.nan)
            draws[m]["R2"].append(r2_score(ds.y[idx, 0], st[:, 0]))
    bs, pairs = [], []
    for m in MODEL_NAMES:
        for k in ("LOD", "R2"):
            v = np.array(draws[m][k]); v = v[np.isfinite(v)]
            bs.append({"model": m, "metric": k, "median": np.median(v), "ci_lo": np.percentile(v, 2.5),
                       "ci_hi": np.percentile(v, 97.5), "valid_draws": len(v)})
    for i, a_ in enumerate(MODEL_NAMES):
        for b_ in MODEL_NAMES[i + 1:]:
            for k in ("LOD", "R2"):
                dd = np.array(draws[a_][k]) - np.array(draws[b_][k]); dd = dd[np.isfinite(dd)]
                pairs.append({"metric": k, "model_A": a_, "model_B": b_, "diff_median": np.median(dd),
                              "ci_lo": np.percentile(dd, 2.5), "ci_hi": np.percentile(dd, 97.5),
                              "p_boot": min(1.0, 2 * min(np.mean(dd <= 0), np.mean(dd >= 0)))})
    pairs = pd.DataFrame(pairs)
    pairs["p_holm"] = np.nan
    for k, g in pairs.groupby("metric"):
        pairs.loc[g.index, "p_holm"] = holm_bonferroni(g.p_boot.tolist())
    pd.DataFrame(bs).to_csv(OUT / "two_stage_bootstrap.csv", index=False)
    pairs.to_csv(OUT / "pairs.csv", index=False)

    # authentic bias per lot, and low-level identification, per repeat then averaged
    au, lv = [], []
    for m in MODEL_NAMES:
        for ps in PARTITION_SEEDS:
            p = P[(m, ps)]
            pa = 100 - p[:, 0]
            ident = np.where(p[:, 1] > p[:, 2], "peanut", "sunflower")
            for L_ in lots:
                ii = auth & (lot == L_)
                au.append({"model": m, "partition": ps, "lot": L_, "bias_pp": pa[ii].mean(), "sd_pp": pa[ii].std(ddof=1)})
            for (k_, l_), g in pd.DataFrame({"k": kind, "l": np.round(at).astype(int)}).groupby(["k", "l"]):
                if k_ == "none":
                    continue
                ii = g.index.to_numpy()
                lv.append({"model": m, "partition": ps, "adulterant": k_, "level_pct": l_,
                           "identification_acc": np.mean(ident[ii] == k_), "pred_mean": pa[ii].mean(),
                           "abs_err": np.abs(pa[ii] - l_).mean()})
    pd.DataFrame(au).to_csv(OUT / "authentic.csv", index=False)
    pd.DataFrame(lv).to_csv(OUT / "per_level.csv", index=False)
    pd.set_option("display.width", 250)
    print(t4.round(3).to_string(index=False))
    print(pd.DataFrame(bs).round(3).to_string(index=False))
    print(pairs[pairs.metric == "LOD"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
