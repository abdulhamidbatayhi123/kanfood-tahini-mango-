"""R04 -- statistical power with 3-5 seeds/folds, and which comparisons
are claims and which are descriptive rankings.

The corrected resampled t-test of Nadeau & Bengio (2003) was derived for J independent random
train/test splits of one dataset, with the variance inflated by (1/J + n_test/n_train). The published
tests applied it to 3 seeds or 5 folds. Here it is applied to its own design on tahini:

  * J = 30 random grouped 70/30 hold-outs (GroupShuffleSplit by physical sample, seeds 1000..1029;
    none equal to the published partition's seed 42);
  * inside each split the WHOLE selection is repeated on that split's training samples only (the
    published tune_model: 3-fold stratified grouped inner CV over the published grids), so no split's
    test samples inform its configuration -- unlike Supplementary Section S3, which re-used the
    primary partition's configuration;
  * each model fitted once per split (seed = split seed), scored on the tahini fraction.

Reported for all 15 model pairs: mean difference, corrected t-test (Holm within the family) and a
corrected two-one-sided equivalence test (TOST) at margins of 0.01 and 0.02 R^2, so that "no significant
difference" can be separated into "equivalent within the margin" and "undetermined".

Resumable: every (split, stage, model, ...) is a checkpointed job. Run:
  python -m revision.r04_repeated_grouped_splits [n_splits]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

from revision.common import (RESULTS, CFG, MODEL_NAMES, load, fit_predict, inner_score, param_key,
                             ResultStore, run_jobs, log_to, _subset)
from kanfood.split import stratified_group_kfold
from kanfood.metrics import holm_bonferroni

OUT = RESULTS / "r04"
OUT.mkdir(parents=True, exist_ok=True)
J_DEFAULT = 30
_D = {}


def _data():
    if not _D:
        ds, meta = load("tahini")
        _D.update(ds=ds, meta=meta)
    return _D


def split(j):
    ds = _data()["ds"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=1000 + j)
    return next(gss.split(np.zeros(len(ds.groups)), groups=ds.groups))


def job_inner(jb):
    d = _data()
    s = inner_score(d["ds"], np.array(jb["tr"]), np.array(jb["va"]), jb["model"], jb["params"],
                    CFG["tahini"]["preprocess"], CFG["tahini"]["n_pls"], seed=42)
    return {"key": jb["key"], "stage": "inner", "split": jb["split"], "model": jb["model"],
            "params": param_key(jb["params"]), "score": s}


def job_final(jb):
    d = _data()
    te = np.array(jb["te"])
    pred, npar, secs = fit_predict(d["ds"], np.array(jb["tr"]), te, jb["model"], jb["params"],
                                   CFG["tahini"]["preprocess"], CFG["tahini"]["n_pls"], seed=1000 + jb["split"])
    return {"key": jb["key"], "stage": "final", "split": jb["split"], "model": jb["model"],
            "params": param_key(jb["params"]), "r2": float(r2_score(d["ds"].y[te, 0], pred[:, 0])),
            "n_train": int(len(jb["tr"])), "n_test": int(len(te)), "fit_s": secs}


def inner_jobs(J):
    d = _data()
    ds = d["ds"]
    jobs = []
    for j in range(J):
        tr, te = split(j)
        sub = _subset(ds, tr)
        folds = stratified_group_kfold(sub.groups, sub.tagsis, 3, 42)      # exactly tune_model's folds
        for m in MODEL_NAMES:
            for params in CFG["tahini"]["grids"][m]:
                for f, (a, b) in enumerate(folds):
                    jobs.append({"key": f"inner|{j}|{m}|{param_key(params)}|{f}", "split": j, "model": m,
                                 "params": params, "tr": tr[a].tolist(), "va": tr[b].tolist()})
    return jobs


def final_jobs(J, store):
    df = store.frame()
    df = df[df.stage == "inner"]
    jobs = []
    for j in range(J):
        tr, te = split(j)
        for m in MODEL_NAMES:
            g = df[(df.split == j) & (df.model == m)].groupby("params").score.agg(["mean", "count"])
            g = g[g["count"] == 3].sort_values("mean", ascending=False)
            if g.empty:
                continue
            import json
            params = {k: (tuple(v) if isinstance(v, list) else v) for k, v in json.loads(g.index[0]).items()}
            jobs.append({"key": f"final|{j}|{m}", "split": j, "model": m, "params": params,
                         "tr": tr.tolist(), "te": te.tolist()})
    return jobs


def corrected_tests(df, margins=(0.01, 0.02)):
    wide = df.pivot_table(index="split", columns="model", values="r2").dropna()
    rho = float((df.n_test / df.n_train).mean())
    J = len(wide)
    rows = []
    models = [m for m in MODEL_NAMES if m in wide]
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            dlt = (wide[a] - wide[b]).to_numpy()
            se = np.sqrt(dlt.var(ddof=1) * (1.0 / J + rho))
            t = dlt.mean() / se if se > 0 else 0.0
            p = 2 * stats.t.sf(abs(t), df=J - 1)
            row = {"model_A": a, "model_B": b, "J": J, "rho": rho, "mean_diff": dlt.mean(),
                   "sd_diff": dlt.std(ddof=1), "corrected_se": se, "t": t, "p": p,
                   "A_wins": int((dlt > 0).sum())}
            for mg in margins:   # TOST with the corrected SE: equivalent if both one-sided tests reject
                p_lo = stats.t.sf((dlt.mean() + mg) / se, df=J - 1)
                p_hi = stats.t.cdf((dlt.mean() - mg) / se, df=J - 1)
                row[f"tost_p_{mg}"] = max(p_lo, p_hi)
            rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_bonferroni(out.p.tolist())
    for mg in margins:
        out[f"tost_p_{mg}_holm"] = holm_bonferroni(out[f"tost_p_{mg}"].tolist())
    return out, wide


def main():
    J = int(sys.argv[1]) if len(sys.argv) > 1 else J_DEFAULT
    store = ResultStore(OUT / "store.jsonl")
    log = log_to(OUT / "run.log")
    log(f"R04: {J} grouped splits, nested selection")
    run_jobs(job_inner, inner_jobs(J), store, log=log)
    run_jobs(job_final, final_jobs(J, store), store, log=log)
    df = store.frame()
    df = df[df.stage == "final"]
    summ = df.groupby("model").r2.agg(["mean", "std", "median", "min", "max", "count"]).reset_index()
    summ.to_csv(OUT / "summary.csv", index=False)
    tests, wide = corrected_tests(df)
    tests.to_csv(OUT / "corrected_tests.csv", index=False)
    wide.to_csv(OUT / "per_split_r2.csv")
    pd.set_option("display.width", 250)
    print(summ.round(4).to_string(index=False))
    print(tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
