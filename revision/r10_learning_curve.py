"""R10 -- Reviewer 2 comment 4: with 38 training samples, are CNN/MLP/KAN appropriate?

Learning curve over the number of independent PHYSICAL SAMPLES in training (not spectra): from the
published primary training partition (38 samples), draw nested random subsets of n in {10, 15, 20,
25, 30, 38} samples, stratified so that each subset keeps authentic and adulterated samples in the
training proportion, fit every model at its published configuration (preprocessing and PLS projection
re-fitted on the subset), and score on the unchanged 17-sample test partition. Five draws per n.
If a flexible model were starved of data, its curve would still be rising steeply at n = 38 and would
sit below PLS's at small n; if the task is learnable at this size, the curves flatten.

Run: python -m revision.r10_learning_curve
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from revision.common import RESULTS, CFG, load, primary_split, fit_predict, ResultStore, run_jobs, log_to, MODEL_NAMES
from revision.r01_nested_generalisation import published_configs

OUT = RESULTS / "r10"
OUT.mkdir(parents=True, exist_ok=True)
SIZES = (10, 15, 20, 25, 30, 38)
DRAWS = 5
_D = {}


def _data():
    if not _D:
        ds, meta = load("tahini")
        tr, te = primary_split("tahini", ds, meta)
        _D.update(ds=ds, meta=meta, tr=tr, te=te)
    return _D


def subsets():
    d = _data()
    g = d["meta"].group.to_numpy()
    samples = np.unique(g[d["tr"]])
    auth = np.array([s for s in samples if "_" not in s])
    adul = np.array([s for s in samples if "_" in s])
    out = []
    for draw in range(DRAWS):
        rng = np.random.RandomState(1000 + draw)
        pa, pd_ = rng.permutation(auth), rng.permutation(adul)
        for n in SIZES:
            na = max(1, int(round(n * len(auth) / len(samples))))
            chosen = np.concatenate([pa[:na], pd_[: n - na]])      # nested: prefixes of one permutation
            out.append((draw, n, chosen))
    return out


def job(j):
    d = _data()
    ds, g = d["ds"], d["meta"].group.to_numpy()
    tr = np.where(np.isin(g, j["samples"]))[0]
    pred, npar, secs = fit_predict(ds, tr, d["te"], j["model"], j["params"], CFG["tahini"]["preprocess"],
                                   CFG["tahini"]["n_pls"], seed=42 + j["draw"])
    return {"key": j["key"], "model": j["model"], "n_samples": j["n"], "draw": j["draw"],
            "n_spectra": int(len(tr)), "r2": float(r2_score(ds.y[d["te"], 0], pred[:, 0])), "fit_s": secs}


def main():
    pub = published_configs("tahini")
    store = ResultStore(OUT / "store.jsonl")
    log = log_to(OUT / "run.log")
    jobs = [{"key": f"{m}|{n}|{draw}", "model": m, "n": n, "draw": draw, "samples": list(s), "params": pub[m]}
            for draw, n, s in subsets() for m in MODEL_NAMES]
    run_jobs(job, jobs, store, log=log)
    df = store.frame()
    s = df.groupby(["model", "n_samples"]).r2.agg(["mean", "std", "min", "count"]).reset_index()
    s.to_csv(OUT / "learning_curve.csv", index=False)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
