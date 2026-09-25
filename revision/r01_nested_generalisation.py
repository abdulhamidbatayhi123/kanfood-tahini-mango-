"""R01 -- Leave-one-lot-out (tahini) / leave-one-season-out (mango) with the selection stage nested.

Addresses hyperparameters carried over from a primary partition that shares
material with the withheld lot/season = selection leak) and informs a revision question.

Three arms, identical outer folds (each production lot / harvest season withheld in turn):

  F  frozen    -- every model re-fitted at the configuration selected on the PUBLISHED primary
                  partition (tahini: grouped 70/30 hold-out; mango: seasons 1-3). This is the design of
                  Supplementary Section S4 / Table S3 and must reproduce it; it is the control.
  A  nested    -- the published search space (same grids, same preprocessing), but selection is
                  redone inside each outer training set by an inner leave-one-lot-out /
                  leave-one-season-out loop, so the withheld unit never informs any choice.
                  A - F isolates the effect of closing the leak.
  B  nested+pp -- as A, and the preprocessing family is selected in the same inner loop by the
                  fold-MEAN criterion rather than a single-fold maximum).
                  B - A isolates the effect of widening the search.

Selection criterion everywhere = the published one (kanfood.tune._fold_score, seed 42).
Final fits: 3 seeds for stochastic models (RF, MLP, CNN, KAN), 1 for PLS/SVR, as in Table S3.
Every final fit stores its out-of-fold predictions so that R03 (low-level identification, authentic
false positives, lot-nested bootstrap) is computed from exactly these models.

Run:  python -m revision.r01_nested_generalisation tahini F A B
      python -m revision.r01_nested_generalisation mango F A
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from revision.common import (CFG, SEEDS3, STOCHASTIC, RESULTS, MODEL_NAMES, load, primary_split,
                             unit_of, leave_one_unit_out, fit_predict, inner_score, param_key,
                             ResultStore, run_jobs, log_to, REPO)

PP_WIDE = {"tahini": ["snv", "snv+sg1", "sg1", "msc", "raw"],
           "mango": ["sg1", "snv+sg1", "snv", "raw"]}
# Arm B also closes an asymmetry in the published search: SVR and RF were fitted at the shared 12- (tahini)
# or 16-component (mango) PLS front-end while MLP and KAN searched their compression. In B every score
# model searches the same compression values as the MLP/KAN grids. The CNN is excluded from B on cost
# grounds (it reads the full spectrum, so the compression asymmetry does not apply to it).
NC_WIDE = {"tahini": (8, 12, 16), "mango": (12, 16, 24)}
B_MODELS = ["PLS", "SVM", "RF", "MLP", "KAN"]


def grid_for(name, arm, model):
    g = CFG[name]["grids"][model]
    if arm == "B" and model in ("SVM", "RF"):
        g = [dict(p, n_components=n) for p in g for n in NC_WIDE[name]]
    return g
OUT = RESULTS / "r01"
OUT.mkdir(parents=True, exist_ok=True)

_DS = {}


def _data(name):
    if name not in _DS:
        _DS[name] = load(name)
    return _DS[name]


def published_configs(name):
    """Configurations the published benchmark selected on its primary partition (written by
    kanfood.run_experiment / kanfood.run_mango to their meta json)."""
    p = REPO / ("results_phase1/phase1_meta.json" if name == "tahini" else "results_mango/mango_meta.json")
    meta = json.loads(p.read_text(encoding="utf-8"))
    return {m: ast.literal_eval(s) for m, s in meta["tuned_params"].items()}


def seeds_for(model):
    return SEEDS3 if model in STOCHASTIC else (42,)


# ------------------------------------------------------------------------------------ job bodies
def job_inner(j):
    ds, meta = _data(j["dataset"])
    s = inner_score(ds, np.array(j["tr"]), np.array(j["va"]), j["model"], j["params"], j["pp"],
                    CFG[j["dataset"]]["n_pls"], seed=42)
    return {"key": j["key"], "stage": "inner", "arm": j["arm"], "outer": j["outer"], "model": j["model"],
            "pp": j["pp"], "params": param_key(j["params"]), "inner": j["inner"], "score": s}


def job_final(j):
    ds, meta = _data(j["dataset"])
    te = np.array(j["te"])
    pred, npar, secs = fit_predict(ds, np.array(j["tr"]), te, j["model"], j["params"], j["pp"],
                                   CFG[j["dataset"]]["n_pls"], seed=j["seed"])
    r2 = float(r2_score(ds.y[te, 0], pred[:, 0]))
    return {"key": j["key"], "stage": "final", "arm": j["arm"], "outer": j["outer"], "model": j["model"],
            "pp": j["pp"], "params": param_key(j["params"]), "seed": j["seed"], "r2": r2,
            "n_params": npar, "fit_s": secs, "rows": te.tolist(), "pred": np.round(pred, 5).tolist()}


# ----------------------------------------------------------------------------------- orchestration
def outer_folds(name):
    ds, meta = _data(name)
    return leave_one_unit_out(unit_of(name, meta))


def inner_jobs(name, arm, models):
    ds, meta = _data(name)
    units = unit_of(name, meta)
    pps = [CFG[name]["preprocess"]] if arm == "A" else PP_WIDE[name]
    jobs = []
    for u, tr, te in outer_folds(name):
        for iu, itr, iva in leave_one_unit_out(units, tr):
            for m in models:
                for pp in pps:
                    for params in grid_for(name, arm, m):
                        key = f"{name}|{arm}|inner|{u}|{m}|{pp}|{param_key(params)}|{iu}"
                        jobs.append({"key": key, "dataset": name, "arm": arm, "outer": u, "inner": iu,
                                     "model": m, "pp": pp, "params": params,
                                     "tr": itr.tolist(), "va": iva.tolist()})
    return jobs


def selected(store, name, arm, models):
    """Best (pp, params) per (outer fold, model) by the inner fold-MEAN score."""
    df = store.frame()
    df = df[(df.get("stage") == "inner") & (df.arm == arm)]
    out = {}
    agg = df.groupby(["outer", "model", "pp", "params"]).score.agg(["mean", "count"]).reset_index()
    for (u, m), g in agg.groupby(["outer", "model"]):
        g = g.sort_values(["mean", "pp", "params"], ascending=[False, True, True])
        best = g.iloc[0]
        out[(str(u), m)] = (best["pp"], json.loads(best["params"]), float(best["mean"]), int(best["count"]))
    return out


def final_jobs(name, arm, models, store):
    jobs = []
    if arm == "F":
        pub = published_configs(name)
        choice = {(str(u), m): (CFG[name]["preprocess"], pub[m]) for u, _, _ in outer_folds(name) for m in models}
    else:
        sel = selected(store, name, arm, models)
        choice = {k: (v[0], v[1]) for k, v in sel.items()}
    for u, tr, te in outer_folds(name):
        for m in models:
            pp, params = choice[(str(u), m)]
            params = {k: (tuple(v) if isinstance(v, list) else v) for k, v in params.items()}
            for sd in seeds_for(m):
                key = f"{name}|{arm}|final|{u}|{m}|{sd}"
                jobs.append({"key": key, "dataset": name, "arm": arm, "outer": u, "model": m, "pp": pp,
                             "params": params, "seed": sd, "tr": tr.tolist(), "te": te.tolist()})
    return jobs


def summarise(name, store):
    df = store.frame()
    df = df[df.get("stage") == "final"]
    if df.empty:
        return None
    per_fold = (df.groupby(["arm", "model", "outer"])
                .agg(r2=("r2", "mean"), r2_sd_seeds=("r2", "std"), n_seeds=("r2", "size"),
                     pp=("pp", "first"), params=("params", "first")).reset_index())
    per_fold.to_csv(OUT / f"{name}_per_fold.csv", index=False)
    summ = (per_fold.groupby(["arm", "model"]).r2
            .agg(mean="mean", sd="std", median="median", worst="min", folds="size").reset_index())
    summ.to_csv(OUT / f"{name}_summary.csv", index=False)
    wide = per_fold.pivot_table(index=["arm", "model"], columns="outer", values="r2").reset_index()
    wide.to_csv(OUT / f"{name}_per_fold_wide.csv", index=False)
    return summ


def main():
    name = sys.argv[1]
    arms = sys.argv[2:] or ["F", "A", "B"]
    models = [m for m in MODEL_NAMES]
    store = ResultStore(OUT / f"{name}_store.jsonl")
    log = log_to(OUT / f"{name}.log")
    for arm in arms:
        ms = B_MODELS if arm == "B" else models
        if arm in ("A", "B"):
            log(f"[{name}] arm {arm}: inner selection")
            run_jobs(job_inner, inner_jobs(name, arm, ms), store, log=log)
        log(f"[{name}] arm {arm}: final fits")
        run_jobs(job_final, final_jobs(name, arm, ms, store), store, log=log)
        s = summarise(name, store)
        log(s.to_string(index=False) if s is not None else "no results")


if __name__ == "__main__":
    main()
