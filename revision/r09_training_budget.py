"""R09 -- Reviewer 1 comment 7: under the published fixed-step recipe the parameter-matched MLP receives
3,600 (tahini) / 32,200 (mango) gradient updates against the KAN's 300. Is the KAN's comparable
performance a training-budget artefact?

Direct crossover test on the published external splits, five seeds: each architecture is trained at
its OWN published budget and at the OTHER architecture's budget, at matched trainable parameters.

  KAN : published configuration, steps = 300 (own) or = the MLP's update count
  MLP : one hidden layer sized to the KAN's trainable-parameter count (with the published
        BatchNorm/dropout block), epochs = 200 (own; = 3,600 / 32,200 updates at batch 64) or the
        number of epochs giving ~300 updates (the KAN's budget)

If the KAN's parity were a budget artefact in its favour, giving it the MLP's budget could not help and
giving the MLP the KAN's budget would not hurt; the reverse pattern shows the asymmetry favours the MLP.
Run: python -m revision.r09_training_budget
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from revision.common import RESULTS, CFG, load, primary_split, fit_predict, ResultStore, run_jobs, log_to
from revision.r01_nested_generalisation import published_configs

OUT = RESULTS / "r09"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS5 = (42, 7, 2024, 13, 777)
_D = {}


def _data(name):
    if name not in _D:
        ds, meta = load(name)
        tr, te = primary_split(name, ds, meta)
        _D[name] = (ds, tr, te)
    return _D[name]


def mlp_hidden_for(n_in, n_out, target):
    """One hidden layer h with Linear+BatchNorm (published MLPModel block): params = n_in*h + h + 2h + h*n_out + n_out."""
    return max(1, round((target - n_out) / (n_in + 3 + n_out)))


def job(j):
    ds, tr, te = _data(j["dataset"])
    pred, npar, secs = fit_predict(ds, tr, te, j["model"], j["params"], CFG[j["dataset"]]["preprocess"],
                                   CFG[j["dataset"]]["n_pls"], seed=j["seed"])
    return {"key": j["key"], "dataset": j["dataset"], "model": j["model"], "budget": j["budget"],
            "updates": j["updates"], "seed": j["seed"], "n_params": npar,
            "r2": float(r2_score(ds.y[te, 0], pred[:, 0])), "fit_s": secs}


def main():
    store = ResultStore(OUT / "store.jsonl")
    log = log_to(OUT / "run.log")
    jobs = []
    for name in ("tahini", "mango"):
        try:
            pub = published_configs(name)
        except FileNotFoundError:
            log(f"{name}: published configuration not available yet; skipped")
            continue
        ds, tr, te = _data(name)
        kan = dict(pub["KAN"])
        nc = kan.get("n_components", CFG[name]["n_pls"])
        # KAN trainable parameters at the published configuration (measured, not estimated)
        _, kan_params, _ = fit_predict(ds, tr[:400], te[:10], "KAN", dict(kan, steps=1),
                                       CFG[name]["preprocess"], CFG[name]["n_pls"], seed=42)
        h = mlp_hidden_for(nc, ds.y.shape[1], kan_params)
        batches = math.ceil(len(tr) / 64)
        mlp_updates = 200 * batches
        epochs_kan_budget = max(1, round(300 / batches))
        cells = [("KAN", "own", 300, dict(kan, steps=300)),
                 ("KAN", "mlp_budget", mlp_updates, dict(kan, steps=mlp_updates)),
                 ("MLP", "own", mlp_updates, {"hidden": (h,), "n_components": nc, "epochs": 200}),
                 ("MLP", "kan_budget", epochs_kan_budget * batches,
                  {"hidden": (h,), "n_components": nc, "epochs": epochs_kan_budget})]
        log(f"{name}: KAN params {kan_params}; matched MLP hidden {h}; MLP updates {mlp_updates}")
        for m, b, u, p in cells:
            for sd in SEEDS5:
                jobs.append({"key": f"{name}|{m}|{b}|{sd}", "dataset": name, "model": m, "budget": b,
                             "updates": u, "params": p, "seed": sd})
    run_jobs(job, jobs, store, log=log)
    df = store.frame()
    s = df.groupby(["dataset", "model", "budget", "updates", "n_params"]).r2.agg(["mean", "std", "min", "max", "count"]).reset_index()
    s.to_csv(OUT / "summary.csv", index=False)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
