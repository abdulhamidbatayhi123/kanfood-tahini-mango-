"""R13 -- a revision question: Supplementary Table S4 shows eight single-factor variants scoring above the
published tahini KAN configuration in cross-validation (best: SNV + first-derivative SG, +0.040). Do the
conclusions depend on a knowingly sub-optimal configuration?

The KAN is refitted with all the favourable single factors combined -- SNV + first-derivative SG (window 11),
grid 3, spline order 4, hidden width 5 -- and compared with the published configuration (SNV, grid 5, order 3,
width 3) under identical protocols:
  * external test (the group-independent test set of Table 2), five seeds;
  * leave-one-lot-out (five folds, three seeds), configuration held fixed (both arms equally).
PLS and the MLP at their published configurations are carried as references on the same rows.
The factors were chosen on the primary training partition's cross-validation (Table S4), so the external test
is used once. KANModel hard-codes spline order 3; `KANk` below differs only in exposing k.
Run: python -m revision.r13_kan_best_factors
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from kan import KAN

from revision.common import RESULTS, load, primary_split, leave_one_unit_out, fit_predict, silent, ResultStore, run_jobs, log_to
from kanfood.preprocess import Preprocessor
from kanfood.features import PLSFeatures
from kanfood.models import KANModel

OUT = RESULTS / "r13"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS5 = (42, 7, 2024, 13, 777)
SEEDS3 = (42, 7, 2024)
CONFIGS = {
    "published": dict(pp="snv", grid=5, k=3, width=(3,), nc=8),
    "best_factors": dict(pp="snv+sg1", grid=3, k=4, width=(5,), nc=8),
}
_D = {}


class KANk(KANModel):
    """Published KANModel with the spline order exposed (identical otherwise)."""

    def __init__(self, k=3, **kw):
        super().__init__(**kw)
        self.k = k

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self._norm = y.shape[1] > 1
        self.sx = MinMaxScaler(feature_range=(-1, 1))
        self.sy = StandardScaler()
        Xs, ys = self.sx.fit_transform(X), self.sy.fit_transform(y)
        t = lambda a: torch.tensor(a, dtype=torch.float32)
        ds = {"train_input": t(Xs), "train_label": t(ys), "test_input": t(Xs), "test_label": t(ys)}
        width = [X.shape[1]] + list(self.width_hidden) + [y.shape[1]]
        self.model = KAN(width=width, grid=self.grid, k=self.k, seed=self.seed, device="cpu",
                         auto_save=False, grid_range=[-1, 1])
        self.model.fit(ds, opt="Adam", steps=self.steps, lr=0.005, batch=128, lamb=self.lamb)
        return self


def _data():
    if not _D:
        ds, meta = load("tahini")
        tr, te = primary_split("tahini", ds, meta)
        _D.update(ds=ds, meta=meta, tr=tr, te=te)
    return _D


def fit_kan(ds, tr, te, cfg, seed):
    pp = Preprocessor(cfg["pp"]).fit(ds.X[tr])
    Str, Ste = pp.transform(ds.X[tr]), pp.transform(ds.X[te])
    f = PLSFeatures(cfg["nc"]).fit(Str, ds.y[tr])
    m = KANk(k=cfg["k"], width_hidden=cfg["width"], grid=cfg["grid"], seed=seed)
    silent(lambda: m.fit(f.transform(Str), ds.y[tr]))
    return m.predict(f.transform(Ste)), m.n_params()


def job(j):
    d = _data()
    ds = d["ds"]
    tr, te = np.array(j["tr"]), np.array(j["te"])
    if j["model"] == "KAN":
        pred, npar = fit_kan(ds, tr, te, CONFIGS[j["cfg"]], j["seed"])
    else:
        pred, npar, _ = fit_predict(ds, tr, te, j["model"], j["params"], "snv", 12, seed=j["seed"])
    return {"key": j["key"], "design": j["design"], "fold": j["fold"], "model": j["model"], "cfg": j["cfg"],
            "seed": j["seed"], "n_params": npar, "r2": float(r2_score(ds.y[te, 0], pred[:, 0]))}


def main():
    from revision.r01_nested_generalisation import published_configs
    pub = published_configs("tahini")
    d = _data()
    jobs = []
    designs = [("external", "test", d["tr"], d["te"], SEEDS5)]
    designs += [("lolo", u, tr, te, SEEDS3) for u, tr, te in leave_one_unit_out(d["meta"].lot.to_numpy())]
    for design, fold, tr, te, seeds in designs:
        for cfg in CONFIGS:
            for sd in seeds:
                jobs.append({"key": f"{design}|{fold}|KAN|{cfg}|{sd}", "design": design, "fold": fold, "model": "KAN",
                             "cfg": cfg, "seed": sd, "tr": tr.tolist(), "te": te.tolist()})
        for m in ("PLS", "MLP"):
            for sd in (seeds if m == "MLP" else (42,)):
                jobs.append({"key": f"{design}|{fold}|{m}|published|{sd}", "design": design, "fold": fold, "model": m,
                             "cfg": "published", "params": pub[m], "seed": sd, "tr": tr.tolist(), "te": te.tolist()})
    store = ResultStore(OUT / "store.jsonl")
    run_jobs(job, jobs, store, n_workers=2, log=log_to(OUT / "run.log"))
    df = store.frame()
    ext = df[df.design == "external"].groupby(["model", "cfg"]).r2.agg(["mean", "std", "min", "max", "count"]).reset_index()
    lolo = (df[df.design == "lolo"].groupby(["model", "cfg", "fold"]).r2.mean().reset_index()
            .groupby(["model", "cfg"]).r2.agg(["mean", "std", "min", "count"]).reset_index())
    ext.to_csv(OUT / "external.csv", index=False)
    lolo.to_csv(OUT / "lolo.csv", index=False)
    df[df.design == "lolo"].groupby(["model", "cfg", "fold"]).r2.mean().unstack().to_csv(OUT / "lolo_per_fold.csv")
    print(ext.round(4).to_string(index=False)); print(lolo.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
