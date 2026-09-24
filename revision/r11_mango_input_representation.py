"""R11 -- Reviewer 2 comment 5: why compress the non-linear mango spectra with LINEAR PLS before the
KAN/MLP? Why not a method that handles non-linearity?

Same external test (season 4), same preprocessing (SG1, window 11), same model configurations as the
published benchmark; only the input representation changes:

  pls      k PLS latent scores (published; supervised, linear)
  full     all 103 preprocessed channels, no compression at all (mango has only 103 channels, so the
           non-linear models can be given the whole spectrum -- the most direct answer)
  pca      k principal-component scores (unsupervised, linear) -- isolates supervision
  kpca     k kernel-PCA scores, RBF kernel (unsupervised, non-linear), fitted on a fixed random subset of
           3000 training spectra (exact kernel PCA on 10,243 spectra is a 10k x 10k eigenproblem) with
           gamma chosen by the median heuristic; k and the subset are fixed before seeing the test set.

k equals each model's published compression. Three seeds per cell. Scaling inside the models is
unchanged (KAN: min-max to the spline grid; MLP: standardisation), so each representation reaches the
network on the same footing.        Run: python -m revision.r11_mango_input_representation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import r2_score, pairwise_distances

from revision.common import RESULTS, CFG, SEEDS3, load, primary_split, silent, ResultStore, run_jobs, log_to
from revision.r01_nested_generalisation import published_configs
from kanfood.preprocess import Preprocessor
from kanfood.features import PLSFeatures
from kanfood.models import build_model

OUT = RESULTS / "r11"
OUT.mkdir(parents=True, exist_ok=True)
REPS = ("pls", "full", "pca", "kpca")
MODELS = ("KAN", "MLP", "SVM")
_D = {}


def _data():
    if not _D:
        ds, meta = load("mango")
        tr, te = primary_split("mango", ds, meta)
        pp = Preprocessor(CFG["mango"]["preprocess"]).fit(ds.X[tr])
        _D.update(ds=ds, tr=tr, te=te, Str=pp.transform(ds.X[tr]), Ste=pp.transform(ds.X[te]))
    return _D


def represent(rep, k):
    d = _data()
    Str, Ste, ytr = d["Str"], d["Ste"], d["ds"].y[d["tr"]]
    if rep == "full":
        return Str, Ste
    if rep == "pls":
        f = PLSFeatures(k).fit(Str, ytr)
        return f.transform(Str), f.transform(Ste)
    mu, sd = Str.mean(0), Str.std(0) + 1e-12
    Ztr, Zte = (Str - mu) / sd, (Ste - mu) / sd
    if rep == "pca":
        p = PCA(n_components=k, random_state=0).fit(Ztr)
        return p.transform(Ztr), p.transform(Zte)
    rng = np.random.RandomState(0)
    sub = rng.choice(len(Ztr), 3000, replace=False)
    med = np.median(pairwise_distances(Ztr[sub][:1000]) ** 2)
    kp = KernelPCA(n_components=k, kernel="rbf", gamma=1.0 / med, random_state=0).fit(Ztr[sub])
    return kp.transform(Ztr), kp.transform(Zte)


def job(j):
    d = _data()
    Xtr, Xte = represent(j["rep"], j["k"])
    mp = {k: v for k, v in j["params"].items() if k != "n_components"}
    model = build_model(j["model"], Xtr.shape[1], 1, seed=j["seed"], **mp)
    silent(lambda: model.fit(Xtr, d["ds"].y[d["tr"]]))
    pred = model.predict(Xte)
    return {"key": j["key"], "model": j["model"], "rep": j["rep"], "k": j["k"], "input_dim": int(Xtr.shape[1]),
            "seed": j["seed"], "r2": float(r2_score(d["ds"].y[d["te"], 0], pred[:, 0])),
            "rmsep": float(np.sqrt(np.mean((pred[:, 0] - d["ds"].y[d["te"], 0]) ** 2))),
            "n_params": model.n_params()}


def main():
    pub = published_configs("mango")
    store = ResultStore(OUT / "store.jsonl")
    log = log_to(OUT / "run.log")
    jobs = []
    for m in MODELS:
        k = pub[m].get("n_components", CFG["mango"]["n_pls"])
        for rep in REPS:
            for sd in (SEEDS3 if m != "SVM" else (42,)):
                jobs.append({"key": f"{m}|{rep}|{sd}", "model": m, "rep": rep, "k": k, "seed": sd, "params": pub[m]})
    run_jobs(job, jobs, store, n_workers=3, log=log)
    df = store.frame()
    s = df.groupby(["model", "rep", "input_dim"]).agg(r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
                                                      rmsep=("rmsep", "mean"), n=("r2", "size")).reset_index()
    s.to_csv(OUT / "summary.csv", index=False)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
