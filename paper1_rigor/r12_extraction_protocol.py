"""How much of the closed-form equation's accuracy loss is the extraction procedure, not the model?

The published chain converts a trained KAN to a closed form with pykan's default
`auto_symbolic`, and reports a large drop against the spline network it came from
(tahini 0.9834 -> 0.9479). Two properties of that default are responsible and neither is
intrinsic to the architecture:

  A. `auto_symbolic` snaps each edge by least-squares fitting `c*f(a*x+b)+d` to `self.acts`,
     the activations of the LAST FORWARD PASS. Inside `MultKAN.fit(..., batch=128)` that is a
     random 128-row mini-batch, so every edge is snapped against 128 calibration rows rather
     than the whole calibration set.
  B. pykan's symbolic edges keep `a, b, c, d` as trainable parameters, and the documented
     workflow is train -> prune -> auto_symbolic -> TRAIN AGAIN so those parameters are fitted
     jointly. The published chain stops before that final step.

Four protocols are compared. Selection is by grouped cross-validation INSIDE the training
partition -- the external test set is touched only once, at the end, with the protocol the inner
CV selected. Every protocol is run at several seeds because the symbolic step is not
reproducible run to run (that non-reproducibility is itself one of the reported results).
"""
import argparse
import time
import numpy as np
import pandas as pd

from kanfood.split import stratified_group_kfold
from paper1_rigor.common import (seed_list, TAHINI, MANGO, SEED, tahini_split, mango_split, project, save)
from paper1_rigor.equation import extract

INNER_SEEDS = [42, 7, 2024]
FINAL_SEEDS = [42, 7, 2024, 100, 13, 999, 123, 50, 777, 888]
# Raised on the command line for the *Foods* revision (reviewer 1 point 8, reviewer 2 point 8).
TAG = ""

PROTOCOLS = {
    "P0 published (mini-batch snap, no refit)": dict(full_batch_symbolic=False, refit_steps=0),
    "P1 full-calibration snap":                 dict(full_batch_symbolic=True,  refit_steps=0),
    "P2 post-symbolic refit only":              dict(full_batch_symbolic=False, refit_steps=200),
    "P3 full snap + post-symbolic refit":       dict(full_batch_symbolic=True,  refit_steps=200),
}


def one(Z_tr, y_tr, Z_te, y_te, normalise, seed, **kw):
    r = extract(Z_tr, y_tr, Z_te, y_te, normalise=normalise, seed=seed, **kw)
    r.pop("model", None)
    return r


def inner_cv(label, ds, tr, cfg, nc, normalise):
    """Protocol selection on the training partition only."""
    folds = stratified_group_kfold(ds.groups[tr], ds.tagsis[tr], n_splits=5, seed=SEED)
    rows = []
    for pname, kw in PROTOCOLS.items():
        for fi, (a, b) in enumerate(folds, 1):
            Z_a, Z_b, *_ = project(ds, tr[a], tr[b], cfg["preprocess"], nc)
            for seed in INNER_SEEDS:
                r = one(Z_a, ds.y[tr[a]], Z_b, ds.y[tr[b], 0], normalise, seed, **kw)
                rows.append(dict(dataset=label, stage="inner_cv", protocol=pname, fold=fi,
                                 seed=seed, **{k: r[k] for k in
                                               ("r2_spline", "r2_symbolic", "r2_fidelity",
                                                "n_terms", "n_vars")}))
            print(f"    {pname:42s} fold {fi}  symbolic "
                  f"{np.nanmean([x['r2_symbolic'] for x in rows[-len(INNER_SEEDS):]]):.4f}",
                  flush=True)
    return rows


def final_test(label, ds, tr, te, cfg, nc, normalise, chosen):
    Z_tr, Z_te, *_ = project(ds, tr, te, cfg["preprocess"], nc)
    rows = []
    for pname in ("P0 published (mini-batch snap, no refit)", chosen):
        for seed in FINAL_SEEDS:
            r = one(Z_tr, ds.y[tr], Z_te, ds.y[te, 0], normalise, seed, **PROTOCOLS[pname])
            rows.append(dict(dataset=label, stage="external", protocol=pname, fold=np.nan,
                             seed=seed, equation=r.get("equation", ""),
                             **{k: r[k] for k in ("r2_spline", "r2_symbolic", "r2_fidelity",
                                                  "n_terms", "n_vars")}))
            print(f"    {pname:42s} seed {seed:<5} spline {r['r2_spline']:.4f}  "
                  f"symbolic {r['r2_symbolic']:.4f}  fidelity {r['r2_fidelity']:.4f}  "
                  f"{r['n_terms']} terms", flush=True)
    return rows


def run(label, splitter, cfg, nc_key, normalise):
    ds, tr, te = splitter()
    nc = cfg[nc_key]
    print(f"\n=== {label}: extraction protocol (nc={nc}, width (3,)) ===", flush=True)
    print("  -- protocol selection by grouped CV inside the training partition --", flush=True)
    rows = inner_cv(label, ds, tr, cfg, nc, normalise)
    df = pd.DataFrame(rows)
    means = df.groupby("protocol")["r2_symbolic"].mean()
    chosen = means.idxmax()
    print(f"  inner-CV symbolic R2 by protocol:\n{means.round(4).to_string()}", flush=True)
    print(f"  -> selected: {chosen}", flush=True)
    print("  -- external test, selected protocol vs published protocol --", flush=True)
    rows += final_test(label, ds, tr, te, cfg, nc, normalise, chosen)
    return rows


def main():
    global FINAL_SEEDS, TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(FINAL_SEEDS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--datasets", default="tahini,mango")
    a = ap.parse_args()
    FINAL_SEEDS = seed_list(a.seeds)
    TAG = ("_" + a.tag) if a.tag else ""
    want = [d for d in a.datasets.split(",") if d]
    print(f"r12: {len(FINAL_SEEDS)} seeds {FINAL_SEEDS}, tag={TAG!r}, datasets={want}", flush=True)
    t0 = time.time()
    rows = []
    if "tahini" in want:
        rows += run("tahini", tahini_split, TAHINI, "kan_nc", True)
    if "mango" in want:
        rows += run("mango", mango_split, MANGO, "kan_eq_nc", False)
    df = pd.DataFrame(rows)
    save(df, f"r12_extraction_protocol_raw{TAG}.csv")
    summ = (df.groupby(["dataset", "stage", "protocol"])
              .agg(n=("r2_symbolic", "size"),
                   spline_mean=("r2_spline", "mean"),
                   symbolic_mean=("r2_symbolic", "mean"), symbolic_sd=("r2_symbolic", "std"),
                   symbolic_min=("r2_symbolic", "min"), symbolic_max=("r2_symbolic", "max"),
                   fidelity_mean=("r2_fidelity", "mean"),
                   terms_median=("n_terms", "median"), vars_median=("n_vars", "median"))
              .reset_index())
    save(summ, f"r12_extraction_protocol_summary{TAG}.csv")
    print("\n", summ.to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
