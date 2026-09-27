"""Is the symbolic conversion reproducible at a fixed seed, and does the exact route share the problem?

`r25` established the fact: on tahini the trained spline network is bit-identical across repeated
runs at a fixed seed (external R2 0.959643 at seed 777, 0.982985 at seed 13, every time), while the
equation extracted from it is not, and at seed 777 the extracted equation ranges from -0.75 to
+0.95 across six runs of the same nominal configuration. This script asks the two questions that
turn that fact into something a reader can act on.

**Where does the non-determinism enter?** Section 2.5 already names the candidate: the symbolic
conversion chooses among candidate elementary functions by comparing fits, and that choice is
discrete. Two near-tied candidates separated by 1e-12 can be ordered either way by a threaded
floating-point reduction, and a different function on one edge changes everything downstream. If
that is the mechanism, forcing a single compute thread should make the conversion reproducible.
Both arms seed torch AND numpy identically before every run, so anything that survives is not a
seeding omission.

**Does the exact route share it?** The paper's recommendation is an already-elementary basis, for
which printing is a change of notation rather than a fit. Its training is still stochastic, so its
accuracy will vary run to run like any fitted model -- but it has no conversion step to be unstable
in, and its printed equation cannot disagree with its network. The control arm fits the
inner-CV-selected Chebyshev basis under the identical repeat protocol, so that the two routes are
compared under the same perturbation rather than by assertion.

Output: `r26_conversion_determinism.csv`, one row per run.
"""
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from paper1_rigor import basis_kan as BK
from paper1_rigor.common import TAHINI, ckpt, tahini_split, project, save
from paper1_rigor.equation import extract

sys.stdout.reconfigure(encoding="utf-8")

SEEDS = [777, 13]
REPEATS = 3
THREADS = [1, 4]
CONVERSION = dict(full_batch_symbolic=True, refit_steps=200)
# The arm inner cross-validation selected in `r19` on the tahini latent scores.
EXACT = dict(basis="cheby", degree=4, lamb=1e-4, prune=0.05)


def exact_route(A_tr, y_tr, A_te, y_te1, seed):
    """The inner-CV-selected elementary-basis arm, scored the way `r19.fit_score` scores it."""
    sx, sy = MinMaxScaler((-1, 1)).fit(A_tr), StandardScaler().fit(y_tr)
    Xtr, Xte = sx.transform(A_tr), np.clip(sx.transform(A_te), -1, 1)
    m = BK.TwoLayerKAN(A_tr.shape[1], basis=EXACT["basis"], degree=EXACT["degree"],
                       n_out=y_tr.shape[1], hidden=3)
    BK.fit(m, Xtr, sy.transform(y_tr), lamb=EXACT["lamb"], prune_frac=EXACT["prune"], seed=seed)
    pred_scaled = BK.predict(m, Xte)
    from kanfood.metrics import normalize_to_100
    pred = normalize_to_100(sy.inverse_transform(pred_scaled))[:, 0]
    printed = BK.equation(m, var_names=[f"x_{i+1}" for i in range(A_tr.shape[1])])
    try:
        ev = BK.evaluate(printed, Xte, var_names=[f"x_{i+1}" for i in range(A_tr.shape[1])])
        fid = float(r2_score(pred_scaled[:, 0], ev))
    except Exception:
        fid = float("nan")
    return float(r2_score(y_te1, pred)), fid, BK.n_terms(printed)


def main():
    ds, tr, te = tahini_split()
    Z_tr, Z_te, *_ = project(ds, tr, te, TAHINI["preprocess"], TAHINI["kan_nc"])
    y_te1 = ds.y[te, 0]
    rows = []
    # Seed outermost, so that both thread settings for one seed complete before the next seed
    # starts: a run stopped early still carries the whole comparison for at least one seed. And a
    # checkpoint after every single run, because an earlier session lost hours of completed work to
    # a script that wrote its results only at the end.
    for seed in SEEDS:
        for threads in THREADS:
            torch.set_num_threads(threads)
            print(f"\n=== seed {seed}, torch threads = {threads} ===", flush=True)
            for rep in range(REPEATS):
                torch.manual_seed(seed); np.random.seed(seed)
                r = extract(Z_tr, ds.y[tr], Z_te, y_te1, normalise=True, seed=seed, **CONVERSION)
                rows.append(dict(route="B-spline + corrected conversion", threads=threads,
                                 seed=seed, repeat=rep, R2_network=r["r2_spline"],
                                 R2_equation=r["r2_symbolic"], fidelity=r["r2_fidelity"],
                                 n_terms=r["n_terms"]))
                print(f"  converted  seed {seed:<4} rep {rep}  network {r['r2_spline']:.6f}  "
                      f"equation {r['r2_symbolic']:9.4f}  fidelity {r['r2_fidelity']:9.4f}",
                      flush=True)
                ckpt(rows, "r26_conversion_determinism_partial.csv")

                torch.manual_seed(seed); np.random.seed(seed)
                a, fid, nt = exact_route(Z_tr, ds.y[tr], Z_te, y_te1, seed)
                rows.append(dict(route="elementary basis (cheby)", threads=threads, seed=seed,
                                 repeat=rep, R2_network=a, R2_equation=a, fidelity=fid,
                                 n_terms=nt))
                print(f"  exact      seed {seed:<4} rep {rep}  network {a:.6f}  "
                      f"equation {a:9.4f}  fidelity {fid:9.6f}", flush=True)
                ckpt(rows, "r26_conversion_determinism_partial.csv")
    df = pd.DataFrame(rows)
    save(df, "r26_conversion_determinism.csv")
    print()
    print(df.groupby(["route", "threads", "seed"])["R2_equation"]
            .agg(["min", "max", lambda v: v.max() - v.min()])
            .rename(columns={"<lambda_0>": "range"}).round(6).to_string())


if __name__ == "__main__":
    main()
