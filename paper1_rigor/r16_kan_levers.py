"""How good is the KAN when its own recommended machinery is actually switched on?

The comparison the manuscript rests on has so far been run against a KAN that uses none of the
mechanisms pykan itself recommends. Four were verified present in the installed pykan 0.2.8 and
never used:

  1. GRID REFINEMENT. `KAN.refine(new_grid)` exists and is never called. pykan's guidance is to
     train at a coarse grid and refine upward, and refining gradually also acts as a form of path
     regularisation. The published pipeline trains once at grid 5. This also means the ablation
     (`r02`) may have tested the wrong thing when it initialised at grid 20 from scratch and
     reported that a fine grid is catastrophic (tahini R2 0.716); the canonical way to reach a
     fine grid is to refine up to it, and that arm is run here.
  2. OPTIMISER. pykan's default is LBFGS at lr 1.0; the published pipeline overrides it with Adam
     at 0.005. Never compared on these data.
  3. MULTIPLICATION NODES. `kan.KAN` IS `MultKAN`, so a hidden layer may be declared as
     [n_add, n_mult]. The published pipeline always builds pure-addition layers. Multiplication
     nodes are the KAN 2.0 feature intended specifically to make symbolic discovery compact -- a
     product needs one node instead of an approximation -- which is exactly this paper's use.
  4. SYMBOLIC SNAPPING. `auto_symbolic(weight_simple=0.8, r2_threshold=0.0)` has never been
     tuned. The default weights simplicity 0.8 against fit, and the symbolic step is precisely
     where the equation's accuracy is lost, so this is a direct lever on the central claim. It
     is swept and reported as an accuracy-against-length frontier rather than as a single tuned
     value, because the trade-off is the honest object here.

Search protocol
---------------
A greedy staged search, not a full grid, so the number of fits stays inspectable: topology first,
then the grid/refinement schedule at the winning topology, then the optimiser. Every stage is
scored by grouped inner cross-validation INSIDE the training partition, on the closed-form
equation's R2, because that is the quantity the paper reports. The external test set is used once
at the end, at the configuration the inner CV selected, alongside the published configuration so
that the change is visible rather than silently substituted.
"""
import time
import warnings

import numpy as np
import pandas as pd

from kanfood.split import stratified_group_kfold
from paper1_rigor.common import (ckpt, TAHINI, MANGO, SEED, tahini_split, mango_split, project, save)
from paper1_rigor.equation import extract

warnings.filterwarnings("ignore")

INNER_FOLDS = 3
INNER_SEEDS = [42, 7]
FINAL_SEEDS = [42, 7, 2024, 100, 13, 999, 123, 50, 777, 888]

TOPOLOGIES = [("(3,) published", (3,)),
              ("() additive", ()),
              ("(5,)", (5,)),
              ("[2 add, 1 mult]", ([2, 1],)),
              ("[3 add, 1 mult]", ([3, 1],)),
              ("[2 add, 2 mult]", ([2, 2],))]

GRIDS = [("G5 fixed (published)", dict(grid=5)),
         ("G10 fixed", dict(grid=10)),
         ("G20 fixed", dict(grid=20)),
         ("refine 5->10", dict(grid=5, refine_grids=(10,))),
         ("refine 5->10->20", dict(grid=5, refine_grids=(10, 20)))]

OPTS = [("Adam lr 0.005 (published)", dict(opt="Adam", lr=0.005, steps=250, prune_steps=60)),
        ("LBFGS lr 1.0 (pykan default)", dict(opt="LBFGS", lr=1.0, steps=40, prune_steps=15))]

SYMBOLIC = [(ws, th) for ws in (0.0, 0.2, 0.5, 0.8, 1.0) for th in (0.0, 0.9, 0.99)]


def inner_score(ds, tr, cfg, nc, normalise, kw, seeds=INNER_SEEDS):
    """Mean closed-form R2 over grouped inner folds; the spline R2 is carried alongside."""
    folds = stratified_group_kfold(ds.groups[tr], ds.tagsis[tr], n_splits=INNER_FOLDS, seed=SEED)
    sym, spl, terms = [], [], []
    for a, b in folds:
        Z_a, Z_b, *_ = project(ds, tr[a], tr[b], cfg["preprocess"], nc)
        for s in seeds:
            r = extract(Z_a, ds.y[tr[a]], Z_b, ds.y[tr[b], 0], normalise=normalise, seed=s, **kw)
            r.pop("model", None)
            sym.append(r["r2_symbolic"])
            spl.append(r["r2_spline"])
            terms.append(r["n_terms"])
    return (float(np.nanmean(sym)), float(np.nanstd(sym)), float(np.nanmean(spl)),
            float(np.nanmedian(terms)))


def stage(label, ds, tr, cfg, nc, normalise, name, options, base, rows):
    print(f"  -- {name} --", flush=True)
    best, best_kw, best_score = None, None, -np.inf
    for tag, kw in options:
        merged = dict(base)
        merged.update(kw if isinstance(kw, dict) else {"width": kw})
        t0 = time.time()
        sym, sd, spl, tm = inner_score(ds, tr, cfg, nc, normalise, merged)
        rows.append(dict(dataset=label, stage=name, option=tag, inner_symbolic=sym,
                         inner_symbolic_sd=sd, inner_spline=spl, terms_median=tm,
                         seconds=round(time.time() - t0)))
        print(f"    {tag:30s} equation R2 {sym:.4f} +- {sd:.4f}   spline {spl:.4f}   "
              f"{tm:.0f} terms   ({time.time() - t0:.0f}s)", flush=True)
        if sym > best_score:
            best, best_kw, best_score = tag, merged, sym
    print(f"    -> selected {best} ({best_score:.4f})", flush=True)
    return best_kw


def run(label, splitter, cfg, nc_key, normalise):
    ds, tr, te = splitter()
    nc = cfg[nc_key]
    rows = []
    print(f"\n=== {label}: KAN lever search (nc={nc}) ===", flush=True)

    base = dict(steps=250, prune_steps=60, refit_steps=200, full_batch_symbolic=True)
    base = stage(label, ds, tr, cfg, nc, normalise, "topology",
                 [(t, dict(width=w)) for t, w in TOPOLOGIES], base, rows)
    base = stage(label, ds, tr, cfg, nc, normalise, "grid schedule", GRIDS, base, rows)
    base = stage(label, ds, tr, cfg, nc, normalise, "optimiser", OPTS, base, rows)

    print("  -- symbolic snapping frontier --", flush=True)
    front = []
    for ws, th in SYMBOLIC:
        kw = dict(base); kw.update(weight_simple=ws, r2_threshold=th)
        t0 = time.time()
        sym, sd, spl, tm = inner_score(ds, tr, cfg, nc, normalise, kw, seeds=[SEED])
        rows.append(dict(dataset=label, stage="symbolic", option=f"ws={ws} r2th={th}",
                         inner_symbolic=sym, inner_symbolic_sd=sd, inner_spline=spl,
                         terms_median=tm, seconds=round(time.time() - t0)))
        front.append((sym, tm, ws, th))
        print(f"    weight_simple={ws:<4} r2_threshold={th:<5} equation R2 {sym:.4f}  "
              f"{tm:.0f} terms   ({time.time() - t0:.0f}s)", flush=True)
    best_sym = max(front)
    base.update(weight_simple=best_sym[2], r2_threshold=best_sym[3])
    print(f"    -> selected weight_simple={best_sym[2]} r2_threshold={best_sym[3]}", flush=True)

    print("  -- external test: selected configuration vs published --", flush=True)
    Z_tr, Z_te, *_ = project(ds, tr, te, cfg["preprocess"], nc)
    ext = []
    arms = [("selected", base),
            ("published", dict(width=(3,), grid=5, steps=250, prune_steps=60, legacy=True))]
    for tag, kw in arms:
        for s in FINAL_SEEDS:
            r = extract(Z_tr, ds.y[tr], Z_te, ds.y[te, 0], normalise=normalise, seed=s, **kw)
            r.pop("model", None)
            ext.append(dict(dataset=label, config=tag, seed=s, n_params=r["n_params"],
                            r2_spline=r["r2_spline"], r2_symbolic=r["r2_symbolic"],
                            r2_fidelity=r["r2_fidelity"], n_terms=r["n_terms"],
                            n_vars=r["n_vars"], equation=r.get("equation", "")))
            print(f"    [{tag:9s}] seed {s:<5} spline {r['r2_spline']:.4f}  "
                  f"equation {r['r2_symbolic']:.4f}  {r['n_terms']} terms", flush=True)
    return rows, ext, base


def main():
    t0 = time.time()
    rows, ext, cfg_t = run("tahini", tahini_split, TAHINI, "kan_nc", True)
    ckpt(rows, "r16_lever_search_tahini.csv")
    ckpt(ext, "r16_lever_external_tahini.csv")
    r2, e2, cfg_m = run("mango", mango_split, MANGO, "kan_eq_nc", False)
    save(pd.DataFrame(rows + r2), "r16_lever_search.csv")
    save(pd.DataFrame(ext + e2), "r16_lever_external.csv")
    chosen = pd.DataFrame([
        dict(dataset="tahini", **{k: str(v) for k, v in cfg_t.items()}),
        dict(dataset="mango", **{k: str(v) for k, v in cfg_m.items()})])
    save(chosen, "r16_selected_config.csv")
    summ = (pd.DataFrame(ext + e2).groupby(["dataset", "config"])
            .agg(n=("seed", "size"), params=("n_params", "median"),
                 spline_mean=("r2_spline", "mean"),
                 eq_mean=("r2_symbolic", "mean"), eq_sd=("r2_symbolic", "std"),
                 eq_min=("r2_symbolic", "min"), eq_max=("r2_symbolic", "max"),
                 fidelity=("r2_fidelity", "mean"), terms=("n_terms", "median"))
            .reset_index())
    save(summ, "r16_lever_external_summary.csv")
    print("\n", summ.to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
