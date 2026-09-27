"""Stability of the closed-form equation and of the explanation it implies.

A calibration equation is only useful if it is reproducible. The whole interpretability chain
(preprocessing -> PLS projection -> KAN fit -> prune -> symbolic conversion) is repeated under
four sources of variation: random seed, cross-validation fold, preprocessing choice, and -- new
here -- repeated runs at a FIXED seed, because pykan's symbolic conversion is not deterministic
across runs. Every R2 is computed on data the model never saw.

This script previously re-implemented the extraction chain and produced impossible numbers
(symbolic R2 near zero everywhere, zero retained variables in every run). Both defects are
root-caused in `paper1_rigor.equation`, which is now the single implementation and which is
checked against the published equations before use. Two rules follow from that diagnosis and are
enforced here by construction:

  * the intrinsic-importance probe is computed inside `extract()` on a deep copy, never on the
    model that is about to be pruned;
  * retained variables are read from the symbolic mask after conversion, not the spline mask.

Both the published extraction protocol (P0) and the corrected protocol (P3, full-calibration
snapping plus the post-symbolic refit) are reported, so the stability of the equation can be read
separately from the improvement in its accuracy.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from kanfood.split import stratified_group_kfold
from paper1_rigor.common import (ckpt, TAHINI, MANGO, SEED, SEEDS10, tahini_split, mango_split,
                                 project, save)
from paper1_rigor.equation import extract

PROTOCOLS = {"P0 published": dict(legacy=True),
             "P3 corrected": dict(full_batch_symbolic=True, refit_steps=200)}
REPEATS_AT_FIXED_SEED = 5


def one_run(ds, tr_idx, te_idx, preprocess, nc, seed, normalise, **kw):
    Z_tr, Z_te, *_ = project(ds, tr_idx, te_idx, preprocess, nc)
    r = extract(Z_tr, ds.y[tr_idx], Z_te, ds.y[te_idx, 0], normalise=normalise, seed=seed,
                want_sensitivity=True, **kw)
    r.pop("model", None)
    return r


def agreement(importances):
    """Mean pairwise top-3 overlap and Spearman rank correlation of importance vectors."""
    ov, sp = [], []
    for a, b in itertools.combinations([v for v in importances if v is not None], 2):
        ta, tb = set(np.argsort(a)[-3:]), set(np.argsort(b)[-3:])
        ov.append(len(ta & tb) / 3.0)
        s = spearmanr(a, b).statistic
        if np.isfinite(s):
            sp.append(s)
    return (float(np.mean(ov)) if ov else np.nan, float(np.mean(sp)) if sp else np.nan)


def run(label, ds, tr, te, preprocess, nc, alt_preprocess, normalise):
    rows, agr = [], []
    print("\n=== " + label + ": stability of the equation (nc=" + str(nc) + ", width (3,)) ===",
          flush=True)

    for pname, kw in PROTOCOLS.items():
        imps = {"seed": [], "fold": [], "preprocess": [], "repeat": []}

        print("  [" + pname + "] seeds ...", flush=True)
        for s in SEEDS10:
            r = one_run(ds, tr, te, preprocess, nc, s, normalise, **kw)
            imps["seed"].append(r["sensitivity"])
            rows.append(dict(dataset=label, protocol=pname, source="seed", variant=str(s), **{
                k: r.get(k) for k in ("r2_spline", "r2_symbolic", "r2_fidelity", "n_terms",
                                      "n_vars", "retained", "equation")}))
            print("    seed " + str(s).ljust(5) + " spline " + format(r["r2_spline"], ".4f") +
                  "  symbolic " + format(r["r2_symbolic"], ".4f") +
                  "  vars " + str(r.get("n_vars")), flush=True)

        print("  [" + pname + "] repeats at the fixed published seed ...", flush=True)
        for i in range(REPEATS_AT_FIXED_SEED):
            r = one_run(ds, tr, te, preprocess, nc, SEED, normalise, **kw)
            imps["repeat"].append(r["sensitivity"])
            rows.append(dict(dataset=label, protocol=pname, source="repeat",
                             variant="run" + str(i + 1), **{
                                 k: r.get(k) for k in ("r2_spline", "r2_symbolic", "r2_fidelity",
                                                       "n_terms", "n_vars", "retained",
                                                       "equation")}))
            print("    run " + str(i + 1) + "      symbolic " + format(r["r2_symbolic"], ".4f"),
                  flush=True)

        print("  [" + pname + "] folds ...", flush=True)
        folds = stratified_group_kfold(ds.groups[tr], ds.tagsis[tr], n_splits=5, seed=SEED)
        for i, (f_tr, f_va) in enumerate(folds, 1):
            r = one_run(ds, tr[f_tr], tr[f_va], preprocess, nc, SEED, normalise, **kw)
            imps["fold"].append(r["sensitivity"])
            rows.append(dict(dataset=label, protocol=pname, source="fold", variant="fold" + str(i),
                             **{k: r.get(k) for k in ("r2_spline", "r2_symbolic", "r2_fidelity",
                                                      "n_terms", "n_vars", "retained",
                                                      "equation")}))
            print("    fold " + str(i) + "     spline " + format(r["r2_spline"], ".4f") +
                  "  symbolic " + format(r["r2_symbolic"], ".4f"), flush=True)

        print("  [" + pname + "] preprocessing ...", flush=True)
        for p in alt_preprocess:
            r = one_run(ds, tr, te, p, nc, SEED, normalise, **kw)
            imps["preprocess"].append(r["sensitivity"])
            rows.append(dict(dataset=label, protocol=pname, source="preprocess", variant=p, **{
                k: r.get(k) for k in ("r2_spline", "r2_symbolic", "r2_fidelity", "n_terms",
                                      "n_vars", "retained", "equation")}))
            print("    " + p.ljust(9) + " spline " + format(r["r2_spline"], ".4f") +
                  "  symbolic " + format(r["r2_symbolic"], ".4f"), flush=True)

        for k in ("seed", "fold", "preprocess", "repeat"):
            ov, sp = agreement(imps[k])
            agr.append(dict(dataset=label, protocol=pname, source=k, top3_overlap=ov, spearman=sp))
            print("  agreement across " + k.ljust(11) + " top-3 overlap " + format(ov, ".2f") +
                  "   Spearman " + format(sp, ".2f"), flush=True)
    return rows, agr


def main():
    ds, tr, te = tahini_split()
    rows, agr = run("tahini", ds, tr, te, TAHINI["preprocess"], TAHINI["kan_nc"],
                    ["sg1", "snv+sg1", "msc"], normalise=True)
    ckpt(rows, "r03_stability_runs_tahini.csv")
    ckpt(agr, "r03_stability_agreement_tahini.csv")
    dm, mtr, mte = mango_split()
    r2, a2 = run("mango", dm, mtr, mte, MANGO["preprocess"], MANGO["kan_eq_nc"],
                 ["snv", "snv+sg1", "msc"], normalise=False)
    df = pd.DataFrame(rows + r2)
    save(df, "r03_stability_runs.csv")
    save(pd.DataFrame(agr + a2), "r03_stability_agreement.csv")
    summ = (df.groupby(["dataset", "protocol", "source"])
              .agg(n=("r2_symbolic", "size"),
                   spline_mean=("r2_spline", "mean"),
                   symbolic_mean=("r2_symbolic", "mean"), symbolic_sd=("r2_symbolic", "std"),
                   symbolic_min=("r2_symbolic", "min"), symbolic_max=("r2_symbolic", "max"),
                   fidelity_mean=("r2_fidelity", "mean"),
                   terms_median=("n_terms", "median"), vars_median=("n_vars", "median"))
              .reset_index())
    save(summ, "r03_stability_summary.csv")
    print("\n", summ.to_string(index=False))


if __name__ == "__main__":
    main()
