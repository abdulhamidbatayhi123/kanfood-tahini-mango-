"""Does the machine account for the r10 / r10b discrepancy? Measured at the configuration where
that discrepancy actually happened.

Why this exists
---------------
`r20` asked how reproducible a run of this pipeline is when everything the user controls is held
fixed, and answered it at the *published* KAN configuration (grid 5, k 3): the published training
recipe reproduces to about 3e-8, and the convergence harness spreads by at most 1.6e-3 across
thread counts. Section 3.9 then used that to explain why `r10` and `r10b` disagreed by 0.020 on
the tahini KAN.

Two things are wrong with that explanation, and both are visible in the CSVs already in this
repository.

1. **The magnitude does not fit.** 1.6e-3 is an order of magnitude too small to account for 0.020.
2. **The stated mechanism is contradicted.** Section 3.9 says early stopping moved the stopping
   point. Comparing `r10_fair_capacity_raw.csv` with `r10b_config_vs_harness_raw.csv` seed by
   seed, the number of updates used is *identical* in all five seeds -- 1000, 2000, 3200, 1100,
   2400 -- while the external R2 differs by up to 0.039 (seed 100: 0.8707 against 0.9099). The
   runs stopped at the same place and arrived at different models.

The reason `r20` could not see this is that it tested a different configuration. `r10` and `r10b`
diverged at the inner-CV-selected configuration for tahini, grid 3 / k 4 (`SELECTED_KAN`), not at
grid 5 / k 3. This script repeats `r20`'s design at that configuration, so that the discrepancy is
measured where it occurred rather than extrapolated from somewhere else.

What is measured
----------------
Tahini, KAN, at SELECTED_KAN["tahini"], under both training rules:

  * repeats at a fixed seed inside one process   -> pure floating-point non-determinism,
  * across thread counts at each seed            -> the effect that separated r10 from r10b,
  * the five r10/r10b seeds                      -> so the per-seed values are directly comparable
                                                    with both of those runs.

The worker, the training functions and the split all come from `r20` and `r10`/`r10b` unchanged;
only the configuration passed to them differs. `r22_vs_r10.csv` puts this run's per-seed external
R2 next to the two runs it is trying to explain.
"""
import time

import numpy as np
import pandas as pd

from paper1_rigor.common import save, ckpt, OUT
from paper1_rigor.r10b_config_vs_harness import SELECTED_KAN
from paper1_rigor.r20_reproducibility import run_worker, THREAD_COUNTS, SEEDS as R20_SEEDS

# the seeds r10 and r10b both used, so the comparison is like for like
SEEDS = [42, 7, 2024, 100, 13]
REPEATS = 3
CFG = SELECTED_KAN["tahini"]


def spread(g):
    return pd.Series({"n": len(g), "mean": g.R2.mean(), "sd": g.R2.std(),
                      "min": g.R2.min(), "max": g.R2.max(),
                      "range": g.R2.max() - g.R2.min(),
                      "updates_min": g.updates.min(), "updates_max": g.updates.max()})


def main():
    t0 = time.time()
    print(f"=== tahini KAN at the inner-CV-selected configuration {CFG} ===", flush=True)
    print("    (r20 measured grid 5 / k 3; r10 and r10b diverged at this one)", flush=True)
    rows = []
    for harness in ("convergence", "published"):
        for threads in THREAD_COUNTS:
            for seed in SEEDS:
                reps = REPEATS if seed == SEEDS[0] else 1
                for rep in range(reps):
                    r = run_worker(threads, "KAN", harness, seed, kan_cfg=CFG)
                    rows.append(dict(architecture="KAN", harness=harness, threads=threads,
                                     seed=seed, repeat=rep + 1, **r))
                    print(f"  {harness:12s} threads={threads} seed={seed:<5} rep {rep + 1}  "
                          f"R2 {r['R2']:.4f}  updates {r['updates']}", flush=True)
                    ckpt(rows, "r22_repro_selected_config_partial.csv")
    df = pd.DataFrame(rows)
    save(df, "r22_repro_selected_config_raw.csv")

    parts = []
    first = SEEDS[0]
    p = (df[df.seed == first].groupby(["harness", "threads"]).apply(spread, include_groups=False)
         .reset_index())
    p["source"] = "repeats at one seed, one thread count"
    parts.append(p)
    p = df.groupby(["harness", "seed"]).apply(spread, include_groups=False).reset_index()
    p["source"] = "thread count, seed fixed"
    parts.append(p)
    p = df.groupby(["harness", "threads"]).apply(spread, include_groups=False).reset_index()
    p["source"] = "seed, thread count fixed"
    parts.append(p)
    summ = pd.concat(parts, ignore_index=True)
    save(summ, "r22_repro_selected_config_summary.csv")
    print("\n", summ.to_string(index=False))

    # ---- the point of the experiment: put this run beside the two it is explaining ----------
    a = pd.read_csv(OUT / "r10_fair_capacity_raw.csv")
    b = pd.read_csv(OUT / "r10b_config_vs_harness_raw.csv")
    a = a[(a.dataset.astype(str).str.contains("tahini")) & (a.architecture == "KAN")]
    b = b[(b.dataset.astype(str).str.contains("tahini")) & (b.architecture == "KAN")
          & (b.configuration == "inner-CV selected") & (b.harness == "convergence harness")]
    here = df[df.harness == "convergence"]
    cmp_rows = []
    for s in SEEDS:
        h = here[here.seed == s]
        ra = a[a.seed == s]["external_R2"]
        rb = b[b.seed == s]["external_R2"]
        cmp_rows.append(dict(
            seed=s,
            r10_external_R2=float(ra.iloc[0]) if len(ra) else np.nan,
            r10b_external_R2=float(rb.iloc[0]) if len(rb) else np.nan,
            r22_min=float(h.R2.min()), r22_max=float(h.R2.max()),
            r22_range=float(h.R2.max() - h.R2.min()),
            r10_vs_r10b=abs(float(ra.iloc[0]) - float(rb.iloc[0])) if len(ra) and len(rb) else np.nan,
            updates=";".join(str(u) for u in sorted(h.updates.unique()))))
    cmp = pd.DataFrame(cmp_rows)
    save(cmp, "r22_vs_r10.csv")
    print("\n--- this run against the two it is explaining ---")
    print(cmp.to_string(index=False))
    print(f"\n  max |r10 - r10b| per seed : {cmp.r10_vs_r10b.max():.4f}")
    print(f"  max within-r22 range      : {cmp.r22_range.max():.4f}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
