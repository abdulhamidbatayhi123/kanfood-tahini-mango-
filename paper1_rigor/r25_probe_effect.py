"""Why two of our own runs of the corrected extraction disagree by a whole unit of R2.

`r12` and `r03` both run the corrected protocol (`full_batch_symbolic=True, refit_steps=200`) on
the tahini training partition at the same ten seeds, through the same `equation.extract`. Their
**published** protocol agrees to four decimal places on all ten seeds. Their **corrected** protocol
does not: at seed 777 `r12` reports an unusable equation (R2 = -0.0385) and `r03` a good one
(0.9579), and five other seeds differ in the third decimal.

The only difference between the two callers is that `r03` passes `want_sensitivity=True`, which
runs the intrinsic-importance probe on a **deep copy** of the network before pruning. That probe
cannot corrupt the model -- the deep copy exists precisely because an earlier version did corrupt
it -- but it runs forward passes, and anything that advances the global random state before the
pruning refit and the post-symbolic refit changes both of those stochastic steps. The published
protocol has neither refit, which is exactly why it is unaffected.

This script tests that explanation instead of assuming it: one training partition, one seed, the
corrected protocol run with the probe on and off, twice each. If the explanation is right, the two
settings reproduce the two published numbers and each setting repeats itself.

Output: `r25_probe_effect.csv`.
"""
import sys

import numpy as np
import pandas as pd
import torch

from paper1_rigor.common import TAHINI, tahini_split, project, save
from paper1_rigor.equation import extract

sys.stdout.reconfigure(encoding="utf-8")

SEEDS = [777, 13]          # the seed the two runs disagree on most, and one that agrees closely
REPEATS = 2
PROTOCOL = dict(full_batch_symbolic=True, refit_steps=200)
# The two published values this is trying to reproduce, from the CSVs (not from a log).
REFERENCE = {(777, False): -0.0385, (777, True): 0.9579,
             (13, False): 0.9838, (13, True): 0.9426}


def main():
    ds, tr, te = tahini_split()
    Z_tr, Z_te, *_ = project(ds, tr, te, TAHINI["preprocess"], TAHINI["kan_nc"])
    rows = []
    for seed in SEEDS:
        for probe in (False, True):
            for rep in range(REPEATS):
                torch.manual_seed(seed)
                np.random.seed(seed)
                r = extract(Z_tr, ds.y[tr], Z_te, ds.y[te, 0], normalise=True, seed=seed,
                            want_sensitivity=probe, **PROTOCOL)
                rows.append(dict(seed=seed, want_sensitivity=probe, repeat=rep,
                                 r2_spline=r["r2_spline"], r2_symbolic=r["r2_symbolic"],
                                 r2_fidelity=r["r2_fidelity"], n_terms=r["n_terms"],
                                 reference=REFERENCE.get((seed, probe), np.nan)))
                print(f"  seed {seed}  probe={str(probe):5s}  rep {rep}  "
                      f"spline {r['r2_spline']:.4f}  symbolic {r['r2_symbolic']:.4f}  "
                      f"fidelity {r['r2_fidelity']:.4f}   (published {REFERENCE.get((seed, probe))})",
                      flush=True)
    df = pd.DataFrame(rows)
    save(df, "r25_probe_effect.csv")
    print()
    print(df.groupby(["seed", "want_sensitivity"])["r2_symbolic"]
            .agg(["mean", "min", "max"]).round(4).to_string())


if __name__ == "__main__":
    main()
