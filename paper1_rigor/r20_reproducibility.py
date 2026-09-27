"""How reproducible is a result from this pipeline, holding everything the user controls fixed?

Why this exists
---------------
`r10` and `r10b` ran the same architecture, at the same configuration, under the same harness,
with the same five seeds, on the same split -- and did not agree: the tahini KAN averaged 0.949 in
one and 0.969 in the other. Nothing a user controls differed. The only difference was the number
of BLAS/torch threads the job was given, which changes the order of floating-point reductions by
amounts far below any tolerance one would normally care about.

That matters here for a specific reason. The convergence harness stops training when a validation
score stops improving, so an arbitrarily small numerical difference can move the stopping point by
a whole chunk of updates, and the resulting models are then genuinely different. A property of the
*training rule* is being converted into apparent variability of the *architecture*, and a paper
that reports one run without saying so is reporting a number it cannot reproduce.

What is measured
----------------
For each architecture and each training rule, the identical run is repeated:

  * REPEATS at a fixed seed inside one process   -> pure floating-point non-determinism,
  * across THREAD_COUNTS                          -> the effect that separated r10 from r10b,
  * across seeds                                  -> the variability normally reported.

The three are reported side by side so the manuscript can state which part of the spread is the
model and which part is the machine. Both training rules are run for both architectures, because
the claim under test is that the convergence rule amplifies this for the KAN specifically.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
THREAD_COUNTS = [2, 4]
SEEDS = [42, 7, 2024]
REPEATS = 3

WORKER = r'''
import os, sys, json
sys.path.insert(0, os.getcwd())
th = sys.argv[1]
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = th
import warnings; warnings.filterwarnings("ignore")
import torch; torch.set_num_threads(int(th))
import numpy as np
from sklearn.metrics import r2_score
from paper1_rigor.common import TAHINI, tahini_split, project
from paper1_rigor.r10_fair_capacity import (train_kan_to_convergence, train_mlp_to_convergence,
                                            match_width, kan_params)
from paper1_rigor.r10b_config_vs_harness import (kan_published_recipe, mlp_published_recipe,
                                                 PUBLISHED_KAN, SELECTED_MLP, _post)

arch, harness, seed = sys.argv[2], sys.argv[3], int(sys.argv[4])
# argv[5], when given, overrides the KAN configuration. Absent, PUBLISHED_KAN is used and
# the worker behaves exactly as it did for r20.
KAN_CFG = json.loads(sys.argv[5]) if len(sys.argv) > 5 else PUBLISHED_KAN
ds, tr, te = tahini_split()
nc, width = TAHINI["kan_nc"], TAHINI["kan_width"]
n_out = ds.y.shape[1]
hidden, _ = match_width(kan_params(nc, width, n_out, grid=5, k=3), nc, n_out)
Z_tr, Z_te, *_ = project(ds, tr, te, TAHINI["preprocess"], nc)
y_tr, y_te, g_tr = ds.y[tr], ds.y[te, 0], ds.groups[tr]

if arch == "KAN":
    if harness == "published":
        pr, upd = kan_published_recipe(Z_tr, y_tr, width, KAN_CFG, seed, n_out)
    else:
        pr, upd, _ = train_kan_to_convergence(Z_tr, y_tr, g_tr, width, KAN_CFG, seed, n_out)
else:
    cfg = SELECTED_MLP["tahini"]
    if harness == "published":
        pr, upd = mlp_published_recipe(Z_tr, y_tr, hidden, cfg, seed, n_out)
    else:
        pr, upd, _ = train_mlp_to_convergence(Z_tr, y_tr, g_tr, hidden, cfg, seed, n_out)
print(json.dumps({"R2": float(r2_score(y_te, _post(pr(Z_te), True))), "updates": int(upd)}))
'''


def run_worker(threads, arch, harness, seed, kan_cfg=None):
    """One run, in its own process, so the thread count really is what changes."""
    p = ROOT / "results_rigor" / "_repro_worker.py"
    p.write_text(WORKER, encoding="utf-8")
    argv = [sys.executable, str(p), str(threads), arch, harness, str(seed)]
    if kan_cfg is not None:
        import json as _json
        argv.append(_json.dumps(kan_cfg))
    out = subprocess.run(argv,
                         cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    for line in reversed(out.stdout.strip().splitlines()):
        if line.startswith("{"):
            import json
            return json.loads(line)
    raise RuntimeError(f"worker failed: {out.stdout[-400:]} {out.stderr[-400:]}")


def main():
    t0 = time.time()
    rows = []
    for arch in ("KAN", "MLP"):
        for harness in ("published", "convergence"):
            for threads in THREAD_COUNTS:
                for seed in SEEDS:
                    reps = REPEATS if (seed == SEEDS[0]) else 1
                    for rep in range(reps):
                        r = run_worker(threads, arch, harness, seed)
                        rows.append(dict(architecture=arch, harness=harness, threads=threads,
                                         seed=seed, repeat=rep + 1, **r))
                        print(f"  {arch} {harness:12s} threads={threads} seed={seed:<5} "
                              f"rep {rep+1}  R2 {r['R2']:.4f}  updates {r['updates']}", flush=True)
    df = pd.DataFrame(rows)
    from paper1_rigor.common import save
    save(df, "r20_reproducibility_raw.csv")

    def spread(g):
        return pd.Series({"n": len(g), "mean": g.R2.mean(), "sd": g.R2.std(),
                          "min": g.R2.min(), "max": g.R2.max(),
                          "range": g.R2.max() - g.R2.min(),
                          "updates_min": g.updates.min(), "updates_max": g.updates.max()})

    parts = []
    fixed = df[(df.seed == SEEDS[0])]
    parts.append(fixed.groupby(["architecture", "harness", "threads"]).apply(
        spread, include_groups=False).reset_index().assign(source="repeats at one seed, one thread count"))
    parts.append(df[df.repeat == 1].groupby(["architecture", "harness", "seed"]).apply(
        spread, include_groups=False).reset_index().assign(source="thread count, seed fixed"))
    parts.append(df[(df.repeat == 1)].groupby(["architecture", "harness", "threads"]).apply(
        spread, include_groups=False).reset_index().assign(source="seed, thread count fixed"))
    summ = pd.concat(parts, ignore_index=True)
    save(summ, "r20_reproducibility_summary.csv")
    print("\n", summ.round(4).to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
