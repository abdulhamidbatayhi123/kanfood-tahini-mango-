"""Is the KAN losing because of its architecture, or because of how it was configured and trained?

Why this exists
---------------
`r10` equalised the parameter count, the size of the hyperparameter search and the convergence
criterion, and found the matched perceptron at least as good on all three comparisons. But it also
produced a result that undercuts its own conclusion: the configuration the KAN's inner
cross-validation selected reached external R2 0.949 on tahini, while the *published pinned*
configuration reaches 0.987 on the same split. If the harness itself disadvantages the KAN, then
`r10` measures the harness, not the architecture -- exactly the mistake `r10` was written to
correct in `r01`, one level up.

Two things are confounded in that number and they are separated here:

  CONFIGURATION -- the published pinned configuration versus the one the inner CV selected.
  HARNESS       -- the published recipe (a fixed step count, fitted on the whole training
                   partition) versus the convergence recipe (early stopping on an inner
                   validation split carved out of the training partition, so the final model
                   sees only 75 % of the training rows).

The full 2 x 2 is run for the KAN, and the perceptron is run under both harnesses at its own
selected configuration, so that every cell of the comparison is populated and the conclusion can
name which factor is responsible. Five seeds throughout; the external test set is scored once per
run and never used for any choice.
"""
import json
import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from kan import KAN
from kanfood.metrics import normalize_to_100
from paper1_rigor.common import (seed_list, TAHINI, MANGO, SEED, tahini_split, mango_split, project,
                                 silent, save)
from paper1_rigor.r10_fair_capacity import (build_mlp, match_width, kan_params, inner_split,
                                            train_mlp_to_convergence, train_kan_to_convergence)

SEEDS = [42, 7, 2024, 100, 13]
# Raised on the command line for the *Foods* revision (reviewer 1 point 7, reviewer 2 point 8).
TAG = ""

# The configurations r10 selected by inner CV, and the published pinned one, per dataset.
PUBLISHED_KAN = dict(grid=5, k=3, lamb=1e-3)
SELECTED_KAN = {"tahini": dict(grid=3, k=4, lamb=1e-3),
                "mango (accuracy KAN)": dict(grid=3, k=3, lamb=1e-3),
                "mango (compact KAN)": dict(grid=5, k=3, lamb=1e-2)}
SELECTED_MLP = {"tahini": dict(lr=3e-3, wd=1e-4, dropout=0.1),
                "mango (accuracy KAN)": dict(lr=1e-3, wd=1e-3, dropout=0.1),
                "mango (compact KAN)": dict(lr=1e-3, wd=1e-3, dropout=0.1)}
# The published recipe's step counts (kanfood.models): 300 Adam steps for the accuracy
# configuration, 200 epochs at batch 64 for the perceptron.
PUBLISHED_KAN_STEPS = 300
PUBLISHED_MLP_EPOCHS = 200


def _post(p, normalise):
    p = np.asarray(p).reshape(len(p), -1)
    return (normalize_to_100(p) if normalise else p)[:, 0]


def kan_published_recipe(Z_tr, y_tr, width, cfg, seed, n_out, steps=PUBLISHED_KAN_STEPS):
    """The published protocol: a fixed number of Adam steps on the WHOLE training partition."""
    torch.manual_seed(seed)
    sx, sy = MinMaxScaler((-1, 1)).fit(Z_tr), StandardScaler().fit(y_tr)
    t = lambda A: torch.tensor(np.asarray(A), dtype=torch.float32)
    Xs, ys = sx.transform(Z_tr), sy.transform(y_tr)
    dsd = {"train_input": t(Xs), "train_label": t(ys), "test_input": t(Xs), "test_label": t(ys)}
    m = silent(lambda: KAN(width=[Z_tr.shape[1]] + list(width) + [n_out], grid=cfg["grid"],
                           k=cfg["k"], seed=seed, device="cpu", auto_save=False,
                           grid_range=[-1, 1]))
    silent(lambda: m.fit(dsd, opt="Adam", steps=steps, lr=0.005, batch=128, lamb=cfg["lamb"]))

    def pred(Z):
        with torch.no_grad():
            return sy.inverse_transform(m(t(sx.transform(Z))).numpy())
    return pred, steps


def mlp_published_recipe(Z_tr, y_tr, hidden, cfg, seed, n_out, epochs=PUBLISHED_MLP_EPOCHS,
                         batch=64):
    """The published protocol: a fixed number of epochs on the WHOLE training partition."""
    torch.manual_seed(seed)
    sx, sy = StandardScaler().fit(Z_tr), StandardScaler().fit(y_tr)
    Xa = torch.tensor(sx.transform(Z_tr), dtype=torch.float32)
    ya = torch.tensor(sy.transform(y_tr), dtype=torch.float32)
    net = build_mlp(Z_tr.shape[1], hidden, n_out, cfg["dropout"])
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    lossf = torch.nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    updates = 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xa), generator=g)
        for i in range(0, len(Xa), batch):
            idx = perm[i:i + batch]
            if idx.numel() < 2:
                continue
            opt.zero_grad()
            lossf(net(Xa[idx]), ya[idx]).backward()
            opt.step()
            updates += 1
    net.eval()

    def pred(Z):
        with torch.no_grad():
            return sy.inverse_transform(
                net(torch.tensor(sx.transform(Z), dtype=torch.float32)).numpy())
    return pred, updates


def run(label, ds, tr, te, cfg_ds, nc, width, normalise):
    n_out = ds.y.shape[1]
    budget = kan_params(nc, width, n_out, grid=5, k=3)
    hidden, p_mlp = match_width(budget, nc, n_out)
    Z_tr, Z_te, *_ = project(ds, tr, te, cfg_ds["preprocess"], nc)
    y_tr, y_te, groups_tr = ds.y[tr], ds.y[te, 0], ds.groups[tr]
    rows = []
    print(f"\n=== {label} (KAN {budget} par, matched MLP {hidden} -> {p_mlp} par) ===", flush=True)

    arms = [
        ("KAN", "published config", "published recipe", PUBLISHED_KAN),
        ("KAN", "published config", "convergence harness", PUBLISHED_KAN),
        ("KAN", "inner-CV selected", "published recipe", SELECTED_KAN[label]),
        ("KAN", "inner-CV selected", "convergence harness", SELECTED_KAN[label]),
        ("MLP", "inner-CV selected", "published recipe", SELECTED_MLP[label]),
        ("MLP", "inner-CV selected", "convergence harness", SELECTED_MLP[label]),
    ]
    for arch, which_cfg, which_harness, cfg in arms:
        for seed in SEEDS:
            t0 = time.time()
            if arch == "KAN":
                if which_harness == "published recipe":
                    pr, upd = kan_published_recipe(Z_tr, y_tr, width, cfg, seed, n_out)
                else:
                    pr, upd, _ = train_kan_to_convergence(Z_tr, y_tr, groups_tr, width, cfg,
                                                          seed, n_out)
            else:
                if which_harness == "published recipe":
                    pr, upd = mlp_published_recipe(Z_tr, y_tr, hidden, cfg, seed, n_out)
                else:
                    pr, upd, _ = train_mlp_to_convergence(Z_tr, y_tr, groups_tr, hidden, cfg,
                                                          seed, n_out)
            ext = float(r2_score(y_te, _post(pr(Z_te), normalise)))
            rows.append(dict(dataset=label, architecture=arch, configuration=which_cfg,
                             harness=which_harness, config=json.dumps(cfg), seed=seed,
                             n_params=budget if arch == "KAN" else p_mlp,
                             updates_used=upd, external_R2=ext,
                             seconds=round(time.time() - t0, 1)))
        v = [r["external_R2"] for r in rows[-len(SEEDS):]]
        print(f"  {arch:3s} {which_cfg:18s} {which_harness:20s} "
              f"R2 {np.mean(v):.4f} +- {np.std(v, ddof=1):.4f}   "
              f"median {np.median(v):.4f}   min {np.min(v):.4f}", flush=True)
    return rows


def main():
    global SEEDS, TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--datasets", default="tahini,mango")
    a = ap.parse_args()
    SEEDS = seed_list(a.seeds)
    TAG = ("_" + a.tag) if a.tag else ""
    want = [d for d in a.datasets.split(",") if d]
    print(f"r10b: {len(SEEDS)} seeds {SEEDS}, tag={TAG!r}, datasets={want}", flush=True)
    t0 = time.time()
    rows = []
    if "tahini" in want:
        ds, tr, te = tahini_split()
        rows += run("tahini", ds, tr, te, TAHINI, TAHINI["kan_nc"], TAHINI["kan_width"], True)
    if "mango" in want:
        dm, mtr, mte = mango_split()
        rows += run("mango (accuracy KAN)", dm, mtr, mte, MANGO, MANGO["kan_nc"],
                    MANGO["kan_width"], False)
        rows += run("mango (compact KAN)", dm, mtr, mte, MANGO, MANGO["kan_eq_nc"],
                    MANGO["kan_eq_width"], False)
    df = pd.DataFrame(rows)
    save(df, f"r10b_config_vs_harness_raw{TAG}.csv")
    summ = (df.groupby(["dataset", "architecture", "configuration", "harness"])
              .agg(n=("seed", "size"), params=("n_params", "first"),
                   updates_median=("updates_used", "median"),
                   ext_R2_mean=("external_R2", "mean"), ext_R2_sd=("external_R2", "std"),
                   ext_R2_median=("external_R2", "median"), ext_R2_min=("external_R2", "min"))
              .reset_index())
    save(summ, f"r10b_config_vs_harness_summary{TAG}.csv")
    print("\n", summ.to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
