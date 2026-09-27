"""The fair equal-capacity comparison: same parameters, same convergence, same search budget.

Why this script exists
----------------------
`r01_matched_mlp.py` matched the two architectures on trainable parameters and found the MLP ahead
on both datasets. That comparison is confounded three ways, all of them favouring the MLP:

  1. Optimisation budget. `MLPModel` runs 200 epochs at batch 64; `KANModel` runs 300 Adam steps at
     batch 128. On the tahini training partition that is 3600 gradient updates against 300 (12x);
     on mango, 32,200 against 300 (107x).
  2. Training recipe. The MLP gets BatchNorm, Dropout(0.1) and weight decay. The KAN gets a single
     activation penalty and nothing else.
  3. Configuration. The KAN is pinned at the published configuration, which the ablation (r02) shows
     is NOT optimal -- on tahini, six single-factor variants beat it, by up to +0.040 R2. The MLP's
     hidden width, by contrast, was chosen to fit the budget.

None of that says anything about the architectures. This script removes all three confounds and asks
the question properly:

    at an equal trainable-parameter count, with both models trained to convergence rather than to a
    fixed step count, and with a hyperparameter search of identical size, does either architecture
    have an accuracy advantage?

Whatever the answer is, it is reportable. If the MLP still wins, the paper says so and rests its
contribution on the closed-form equation. If they tie, the accuracy story is clean again.

Protocol notes
--------------
* Data loading, preprocessing, PLS compression and the splits all come from `kanfood` -- only the
  *training harness* is new, because changing the training protocol is the entire point.
* Convergence is defined by early stopping on an inner validation split carved out of the training
  partition by group, never on the test set. The number of updates actually used is recorded so the
  budget can be inspected rather than assumed.
* Selection is by inner grouped cross-validation on the training partition only.
"""
import itertools
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from kan import KAN
from kanfood.split import stratified_group_kfold
from paper1_rigor.common import (TAHINI, MANGO, SEED, ROOT, tahini_split, mango_split, project,
                                 silent, save)

SEEDS = [42, 7, 2024, 100, 13]
MAX_UPDATES = 6000          # generous ceiling; early stopping is expected to fire long before
CHUNK = 100                 # updates between validation checks
PATIENCE = 8                # chunks without improvement before stopping
INNER_VAL_FRAC = 0.25

# Six configurations each -- the searches are the same size by construction.
KAN_GRID = [dict(grid=g, k=k, lamb=lam)
            for g, k in [(3, 3), (5, 3), (3, 4)]
            for lam in (1e-3, 1e-2)]
MLP_GRID = [dict(lr=lr, wd=wd, dropout=do)
            for lr, wd, do in [(1e-3, 1e-4, 0.1), (1e-3, 1e-4, 0.0), (3e-3, 1e-4, 0.1),
                               (1e-3, 1e-3, 0.1), (3e-3, 0.0, 0.0), (1e-3, 0.0, 0.0)]]
assert len(KAN_GRID) == len(MLP_GRID)


# --------------------------------------------------------------------------- models
def build_mlp(n_in, hidden, n_out, dropout):
    layers, prev = [], n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, n_out))
    return nn.Sequential(*layers)


def mlp_params(n_in, hidden, n_out):
    tot, prev = 0, n_in
    for h in hidden:
        tot += prev * h + h + 2 * h
        prev = h
    return tot + prev * n_out + n_out


def match_width(target, n_in, n_out):
    best = None
    for h in range(2, 4096):
        p = mlp_params(n_in, (h,), n_out)
        if best is None or abs(p - target) < abs(best[1] - target):
            best = ((h,), p)
    return best


def kan_params(n_in, width, n_out, grid, k, seed=SEED):
    m = silent(lambda: KAN(width=[n_in] + list(width) + [n_out], grid=grid, k=k, seed=seed,
                           device="cpu", auto_save=False, grid_range=[-1, 1]))
    return int(sum(p.numel() for p in m.parameters()))


# --------------------------------------------------------------- convergence harness
def inner_split(groups, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_FRAC, random_state=seed)
    return next(gss.split(np.zeros(len(groups)), groups=groups))


def train_mlp_to_convergence(Z_tr, y_tr, groups_tr, hidden, cfg, seed, n_out):
    """Mini-batch Adam with early stopping on a grouped inner validation split."""
    torch.manual_seed(seed)
    a, b = inner_split(groups_tr, seed)
    sx, sy = StandardScaler().fit(Z_tr[a]), StandardScaler().fit(y_tr[a])
    Xa = torch.tensor(sx.transform(Z_tr[a]), dtype=torch.float32)
    ya = torch.tensor(sy.transform(y_tr[a]), dtype=torch.float32)
    Xb = torch.tensor(sx.transform(Z_tr[b]), dtype=torch.float32)
    yb_true = y_tr[b, 0]

    net = build_mlp(Z_tr.shape[1], hidden, n_out, cfg["dropout"])
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    lossf = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad, updates = -np.inf, None, 0, 0
    while updates < MAX_UPDATES:
        net.train()
        for _ in range(CHUNK):
            idx = torch.randint(0, len(Xa), (min(64, len(Xa)),), generator=g)
            if idx.numel() < 2:
                continue
            opt.zero_grad()
            lossf(net(Xa[idx]), ya[idx]).backward()
            opt.step()
            updates += 1
        net.eval()
        with torch.no_grad():
            pv = sy.inverse_transform(net(Xb).numpy())[:, 0]
        score = r2_score(yb_true, pv)
        if score > best + 1e-4:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    def predict(Z):
        with torch.no_grad():
            return sy.inverse_transform(net(torch.tensor(sx.transform(Z), dtype=torch.float32)).numpy())
    return predict, updates, best


def train_kan_to_convergence(Z_tr, y_tr, groups_tr, width, cfg, seed, n_out):
    """The same early-stopping rule applied to the KAN, in chunks of `CHUNK` Adam steps."""
    torch.manual_seed(seed)
    a, b = inner_split(groups_tr, seed)
    sx = MinMaxScaler((-1, 1)).fit(Z_tr[a])
    sy = StandardScaler().fit(y_tr[a])
    Xa, ya = sx.transform(Z_tr[a]), sy.transform(y_tr[a])
    t = lambda A: torch.tensor(A, dtype=torch.float32)
    dsd = {"train_input": t(Xa), "train_label": t(ya), "test_input": t(Xa), "test_label": t(ya)}
    m = silent(lambda: KAN(width=[Z_tr.shape[1]] + list(width) + [n_out], grid=cfg["grid"],
                           k=cfg["k"], seed=seed, device="cpu", auto_save=False,
                           grid_range=[-1, 1]))
    yb_true = y_tr[b, 0]

    def pred(Z):
        with torch.no_grad():
            return sy.inverse_transform(m(t(sx.transform(Z))).numpy())

    best, best_state, bad, updates = -np.inf, None, 0, 0
    while updates < MAX_UPDATES:
        silent(lambda: m.fit(dsd, opt="Adam", steps=CHUNK, lr=0.005, batch=128, lamb=cfg["lamb"]))
        updates += CHUNK
        score = r2_score(yb_true, pred(Z_tr[b])[:, 0])
        if score > best + 1e-4:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        m.load_state_dict(best_state)
    return pred, updates, best


# --------------------------------------------------------------------------- driver
def evaluate(label, ds, tr, te, cfg_ds, nc, width):
    n_out = ds.y.shape[1]
    budget = kan_params(nc, width, n_out, grid=5, k=3)
    hidden, p_mlp = match_width(budget, nc, n_out)
    Z_tr, Z_te, _, _, _ = project(ds, tr, te, cfg_ds["preprocess"], nc)
    y_tr, y_te, groups_tr = ds.y[tr], ds.y[te, 0], ds.groups[tr]
    folds = stratified_group_kfold(groups_tr, ds.tagsis[tr], n_splits=3, seed=SEED)

    print(f"\n=== {label} ===")
    print(f"  parameter budget {budget} (KAN {tuple(width)} on {nc} scores); "
          f"matched MLP hidden {hidden} -> {p_mlp}")

    rows = []
    for arch, grid, trainer, shape in [("KAN", KAN_GRID, train_kan_to_convergence, width),
                                       ("MLP", MLP_GRID, train_mlp_to_convergence, hidden)]:
        # --- selection: inner grouped CV on the training partition only, one seed
        scores = []
        for cfg in grid:
            fold_r2 = []
            for a, b in folds:
                pr, _, _ = trainer(Z_tr[a], y_tr[a], groups_tr[a], shape, cfg, SEED, n_out)
                fold_r2.append(r2_score(y_tr[b, 0], pr(Z_tr[b])[:, 0]))
            scores.append(float(np.mean(fold_r2)))
            print(f"    {arch} {cfg} -> inner CV R2 {scores[-1]:.4f}")
        chosen = grid[int(np.argmax(scores))]
        print(f"  {arch} selected: {chosen}  (inner CV R2 {max(scores):.4f})")

        # --- evaluation at the selected configuration, several seeds
        for seed in SEEDS:
            pr, upd, val = trainer(Z_tr, y_tr, groups_tr, shape, chosen, seed, n_out)
            ext = r2_score(y_te, pr(Z_te)[:, 0])
            rows.append(dict(dataset=label, architecture=arch,
                             n_params=budget if arch == "KAN" else p_mlp,
                             config=json.dumps(chosen), seed=seed, updates_used=upd,
                             inner_val_R2=val, external_R2=ext))
            print(f"    seed {seed:<5} updates {upd:5d}  inner-val R2 {val:.4f}  external R2 {ext:.4f}")
    return rows


def main():
    t0 = time.time()
    rows = []
    ds, tr, te = tahini_split()
    rows += evaluate("tahini", ds, tr, te, TAHINI, TAHINI["kan_nc"], TAHINI["kan_width"])
    dm, mtr, mte = mango_split()
    rows += evaluate("mango (accuracy KAN)", dm, mtr, mte, MANGO, MANGO["kan_nc"], MANGO["kan_width"])
    rows += evaluate("mango (compact KAN)", dm, mtr, mte, MANGO, MANGO["kan_eq_nc"],
                     MANGO["kan_eq_width"])

    df = pd.DataFrame(rows)
    save(df, "r10_fair_capacity_raw.csv")
    summ = (df.groupby(["dataset", "architecture"])
              .agg(n_params=("n_params", "first"), config=("config", "first"),
                   updates_median=("updates_used", "median"),
                   ext_R2_mean=("external_R2", "mean"), ext_R2_sd=("external_R2", "std"))
              .reset_index())
    save(summ, "r10_fair_capacity_summary.csv")
    print("\n", summ.to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
