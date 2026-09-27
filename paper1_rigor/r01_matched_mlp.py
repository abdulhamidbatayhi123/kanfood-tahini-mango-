"""Parameter-matched multilayer perceptron.

A smaller model matching a larger, differently tuned one shows only that the task did not need
the larger model's capacity. The architecture claim requires an equal-capacity comparison: an MLP
built to the KAN's *actual* trainable-parameter count, trained under the identical protocol and
evaluated on the identical folds.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

from kanfood.models import build_model
from kanfood.split import stratified_group_kfold
from kanfood.metrics import regression_metrics
from paper1_rigor.common import (TAHINI, MANGO, SEED, tahini_split, mango_split, project,
                                 y_primary, save)

SEEDS = [42, 7, 2024]


def mlp_params(n_in, hidden, n_out):
    """Exact trainable-parameter count of kanfood.models.MLPModel for a given shape."""
    layers, prev, tot = [], n_in, 0
    for h in hidden:
        tot += prev * h + h          # Linear
        tot += 2 * h                 # BatchNorm1d (weight + bias)
        prev = h
    tot += prev * n_out + n_out      # output Linear
    return tot


def match_width(target, n_in, n_out, two_layer=False):
    """Smallest-error hidden shape whose parameter count matches `target`."""
    best = None
    if two_layer:
        for h1 in range(4, 1024):
            for h2 in (max(2, h1 // 2),):
                p = mlp_params(n_in, (h1, h2), n_out)
                if best is None or abs(p - target) < abs(best[1] - target):
                    best = ((h1, h2), p)
    else:
        for h in range(2, 4096):
            p = mlp_params(n_in, (h,), n_out)
            if best is None or abs(p - target) < abs(best[1] - target):
                best = ((h,), p)
    return best


def kan_param_count(n_in, width_hidden, n_out, grid=5, k=3, seed=SEED):
    from kan import KAN
    import os, sys
    from contextlib import redirect_stdout, redirect_stderr
    with open(os.devnull, "w") as dn, redirect_stdout(dn), redirect_stderr(dn):
        m = KAN(width=[n_in] + list(width_hidden) + [n_out], grid=grid, k=k, seed=seed,
                device="cpu", auto_save=False, grid_range=[-1, 1])
    return int(sum(p.numel() for p in m.parameters()))


def evaluate(ds, tr, te, cfg, nc, kan_width, label, folds_groups):
    """Grouped-CV (on train) and external-test R2 for the KAN and its parameter-matched MLPs."""
    n_out = ds.y.shape[1]
    n_kan = kan_param_count(nc, kan_width, n_out)
    m1, p1 = match_width(n_kan, nc, n_out, two_layer=False)
    m2, p2 = match_width(n_kan, nc, n_out, two_layer=True)
    print(f"\n[{label}] KAN {tuple(kan_width)} on {nc} PLS scores -> {n_kan} parameters")
    print(f"   matched MLP (1 hidden layer): {m1} -> {p1} parameters")
    print(f"   matched MLP (2 hidden layers): {m2} -> {p2} parameters")

    Z_tr, Z_te, _, _, _ = project(ds, tr, te, cfg["preprocess"], nc)
    y_tr, y_te = ds.y[tr], y_primary(ds, te)

    arms = {"KAN": ("KAN", dict(width_hidden=kan_width, grid=5)),
            "MLP-matched-1L": ("MLP", dict(hidden=m1)),
            "MLP-matched-2L": ("MLP", dict(hidden=m2))}
    rows = []

    # inner grouped folds on the TRAIN partition only (same splitter as the published benchmark)
    folds = stratified_group_kfold(folds_groups, ds.tagsis[tr], n_splits=5, seed=SEED)
    for arm, (name, kw) in arms.items():
        cv, ext = [], []
        for seed in SEEDS:
            fold_r2 = []
            for a, b in folds:
                m = build_model(name, Z_tr.shape[1], n_out, seed=seed, **kw).fit(Z_tr[a], y_tr[a])
                pr = m.predict(Z_tr[b])
                fold_r2.append(regression_metrics(y_tr[b], pr, ds.target_names)["R2_mean"])
            cv.append(float(np.mean(fold_r2)))
            m = build_model(name, Z_tr.shape[1], n_out, seed=seed, **kw).fit(Z_tr, y_tr)
            ext.append(r2_score(y_te, m.predict(Z_te)[:, 0]))
            rows.append(dict(dataset=label, arm=arm, seed=seed,
                             n_params=n_kan if arm == "KAN" else (p1 if "1L" in arm else p2),
                             cv_R2=cv[-1], external_R2=ext[-1]))
        print(f"   {arm:15s} CV R2 = {np.mean(cv):.4f} +/- {np.std(cv):.4f}   "
              f"external R2 = {np.mean(ext):.4f} +/- {np.std(ext):.4f}")
    return rows


def main():
    rows = []

    ds, tr, te = tahini_split()
    rows += evaluate(ds, tr, te, TAHINI, TAHINI["kan_nc"], TAHINI["kan_width"],
                     "tahini", ds.groups[tr])

    dm, mtr, mte = mango_split()
    rows += evaluate(dm, mtr, mte, MANGO, MANGO["kan_nc"], MANGO["kan_width"],
                     "mango (accuracy KAN)", dm.groups[mtr])
    rows += evaluate(dm, mtr, mte, MANGO, MANGO["kan_eq_nc"], MANGO["kan_eq_width"],
                     "mango (compact KAN)", dm.groups[mtr])

    df = pd.DataFrame(rows)
    save(df, "r01_matched_mlp_raw.csv")
    summ = (df.groupby(["dataset", "arm"])
              .agg(n_params=("n_params", "first"),
                   cv_R2_mean=("cv_R2", "mean"), cv_R2_sd=("cv_R2", "std"),
                   ext_R2_mean=("external_R2", "mean"), ext_R2_sd=("external_R2", "std"))
              .reset_index())
    save(summ, "r01_matched_mlp_summary.csv")
    print("\n", summ.to_string(index=False))


if __name__ == "__main__":
    main()
