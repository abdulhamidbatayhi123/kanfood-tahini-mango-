"""The competitors an interpretable-calibration paper has to beat, or concede to.

Three families are missing from the original benchmark and each is a live reviewer objection.

1. ADDITIVE GLASS-BOXES. A depth-1 KAN is a generalised additive model, so the explainable
   boosting machine (EBM) and a spline GAM occupy exactly the same quadrant as the KAN, are
   older, and are widely used. Omitting them would let a reviewer say the comparison was made
   only against models the KAN was always going to beat on transparency.
2. THE SPLINE-ACTIVATION MLP. The controlled KAN-vs-MLP literature reports that an MLP given
   B-spline activations matches or surpasses a KAN even on symbolic tasks, i.e. that the
   advantage attributed to the architecture belongs to the basis. That ablation is run here:
   the same MLP, the same budget, with each hidden unit's activation replaced by a learnable
   B-spline on the same grid the KAN uses. Note what it does NOT give: because the activation
   sits on the NODE, the network still mixes its inputs inside every unit, so there is no
   per-edge univariate function to convert and no closed form comes out.
3. SYMBOLIC REGRESSION. "Why not simply run symbolic regression?" is the obvious question, and
   it deserves a measured answer rather than an argument. Genetic-programming symbolic
   regression is run on exactly the inputs the KAN receives.

Gradient boosting (XGBoost) is added as a strong opaque reference so the frontier has a
top-accuracy anchor that nobody claims is interpretable.

Every model is evaluated on the published splits, on both input representations that the paper
uses -- the PLS latent scores and the full preprocessed spectrum -- so that the frontier figure
can place each model on comparable axes.
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

from kan.spline import B_batch
from kanfood.metrics import normalize_to_100
from kanfood.preprocess import Preprocessor
from paper1_rigor.common import (seed_list, TAHINI, MANGO, SEED, tahini_split, mango_split,
                                 project, save, ckpt)

warnings.filterwarnings("ignore")

SEEDS = [42, 7, 2024]
# Raised on the command line for the *Foods* revision; `seed_list(n)` is a prefix of the published
# list, so a larger budget extends this result rather than redrawing it (reviewer 2, point 8).
TAG = ""


# --------------------------------------------------------------------- spline-activation MLP
class SplineActivation(nn.Module):
    """A learnable B-spline activation, one per hidden unit, on the same grid family as the KAN.

    This is the node-wise counterpart of a KAN edge function: the spline is applied to the unit's
    pre-activation, after the inputs have already been mixed by the linear layer. That mixing is
    exactly why no closed form in the original variables can be read off the trained network.
    """

    def __init__(self, width, grid=5, k=3, rng=(-3.0, 3.0)):
        super().__init__()
        self.k, self.width = k, width
        g = torch.linspace(rng[0], rng[1], grid + 1)
        step = (rng[1] - rng[0]) / grid
        ext = torch.cat([g[0] - step * torch.arange(k, 0, -1), g, g[-1] + step * torch.arange(1, k + 1)])
        self.register_buffer("grid", ext[None, :].repeat(width, 1))
        self.coef = nn.Parameter(torch.randn(width, grid + k) * 0.1)

    def forward(self, x):                      # x: (batch, width)
        b = B_batch(x, self.grid, k=self.k)     # (batch, width, grid + k)
        return torch.einsum("bwc,wc->bw", b, self.coef) + torch.nn.functional.silu(x)


def build_spline_mlp(n_in, hidden, n_out, grid=5, k=3):
    layers, prev = [], n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), SplineActivation(h, grid, k)]
        prev = h
    layers.append(nn.Linear(prev, n_out))
    return nn.Sequential(*layers)


def train_torch(net, Xa, ya, epochs, lr, wd, seed, batch=64):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    n = Xa.shape[0]
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            if idx.numel() < 2:
                continue
            opt.zero_grad()
            lossf(net(Xa[idx]), ya[idx]).backward()
            opt.step()
    return net


# ------------------------------------------------------------------------------- model zoo
def _post(p, normalise):
    p = np.asarray(p).reshape(len(p), -1)
    return (normalize_to_100(p) if normalise else p)[:, 0]


def run_model(name, A_tr, y_tr, A_te, normalise, seed):
    from sklearn.preprocessing import StandardScaler
    if name == "EBM":
        from interpret.glassbox import ExplainableBoostingRegressor
        return ExplainableBoostingRegressor(random_state=seed, interactions=0, n_jobs=1
                                            ).fit(A_tr, y_tr[:, 0]).predict(A_te)
    if name == "EBM+int":
        from interpret.glassbox import ExplainableBoostingRegressor
        return ExplainableBoostingRegressor(random_state=seed, interactions=8, n_jobs=1
                                            ).fit(A_tr, y_tr[:, 0]).predict(A_te)
    if name == "GAM":
        from pygam import LinearGAM
        return LinearGAM().gridsearch(A_tr, y_tr[:, 0], progress=False).predict(A_te)
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=seed,
                            n_jobs=2).fit(A_tr, y_tr[:, 0]).predict(A_te)
    if name in ("spline-MLP", "MLP(ref)"):
        sx, sy = StandardScaler().fit(A_tr), StandardScaler().fit(y_tr)
        Xa = torch.tensor(sx.transform(A_tr), dtype=torch.float32)
        ya = torch.tensor(sy.transform(y_tr), dtype=torch.float32)
        hidden = (64, 32) if A_tr.shape[1] <= 32 else (128, 64)
        if name == "spline-MLP":
            net = build_spline_mlp(A_tr.shape[1], hidden, y_tr.shape[1])
        else:
            layers, prev = [], A_tr.shape[1]
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)]
                prev = h
            layers.append(nn.Linear(prev, y_tr.shape[1]))
            net = nn.Sequential(*layers)
        train_torch(net, Xa, ya, epochs=200, lr=1e-3, wd=1e-4, seed=seed)
        net.eval()
        with torch.no_grad():
            out = net(torch.tensor(sx.transform(A_te), dtype=torch.float32)).numpy()
        return _post(sy.inverse_transform(out), normalise)
    if name == "SR (gplearn)":
        from gplearn.genetic import SymbolicRegressor
        m = SymbolicRegressor(population_size=2000, generations=20, stopping_criteria=0.001,
                              function_set=("add", "sub", "mul", "div", "sqrt", "log", "sin"),
                              parsimony_coefficient=0.001, random_state=seed, n_jobs=2,
                              verbose=0)
        m.fit(A_tr, y_tr[:, 0])
        return m.predict(A_te), str(m._program)
    raise ValueError(name)


MODELS = ["EBM", "EBM+int", "GAM", "XGBoost", "spline-MLP", "MLP(ref)", "SR (gplearn)"]


def run(label, splitter, cfg, nc, normalise, only_reps=None):
    ds, tr, te = splitter()
    y_tr, y_te = ds.y[tr], ds.y[te, 0]
    Z_tr, Z_te, _, S_tr, S_te = project(ds, tr, te, cfg["preprocess"], nc)
    reps = {"PLS scores": (Z_tr, Z_te), "full spectrum": (S_tr, S_te)}
    if only_reps is not None:
        reps = {k: v for k, v in reps.items() if k in only_reps}
    rows, omitted = [], []
    print("\n=== " + label + ": additional baselines ===", flush=True)
    for rep, (A_tr, A_te) in reps.items():
        for name in MODELS:
            # Cells that are not computable are recorded, never dropped in silence: an
            # empty cell in a published frontier table has to say why it is empty.
            why = None
            if name == "SR (gplearn)" and rep == "full spectrum":
                why = "symbolic regression over %d channels is not meaningful" % A_tr.shape[1]
            elif name == "GAM" and A_tr.shape[1] > 60:
                why = "a per-channel spline GAM on %d channels is not a model" % A_tr.shape[1]
            elif name == "EBM+int" and A_tr.shape[1] > 60:
                # EBM's pairwise stage searches O(p^2) candidate interactions. On the latent
                # scores (p=8) it already costs 10x the additive fit; at p=1762 that is ~1.6M
                # pairs, projected at ~38 h per seed against 3.8 h for the additive fit as
                # measured on this machine. Outside the study's compute budget.
                why = ("pairwise-interaction search over %d channels (~%.1fM candidate pairs) "
                       "exceeds the compute budget"
                       % (A_tr.shape[1], A_tr.shape[1] * (A_tr.shape[1] - 1) / 2e6))
            if why:
                omitted.append(dict(dataset=label, representation=rep, model=name, reason=why))
                print("    " + rep.ljust(14) + name.ljust(14) + " OMITTED: " + why, flush=True)
                continue
            for seed in SEEDS:
                t0 = time.time()
                eq = ""
                try:
                    out = run_model(name, A_tr, y_tr, A_te, normalise, seed)
                    if isinstance(out, tuple):
                        out, eq = out
                    r2 = float(r2_score(y_te, out))
                except Exception as e:
                    r2, eq = float("nan"), "FAILED " + type(e).__name__ + ": " + str(e)[:120]
                rows.append(dict(dataset=label, representation=rep, model=name, seed=seed,
                                 R2=r2, seconds=round(time.time() - t0, 1), expression=eq))
                print("    " + rep.ljust(14) + name.ljust(14) + " seed " + str(seed).ljust(5) +
                      " R2 " + format(r2, ".4f") + "  (" + format(time.time() - t0, ".0f") + "s)",
                      flush=True)
            ckpt(rows, "r14_glassbox_partial_" + label + "_" + rep.split()[0] + TAG + ".csv")
    return rows, omitted


def main():
    global SEEDS, TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--datasets", default="tahini,mango")
    ap.add_argument("--reps", default="PLS scores,full spectrum")
    a = ap.parse_args()
    SEEDS = seed_list(a.seeds)
    TAG = ("_" + a.tag) if a.tag else ""
    want_ds = [d for d in a.datasets.split(",") if d]
    want_reps = [r for r in a.reps.split(",") if r]
    print(f"r14: {len(SEEDS)} seeds {SEEDS}, tag={TAG!r}, datasets={want_ds}, reps={want_reps}",
          flush=True)
    t0 = time.time()
    # The latent-scores frontier is the comparison the paper actually rests on, and it is
    # cheap; the full-spectrum arm costs hours per cell. Both datasets are therefore run at
    # the latent representation first, so an interrupted job still leaves the central result
    # complete rather than one dataset half-finished.
    rows, om = [], []
    for reps in ([r] for r in want_reps):
        for lab, sp, cf in (("tahini", tahini_split, TAHINI), ("mango", mango_split, MANGO)):
            if lab not in want_ds:
                continue
            r, o = run(lab, sp, cf, cf["kan_nc"], lab == "tahini", only_reps=reps)
            rows += r
            om += o
            save(pd.DataFrame(rows), f"r14_glassbox_baselines_raw_progress{TAG}.csv")
    save(pd.DataFrame(om), f"r14_glassbox_omitted{TAG}.csv")
    df = pd.DataFrame(rows)
    save(df, f"r14_glassbox_baselines_raw{TAG}.csv")
    # Median, not mean: the published table's column and caption both say median, and with three
    # seeds a single slow run moves the mean by more than a rounding step.
    summ = (df.groupby(["dataset", "representation", "model"])
              .agg(R2_mean=("R2", "mean"), R2_sd=("R2", "std"), seconds=("seconds", "median"))
              .reset_index())
    save(summ, f"r14_glassbox_baselines_summary{TAG}.csv")
    print("\n", summ.to_string(index=False))
    print("\ntotal " + format(time.time() - t0, ".0f") + "s")


if __name__ == "__main__":
    main()
