"""Shared protocol helpers for the tahini/mango rigor experiments.

Every experiment in this folder imports the *published* pipeline (kanfood.*) rather than
re-implementing it, and reproduces a published number as a guard before reporting anything new.
"""
import os
import sys
import json

# Thread cap for BLAS and torch. Leave a couple of cores free so the machine stays usable.
# `KAN_THREADS` lets two long experiments share one machine without oversubscribing it; the
# default is the 8 every published result was produced at, so an unset variable changes nothing.
_THREADS = os.environ.get("KAN_THREADS") or os.environ.get("OMP_NUM_THREADS") or "8"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)
os.environ.setdefault("TQDM_DISABLE", "1")   # pykan's per-step bar otherwise floods the logs
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(int(_THREADS))

from kanfood.data import load_tahini, load_mango, SpectralDataset
from kanfood.preprocess import Preprocessor
from kanfood.features import PLSFeatures
from kanfood.split import group_holdout_split, stratified_group_kfold

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_rigor"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
SEEDS10 = [42, 100, 2024, 7, 123, 999, 13, 50, 777, 888]

# ---- canonical, benchmark-selected configurations (results_*/*_meta.json) -------------------
TAHINI_META = json.loads((ROOT / "results_phase1" / "phase1_meta.json").read_text(encoding="utf-8"))
MANGO_META = json.loads((ROOT / "results_mango" / "mango_meta.json").read_text(encoding="utf-8"))

TAHINI = dict(
    preprocess="snv",
    kan_nc=int(TAHINI_META["kan_n_components"]),          # 8
    kan_width=(3,),
    mlp_hidden=(64, 32), mlp_nc=8,
    pls_nc=12,
)
MANGO = dict(
    preprocess="sg1",
    kan_nc=int(MANGO_META["kan_n_components"]),           # 24
    kan_width=(16, 8),
    kan_eq_nc=12, kan_eq_width=(3,),                      # compact, equation-bearing variant
    mlp_hidden=(128, 64), mlp_nc=16,
    pls_nc=24,
    wl_band=(684.0, 990.0),                               # F750 useful NIR band (run_mango.WL_LO/HI)
)


def silent(fn):
    with open(os.devnull, "w") as dn, redirect_stdout(dn), redirect_stderr(dn):
        return fn()


# ---- dataset accessors that mirror the published splits ------------------------------------
def tahini_split():
    """The published tahini partition: grouped hold-out by `isim`, 30 % test, seed 42."""
    ds = load_tahini()
    tr, te = group_holdout_split(ds.groups, test_size=0.30, seed=SEED)
    assert set(ds.groups[tr]).isdisjoint(set(ds.groups[te])), "leakage"
    return ds, tr, te


def mango_split():
    """The published mango partition: Seasons 1-3 (Cal+Tuning) train, Season 4 (Val Ext) test."""
    ds = load_mango()
    lo, hi = MANGO["wl_band"]
    keep = (ds.wavenumbers >= lo) & (ds.wavenumbers <= hi)
    ds = SpectralDataset(ds.X[:, keep], ds.y, ds.groups, ds.tagsis, ds.wavenumbers[keep],
                         ds.target_names, ds.name, ds.sets)
    te = np.where(ds.sets == "Val Ext")[0]
    tr = np.where(ds.sets != "Val Ext")[0]
    assert set(ds.groups[tr]).isdisjoint(set(ds.groups[te])), "population leakage"
    return ds, tr, te


def project(ds, tr, te, preprocess, n_components, window=11, poly=2):
    """Fit preprocessing + PLS compression on TRAIN only; return latent scores for both sides.

    `window`/`poly` are the Savitzky-Golay settings; the defaults are the published ones, so
    calls that do not pass them are unchanged.
    """
    pp = Preprocessor(preprocess, window=window, poly=poly).fit(ds.X[tr])
    Sp_tr, Sp_te = pp.transform(ds.X[tr]), pp.transform(ds.X[te])
    plsf = PLSFeatures(n_components).fit(Sp_tr, ds.y[tr])
    return plsf.transform(Sp_tr), plsf.transform(Sp_te), plsf, Sp_tr, Sp_te


def y_primary(ds, idx):
    """The reported target: tahini fraction (%) for tahini, dry matter (%) for mango."""
    return ds.y[idx, 0]


def save(df, name):
    p = OUT / name
    df.to_csv(p, index=False)
    print(f"  -> {p.relative_to(ROOT)}")
    return p


def ckpt(obj, name):
    """Write a partial result the moment it exists.

    Every long experiment in this package used to call `save` only at the end of `main`, so a
    killed console discarded hours of completed work and left its numbers reachable only in a
    log -- which rule 1 forbids quoting. Checkpoints are written per dataset and per stage so a
    partial run is still CSV-traceable. `_partial` files are inputs to nothing; the final
    `save` calls remain the authority.
    """
    df = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(list(obj))
    if len(df) == 0:
        return None
    p = OUT / name
    df.to_csv(p, index=False)
    print(f"  [ckpt] {p.relative_to(ROOT)} ({len(df)} rows)", flush=True)
    return p


# 30 seeds, for the high-power re-runs the *Foods* revision needed (reviewer 2, major point 8: the
# submitted comparisons rest on 3-5 seeds, which the paper itself describes as having "very little
# power"). The first 3, 5 and 10 entries are exactly the seed lists the submitted results were
# produced at, in their original order, so a re-run at `--seeds 3` reproduces the published numbers
# and a re-run at `--seeds 20` extends them rather than replacing them with a different draw.
SEEDS30 = [42, 7, 2024,                                             # the published 3
           100, 13,                                                 # -> the published 5
           999, 123, 50, 777, 888,                                  # -> the published 10
           11, 17, 23, 31, 47, 61, 73, 89, 97, 101,
           211, 307, 401, 503, 601, 701, 809, 907, 1009, 1103]
assert len(SEEDS30) == 30 and len(set(SEEDS30)) == 30


def seed_list(n):
    """The first `n` of SEEDS30, so that nested seed budgets are prefixes of one another."""
    if n > len(SEEDS30):
        raise ValueError(f"only {len(SEEDS30)} seeds are defined; asked for {n}")
    return list(SEEDS30[:n])
