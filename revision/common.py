"""Shared helpers for the revision experiments (foods-4561619, round 1).

Everything here is a thin wrapper around the published `kanfood` pipeline -- the same loaders,
`Preprocessor`, `PLSFeatures`, `build_model` and `tune._fold_score` that produced Tables 2 and 3 --
so a new experiment differs from the published benchmark only in *which rows* it trains and tests
on and *where* the configuration is selected, never in how a model is fitted or scored.

Added here:
  * `load(name)`      -> (ds, meta): the dataset exactly as the benchmark loads it, plus an aligned
                         metadata frame (tahini production lot; mango season/cultivar/type/...).
  * `fit_predict`     -> one fit on `tr`, predictions on `te`, mirroring run_experiment.py line by line.
  * `select_config`   -> configuration selection over (preprocessing x hyperparameter grid) using any
                         caller-supplied inner folds (e.g. leave-one-lot-out *inside* an outer fold).
  * `ResultStore`     -> append-only JSONL checkpoint so long runs resume after interruption.
  * `run_jobs`        -> process-level parallelism with one torch thread per worker.
"""
from __future__ import annotations

import json
import os
import sys
import time
import hashlib
import warnings
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import numpy as np
import pandas as pd

from kanfood.data import load_tahini, load_mango, SpectralDataset, MANGO_PATH
from kanfood.preprocess import Preprocessor
from kanfood.features import PLSFeatures
from kanfood.models import build_model, front_end_components, MODEL_NAMES
from kanfood.tune import GRIDS, _fold_score
from kanfood.run_mango import MANGO_GRIDS, WL_LO, WL_HI

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "revision" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Published pipeline constants (run_experiment.py / run_mango.py).
CFG = {
    "tahini": {"preprocess": "snv", "n_pls": 12, "grids": GRIDS},
    "mango": {"preprocess": "sg1", "n_pls": 16, "grids": MANGO_GRIDS},
}
SEEDS3 = (42, 7, 2024)          # the three seeds used for stochastic models in Table S5
STOCHASTIC = {"MLP", "CNN", "KAN", "RF"}   # RF is seeded; PLS and SVR are deterministic


# ----------------------------------------------------------------------------------------- data
def _subset(ds: SpectralDataset, idx) -> SpectralDataset:
    sets = None if ds.sets is None else ds.sets[idx]
    return SpectralDataset(ds.X[idx], ds.y[idx], ds.groups[idx], ds.tagsis[idx], ds.wavenumbers,
                           ds.target_names, ds.name, sets)


def load(name: str):
    """Return (ds, meta). `ds` is byte-identical to what the published benchmark uses
    (mango trimmed to 684-990 nm); `meta` is a row-aligned DataFrame of grouping covariates."""
    if name == "tahini":
        ds = load_tahini()
        g = pd.Series(ds.groups)
        meta = pd.DataFrame({
            "group": g,
            "lot": g.str.split("_").str[0],                      # T1..T5 (authentic lot or blend base)
            "adulterant": np.where(ds.y[:, 1] > 0, "peanut", np.where(ds.y[:, 2] > 0, "sunflower", "none")),
            "adulterant_pct": 100.0 - ds.y[:, 0],
        })
        return ds, meta
    if name == "mango":
        full = load_mango()
        keep = (full.wavenumbers >= WL_LO) & (full.wavenumbers <= WL_HI)
        ds = SpectralDataset(full.X[:, keep], full.y, full.groups, full.tagsis, full.wavenumbers[keep],
                             full.target_names, full.name, full.sets)
        raw = pd.read_csv(MANGO_PATH, usecols=["Set", "Season", "Region", "Type", "Cultivar", "Pop", "Temp", "DM"])
        assert len(raw) == len(ds.y) and np.allclose(raw["DM"].to_numpy(), ds.y[:, 0])
        meta = pd.DataFrame({
            "group": raw["Pop"].astype(str).to_numpy(), "season": raw["Season"].to_numpy(),
            "set": raw["Set"].astype(str).to_numpy(), "region": raw["Region"].astype(str).to_numpy(),
            "type": raw["Type"].astype(str).to_numpy(), "cultivar": raw["Cultivar"].astype(str).to_numpy(),
            "temp": raw["Temp"].astype(str).to_numpy(),
        })
        return ds, meta
    raise ValueError(name)


def primary_split(name: str, ds, meta):
    """The published primary partition: tahini grouped 70/30 hold-out (seed 42); mango seasons 1-3 vs 4."""
    if name == "tahini":
        from kanfood.split import group_holdout_split
        return group_holdout_split(ds.groups, test_size=0.30, seed=42)
    tr = np.where(np.isin(ds.sets, ["Cal", "Tuning"]))[0]
    te = np.where(ds.sets == "Val Ext")[0]
    return tr, te


def unit_of(name: str, meta) -> np.ndarray:
    """Top-level exchangeable unit for off-distribution folds: production lot / harvest season."""
    return meta["lot"].to_numpy() if name == "tahini" else meta["season"].astype(str).to_numpy()


def leave_one_unit_out(units: np.ndarray, idx: np.ndarray | None = None):
    """Leave-one-unit-out folds over the rows `idx` (default: all rows). Returns [(unit, tr, te)] as
    absolute row indices, in sorted unit order."""
    idx = np.arange(len(units)) if idx is None else np.asarray(idx)
    u = units[idx]
    return [(v, idx[u != v], idx[u == v]) for v in sorted(np.unique(u))]


# ------------------------------------------------------------------------------------- modelling
def silent(fn):
    with open(os.devnull, "w") as dn, redirect_stdout(dn), redirect_stderr(dn):
        return fn()


def fit_predict(ds, tr, te, name, params, preprocess, n_pls, seed=42):
    """Fit one model on rows `tr`, predict rows `te` -- identical to run_experiment.py/run_mango.py:
    preprocessing fitted on train, spectrum models (PLS, CNN) on the full preprocessed spectrum,
    score models on PLS scores at their own compression. Returns (pred, n_params, fit_seconds)."""
    pp = Preprocessor(preprocess).fit(ds.X[tr])
    Sp_tr, Sp_te = pp.transform(ds.X[tr]), pp.transform(ds.X[te])
    mp = dict(params)
    nc = front_end_components(name, mp, n_pls)
    if nc is None:
        Xi_tr, Xi_te = Sp_tr, Sp_te
    else:
        f = PLSFeatures(nc).fit(Sp_tr, ds.y[tr])
        Xi_tr, Xi_te = f.transform(Sp_tr), f.transform(Sp_te)
        mp = {k: v for k, v in mp.items() if k != "n_components"}
    t0 = time.time()
    model = build_model(name, Xi_tr.shape[1], ds.y.shape[1], seed=seed, **mp)
    silent(lambda: model.fit(Xi_tr, ds.y[tr]))
    pred = model.predict(Xi_te)
    return pred, model.n_params(), time.time() - t0


def inner_score(ds, tr, va, name, params, preprocess, n_pls, seed=42):
    """One inner-fold score with the published selection criterion (tune._fold_score: mean R^2 over
    all targets, i.e. the three composition fractions for tahini, DM for mango)."""
    return float(silent(lambda: _fold_score(ds, tr, va, name, params, preprocess, n_pls, seed, False)))


def param_key(params) -> str:
    return json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(params.items())},
                      sort_keys=True)


def key_hash(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------- checkpointing
class ResultStore:
    """Append-only JSONL store keyed by a string; a job whose key is present is skipped on resume."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.rows[r["key"]] = r

    def has(self, key):
        return key in self.rows

    def add(self, row):
        self.rows[row["key"]] = row
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=_jsonable) + "\n")

    def frame(self):
        return pd.DataFrame(list(self.rows.values()))


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


# ----------------------------------------------------------------------------------- parallelism
def _init_worker():
    import torch
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"


def run_jobs(fn, jobs, store: ResultStore, n_workers=None, log=print):
    """Run `fn(job) -> row dict (must contain 'key')` for every job whose key is not yet stored,
    across `n_workers` processes (default: all CPUs), appending each row as it completes."""
    import multiprocessing as mp
    todo = [j for j in jobs if not store.has(j["key"])]
    log(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run")
    if not todo:
        return
    n_workers = n_workers or os.cpu_count()
    t0 = time.time()
    if n_workers == 1:
        _init_worker()
        for i, j in enumerate(todo, 1):
            store.add(fn(j))
            log(f"  [{i}/{len(todo)}] {j['key']} ({time.time() - t0:.0f}s)")
        return
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers, initializer=_init_worker, maxtasksperchild=20) as pool:
        for i, row in enumerate(pool.imap_unordered(fn, todo), 1):
            store.add(row)
            log(f"  [{i}/{len(todo)}] {row['key']} ({time.time() - t0:.0f}s)")


def log_to(path):
    fh = open(path, "a", encoding="utf-8", buffering=1)

    def _log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
    return _log
