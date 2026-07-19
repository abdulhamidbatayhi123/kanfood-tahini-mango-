import numpy as np
from kanfood.data import SpectralDataset
from kanfood.validate import run_group_cv


def _toy_ds():
    rng = np.random.RandomState(0)
    n_groups, scans, p = 16, 8, 30
    groups = np.repeat([f"S{i}" for i in range(n_groups)], scans)
    X = rng.rand(n_groups * scans, p)
    tahini = (X[:, 3] * 60 + X[:, 7] * 40)
    tahini = np.clip(tahini, 0, 100)
    y = np.stack([tahini, (100 - tahini) * 0.6, (100 - tahini) * 0.4], axis=1)
    tagsis = (tahini < 95).astype(int)
    return SpectralDataset(X, y, groups, tagsis, np.arange(p), ["t", "f", "a"], "toy")


def test_run_group_cv_returns_per_model_scores_without_error():
    ds = _toy_ds()
    res = run_group_cv(ds, model_names=["PLS", "RF", "KAN"], n_splits=3,
                       preprocess="snv", n_components=8, seed=0, fast=True)
    assert set(res.keys()) == {"PLS", "RF", "KAN"}
    assert all(len(v["R2_mean"]) == 3 for v in res.values())   # one score per fold
