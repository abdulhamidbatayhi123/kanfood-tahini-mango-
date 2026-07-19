import numpy as np
from kanfood.data import SpectralDataset
from kanfood.tune import tune_model


def _toy_ds():
    rng = np.random.RandomState(0)
    n_groups, scans, p = 16, 8, 30
    groups = np.repeat([f"S{i}" for i in range(n_groups)], scans)
    X = rng.rand(n_groups * scans, p)
    tahini = np.clip(X[:, 3] * 60 + X[:, 7] * 40, 0, 100)
    y = np.stack([tahini, (100 - tahini) * 0.6, (100 - tahini) * 0.4], axis=1)
    tagsis = (tahini < 95).astype(int)
    return SpectralDataset(X, y, groups, tagsis, np.arange(p), ["t", "f", "a"], "toy")


def test_tune_returns_a_config_from_the_grid():
    ds = _toy_ds()
    grid = [{"n_components": 4}, {"n_components": 8}]
    best, score = tune_model(ds, "PLS", grid=grid, n_splits=3, n_components=6, seed=0, fast=True)
    assert best in grid
    assert -2.0 <= score <= 1.0
