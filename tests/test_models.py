import numpy as np
from kanfood.models import build_model, MODEL_NAMES, front_end_components


def test_score_model_uses_tuned_n_components():
    # A score model carries its PLS-compression dimension as a tuned hyperparameter.
    assert front_end_components("KAN", {"n_components": 8, "grid": 5}, default=12) == 8


def test_score_model_falls_back_to_default_when_untuned():
    assert front_end_components("MLP", {"hidden": (64, 32)}, default=12) == 12


def test_spectrum_models_have_no_pls_front_end():
    # PLS and CNN consume the full spectrum; n_components (if any) is the model's own, not a front-end.
    assert front_end_components("PLS", {"n_components": 16}, default=12) is None
    assert front_end_components("CNN", {}, default=12) is None


def _toy():
    rng = np.random.RandomState(0)
    Xtr, Xte = rng.rand(40, 12), rng.rand(15, 12)
    ytr = rng.rand(40, 3)
    ytr = ytr / ytr.sum(1, keepdims=True) * 100
    return Xtr, ytr, Xte


def test_all_models_predict_normalized_composition():
    Xtr, ytr, Xte = _toy()
    for name in MODEL_NAMES:
        model = build_model(name, input_dim=12, n_targets=3, seed=0, fast=True)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        assert pred.shape == (15, 3)
        assert np.allclose(pred.sum(axis=1), 100.0, atol=1e-3)


def test_models_skip_normalization_for_single_target():
    # A single continuous target (e.g. mango dry-matter %) must NOT be forced to sum to 100;
    # normalize_to_100 would collapse every prediction to the constant 100.
    rng = np.random.RandomState(0)
    Xtr, Xte = rng.rand(40, 12), rng.rand(15, 12)
    ytr = rng.rand(40, 1) * 15 + 10          # DMC-like values ~10-25
    for name in MODEL_NAMES:
        model = build_model(name, input_dim=12, n_targets=1, seed=0, fast=True)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        assert pred.shape == (15, 1)
        assert not np.allclose(pred, 100.0, atol=1.0), f"{name} wrongly normalized a single target"
