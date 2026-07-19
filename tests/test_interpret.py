import numpy as np
from kanfood.interpret import fit_kan


def test_fit_kan_model_can_prune_for_equation_extraction():
    """Regression: pykan's prune() spawns an auto_save=True child that writes ckpt_path/history.txt.
    fit_kan must point ckpt_path at an existing directory, otherwise symbolic-equation extraction
    (which prunes) crashes with FileNotFoundError: './model/history.txt'."""
    rng = np.random.RandomState(0)
    Zs = rng.uniform(-1, 1, size=(40, 4)).astype("float32")
    ys = rng.standard_normal((40, 1)).astype("float32")
    m, _ = fit_kan(Zs, ys, width_hidden=(2,), grid=3, steps=3, seed=0)
    pruned = m.prune()          # must not raise FileNotFoundError
    assert pruned is not None
