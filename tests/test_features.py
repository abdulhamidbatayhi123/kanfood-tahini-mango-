import numpy as np
from kanfood.features import MIFeatureSelector, PLSFeatures


def test_pls_features_fit_on_train_and_shape():
    rng = np.random.RandomState(0)
    Xtr, ytr = rng.rand(40, 25), rng.rand(40, 3)
    pf = PLSFeatures(n_components=6).fit(Xtr, ytr)
    assert pf.transform(Xtr).shape == (40, 6)
    Xte = rng.rand(8, 25)
    assert pf.transform(Xte).shape == (8, 6)        # reuses train-fit PLS on test


def test_selector_is_fit_on_train_and_reused():
    rng = np.random.RandomState(0)
    Xtr = rng.rand(60, 40)
    ytr = Xtr[:, 5] * 3 + Xtr[:, 9] * 2 + rng.rand(60) * 0.01   # cols 5,9 informative
    sel = MIFeatureSelector(n_features=10, seed=0).fit(Xtr, ytr)
    assert sel.indices_ is not None and len(sel.indices_) == 10
    assert {5, 9}.issubset(set(sel.indices_.tolist()))
    Xte = rng.rand(7, 40)
    assert sel.transform(Xte).shape == (7, 10)
    # transform uses stored indices, independent of test content
    assert np.array_equal(sel.transform(Xte), Xte[:, sel.indices_])
