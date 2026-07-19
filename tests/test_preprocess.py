import numpy as np
from kanfood.preprocess import snv, Preprocessor


def test_snv_is_row_wise_zero_mean_unit_std():
    X = np.random.RandomState(0).rand(8, 50) * 10 + 3
    Z = snv(X)
    assert np.allclose(Z.mean(axis=1), 0, atol=1e-9)
    assert np.allclose(Z.std(axis=1), 1, atol=1e-6)


def test_msc_reference_is_frozen_after_fit():
    rng = np.random.RandomState(1)
    Xtr, Xte = rng.rand(20, 30), rng.rand(7, 30)
    p = Preprocessor("msc").fit(Xtr)
    ref_before = p.reference_.copy()
    _ = p.transform(Xte)                      # transforming test must NOT change reference
    assert np.array_equal(p.reference_, ref_before)
    assert np.allclose(p.reference_, Xtr.mean(axis=0))


def test_methods_preserve_shape():
    X = np.random.RandomState(2).rand(10, 60)
    for m in ["raw", "snv", "msc", "sg1", "sg2", "snv+sg1"]:
        out = Preprocessor(m).fit_transform(X)
        assert out.shape == X.shape
