import numpy as np
from kanfood.metrics import (normalize_to_100, rpd, regression_metrics,
                             classification_from_tahini, bootstrap_ci)
from sklearn.metrics import r2_score


def test_normalize_to_100_rows_sum_to_100():
    pred = np.array([[80.0, 10.0, 10.0], [-5.0, 50.0, 30.0]])
    out = normalize_to_100(pred)
    assert np.allclose(out.sum(axis=1), 100.0)
    assert (out >= 0).all()


def test_rpd_matches_definition():
    yt = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    yp = yt + 1.0                       # rmse = 1
    assert rpd(yt, yp) == np.std(yt)    # /1


def test_regression_metrics_keys_and_means():
    yt = np.random.RandomState(0).rand(30, 3) * 100
    yp = yt + np.random.RandomState(1).rand(30, 3)
    m = regression_metrics(yt, yp, ["a", "b", "c"])
    assert "R2_a" in m and "R2_mean" in m and "RPD_a" in m


def test_classification_threshold():
    pred_tahini = np.array([99.0, 90.0, 100.0])
    assert classification_from_tahini(pred_tahini, 95.0).tolist() == [0, 1, 0]


def test_bootstrap_ci_orders_correctly():
    yt = np.random.RandomState(0).rand(50)
    yp = yt + np.random.RandomState(1).rand(50) * 0.1
    mean, lo, hi = bootstrap_ci(yt, yp, r2_score, n_boot=200, seed=0)
    assert lo <= mean <= hi
