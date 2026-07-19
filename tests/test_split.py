import numpy as np
from kanfood.split import group_holdout_split, stratified_group_kfold


def _toy_groups(n_groups=20, scans=12):
    groups = np.repeat([f"S{i}" for i in range(n_groups)], scans)
    y_class = np.repeat([i % 2 for i in range(n_groups)], scans)  # class fixed per group
    return groups, y_class


def test_holdout_has_no_sample_overlap():
    groups, _ = _toy_groups()
    tr, te = group_holdout_split(groups, test_size=0.3, seed=0)
    assert set(groups[tr]).isdisjoint(set(groups[te]))
    assert len(tr) + len(te) == len(groups)


def test_kfold_folds_have_no_sample_overlap():
    groups, y_class = _toy_groups()
    folds = stratified_group_kfold(groups, y_class, n_splits=5, seed=0)
    assert len(folds) == 5
    for tr, va in folds:
        assert set(groups[tr]).isdisjoint(set(groups[va]))
