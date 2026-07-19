import os
import numpy as np
import pytest
from kanfood.data import _parse_wavenumber, load_tahini, load_mango, TAHINI_PATH, MANGO_PATH


def test_parse_wavenumber_handles_comma_decimal():
    assert _parse_wavenumber("2920,026") == pytest.approx(2920.026)
    assert _parse_wavenumber("isim") is None


@pytest.mark.skipif(not os.path.exists(TAHINI_PATH), reason="tahini xlsx not present")
def test_load_tahini_shapes_and_groups():
    ds = load_tahini()
    assert ds.X.shape[0] == ds.y.shape[0] == len(ds.groups) == len(ds.tagsis)
    assert ds.X.shape[1] == len(ds.wavenumbers)
    # FTIR cm-1 range, ascending
    assert ds.wavenumbers[0] < ds.wavenumbers[-1]
    assert 590 < ds.wavenumbers.min() < 620
    assert 3900 < ds.wavenumbers.max() < 4050
    assert ds.y.shape[1] == 3
    # replicate structure: far fewer unique samples than rows
    assert len(np.unique(ds.groups)) < ds.X.shape[0] / 5


@pytest.mark.skipif(not os.path.exists(MANGO_PATH), reason="mango csv not present")
def test_load_mango_shapes_groups_sets():
    ds = load_mango()
    assert ds.X.shape[0] == ds.y.shape[0] == len(ds.groups) == len(ds.sets)
    assert ds.X.shape[1] == len(ds.wavenumbers)
    assert ds.y.shape[1] == 1 and ds.target_names == ["DM"]      # single DMC target
    assert ds.wavenumbers[0] < ds.wavenumbers[-1]                # ascending nm
    assert 280 < ds.wavenumbers.min() < 320 and 1150 < ds.wavenumbers.max() < 1250
    assert set(np.unique(ds.sets)) <= {"Cal", "Tuning", "Val Ext"}
    assert len(np.unique(ds.groups)) == 112                      # populations (leakage-safe grouping)
    assert np.isfinite(ds.X).all() and np.isfinite(ds.y).all()
    assert 5 < ds.y.min() and ds.y.max() < 35                    # plausible dry-matter %
