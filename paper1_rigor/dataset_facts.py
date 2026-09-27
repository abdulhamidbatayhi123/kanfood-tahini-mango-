"""Every dataset fact quoted in the Materials and Methods, read directly off the data files."""
import numpy as np
import pandas as pd
from kanfood.data import load_tahini, load_mango, MANGO_PATH
from kanfood.split import group_holdout_split


def main():
    ds = load_tahini()
    wn = ds.wavenumbers
    print("TAHINI")
    print("  spectra %d, channels %d" % ds.X.shape)
    print("  cm-1 %.2f - %.2f ; median step %.4f" % (wn.min(), wn.max(), np.median(np.diff(wn))))
    g, c = np.unique(ds.groups, return_counts=True)
    pure = np.array([("_" not in x) for x in g])
    print("  pure lots %d, scans each %s" % (pure.sum(), c[pure].tolist()))
    print("  blends %d, scans per blend min %d max %d mean %.1f"
          % ((~pure).sum(), c[~pure].min(), c[~pure].max(), c[~pure].mean()))
    lv = sorted({float(x.split("_")[1][1:]) for x in g if "_" in x})
    print("  adulteration levels (%%): %s" % lv)
    print("  peanut blends %d, sunflower blends %d"
          % (sum("_P" in x for x in g), sum("_S" in x for x in g)))
    print("  tahini fraction %.0f - %.0f ; pure spectra %d, adulterated %d"
          % (ds.y[:, 0].min(), ds.y[:, 0].max(), (ds.tagsis == 0).sum(), (ds.tagsis == 1).sum()))
    tr, te = group_holdout_split(ds.groups, 0.30, 42)
    print("  hold-out: train %d spectra / %d lots-blends ; test %d / %d"
          % (len(tr), len(np.unique(ds.groups[tr])), len(te), len(np.unique(ds.groups[te]))))
    print("  test ids: %s" % sorted(np.unique(ds.groups[te]).tolist()))

    dm = load_mango()
    print("\nMANGO")
    print("  spectra %d, channels %d, nm %.0f-%.0f"
          % (dm.X.shape[0], dm.X.shape[1], dm.wavenumbers.min(), dm.wavenumbers.max()))
    keep = (dm.wavenumbers >= 684) & (dm.wavenumbers <= 990)
    print("  modelled band: %d channels, step %.2f nm" % (keep.sum(), np.median(np.diff(dm.wavenumbers))))
    print("  sets: %s" % {k: int(v) for k, v in zip(*np.unique(dm.sets, return_counts=True))})
    tr = dm.sets != "Val Ext"; te = dm.sets == "Val Ext"
    # The manuscript quotes the combined calibration+tuning count as one number, so record it as
    # one number rather than leaving the reader (or the number-tracing check) to add it up.
    print("  train spectra (Cal + Tuning) %d ; external test (Val Ext) %d"
          % (int(tr.sum()), int(te.sum())))

    # Gradient updates each architecture receives under the PUBLISHED recipe. The manuscript
    # compares these directly, so they are computed here rather than worked out in the prose:
    # the perceptron runs `epochs` passes of ceil(n/batch) mini-batches, the KAN a fixed number
    # of Adam steps. Settings from kanfood.models (MLPModel epochs=200 batch=64; KANModel
    # steps=300).
    import math
    for name, n_train in (("tahini", 1146), ("mango", int(tr.sum()))):
        mlp_updates = 200 * math.ceil(n_train / 64)
        print("  %s: published recipe gives the MLP %d gradient updates and the KAN 300 (%.0fx)"
              % (name, mlp_updates, mlp_updates / 300))
    print("  populations total %d ; train %d ; test %d ; overlap %d"
          % (len(np.unique(dm.groups)), len(np.unique(dm.groups[tr])),
             len(np.unique(dm.groups[te])), len(set(dm.groups[tr]) & set(dm.groups[te]))))
    print("  DM%% train %.2f-%.2f (mean %.2f sd %.2f) ; test %.2f-%.2f (mean %.2f sd %.2f)"
          % (dm.y[tr, 0].min(), dm.y[tr, 0].max(), dm.y[tr, 0].mean(), dm.y[tr, 0].std(),
             dm.y[te, 0].min(), dm.y[te, 0].max(), dm.y[te, 0].mean(), dm.y[te, 0].std()))
    raw = pd.read_csv(MANGO_PATH)
    for col in ["Season", "Region", "Cultivar", "Type"]:
        v = raw[col].astype(str)
        print("  %s: %d unique -> %s" % (col, v.nunique(), sorted(v.unique())[:12]))
    print(pd.crosstab(raw["Season"], raw["Set"]).to_string())


if __name__ == "__main__":
    main()
