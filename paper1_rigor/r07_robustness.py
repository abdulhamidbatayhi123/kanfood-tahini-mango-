"""Robustness of the tahini result to the choice of grouped hold-out.

The headline tahini numbers come from one grouped split. Ten independent grouped splits show
whether the ordering of the models, and the KAN's position in it, depend on that choice.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from kanfood.preprocess import Preprocessor
from kanfood.features import PLSFeatures
from kanfood.models import build_model
from kanfood.split import group_holdout_split
from paper1_rigor.common import TAHINI, tahini_split, save

SPLIT_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
CFG = dict(TAHINI, svm_C=100.0, svm_gamma=0.05, rf_depth=10, cnn_channels=(16, 32), cnn_lr=1e-3)
MODELS = ["PLS", "SVM", "RF", "MLP", "CNN", "KAN"]


def spec(name):
    return {"PLS": (None, dict(n_components=CFG["pls_nc"])),
            "CNN": (None, dict(channels=CFG["cnn_channels"], lr=CFG["cnn_lr"])),
            "SVM": (CFG["pls_nc"], dict(C=CFG["svm_C"], gamma=CFG["svm_gamma"])),
            "RF": (CFG["pls_nc"], dict(n_estimators=300, max_depth=CFG["rf_depth"])),
            "MLP": (CFG["mlp_nc"], dict(hidden=CFG["mlp_hidden"])),
            "KAN": (CFG["kan_nc"], dict(width_hidden=CFG["kan_width"], grid=5))}[name]


def main():
    ds, _, _ = tahini_split()
    rows = []
    for s in SPLIT_SEEDS:
        tr, te = group_holdout_split(ds.groups, test_size=0.30, seed=s)
        assert set(ds.groups[tr]).isdisjoint(set(ds.groups[te]))
        pp = Preprocessor(CFG["preprocess"]).fit(ds.X[tr])
        S_tr, S_te = pp.transform(ds.X[tr]), pp.transform(ds.X[te])
        cache = {}
        line = []
        for name in MODELS:
            nc, kw = spec(name)
            if nc is None:
                A, B = S_tr, S_te
            else:
                if nc not in cache:
                    pf = PLSFeatures(nc).fit(S_tr, ds.y[tr])
                    cache[nc] = (pf.transform(S_tr), pf.transform(S_te))
                A, B = cache[nc]
            m = build_model(name, A.shape[1], ds.y.shape[1], seed=42, **kw).fit(A, ds.y[tr])
            r2 = r2_score(ds.y[te, 0], m.predict(B)[:, 0])
            rows.append(dict(split_seed=s, model=name, R2=r2,
                             n_test_samples=len(np.unique(ds.groups[te]))))
            line.append(f"{name} {r2:.3f}")
        print(f"  split {s}: " + "  ".join(line))
    df = pd.DataFrame(rows)
    save(df, "r07_robustness_raw.csv")
    summ = df.groupby("model").agg(R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
                                   R2_min=("R2", "min"), R2_max=("R2", "max")).reset_index()
    save(summ, "r07_robustness_summary.csv")
    print("\n", summ.to_string(index=False))


if __name__ == "__main__":
    main()
