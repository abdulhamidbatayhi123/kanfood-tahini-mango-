"""Generalisation beyond the grouped hold-out.

Tahini: leave-one-source-out. The 55 blends derive from five independent tahini lots (T1-T5);
withholding an entire lot asks the calibration to transfer to a tahini it has never seen, which is
strictly harder than withholding blends of lots it has seen.

Mango: leave-one-season-out. The published protocol already tests one unseen season (Season 4);
cycling the held-out season shows whether that particular season was unusually easy or hard.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from kanfood.data import load_mango
from kanfood.preprocess import Preprocessor
from kanfood.features import PLSFeatures
from kanfood.models import build_model
from paper1_rigor.common import TAHINI, MANGO, SEED, tahini_split, mango_split, save

SEEDS = [42, 7, 2024]
MODELS = ["PLS", "SVM", "RF", "MLP", "CNN", "KAN"]
STOCHASTIC = {"MLP", "CNN", "KAN"}


def kw_for(name, cfg):
    return {"PLS": dict(n_components=cfg["pls_nc"]),
            "SVM": dict(C=cfg["svm_C"], gamma=cfg["svm_gamma"]),
            "RF": dict(n_estimators=300, max_depth=cfg["rf_depth"]),
            "MLP": dict(hidden=cfg["mlp_hidden"]),
            "CNN": dict(channels=cfg["cnn_channels"], lr=cfg["cnn_lr"]),
            "KAN": dict(width_hidden=cfg["kan_width"], grid=5)}[name]


def nc_for(name, cfg):
    return {"PLS": None, "CNN": None, "SVM": cfg["pls_nc"], "RF": cfg["pls_nc"],
            "MLP": cfg["mlp_nc"], "KAN": cfg["kan_nc"]}[name]


def one_fold(ds, tr, te, cfg, name, seed):
    pp = Preprocessor(cfg["preprocess"]).fit(ds.X[tr])
    S_tr, S_te = pp.transform(ds.X[tr]), pp.transform(ds.X[te])
    nc = nc_for(name, cfg)
    if nc is None:
        A, B = S_tr, S_te
    else:
        pf = PLSFeatures(nc).fit(S_tr, ds.y[tr])
        A, B = pf.transform(S_tr), pf.transform(S_te)
    m = build_model(name, A.shape[1], ds.y.shape[1], seed=seed, **kw_for(name, cfg)).fit(A, ds.y[tr])
    return r2_score(ds.y[te, 0], m.predict(B)[:, 0])


def run(label, ds, fold_labels, cfg):
    rows = []
    for held in sorted(np.unique(fold_labels)):
        te = np.where(fold_labels == held)[0]
        tr = np.where(fold_labels != held)[0]
        assert set(ds.groups[tr]).isdisjoint(set(ds.groups[te])), "group leakage in the fold"
        print(f"\n  hold out {held}: train {len(tr)} / test {len(te)} spectra")
        for name in MODELS:
            seeds = SEEDS if name in STOCHASTIC else [SEED]
            r2 = [one_fold(ds, tr, te, cfg, name, s) for s in seeds]
            rows.append(dict(dataset=label, held_out=str(held), model=name,
                             R2_mean=float(np.mean(r2)), R2_sd=float(np.std(r2)),
                             R2_seeds=";".join(f"{v:.4f}" for v in r2), n_test=len(te)))
            print(f"    {name:4s} R2 = {np.mean(r2):7.4f}"
                  + (f"  (seeds {', '.join(f'{v:.3f}' for v in r2)})" if len(r2) > 1 else ""))
    return rows


def main():
    tcfg = dict(TAHINI, svm_C=100.0, svm_gamma=0.05, rf_depth=10,
                cnn_channels=(16, 32), cnn_lr=1e-3)
    mcfg = dict(MANGO, svm_C=10.0, svm_gamma=0.05, rf_depth=20,
                cnn_channels=(8, 16), cnn_lr=3e-4)

    print("=== tahini: leave-one-source-out (5 independent tahini lots) ===")
    ds, _, _ = tahini_split()
    sources = np.array([g.split("_")[0] for g in ds.groups])
    assert set(np.unique(sources)) == {"T1", "T2", "T3", "T4", "T5"}, np.unique(sources)
    rows = run("tahini", ds, sources, tcfg)

    print("\n=== mango: leave-one-season-out ===")
    dm, _, _ = mango_split()
    seasons = pd.read_csv(load_mango.__defaults__[0])["Season"].astype(str).to_numpy()
    assert len(seasons) == dm.X.shape[0]
    print("  seasons:", np.unique(seasons))
    rows += run("mango", dm, seasons, mcfg)

    df = pd.DataFrame(rows)
    save(df, "r04_generalisation_raw.csv")
    summ = (df.groupby(["dataset", "model"])
              .agg(mean_R2=("R2_mean", "mean"), median_R2=("R2_mean", "median"),
                   min_R2=("R2_mean", "min"), sd_across_folds=("R2_mean", "std"))
              .reset_index())
    save(summ, "r04_generalisation_summary.csv")
    print("\n", summ.to_string(index=False))


if __name__ == "__main__":
    main()
