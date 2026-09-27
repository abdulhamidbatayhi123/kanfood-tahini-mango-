"""Limit of detection from pooled out-of-fold predictions.

The hold-out estimate of the detection limit has a structural weakness: the blank standard deviation
sigma is computed from the authentic samples on the *test* side, and in the published 38/17 split
only one of the five authentic tahini lots falls there. A cluster bootstrap that resamples whole
physical samples therefore redraws the same 151 scans of the same lot every time, which makes the
interval far tighter than the real uncertainty and ties the point estimate to one lot.

This script replaces that estimate with one built from pooled out-of-fold predictions over repeated
grouped cross-validation, in which every one of the five authentic lots is used as a blank while it
is held out, and resamples whole lots (for sigma) and whole samples (for the slope) in the bootstrap.
The hold-out values are still reported alongside, so the two can be compared rather than silently
swapped.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from kanfood.preprocess import Preprocessor
from kanfood.features import PLSFeatures
from kanfood.models import build_model
from kanfood.split import stratified_group_kfold
from paper1_rigor.common import TAHINI, SEED, ROOT, tahini_split, save

REPEATS = 3
N_BOOT = 2000
MODELS = ["PLS", "SVM", "RF", "MLP", "CNN", "KAN"]
CFG = dict(TAHINI, svm_C=100.0, svm_gamma=0.05, rf_depth=10, cnn_channels=(16, 32), cnn_lr=1e-3)


def spec(name):
    return {"PLS": (None, dict(n_components=CFG["pls_nc"])),
            "CNN": (None, dict(channels=CFG["cnn_channels"], lr=CFG["cnn_lr"])),
            "SVM": (CFG["pls_nc"], dict(C=CFG["svm_C"], gamma=CFG["svm_gamma"])),
            "RF": (CFG["pls_nc"], dict(n_estimators=300, max_depth=CFG["rf_depth"])),
            "MLP": (CFG["mlp_nc"], dict(hidden=CFG["mlp_hidden"])),
            "KAN": (CFG["kan_nc"], dict(width_hidden=CFG["kan_width"], grid=5))}[name]


def lod_loq(adult_true, adult_pred, blank_mask):
    if blank_mask.sum() < 2 or len(np.unique(adult_true)) < 2:
        return np.nan, np.nan
    sigma = float(np.std(adult_pred[blank_mask]))
    slope = float(np.polyfit(adult_true, adult_pred, 1)[0])
    slope = slope if abs(slope) > 1e-6 else 1.0
    return 3.3 * sigma / abs(slope), 10.0 * sigma / abs(slope)


def out_of_fold(ds):
    """Pooled out-of-fold predictions over `REPEATS` independent grouped five-fold partitions.

    All three composition targets are retained, not only the tahini fraction, because the
    adulterant-identification analysis (r09) needs the peanut and sunflower columns and must see
    every concentration level -- the single hold-out contains no blend below 16 % adulterant.
    """
    n_t = ds.y.shape[1]
    preds = {m: np.full((REPEATS, len(ds.y), n_t), np.nan) for m in MODELS}
    for rep in range(REPEATS):
        folds = stratified_group_kfold(ds.groups, ds.tagsis, n_splits=5, seed=SEED + rep)
        for k, (tr, va) in enumerate(folds, 1):
            pp = Preprocessor(CFG["preprocess"]).fit(ds.X[tr])
            S_tr, S_va = pp.transform(ds.X[tr]), pp.transform(ds.X[va])
            cache = {}
            for m in MODELS:
                nc, kw = spec(m)
                if nc is None:
                    A, B = S_tr, S_va
                else:
                    if nc not in cache:
                        pf = PLSFeatures(nc).fit(S_tr, ds.y[tr])
                        cache[nc] = (pf.transform(S_tr), pf.transform(S_va))
                    A, B = cache[nc]
                mod = build_model(m, A.shape[1], ds.y.shape[1], seed=SEED, **kw).fit(A, ds.y[tr])
                preds[m][rep, va] = mod.predict(B)
            print(f"  repeat {rep + 1} fold {k}/5 done", flush=True)
    np.savez_compressed(ROOT / "results_rigor" / "r08_oof_predictions.npz",
                        y=ds.y, groups=ds.groups, tagsis=ds.tagsis,
                        **{f"pred_{m}": preds[m] for m in MODELS})
    return preds


def main():
    ds, tr, te = tahini_split()
    print(f"pooled out-of-fold LOD over {REPEATS} x 5 grouped folds "
          f"({len(np.unique(ds.groups))} samples, "
          f"{len({g for g in ds.groups if '_' not in g})} authentic lots as blanks)")
    preds = out_of_fold(ds)

    adult_true = 100 - ds.y[:, 0]
    blank = ds.tagsis == 0
    lots = np.array([g.split("_")[0] for g in ds.groups])
    blank_lots = np.unique(lots[blank])
    samples = np.unique(ds.groups)
    rng = np.random.RandomState(SEED)
    idx_lot = {l: np.where(blank & (lots == l))[0] for l in blank_lots}
    idx_smp = {s: np.where(ds.groups == s)[0] for s in samples}
    draws = []
    for _ in range(N_BOOT):
        b = np.concatenate([idx_lot[l] for l in rng.choice(blank_lots, len(blank_lots), True)])
        a = np.concatenate([idx_smp[s] for s in rng.choice(samples, len(samples), True)])
        draws.append((b, a))

    rows = []
    for m in MODELS:
        per_rep = []
        for rep in range(REPEATS):
            p = 100 - preds[m][rep][:, 0]
            per_rep.append(lod_loq(adult_true, p, blank)[0])
        p = 100 - np.nanmean(preds[m][:, :, 0], axis=0)
        lod, loq = lod_loq(adult_true, p, blank)
        r2 = r2_score(ds.y[:, 0], 100 - p)
        bl, bq = [], []
        for b, a in draws:
            sigma = float(np.std(p[b]))
            slope = float(np.polyfit(adult_true[a], p[a], 1)[0])
            slope = slope if abs(slope) > 1e-6 else 1.0
            bl.append(3.3 * sigma / abs(slope)); bq.append(10.0 * sigma / abs(slope))
        row = dict(model=m, oof_R2=r2, LOD=lod, LOD_lo=np.percentile(bl, 2.5),
                   LOD_hi=np.percentile(bl, 97.5), LOQ=loq, LOQ_lo=np.percentile(bq, 2.5),
                   LOQ_hi=np.percentile(bq, 97.5),
                   LOD_per_repeat=";".join(f"{v:.2f}" for v in per_rep))
        rows.append(row)
        print(f"  {m:4s} out-of-fold R2 {r2:.3f}   LOD {lod:6.2f} % "
              f"[{row['LOD_lo']:.2f}, {row['LOD_hi']:.2f}]   LOQ {loq:6.2f} %", flush=True)
        assert row["LOD_lo"] <= lod <= row["LOD_hi"], f"CI does not bracket the estimate for {m}"
    save(pd.DataFrame(rows), "r08_lod_oof.csv")


if __name__ == "__main__":
    main()
