"""Limit of detection and quantification with cluster-bootstrap confidence intervals.

The point estimate follows the IUPAC 3-sigma convention exactly as in the published analysis
(LOD = 3.3 sigma / S, LOQ = 10 sigma / S, sigma from the authentic samples' predicted adulterant
fraction, S the calibration slope). The interval resamples whole physical samples and recomputes
BOTH sigma and the slope inside every resample, so the interval reflects the same estimator as the
point value and the replicate structure of the data.
"""
import json
import numpy as np
import pandas as pd

from paper1_rigor.common import SEED, tahini_split, save, OUT as ROUT
from paper1_rigor.common import ROOT

N_BOOT = 2000


def lod_loq(adult_true, adult_pred, pure):
    if pure.sum() < 2:
        return np.nan, np.nan
    sigma = float(np.std(adult_pred[pure]))
    if len(np.unique(adult_true)) < 2:
        return np.nan, np.nan
    slope = float(np.polyfit(adult_true, adult_pred, 1)[0])
    slope = slope if abs(slope) > 1e-6 else 1.0
    return 3.3 * sigma / abs(slope), 10.0 * sigma / abs(slope)


def main():
    a = np.load(ROOT / "results_phase1" / "phase1_artifacts.npz", allow_pickle=True)
    meta = json.loads((ROOT / "results_phase1" / "phase1_meta.json").read_text(encoding="utf-8"))
    ds, tr, te = tahini_split()
    groups_te = ds.groups[te]
    assert len(groups_te) == len(a["y_test"]), "artifact/test-set mismatch"

    adult_true = 100 - a["y_test"][:, 0]
    pure = a["tagsis_test"] == 0
    uniq = np.unique(groups_te)
    idx_by_group = {g: np.where(groups_te == g)[0] for g in uniq}
    rng = np.random.RandomState(SEED)
    draws = [np.concatenate([idx_by_group[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
             for _ in range(N_BOOT)]

    rows = []
    for m in meta["model_names"]:
        adult_pred = 100 - a[f"pred_{m}"][:, 0]
        lod, loq = lod_loq(adult_true, adult_pred, pure)
        bl, bq = [], []
        for idx in draws:
            l, q = lod_loq(adult_true[idx], adult_pred[idx], pure[idx])
            if np.isfinite(l):
                bl.append(l); bq.append(q)
        rows.append(dict(model=m, LOD=lod, LOD_lo=np.percentile(bl, 2.5), LOD_hi=np.percentile(bl, 97.5),
                         LOQ=loq, LOQ_lo=np.percentile(bq, 2.5), LOQ_hi=np.percentile(bq, 97.5),
                         n_boot_valid=len(bl)))
        print(f"  {m:4s} LOD = {lod:6.2f} % [{rows[-1]['LOD_lo']:.2f}, {rows[-1]['LOD_hi']:.2f}]   "
              f"LOQ = {loq:6.2f} % [{rows[-1]['LOQ_lo']:.2f}, {rows[-1]['LOQ_hi']:.2f}]")
        assert rows[-1]["LOD_lo"] <= lod <= rows[-1]["LOD_hi"], f"CI does not bracket the point estimate for {m}"
    save(pd.DataFrame(rows), "r05_lod_ci.csv")


if __name__ == "__main__":
    main()
