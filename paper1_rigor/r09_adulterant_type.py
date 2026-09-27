"""Which adulterant, not just how much.

The tahini design contains two distinct adulterants -- sunflower paste and peanut paste -- and every
model predicts a three-component composition, so the same calibration that answers "how much" also
answers "which". That question is not addressed anywhere in the quantification tables, which report
only the tahini fraction, and it is the part of this dataset that a one-adulterant design cannot ask.

Computed from the saved predictions of the published run; no model is retrained.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, f1_score, accuracy_score, mean_absolute_error

from paper1_rigor.common import ROOT, tahini_split, SEED, save
from paper1_rigor.figstyle import MODEL_ORDER

N_BOOT = 2000


def main():
    a = np.load(ROOT / "results_phase1" / "phase1_artifacts.npz", allow_pickle=True)
    meta = json.loads((ROOT / "results_phase1" / "phase1_meta.json").read_text(encoding="utf-8"))
    names = meta["target_names"]          # [tahini, peanut, sunflower]
    print("targets:", names)

    ds, tr, te = tahini_split()
    y = a["y_test"]
    groups = ds.groups[te]
    adulterated = a["tagsis_test"] == 1
    # true adulterant identity for the adulterated samples: 1 = peanut, 2 = sunflower
    true_id = np.where(y[:, 1] > y[:, 2], 1, 2)[adulterated]

    uniq = np.unique(groups)
    rng = np.random.RandomState(SEED)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    draws = [np.concatenate([idx_by_group[g] for g in rng.choice(uniq, len(uniq), True)])
             for _ in range(N_BOOT)]

    rows = []
    for m in MODEL_ORDER:
        p = a[f"pred_{m}"]
        pred_id = np.where(p[adulterated, 1] > p[adulterated, 2], 1, 2)
        acc = accuracy_score(true_id, pred_id)
        f1 = f1_score(true_id, pred_id, average="macro")
        boot = []
        for idx in draws:
            sub = idx[adulterated[idx]]
            if len(np.unique(true_id_full := np.where(y[sub, 1] > y[sub, 2], 1, 2))) < 1:
                continue
            pid = np.where(p[sub, 1] > p[sub, 2], 1, 2)
            boot.append(accuracy_score(true_id_full, pid))
        # Score the two adulterant fractions where the section says they are scored: on the 257
        # adulterated spectra. Until 2026-08-27 these four columns were computed over all 408 test
        # spectra, including the 151 authentic ones whose true peanut and sunflower fractions are
        # both zero -- which widens the target variance, inflates R2, and absorbs each model's
        # authentic-sample bias into the MAE. It also inverted the ordering: SVR's peanut R2 is
        # 0.973 on adulterated material and 0.936 over the whole partition. Both scopes are kept,
        # and the bias on authentic material is reported separately because that is the quantity
        # the whole-partition figures were confounding the fractions with.
        auth = ~adulterated
        rows.append(dict(
            model=m,
            R2_peanut=r2_score(y[adulterated, 1], p[adulterated, 1]),
            R2_sunflower=r2_score(y[adulterated, 2], p[adulterated, 2]),
            MAE_peanut=mean_absolute_error(y[adulterated, 1], p[adulterated, 1]),
            MAE_sunflower=mean_absolute_error(y[adulterated, 2], p[adulterated, 2]),
            R2_peanut_all=r2_score(y[:, 1], p[:, 1]),
            R2_sunflower_all=r2_score(y[:, 2], p[:, 2]),
            MAE_peanut_all=mean_absolute_error(y[:, 1], p[:, 1]),
            MAE_sunflower_all=mean_absolute_error(y[:, 2], p[:, 2]),
            bias_peanut_authentic=float(np.mean(p[auth, 1] - y[auth, 1])),
            bias_sunflower_authentic=float(np.mean(p[auth, 2] - y[auth, 2])),
            n_authentic_spectra=int(auth.sum()),
            identification_accuracy=acc,
            identification_acc_lo=float(np.percentile(boot, 2.5)),
            identification_acc_hi=float(np.percentile(boot, 97.5)),
            identification_macroF1=f1,
            n_adulterated_spectra=int(adulterated.sum()),
        ))
        print(f"  {m:4s} R2 peanut {rows[-1]['R2_peanut']:.3f}  R2 sunflower "
              f"{rows[-1]['R2_sunflower']:.3f}  identification accuracy {acc:.3f} "
              f"[{rows[-1]['identification_acc_lo']:.3f}, {rows[-1]['identification_acc_hi']:.3f}]")
    df = pd.DataFrame(rows)
    save(df, "r09_adulterant_type.csv")

    # Per-level breakdown. The single hold-out contains no blend below 16 % adulterant, so the
    # level-resolved statement is made on pooled out-of-fold predictions (every one of the 25 levels
    # from 4 % to 100 % appears, each predicted only by models that never saw that blend).
    oof_path = ROOT / "results_rigor" / "r08_oof_predictions.npz"
    if not oof_path.exists():
        print("  r08_oof_predictions.npz not written yet -- skipping the level-resolved analysis")
        return
    o = np.load(oof_path, allow_pickle=True)
    Y, TAG = o["y"], o["tagsis"]
    ad = TAG == 1
    lvl = np.round(100 - Y[:, 0]).astype(int)
    out = []
    for L in sorted(set(lvl[ad])):
        sel = ad & (lvl == L)
        row = dict(adulterant_level=L, n=int(sel.sum()))
        for m in MODEL_ORDER:
            P = np.nanmean(o[f"pred_{m}"], axis=0)
            tid = np.where(Y[sel, 1] > Y[sel, 2], 1, 2)
            pid = np.where(P[sel, 1] > P[sel, 2], 1, 2)
            row[f"{m}_accuracy"] = accuracy_score(tid, pid)
        out.append(row)
    save(pd.DataFrame(out), "r09_adulterant_type_by_level.csv")

    overall = []
    for m in MODEL_ORDER:
        P = np.nanmean(o[f"pred_{m}"], axis=0)
        tid = np.where(Y[ad, 1] > Y[ad, 2], 1, 2)
        pid = np.where(P[ad, 1] > P[ad, 2], 1, 2)
        overall.append(dict(model=m, source="pooled out-of-fold (all 25 levels)",
                            n=int(ad.sum()), accuracy=accuracy_score(tid, pid),
                            macroF1=f1_score(tid, pid, average="macro"),
                            R2_peanut=r2_score(Y[:, 1], P[:, 1]),
                            R2_sunflower=r2_score(Y[:, 2], P[:, 2])))
        print(f"  {m:4s} out-of-fold identification accuracy {overall[-1]['accuracy']:.3f} "
              f"over {ad.sum()} adulterated spectra, 25 levels")
    save(pd.DataFrame(overall), "r09_adulterant_type_oof.csv")


if __name__ == "__main__":
    main()
