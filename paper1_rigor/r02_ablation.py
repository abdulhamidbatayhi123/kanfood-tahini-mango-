"""Architecture and preprocessing ablation for the KAN.

Each configuration differs from the published one in a single respect and is scored by grouped
five-fold cross-validation on the TRAINING partition only, with preprocessing and the PLS
projection refitted inside every fold. Because the number of configurations makes the full
protocol prohibitive, this uses a single grouped five-fold and must be read as change relative to
its own baseline, not against the headline tables.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from kanfood.preprocess import Preprocessor
from kanfood.features import PLSFeatures
from kanfood.models import build_model
from kanfood.split import stratified_group_kfold
from kanfood.metrics import regression_metrics
from paper1_rigor.common import TAHINI, MANGO, SEED, tahini_split, mango_split, save


def cv_score(ds, tr, preprocess, nc, folds, **kan_kw):
    """Grouped-CV R2 with preprocessing and PLS refitted inside every fold."""
    X, y = ds.X[tr], ds.y[tr]
    scores = []
    for a, b in folds:
        pp = Preprocessor(preprocess).fit(X[a])
        Sa, Sb = pp.transform(X[a]), pp.transform(X[b])
        pf = PLSFeatures(nc).fit(Sa, y[a])
        Za, Zb = pf.transform(Sa), pf.transform(Sb)
        m = build_model("KAN", Za.shape[1], y.shape[1], seed=SEED, **kan_kw).fit(Za, y[a])
        scores.append(regression_metrics(y[b], m.predict(Zb), ds.target_names)["R2_mean"])
    return float(np.mean(scores)), float(np.std(scores))


def grid_for(cfg, label):
    base = dict(width_hidden=cfg["kan_width"], grid=5, k=3, lamb=1e-3)
    pre = cfg["preprocess"]
    widths = [(2,), (3,), (4,), (5,)] if label == "tahini" else [(3,), (8, 4), (16, 8), (32, 16)]
    out = [("published", pre, dict(base))]
    for g in (3, 10, 20):
        out.append((f"grid G={g}", pre, {**base, "grid": g}))
    for k in (2, 4):
        out.append((f"spline order k={k}", pre, {**base, "k": k}))
    for w in widths:
        if tuple(w) != tuple(cfg["kan_width"]):
            out.append((f"hidden width {w}", pre, {**base, "width_hidden": w}))
    for lam in (0.0, 1e-2, 1e-1):
        out.append((f"lambda={lam:g}", pre, {**base, "lamb": lam}))
    for p in ("snv", "sg1", "snv+sg1", "msc"):
        if p != pre:
            out.append((f"preprocessing {p}", p, dict(base)))
    return out


def run(label, ds, tr, cfg):
    folds = stratified_group_kfold(ds.groups[tr], ds.tagsis[tr], n_splits=5, seed=SEED)
    nc = cfg["kan_nc"]
    rows = []
    print(f"\n=== {label}: {len(grid_for(cfg, label))} configurations ===")
    for name, pre, kw in grid_for(cfg, label):
        mu, sd = cv_score(ds, tr, pre, nc, folds, **kw)
        rows.append(dict(dataset=label, config=name, preprocess=pre,
                         width=str(kw["width_hidden"]), grid=kw["grid"], k=kw["k"],
                         lamb=kw["lamb"], n_components=nc, cv_R2=mu, cv_R2_sd=sd))
        print(f"  {name:26s} R2 = {mu:7.4f} +/- {sd:.4f}")
    base = [r for r in rows if r["config"] == "published"][0]["cv_R2"]
    for r in rows:
        r["delta_vs_published"] = r["cv_R2"] - base
    return rows


def main():
    ds, tr, te = tahini_split()
    rows = run("tahini", ds, tr, TAHINI)
    dm, mtr, mte = mango_split()
    rows += run("mango", dm, mtr, MANGO)
    df = pd.DataFrame(rows)
    save(df, "r02_ablation.csv")


if __name__ == "__main__":
    main()
