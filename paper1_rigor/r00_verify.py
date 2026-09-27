"""Guard: reproduce the published headline numbers and report the exact trainable-parameter
counts used by every later experiment. Nothing else runs until this matches Tables 1 and 2."""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from kanfood.models import build_model
from paper1_rigor.common import (TAHINI, MANGO, SEED, tahini_split, mango_split, project,
                                 y_primary, save)

PUBLISHED = {  # (dataset, model) -> external-test R2 printed in the manuscript
    ("tahini", "KAN"): 0.9869, ("tahini", "MLP"): 0.9882, ("tahini", "PLS"): 0.9943,
    ("mango", "KAN"): 0.8629, ("mango", "MLP"): 0.8615, ("mango", "PLS"): 0.8452,
}


def run_one(ds, tr, te, cfg, model, kind):
    if model == "PLS":
        from kanfood.preprocess import Preprocessor
        pp = Preprocessor(cfg["preprocess"]).fit(ds.X[tr])
        Sp_tr, Sp_te = pp.transform(ds.X[tr]), pp.transform(ds.X[te])
        m = build_model("PLS", Sp_tr.shape[1], ds.y.shape[1], seed=SEED,
                        n_components=cfg["pls_nc"]).fit(Sp_tr, ds.y[tr])
        pred = m.predict(Sp_te)
    else:
        nc = cfg["kan_nc"] if model == "KAN" else cfg["mlp_nc"]
        Z_tr, Z_te, _, _, _ = project(ds, tr, te, cfg["preprocess"], nc)
        kw = dict(width_hidden=cfg["kan_width"], grid=5) if model == "KAN" else dict(hidden=cfg["mlp_hidden"])
        m = build_model(model, Z_tr.shape[1], ds.y.shape[1], seed=SEED, **kw).fit(Z_tr, ds.y[tr])
        pred = m.predict(Z_te)
    return r2_score(y_primary(ds, te), pred[:, 0]), m.n_params()


def main():
    rows = []
    for kind, loader, cfg in [("tahini", tahini_split, TAHINI), ("mango", mango_split, MANGO)]:
        ds, tr, te = loader()
        print(f"\n[{kind}] train {len(tr)} / test {len(te)} spectra; "
              f"{len(np.unique(ds.groups[tr]))}/{len(np.unique(ds.groups[te]))} groups")
        for model in ["PLS", "MLP", "KAN"]:
            r2, npar = run_one(ds, tr, te, cfg, model, kind)
            pub = PUBLISHED[(kind, model)]
            ok = abs(r2 - pub) < 0.004
            print(f"  {model:4s} R2={r2:.4f} (published {pub:.4f}) {'OK' if ok else '*** MISMATCH ***'}"
                  f"  params={npar}")
            rows.append(dict(dataset=kind, model=model, R2=r2, R2_published=pub,
                             reproduces=ok, n_params=npar))
    df = pd.DataFrame(rows)
    save(df, "r00_reproduction.csv")
    assert df["reproduces"].all(), "pipeline does not reproduce the published table -- stop"
    print("\nAll published numbers reproduced.")


if __name__ == "__main__":
    main()
