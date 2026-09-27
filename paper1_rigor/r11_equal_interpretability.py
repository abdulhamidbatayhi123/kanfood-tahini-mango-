"""Equal-interpretability-budget comparison: PLS at n latent variables vs the KAN at the same n.

Why this script exists
----------------------
The manuscript's central comparison is against PLS, which is also interpretable. As reported so
far the comparison is unequal in PLS's favour: on mango the PLS baseline uses 24 latent variables
while the KAN equation is written in 12. Interpretability for both models is bought with latent
variables, so the honest comparison holds the number of latent variables fixed and asks which
model extracts more from them.

At each n, on exactly the same partition and preprocessing, we report:
  * PLS with n latent variables (the standard chemometric model),
  * the spline KAN on the same n PLS scores,
  * the KAN's closed-form equation on the same n PLS scores, under BOTH the published extraction
    protocol and the corrected one, so that the effect of the protocol is separable from the
    effect of n,
and, for each, an explicit count of what a reader must be handed to evaluate the model:

  `disclosure` = the number of real numbers that have to be published for a third party to
  compute a prediction from a raw spectrum. For PLS at n components that is the n x p loading
  matrix plus n regression coefficients plus the centring/scaling vectors; for the KAN equation
  it is the same projection cost plus the constants in the printed expression. Reporting it makes
  the "both are just one equation" objection quantitative instead of rhetorical.

Everything is imported from the published pipeline (`kanfood.*`) and the shared extraction module
(`paper1_rigor.equation`); only the sweep is new.
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score

from kanfood.metrics import normalize_to_100
from paper1_rigor.common import (ckpt, TAHINI, MANGO, SEED, tahini_split, mango_split, project, save)
from paper1_rigor.equation import extract

warnings.filterwarnings("ignore")

N_GRID = [2, 4, 6, 8, 10, 12, 16, 20, 24]
SEEDS = [42, 7, 2024]
PROTOCOLS = {"published": dict(legacy=True),
             "corrected": dict(full_batch_symbolic=True, refit_steps=200)}


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def n_constants(expr_str):
    """Count the numeric constants a reader must be given to evaluate the printed expression."""
    import re
    if not expr_str or expr_str.startswith("FAILED"):
        return np.nan
    return len(re.findall(r"(?<![\w.])\d+\.?\d*(?:[eE][+-]?\d+)?", expr_str))


def run(label, splitter, cfg, normalise):
    ds, tr, te = splitter()
    y_te1, p = ds.y[te, 0], ds.X.shape[1]
    rows = []
    print(f"\n=== {label}: equal-interpretability-budget sweep (p={p} channels) ===", flush=True)
    for n in N_GRID:
        if n >= min(len(tr), p):
            continue
        t0 = time.time()
        Z_tr, Z_te, _, Sp_tr, Sp_te = project(ds, tr, te, cfg["preprocess"], n)

        pls = PLSRegression(n_components=n).fit(Sp_tr, ds.y[tr])
        q = np.asarray(pls.predict(Sp_te)).reshape(len(Z_te), -1)
        q = normalize_to_100(q) if normalise else q
        rows.append(dict(dataset=label, n_latent=n, model="PLS", protocol="-", seed=SEED,
                         R2=float(r2_score(y_te1, q[:, 0])), RMSEP=rmse(y_te1, q[:, 0]),
                         eq_terms=np.nan, eq_constants=np.nan,
                         disclosure=n * p + n * ds.y.shape[1] + 2 * p, equation=""))

        for pname, kw in PROTOCOLS.items():
            for seed in SEEDS:
                r = extract(Z_tr, ds.y[tr], Z_te, y_te1, normalise=normalise, seed=seed, **kw)
                r.pop("model", None)
                for tag, key in [("KAN spline", "r2_spline"), ("KAN equation", "r2_symbolic")]:
                    eq = r.get("equation", "") if tag == "KAN equation" else ""
                    nc_ = n_constants(eq) if eq else np.nan
                    rows.append(dict(
                        dataset=label, n_latent=n, model=tag, protocol=pname, seed=seed,
                        R2=r[key], RMSEP=np.nan, eq_terms=(r["n_terms"] if eq else np.nan),
                        eq_constants=nc_,
                        disclosure=(n * p + 2 * p + (nc_ if eq else np.nan)),
                        equation=eq))
        sel = [x for x in rows if x["n_latent"] == n]
        got = {m: np.nanmean([x["R2"] for x in sel if x["model"] == m and
                              x["protocol"] in ("-", "corrected")])
               for m in ("PLS", "KAN spline", "KAN equation")}
        print(f"  n={n:<3} PLS {got['PLS']:.4f} | KAN spline {got['KAN spline']:.4f} | "
              f"KAN eq (corrected) {got['KAN equation']:.4f}   ({time.time() - t0:.0f}s)",
              flush=True)
    return rows


def main():
    t0 = time.time()
    rows = run("tahini", tahini_split, TAHINI, True)
    ckpt(rows, "r11_equal_interpretability_tahini.csv")
    rows += run("mango", mango_split, MANGO, False)
    df = pd.DataFrame(rows)
    save(df, "r11_equal_interpretability_raw.csv")
    summ = (df.groupby(["dataset", "n_latent", "model", "protocol"])
              .agg(R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
                   terms_median=("eq_terms", "median"),
                   constants_median=("eq_constants", "median"),
                   disclosure_median=("disclosure", "median"))
              .reset_index())
    save(summ, "r11_equal_interpretability_summary.csv")
    print("\n", summ.to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
