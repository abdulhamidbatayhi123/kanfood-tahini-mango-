"""Does the closed form survive a distribution shift as well as the network it came from?

The question
------------
The paper makes one strong positive claim and one honest negative one, and until now nothing
connected them.

  * Positive (`r19`): an already-elementary edge basis makes the printed equation *exact* --
    fidelity 0.999996-1.000000 against 0.983 for the corrected conversion on latent scores, and
    0.779 on named bands -- at no cost in accuracy.
  * Negative (`r04`): every model in this study degrades when a whole tahini lot or a whole mango
    season is withheld, and the KAN is not the most robust of them.

Both are measured on the model. Neither says what happens to the *equation* -- the object the
paper actually asks a reader to use -- when it meets material the network was never trained on.
That matters because a printed calibration is far more likely to be applied off-distribution than
a model that lives in a repository: it is a formula on a page, and nothing stops anyone typing new
absorbances into it.

There is also a specific, testable prediction to separate the two extraction routes. For an
elementary basis, printing is a change of notation, so equation and network are the *same
function* and their agreement cannot depend on where they are evaluated -- fidelity must stay at 1
however far the inputs move. For `auto_symbolic`, each edge's substituted function was chosen and
fitted against activations seen during training; off-distribution inputs can fall where the
substitute and the spline diverge, so its fidelity is free to decay. If that happens it is a
substantive argument for the elementary basis, independent of accuracy: the converted equation
would be an approximation whose quality is itself unknown outside the calibration range.

What is measured
----------------
The folds of `r04`, unchanged: leave-one-tahini-lot-out (five independent lots) and
leave-one-mango-season-out (four seasons). In every fold, on held-out material:

    R2 of the network · R2 of the printed equation · FIDELITY of the equation to its own network

for three routes -- the elementary basis (exact by construction), the pykan B-spline chain with
the corrected conversion of `r12`, and ordinary least squares as the linear control -- on both
input representations the paper uses. Band selection and every preprocessing and projection fit
happen inside each fold's training partition, so a withheld lot or season is never seen.

The elementary basis is fixed in advance from `r19`'s inner-CV selection on the published training
partition -- Chebyshev for latent scores, raw powers for named bands -- and is not re-selected
per fold, so nothing here is tuned on the material being transferred to.

Scoring convention, and why it is not the pipeline's
----------------------------------------------------
Everything here is scored on the FIRST output alone, without the sum-to-100 normalisation the
multi-output pipeline applies. Two reasons, and they point the same way. First, a printed equation
gives one number -- the tahini fraction -- so a practitioner reading it off a page has nothing to
normalise against; scoring it as though the other two components were also available would measure
something nobody can do. Second, normalising the equation's output against the *network's* other
two outputs would make the equation's score depend on the network it was supposed to replace,
which is exactly the confound this experiment exists to avoid. Both routes are scored identically,
so the comparison between them -- which is the whole point -- is unaffected. Absolute values are
therefore slightly below the corresponding numbers in Section 3.7, which uses the pipeline
convention, and the two should not be read against each other.
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from kanfood.data import load_mango
from kanfood.features import PLSFeatures
from kanfood.preprocess import Preprocessor
from paper1_rigor import basis_kan as BK
from paper1_rigor.common import (ckpt, TAHINI, MANGO, SEED, tahini_split, mango_split, save)
from paper1_rigor.equation import extract
from paper1_rigor.r13_band_equation import score_vector, select

warnings.filterwarnings("ignore")

SEEDS = [42, 7, 2024]
BAND_K = 8
# Fixed in advance from r19's INNER-CV selection on the published training partition
# (r19_basis_inner_tahini.csv, stage 'inner_basis'): cheby wins on latent scores, poly on
# named bands. NOT the best-external arm -- choosing that would be selection on the test
# set -- and NOT re-selected against the held-out fold.
BASIS = {"PLS scores": "cheby", "selected bands": "poly"}
DEGREE, LAMB, PRUNE = 4, 1e-4, 0.05


def elementary(A_tr, y_tr, A_te, y_te1, basis, seed, names):
    """Fit an elementary-basis KAN, print it, and score network and equation on the same rows."""
    sx, sy = MinMaxScaler((-1, 1)).fit(A_tr), StandardScaler().fit(y_tr)
    Xtr, Xte = sx.transform(A_tr), np.clip(sx.transform(A_te), -1, 1)
    m = BK.AdditiveKAN(A_tr.shape[1], basis=basis, degree=DEGREE, n_out=y_tr.shape[1])
    BK.fit(m, Xtr, sy.transform(y_tr), lamb=LAMB, prune_frac=PRUNE, seed=seed)
    pred_scaled = BK.predict(m, Xte)
    # Output 0 only, put back on the original scale with that column's own centre and width.
    c0, s0 = sy.mean_[0], sy.scale_[0]
    net = pred_scaled[:, 0] * s0 + c0
    printed = BK.equation(m, var_names=names)
    try:
        ev = BK.evaluate(printed, Xte, var_names=names)
        fid = float(r2_score(pred_scaled[:, 0], ev))
        eq_r2 = float(r2_score(y_te1, ev * s0 + c0))
    except Exception:
        fid, eq_r2 = float("nan"), float("nan")
    return dict(R2_network=float(r2_score(y_te1, net)), R2_equation=eq_r2, fidelity=fid,
                n_terms=BK.n_terms(printed), n_vars=len(BK.retained(m)))


def one_fold(ds, tr, te, cfg, label, held, min_sep, rows):
    y_te1 = ds.y[te, 0]
    pp = Preprocessor(cfg["preprocess"]).fit(ds.X[tr])
    S_tr, S_te = pp.transform(ds.X[tr]), pp.transform(ds.X[te])

    reps = {}
    # tahini has no separate equation configuration; mango has the compact one (nc = 12)
    nc = cfg.get("kan_eq_nc") or cfg["kan_nc"]
    pf = PLSFeatures(nc).fit(S_tr, ds.y[tr])
    reps["PLS scores"] = (pf.transform(S_tr), pf.transform(S_te),
                          [f"x_{i + 1}" for i in range(nc)])
    idx = select(score_vector(S_tr, ds.y[tr], "vip"), ds.wavenumbers, BAND_K, min_sep)
    reps["selected bands"] = (S_tr[:, idx], S_te[:, idx],
                              [f"A_{ds.wavenumbers[i]:.0f}" for i in idx])

    for rep, (A_tr, A_te, names) in reps.items():
        for seed in SEEDS:
            t0 = time.time()
            r = elementary(A_tr, ds.y[tr], A_te, y_te1, BASIS[rep], seed, names)
            rows.append(dict(dataset=label, held_out=str(held), representation=rep,
                             route="elementary basis (" + BASIS[rep] + ")", seed=seed,
                             n_test=len(te), seconds=round(time.time() - t0, 1), **r))
            print(f"    {rep:14s} elementary  seed {seed:<5} net {r['R2_network']:7.4f}  "
                  f"eq {r['R2_equation']:7.4f}  fidelity {r['fidelity']:9.6f}", flush=True)

            t0 = time.time()
            try:
                # normalise=False for the same reason the elementary route is scored that
                # way: one printed equation yields one number. See the module docstring.
                e = extract(A_tr, ds.y[tr], A_te, y_te1, normalise=False, seed=seed,
                            width=(3,), steps=400, prune_steps=100, refit_steps=200,
                            full_batch_symbolic=True)
                r2 = dict(R2_network=e["r2_spline"], R2_equation=e["r2_symbolic"],
                          fidelity=e["r2_fidelity"], n_terms=e["n_terms"], n_vars=e["n_vars"])
            except Exception as ex:
                r2 = dict(R2_network=np.nan, R2_equation=np.nan, fidelity=np.nan,
                          n_terms=np.nan, n_vars=np.nan)
                print("      pykan chain FAILED: " + type(ex).__name__, flush=True)
            rows.append(dict(dataset=label, held_out=str(held), representation=rep,
                             route="B-spline + corrected conversion", seed=seed,
                             n_test=len(te), seconds=round(time.time() - t0, 1), **r2))
            print(f"    {rep:14s} conversion  seed {seed:<5} net {r2['R2_network']:7.4f}  "
                  f"eq {r2['R2_equation']:7.4f}  fidelity {r2['fidelity']:9.6f}", flush=True)

        # the linear control: exact by construction, so fidelity is 1 by definition
        lin = LinearRegression().fit(A_tr, ds.y[tr])
        pred = np.asarray(lin.predict(A_te)).reshape(len(A_te), -1)[:, 0]
        rows.append(dict(dataset=label, held_out=str(held), representation=rep,
                         route="OLS on the same inputs", seed=SEED, n_test=len(te), seconds=0.0,
                         R2_network=float(r2_score(y_te1, pred)),
                         R2_equation=float(r2_score(y_te1, pred)), fidelity=1.0,
                         n_terms=A_tr.shape[1] + 1, n_vars=A_tr.shape[1]))
        print(f"    {rep:14s} OLS                     "
              f"    {r2_score(y_te1, pred):7.4f}", flush=True)


def run(label, ds, fold_labels, cfg, min_sep):
    rows = []
    print(f"\n=== {label}: the equation under distribution shift ===", flush=True)
    for held in sorted(np.unique(fold_labels)):
        te = np.where(fold_labels == held)[0]
        tr = np.where(fold_labels != held)[0]
        assert set(ds.groups[tr]).isdisjoint(set(ds.groups[te])), "group leakage in the fold"
        print(f"\n  hold out {held}: train {len(tr)} / test {len(te)} spectra", flush=True)
        one_fold(ds, tr, te, cfg, label, held, min_sep, rows)
        ckpt(rows, f"r23_shift_{label}_partial.csv")
    return rows


def main():
    t0 = time.time()
    ds, _, _ = tahini_split()
    sources = np.array([g.split("_")[0] for g in ds.groups])
    rows = run("tahini", ds, sources, TAHINI, 15.0)
    save(pd.DataFrame(rows), "r23_shift_tahini.csv")

    dm, _, _ = mango_split()
    seasons = pd.read_csv(load_mango.__defaults__[0])["Season"].astype(str).to_numpy()
    rows += run("mango", dm, seasons, MANGO, 9.0)

    df = pd.DataFrame(rows)
    save(df, "r23_shift_raw.csv")
    summ = (df.groupby(["dataset", "representation", "route"])
              .agg(folds=("held_out", "nunique"), n=("R2_network", "size"),
                   net_mean=("R2_network", "mean"), net_min=("R2_network", "min"),
                   eq_mean=("R2_equation", "mean"), eq_min=("R2_equation", "min"),
                   fidelity_mean=("fidelity", "mean"), fidelity_min=("fidelity", "min"),
                   terms=("n_terms", "median"))
              .reset_index())
    summ["equation_minus_network"] = summ.eq_mean - summ.net_mean
    save(summ, "r23_shift_summary.csv")
    print("\n", summ.to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
