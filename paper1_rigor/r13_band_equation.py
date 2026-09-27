"""The calibration equation written in named absorbance bands instead of latent variables.

Motivation
----------
The equation reported so far is a function of PLS scores x_1..x_n. Each score is a dense linear
combination of 1762 (tahini) or 103 (mango) channels, so evaluating the equation still requires
n x p unpublished loadings. In Lipton's terms the model is simulatable but not decomposable, and
that is the weakest point of the interpretability claim: a reviewer can fairly answer that a PLS
model is *also* one linear equation, and that neither is readable at the level of chemistry.

This experiment removes the objection at the source. A small set of individual channels is
selected inside the training folds, and every model -- including the KAN -- is fitted on those
channels alone, so the resulting equation reads

    tahini fraction (%) = f1(A_3009) + f2(A_2922) + f3(A_1744) + ...

with each argument a single named vibrational band.

Protocol
--------
* Variable selection (PLS-VIP or mutual information) is fitted on the training rows of each inner
  fold only, never on the whole training partition and never on the test set. The score vector is
  computed ONCE per (fold, method) and the whole k grid is read off it, which is cheaper than and
  exactly equivalent to recomputing it for every k.
* A minimum spectral separation is enforced so that the selected channels are distinct bands
  rather than neighbouring points of one band.
* (method, k) is chosen for EACH model family separately by grouped inner cross-validation on the
  training partition; the external test set is used once, at the end.
* Only the additive KAN -- the variant that actually yields a per-band equation -- is carried
  through the inner sweep. The wider variants are evaluated at the selected configuration on the
  external test, where they act as the capacity control.
* A LINEAR model on exactly the same channels is carried throughout. It is the control that
  decides whether interpretable non-linearity earns its place: if the KAN does not beat ordinary
  least squares on the identical inputs, the non-linearity is not doing any work and the paper
  says so.
* The stability of the selected band set across folds is logged, because a band equation is only
  chemically meaningful if the same bands keep being chosen.
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score

from kanfood.metrics import normalize_to_100
from kanfood.preprocess import Preprocessor
from kanfood.split import stratified_group_kfold
from paper1_rigor.common import TAHINI, MANGO, SEED, tahini_split, mango_split, save
from paper1_rigor.equation import extract

warnings.filterwarnings("ignore")

K_GRID = [4, 6, 8, 12, 16]
METHODS = ["vip", "mi"]
INNER_FOLDS = 3        # the inner sweep only SELECTS (method, k); the external evaluation below
INNER_SEEDS = [42]     # is the one that gets reported, and it keeps its full budget
FINAL_SEEDS = [42, 7, 2024, 100, 13]
# The inner sweep uses the published step counts; the external evaluation trains longer.
INNER_STEPS = dict(steps=250, prune_steps=60, refit_steps=200)
FINAL_STEPS = dict(steps=400, prune_steps=100, refit_steps=200)

# Chemically anchored bands, reported as a separate variant rather than tuned.
# Lipid assignments follow Guillen and Cabo (1997); amide I/II follow Barth (2007).
TAHINI_ANCHORS = (3008.0, 2922.0, 2853.0, 1743.0, 1653.0, 1465.0, 1163.0, 722.0)
# Short-wave NIR assignments follow Subedi and Walsh (2011): O-H and C-H overtone regions.
MANGO_ANCHORS = (700.0, 750.0, 800.0, 840.0, 880.0, 910.0, 940.0, 970.0)


def vip_scores(S, y, n_components=10):
    pls = PLSRegression(n_components=min(n_components, S.shape[1], S.shape[0] - 1)).fit(S, y)
    t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    wn = w / np.sqrt((w ** 2).sum(axis=0))
    return np.sqrt(p * ((wn ** 2) @ s).ravel() / s.sum())


def mi_scores(S, y, seed=SEED):
    """Mutual information, evaluated on a subsampled axis and interpolated back (speed)."""
    step = max(1, S.shape[1] // 400)
    idx = np.arange(0, S.shape[1], step)
    v = mutual_info_regression(S[:, idx], y[:, 0], random_state=seed)
    return np.interp(np.arange(S.shape[1]), idx, v)


def score_vector(S_tr, y_tr, method):
    return vip_scores(S_tr, y_tr) if method == "vip" else mi_scores(S_tr, y_tr)


def select(score, axis, k, min_sep, anchors=()):
    """Top-k channels by a precomputed score, subject to a minimum spectral separation."""
    chosen = [int(np.argmin(np.abs(axis - a))) for a in anchors][:k]
    for i in np.argsort(-score):
        if len(chosen) >= k:
            break
        if all(abs(axis[i] - axis[c]) >= min_sep for c in chosen):
            chosen.append(int(i))
    return sorted(chosen[:k], key=lambda i: -axis[i])


def _post(p, normalise):
    p = np.asarray(p).reshape(len(p), -1)
    return (normalize_to_100(p) if normalise else p)[:, 0]


def fit_predict(name, A_tr, y_tr, A_te, normalise, seed):
    """Every model that consumes the selected channels directly."""
    if name == "OLS":
        return _post(LinearRegression().fit(A_tr, y_tr).predict(A_te), normalise)
    if name == "Ridge":
        return _post(RidgeCV(alphas=np.logspace(-4, 4, 25)).fit(A_tr, y_tr).predict(A_te), normalise)
    if name == "PLS":
        n = min(A_tr.shape[1], 10)
        return _post(PLSRegression(n_components=n).fit(A_tr, y_tr).predict(A_te), normalise)
    if name == "EBM":
        from interpret.glassbox import ExplainableBoostingRegressor
        m = ExplainableBoostingRegressor(random_state=seed, interactions=0, n_jobs=1).fit(A_tr, y_tr[:, 0])
        return m.predict(A_te)
    if name == "EBM+int":
        from interpret.glassbox import ExplainableBoostingRegressor
        m = ExplainableBoostingRegressor(random_state=seed, interactions=5, n_jobs=1).fit(A_tr, y_tr[:, 0])
        return m.predict(A_te)
    if name == "GAM":
        from pygam import LinearGAM
        return LinearGAM().gridsearch(A_tr, y_tr[:, 0], progress=False).predict(A_te)
    raise ValueError(name)


DIRECT_MODELS = ["OLS", "Ridge", "PLS", "EBM", "EBM+int", "GAM"]
KAN_INNER = {"KAN additive": ()}
KAN_ALL = {"KAN additive": (), "KAN (2,)": (2,), "KAN (3,)": (3,)}


def evaluate_block(A_tr, y_tr, A_te, y_te1, normalise, seed, kan_variants, steps=None):
    out = []
    for name in DIRECT_MODELS:
        try:
            r2 = float(r2_score(y_te1, fit_predict(name, A_tr, y_tr, A_te, normalise, seed)))
        except Exception:
            r2 = float("nan")
        out.append(dict(model=name, R2=r2, n_terms=np.nan, n_vars=A_tr.shape[1], equation="",
                        fidelity=np.nan))
    for name, width in kan_variants.items():
        try:
            r = extract(A_tr, y_tr, A_te, y_te1, normalise=normalise, seed=seed,
                        width=width, full_batch_symbolic=True, **(steps or FINAL_STEPS))
            out.append(dict(model=name + " spline", R2=r["r2_spline"], n_terms=np.nan,
                            n_vars=A_tr.shape[1], equation="", fidelity=np.nan))
            out.append(dict(model=name + " equation", R2=r["r2_symbolic"], n_terms=r["n_terms"],
                            n_vars=r["n_vars"], equation=r.get("equation", ""),
                            fidelity=r["r2_fidelity"]))
        except Exception as e:
            out.append(dict(model=name + " equation", R2=float("nan"), n_terms=np.nan,
                            n_vars=A_tr.shape[1], equation="FAILED " + type(e).__name__,
                            fidelity=np.nan))
    return out


def run(label, splitter, cfg, min_sep, normalise, anchors):
    ds, tr, te = splitter()
    axis = ds.wavenumbers
    y_tr_all, y_te1 = ds.y[tr], ds.y[te, 0]
    folds = stratified_group_kfold(ds.groups[tr], ds.tagsis[tr], n_splits=INNER_FOLDS,
                                   seed=SEED)
    rows, sel_log = [], []

    print("\n=== " + label + ": band-equation sweep ===", flush=True)
    print("  -- inner grouped CV on the training partition --", flush=True)
    for fi, (a, b) in enumerate(folds, 1):
        pp = Preprocessor(cfg["preprocess"]).fit(ds.X[tr[a]])
        S_a, S_b = pp.transform(ds.X[tr[a]]), pp.transform(ds.X[tr[b]])
        for method in METHODS:
            t0 = time.time()
            sc = score_vector(S_a, ds.y[tr[a]], method)
            for k in K_GRID:
                idx = select(sc, axis, k, min_sep)
                for seed in INNER_SEEDS:
                    blk = evaluate_block(S_a[:, idx], ds.y[tr[a]], S_b[:, idx], ds.y[tr[b], 0],
                                         normalise, seed, KAN_INNER, steps=INNER_STEPS)
                    for o in blk:
                        o.update(k=k, method=method,
                                 bands=";".join(format(axis[i], ".0f") for i in idx))
                        rows.append(dict(dataset=label, stage="inner_cv", fold=fi, seed=seed, **o))
                sel_log.append(dict(dataset=label, method=method, k=k, fold=fi,
                                    bands=";".join(format(axis[i], ".1f") for i in idx)))
            print("    fold " + str(fi) + " " + method + ": " + str(len(K_GRID)) +
                  " k values in " + format(time.time() - t0, ".0f") + "s", flush=True)
            save(pd.DataFrame(rows), "r13_" + label + "_partial.csv")

    inner = pd.DataFrame([r for r in rows if r["stage"] == "inner_cv"])
    piv = inner.groupby(["model", "method", "k"])["R2"].mean().reset_index()
    chosen = piv.sort_values("R2", ascending=False).groupby("model").head(1).set_index("model")
    print("  -- selected (method, k) per model --", flush=True)
    print(chosen.round(4).to_string(), flush=True)
    save(piv, "r13_" + label + "_inner_grid.csv")

    print("  -- external test at each selected configuration --", flush=True)
    pp = Preprocessor(cfg["preprocess"]).fit(ds.X[tr])
    S_tr, S_te = pp.transform(ds.X[tr]), pp.transform(ds.X[te])
    cache = {m: score_vector(S_tr, y_tr_all, m) for m in METHODS}
    for method, k in sorted({(r["method"], int(r["k"])) for _, r in chosen.iterrows()}):
        idx = select(cache[method], axis, k, min_sep)
        bands = ";".join(format(axis[i], ".0f") for i in idx)
        for seed in FINAL_SEEDS:
            blk = evaluate_block(S_tr[:, idx], y_tr_all, S_te[:, idx], y_te1, normalise, seed,
                                 KAN_ALL)
            for o in blk:
                o.update(k=k, method=method, bands=bands)
                rows.append(dict(dataset=label, stage="external", fold=np.nan, seed=seed, **o))
        print("    method=" + method + " k=" + str(k) + " done  bands=" + bands, flush=True)

    idx = select(cache["vip"], axis, len(anchors), min_sep, anchors=anchors)
    bands = ";".join(format(axis[i], ".0f") for i in idx)
    for seed in FINAL_SEEDS:
        blk = evaluate_block(S_tr[:, idx], y_tr_all, S_te[:, idx], y_te1, normalise, seed, KAN_ALL)
        for o in blk:
            o.update(k=len(anchors), method="anchored", bands=bands)
            rows.append(dict(dataset=label, stage="external_anchored", fold=np.nan, seed=seed, **o))
    print("    chemistry-anchored (" + str(len(anchors)) + " bands) done  bands=" + bands,
          flush=True)
    return rows, sel_log


DATASETS = {"tahini": (tahini_split, TAHINI, 15.0, True, TAHINI_ANCHORS),
            "mango": (mango_split, MANGO, 9.0, False, MANGO_ANCHORS)}


def main():
    t0 = time.time()
    import sys as _sys
    only = [a for a in _sys.argv[1:] if a in DATASETS]
    if only:                                   # allow the two datasets to run as separate jobs
        allrows, allsel = [], []
        for name in only:
            r, s_ = run(name, *DATASETS[name])
            allrows += r
            allsel += s_
            save(pd.DataFrame(r), "r13_band_equation_raw_" + name + ".csv")
        df = pd.DataFrame(allrows)
        save(pd.DataFrame(allsel), "r13_band_selection_log_" + "_".join(only) + ".csv")
        summ = (df[df.stage != "inner_cv"]
                .groupby(["dataset", "stage", "model", "method", "k"])
                .agg(R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
                     terms_median=("n_terms", "median"), vars_median=("n_vars", "median"),
                     fidelity_mean=("fidelity", "mean"))
                .reset_index())
        save(summ, "r13_band_equation_summary_" + "_".join(only) + ".csv")
        print(summ.to_string(index=False))
        print("total " + format(time.time() - t0, ".0f") + "s")
        return
    rows, sel = run("tahini", tahini_split, TAHINI, 15.0, True, TAHINI_ANCHORS)
    save(pd.DataFrame(rows), "r13_band_equation_raw_tahini.csv")
    r2, s2 = run("mango", mango_split, MANGO, 9.0, False, MANGO_ANCHORS)
    df = pd.DataFrame(rows + r2)
    save(df, "r13_band_equation_raw.csv")
    save(pd.DataFrame(sel + s2), "r13_band_selection_log.csv")
    summ = (df[df.stage != "inner_cv"]
            .groupby(["dataset", "stage", "model", "method", "k"])
            .agg(R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
                 terms_median=("n_terms", "median"), vars_median=("n_vars", "median"),
                 fidelity_mean=("fidelity", "mean"))
            .reset_index())
    save(summ, "r13_band_equation_summary.csv")
    print("\n", summ.to_string(index=False))
    print("\ntotal " + format(time.time() - t0, ".0f") + "s")


if __name__ == "__main__":
    main()
