"""Was the published preprocessing the right one? A widened, honestly-selected search.

Why this exists
---------------
The published pipeline fixes one preprocessing per dataset -- SNV for tahini, first-derivative
Savitzky-Golay for mango -- chosen from a grid that contained neither the SECOND derivative nor
any Savitzky-Golay window other than 11 points. Two independent reasons say that grid was too
narrow:

  * The ablation (`r02`) already found that on tahini the published SNV is NOT the best option:
    SNV+SG1 beat it by +0.040 R2 and SG1 alone by +0.034, at a fixed architecture.
  * On mango, every published calibration that beats the global PLS benchmark uses the SECOND
    derivative of absorbance over the trimmed 684-990 nm range. Anderson et al.'s global PLS
    (11 latent variables, second derivative) reports RMSEP 1.014 %; our pipeline's PLS reaches
    1.050 %, i.e. it reproduces that benchmark, so the gap to the better published models is not
    in the split or the data but somewhere in the model chain, and preprocessing is the first
    place to look.

Widening a search grid is only legitimate if the selection stays inside the training partition,
so that is enforced here: every configuration is scored by grouped inner cross-validation on the
training rows, and the external test set is touched once at the end, at the configuration the
inner CV chose.

Protocol
--------
Stage 1  A dense preprocessing x window x latent-dimension grid, scored with PLS, which is cheap
         and is also the model whose published benchmark we are trying to reproduce.
Stage 2  The best `N_CARRY` preprocessing configurations from stage 1 are re-scored with the
         expensive models (both KAN widths and the MLP), because the best transform for a linear
         model need not be the best for a non-linear one.
Stage 3  External test at each model's own inner-CV-selected configuration, several seeds, with
         the published configuration carried alongside so the change is visible rather than
         silently swapped in.
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from kanfood.features import PLSFeatures
from kanfood.metrics import normalize_to_100
from kanfood.models import build_model
from kanfood.preprocess import Preprocessor
from kanfood.split import stratified_group_kfold
from paper1_rigor.common import (TAHINI, MANGO, SEED, OUT, tahini_split, mango_split, save)

warnings.filterwarnings("ignore")

PLAIN = ["raw", "snv", "msc"]
DERIV = ["sg1", "sg2", "snv+sg1", "snv+sg2", "msc+sg1", "msc+sg2", "sg1+snv", "sg2+snv"]
WINDOWS = [7, 11, 15, 21]
NC_GRID = [4, 6, 8, 10, 12, 16, 20, 24, 30]
N_CARRY = 3        # the top three preprocessing configurations, not six: the expensive models
                   # cost hours per configuration on mango and the tahini sweep already showed
                   # the choice moves the external result by less than 0.002 R2
INNER_FOLDS = 3
SEEDS = [42, 7, 2024]


def configs():
    out = [(m, 11, 2) for m in PLAIN]
    for m in DERIV:
        for w in WINDOWS:
            out.append((m, w, 2))
    return out


def _post(p, normalise):
    p = np.asarray(p).reshape(len(p), -1)
    return (normalize_to_100(p) if normalise else p)[:, 0]


def transform_pair(ds, a, b, method, window, poly):
    pp = Preprocessor(method, window=window, poly=poly).fit(ds.X[a])
    return pp.transform(ds.X[a]), pp.transform(ds.X[b])


def score_model(name, S_a, y_a, S_b, y_b1, nc, normalise, seed, **kw):
    """One fit/score at a fixed representation. PLS consumes the spectrum; the rest consume
    `nc` PLS scores, exactly as in the published pipeline."""
    if name == "PLS":
        A, B = S_a, S_b
        mdl = build_model("PLS", A.shape[1], y_a.shape[1], seed=seed, n_components=nc)
    else:
        pf = PLSFeatures(nc).fit(S_a, y_a)
        A, B = pf.transform(S_a), pf.transform(S_b)
        mdl = build_model(name, A.shape[1], y_a.shape[1], seed=seed, **kw)
    mdl.fit(A, y_a)
    return float(r2_score(y_b1, _post(mdl.predict(B), normalise)))


def stage1(label, ds, tr, normalise):
    """Dense preprocessing sweep, scored with PLS on grouped inner folds."""
    folds = stratified_group_kfold(ds.groups[tr], ds.tagsis[tr], n_splits=INNER_FOLDS, seed=SEED)
    rows = []
    for method, window, poly in configs():
        t0 = time.time()
        per_nc = {n: [] for n in NC_GRID}
        for a, b in folds:
            S_a, S_b = transform_pair(ds, tr[a], tr[b], method, window, poly)
            for nc in NC_GRID:
                if nc >= min(S_a.shape):
                    continue
                per_nc[nc].append(score_model("PLS", S_a, ds.y[tr[a]], S_b, ds.y[tr[b], 0], nc,
                                              normalise, SEED))
        for nc, v in per_nc.items():
            if v:
                rows.append(dict(dataset=label, stage="s1_pls", method=method, window=window,
                                 poly=poly, nc=nc, model="PLS", R2=float(np.mean(v)),
                                 R2_sd=float(np.std(v))))
        best = max((r["R2"] for r in rows if r["method"] == method and r["window"] == window),
                   default=float("nan"))
        print(f"    {method:9s} w={window:<3} best inner-CV R2 {best:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    return rows


def stage2(label, ds, tr, normalise, carry, model_specs):
    folds = stratified_group_kfold(ds.groups[tr], ds.tagsis[tr], n_splits=INNER_FOLDS, seed=SEED)
    rows = []
    for method, window, poly in carry:
        for a, b in folds:
            S_a, S_b = transform_pair(ds, tr[a], tr[b], method, window, poly)
            for mname, nc, kw in model_specs:
                v = [score_model(mname, S_a, ds.y[tr[a]], S_b, ds.y[tr[b], 0], nc, normalise, s,
                                 **kw) for s in [SEED]]
                rows.append(dict(dataset=label, stage="s2_models", method=method, window=window,
                                 poly=poly, nc=nc, model=mname + f" nc={nc}",
                                 R2=float(np.mean(v)), R2_sd=float(np.std(v))))
        print(f"    carried {method:9s} w={window:<3} done", flush=True)
    return rows


def stage3(label, ds, tr, te, normalise, chosen, published, model_specs):
    rows = []
    y_te1 = ds.y[te, 0]
    for tag, (method, window, poly) in [("selected", chosen), ("published", published)]:
        S_tr, S_te = transform_pair(ds, tr, te, method, window, poly)
        for mname, nc, kw in model_specs:
            v = [score_model(mname, S_tr, ds.y[tr], S_te, y_te1, nc, normalise, s, **kw)
                 for s in ([SEED] if mname == "PLS" else SEEDS)]
            rows.append(dict(dataset=label, stage="s3_external", config=tag, method=method,
                             window=window, poly=poly, nc=nc, model=mname + f" nc={nc}",
                             R2=float(np.mean(v)), R2_sd=float(np.std(v))))
            print(f"    [{tag:9s}] {method:9s} w={window:<3} {mname:4s} nc={nc:<3} "
                  f"external R2 {np.mean(v):.4f}", flush=True)
    return rows


def run(label, splitter, cfg, normalise, model_specs, published):
    ds, tr, te = splitter()
    print(f"\n=== {label}: preprocessing search ===", flush=True)
    print("  -- stage 1: dense grid, scored with PLS on inner grouped folds --", flush=True)
    rows = stage1(label, ds, tr, normalise)
    s1 = pd.DataFrame(rows)
    best_per_cfg = (s1.groupby(["method", "window", "poly"])["R2"].max()
                      .sort_values(ascending=False))
    carry = list(best_per_cfg.head(N_CARRY).index)
    print("  -- carried forward --", flush=True)
    print(best_per_cfg.head(N_CARRY).round(4).to_string(), flush=True)

    print("  -- stage 2: expensive models on the carried configurations --", flush=True)
    rows += stage2(label, ds, tr, normalise, carry, model_specs)
    s2 = pd.DataFrame([r for r in rows if r["stage"] == "s2_models"])
    per_model = s2.sort_values("R2", ascending=False).groupby("model").head(1)
    print(per_model[["model", "method", "window", "nc", "R2"]].round(4).to_string(index=False),
          flush=True)
    chosen = tuple(per_model.sort_values("R2", ascending=False)
                   .iloc[0][["method", "window", "poly"]].tolist())

    print(f"  -- stage 3: external test, selected {chosen} vs published {published} --",
          flush=True)
    rows += stage3(label, ds, tr, te, normalise, chosen, published, model_specs)
    return rows, per_model


def main():
    t0 = time.time()
    tahini_specs = [("PLS", TAHINI["pls_nc"], {}),
                    ("KAN", TAHINI["kan_nc"], dict(width_hidden=(3,), grid=5)),
                    ("MLP", TAHINI["mlp_nc"], dict(hidden=TAHINI["mlp_hidden"]))]
    mango_specs = [("PLS", MANGO["pls_nc"], {}),
                   ("KAN", MANGO["kan_nc"], dict(width_hidden=(16, 8), grid=5)),
                   ("KAN", MANGO["kan_eq_nc"], dict(width_hidden=(3,), grid=5)),
                   ("MLP", MANGO["mlp_nc"], dict(hidden=MANGO["mlp_hidden"]))]

    # Save after each dataset: the mango stage costs hours and an interruption must not discard
    # the tahini result along with it. And reload rather than recompute when those files already
    # exist -- after a killed session the tahini arm would otherwise spend an hour reproducing
    # numbers that are already on disk.
    raw_t, sel_t = OUT / "r17_preprocessing_raw_tahini.csv", OUT / "r17_preprocessing_selected_tahini.csv"
    if raw_t.exists() and sel_t.exists():
        print("  [resume] the tahini arm is already on disk; skipping it", flush=True)
        rows = pd.read_csv(raw_t).to_dict("records")
        t_best = pd.read_csv(sel_t)
    else:
        rows, t_best = run("tahini", tahini_split, TAHINI, True, tahini_specs, ("snv", 11, 2))
        save(pd.DataFrame(rows), "r17_preprocessing_raw_tahini.csv")
        save(t_best, "r17_preprocessing_selected_tahini.csv")
    r2, m_best = run("mango", mango_split, MANGO, False, mango_specs, ("sg1", 11, 2))
    df = pd.DataFrame(rows + r2)
    save(df, "r17_preprocessing_raw.csv")
    save(pd.concat([t_best, m_best], ignore_index=True), "r17_preprocessing_selected.csv")
    ext = df[df.stage == "s3_external"]
    save(ext, "r17_preprocessing_external.csv")
    print("\n", ext.to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
