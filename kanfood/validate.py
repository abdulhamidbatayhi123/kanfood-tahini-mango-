import time
import numpy as np
from kanfood.split import stratified_group_kfold
from kanfood.preprocess import Preprocessor
from kanfood.features import PLSFeatures
from kanfood.models import build_model, front_end_components
from kanfood.metrics import regression_metrics, classification_from_tahini, classification_metrics


def run_group_cv(ds, model_names, n_splits=5, preprocess="snv", n_components=12,
                 seed=42, fast=False, threshold=95.0, params=None):
    """Nested-safe group CV. Per fold: preprocessing + PLS compression fit on TRAIN only.
    Spectrum models (PLS, CNN) see the full preprocessed spectrum; the rest see PLS scores at
    their own (possibly tuned) compression dimension. `params` = {model_name: hyperparam dict}."""
    params = params or {}
    folds = stratified_group_kfold(ds.groups, ds.tagsis, n_splits=n_splits, seed=seed)
    results = {m: {"R2_mean": [], "MAE_mean": [], "RPD_mean": [], "F1": [], "Acc": [], "time_s": []}
               for m in model_names}
    for tr, va in folds:
        pp = Preprocessor(preprocess).fit(ds.X[tr])
        Sp_tr, Sp_va = pp.transform(ds.X[tr]), pp.transform(ds.X[va])
        pls_cache = {}   # n_components -> (Z_tr, Z_va); PLS fit once per distinct compression size
        for name in model_names:
            mp = params.get(name, {})
            nc = front_end_components(name, mp, n_components)
            if nc is None:
                Xi_tr, Xi_va = Sp_tr, Sp_va
            else:
                if nc not in pls_cache:
                    plsf = PLSFeatures(nc).fit(Sp_tr, ds.y[tr])
                    pls_cache[nc] = (plsf.transform(Sp_tr), plsf.transform(Sp_va))
                Xi_tr, Xi_va = pls_cache[nc]
                mp = {k: v for k, v in mp.items() if k != "n_components"}
            t0 = time.time()
            model = build_model(name, Xi_tr.shape[1], ds.y.shape[1], seed=seed, fast=fast, **mp)
            model.fit(Xi_tr, ds.y[tr])
            pred = model.predict(Xi_va)
            dt = time.time() - t0
            reg = regression_metrics(ds.y[va], pred, ds.target_names)
            clf = classification_metrics(ds.tagsis[va], classification_from_tahini(pred[:, 0], threshold))
            results[name]["R2_mean"].append(reg["R2_mean"])
            results[name]["MAE_mean"].append(reg["MAE_mean"])
            results[name]["RPD_mean"].append(reg["RPD_mean"])
            results[name]["F1"].append(clf["F1"])
            results[name]["Acc"].append(clf["Acc"])
            results[name]["time_s"].append(dt)
    return results
