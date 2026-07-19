import numpy as np
from kanfood.split import stratified_group_kfold
from kanfood.preprocess import Preprocessor
from kanfood.features import PLSFeatures
from kanfood.models import build_model, front_end_components
from kanfood.metrics import regression_metrics

# Documented hyperparameter grids, selected by 3-fold inner GroupKFold on TRAIN only (nested CV).
# EQUAL-BUDGET fairness (rigor audit 1.1): every model receives a genuine, comparable search over
# its principal hyperparameters -- the deep baselines (CNN, MLP) are NOT run at a single fixed
# architecture. PLS: latent dimensionality; SVM: C, gamma; RF: number/depth of trees; MLP:
# hidden-layer sizes x PLS-compression; CNN: channel widths x learning rate; KAN: hidden width x
# PLS-compression. On tahini KAN's architecture is fixed to the compact, symbolically-extractable
# width-3 layer (the model the equation is read off), so KAN in fact gets the SMALLEST search here.
GRIDS = {
    "PLS": [{"n_components": k} for k in (8, 12, 16, 20)],                                  # 4
    "SVM": [{"C": c, "gamma": g} for c in (1.0, 10.0, 100.0) for g in ("scale", 0.05)],     # 6
    "RF":  [{"n_estimators": 300, "max_depth": d} for d in (None, 10, 20)],                 # 3
    "MLP": [{"hidden": h, "n_components": n} for h in ((128, 64), (64, 32)) for n in (8, 12, 16)],  # 6
    "CNN": [{"channels": c, "lr": lr} for c in ((8, 16), (16, 32), (32, 64)) for lr in (1e-3, 3e-4)],  # 6
    "KAN": [{"width_hidden": (3,), "grid": 5, "n_components": n} for n in (8, 12, 16)],      # 3
}


def _fold_score(ds, tr, va, name, params, preprocess, n_components, seed, fast):
    pp = Preprocessor(preprocess).fit(ds.X[tr])
    Sp_tr, Sp_va = pp.transform(ds.X[tr]), pp.transform(ds.X[va])
    nc = front_end_components(name, params, n_components)
    if nc is None:                       # spectrum model: full spectrum; n_components is the model's own
        Xi_tr, Xi_va, mp = Sp_tr, Sp_va, params
    else:                                # score model: PLS-score front-end at this model's compression
        plsf = PLSFeatures(nc).fit(Sp_tr, ds.y[tr])
        Xi_tr, Xi_va = plsf.transform(Sp_tr), plsf.transform(Sp_va)
        mp = {k: v for k, v in params.items() if k != "n_components"}
    model = build_model(name, Xi_tr.shape[1], ds.y.shape[1], seed=seed, fast=fast, **mp)
    model.fit(Xi_tr, ds.y[tr])
    return regression_metrics(ds.y[va], model.predict(Xi_va), ds.target_names)["R2_mean"]


def tune_model(ds_train, name, grid=None, n_splits=3, preprocess="snv",
               n_components=12, seed=42, fast=False):
    """Return (best_params, best_inner_cv_R2) chosen by inner GroupKFold on the TRAIN set only."""
    grid = grid if grid is not None else GRIDS[name]
    folds = stratified_group_kfold(ds_train.groups, ds_train.tagsis, n_splits, seed)
    best_params, best_score = grid[0], -np.inf
    for params in grid:
        scores = [_fold_score(ds_train, tr, va, name, params, preprocess, n_components, seed, fast)
                  for tr, va in folds]
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score, best_params = mean_score, params
    return best_params, best_score
