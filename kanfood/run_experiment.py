import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from kanfood.data import load_tahini, SpectralDataset
from kanfood._paths import data_path
from kanfood.split import group_holdout_split, stratified_group_kfold
from kanfood.preprocess import Preprocessor
from kanfood.features import PLSFeatures
from kanfood.models import build_model, MODEL_NAMES, front_end_components
from kanfood.metrics import (regression_metrics, classification_from_tahini,
                             classification_metrics, bootstrap_ci,
                             corrected_resampled_ttest, holm_bonferroni)
from kanfood.validate import run_group_cv
from kanfood.tune import tune_model
from kanfood.figures import plot_mean_spectra, plot_cv_box
from sklearn.metrics import r2_score, mean_absolute_error

OUT = data_path("KANFOOD_OUT", "results_phase1")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
PREPROCESS = "snv"
N_PLS = 12
# Full run by default; env overrides allow a fast end-to-end smoke of the pipeline wiring.
N_SPLITS = int(os.environ.get("KANFOOD_N_SPLITS", "5"))
FAST = os.environ.get("KANFOOD_FAST", "0") == "1"


def _subset(ds, idx):
    return SpectralDataset(ds.X[idx], ds.y[idx], ds.groups[idx], ds.tagsis[idx],
                           ds.wavenumbers, ds.target_names, ds.name)


def _choose_threshold(ds_train, seed):
    """Pick the detection threshold on TRAIN CV only (never the test set), using the PLS baseline."""
    folds = stratified_group_kfold(ds_train.groups, ds_train.tagsis, n_splits=3, seed=seed)
    cached = []
    for tr, va in folds:
        pp = Preprocessor(PREPROCESS).fit(ds_train.X[tr])
        Sp_tr, Sp_va = pp.transform(ds_train.X[tr]), pp.transform(ds_train.X[va])
        model = build_model("PLS", Sp_tr.shape[1], ds_train.y.shape[1], seed=seed)
        model.fit(Sp_tr, ds_train.y[tr])
        cached.append((model.predict(Sp_va)[:, 0], ds_train.tagsis[va]))
    best_t, best_f1 = 95.0, -1.0
    for t in np.arange(85, 100.5, 0.5):
        f1s = [classification_metrics(tg, classification_from_tahini(pt, t))["F1"] for pt, tg in cached]
        if np.mean(f1s) > best_f1:
            best_f1, best_t = float(np.mean(f1s)), float(t)
    return best_t


def _write_results_md(ds, tr, te, threshold, best, cv_rows, test_df, stat_rows):
    tuned = "\n".join(f"  - {k}: {v}" for k, v in best.items())
    lines = [
        "# Phase 1 - Tahini Benchmark (leakage-free, tuned, SNV -> PLS -> model)\n",
        f"- Samples: {ds.X.shape[0]} spectra, {len(np.unique(ds.groups))} physical samples (isim)",
        f"- Split: GROUP hold-out by isim - train {len(tr)} / test {len(te)} spectra, no shared sample",
        f"- Pipeline: SNV -> PLS latent scores -> model. Score models use a {N_PLS}-component PLS "
        f"front-end by default; KAN's compression is a tuned hyperparameter. PLS & 1-D CNN use the full spectrum.",
        "- Hyperparameters chosen by 3-fold inner GroupKFold on TRAIN only (equal light tuning):",
        tuned,
        f"- Detection threshold {threshold}% (chosen on train CV).\n",
        "## Group-CV summary (train, tuned configs)\n",
        pd.DataFrame(cv_rows).to_markdown(index=False),
        "\n## Held-out test (group-independent)\n",
        test_df.to_markdown(index=False),
        "\n## KAN vs baselines (paired t-test on CV folds)\n",
        pd.DataFrame(stat_rows).to_markdown(index=False),
        "\n## Honest reading\n",
        "- Leakage-free (group split); every model tuned with the same inner-CV budget and trained to convergence.",
        "- KAN is a single, stable model (no ensembling) -> one equation for interpretability.",
        "- Tahini blending is ~linear, so PLS is very strong; KAN's value is interpretability + cross-food transfer, not winning tahini accuracy.",
    ]
    (OUT / "RESULTS_phase1.md").write_text("\n".join(lines), encoding="utf-8")


def _save_artifacts(ds, te, test_preds, plsf, threshold, best, cv, kan_nc):
    """Persist every raw result so report.py can regenerate any publication figure without retraining.
    `plsf` is KAN's PLS front-end (its loadings are the interpretable inputs in report/interpret)."""
    arts = {
        "wavenumbers": ds.wavenumbers,
        "y_test": ds.y[te],
        "tagsis_test": ds.tagsis[te],
        "pls_loadings": plsf.pls.x_loadings_,   # (n_wavenumbers, n_components)
    }
    for col, key in [(0, "mean_tahini"), (2, "mean_sunflower"), (1, "mean_peanut")]:
        mask = ds.y[:, col] >= 99.0
        arts[key] = ds.X[mask].mean(0) if mask.sum() else np.zeros(ds.X.shape[1])
    for name, pred in test_preds.items():
        arts[f"pred_{name}"] = pred
    np.savez(OUT / "phase1_artifacts.npz", **arts)
    meta = {"threshold": threshold,
            "tuned_params": {k: str(v) for k, v in best.items()},
            "kan_n_components": int(kan_nc),   # canonical KAN compression; interpret/phase2 read this
            "target_names": ds.target_names,
            "model_names": list(test_preds.keys()),
            "cv_r2": {m: list(map(float, cv[m]["R2_mean"])) for m in cv}}
    (OUT / "phase1_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    ds = load_tahini()
    print(f"Loaded {ds.name}: X={ds.X.shape}, groups={len(np.unique(ds.groups))}, "
          f"cm-1 {ds.wavenumbers.min():.0f}-{ds.wavenumbers.max():.0f}")
    plot_mean_spectra(ds, OUT / "Fig_mean_spectra.png")

    tr, te = group_holdout_split(ds.groups, test_size=0.30, seed=SEED)
    assert set(ds.groups[tr]).isdisjoint(set(ds.groups[te])), "leakage!"
    train_ds = _subset(ds, tr)
    print(f"Group hold-out: train {len(tr)} / test {len(te)} spectra "
          f"({len(np.unique(ds.groups[tr]))}/{len(np.unique(ds.groups[te]))} samples)")

    print("Tuning each model (3-fold inner GroupKFold on TRAIN)...")
    best = {}
    for name in MODEL_NAMES:
        p, s = tune_model(train_ds, name, n_splits=3, preprocess=PREPROCESS,
                          n_components=N_PLS, seed=SEED, fast=FAST)
        best[name] = p
        print(f"  {name}: {p}  (inner-CV R2={s:.4f})")

    print(f"Report CV ({N_SPLITS}-fold) with tuned configs...")
    cv = run_group_cv(train_ds, MODEL_NAMES, n_splits=N_SPLITS, preprocess=PREPROCESS,
                      n_components=N_PLS, seed=SEED, fast=FAST, params=best)
    cv_rows = [{"Method": m,
                "R2_mean": round(np.mean(cv[m]["R2_mean"]), 4), "R2_std": round(np.std(cv[m]["R2_mean"]), 4),
                "MAE_mean": round(np.mean(cv[m]["MAE_mean"]), 3), "RPD_mean": round(np.mean(cv[m]["RPD_mean"]), 3),
                "F1_mean": round(np.mean(cv[m]["F1"]), 4), "time_s": round(np.mean(cv[m]["time_s"]), 2)}
               for m in MODEL_NAMES]
    pd.DataFrame(cv_rows).to_csv(OUT / "Table_cv_summary.csv", index=False)
    plot_cv_box(cv, MODEL_NAMES, OUT / "Fig_cv_box.png")

    threshold = _choose_threshold(train_ds, seed=SEED)
    print(f"Detection threshold chosen on train CV: {threshold}%")

    pp = Preprocessor(PREPROCESS).fit(ds.X[tr])
    Sp_tr, Sp_te = pp.transform(ds.X[tr]), pp.transform(ds.X[te])
    pls_cache = {}   # n_components -> (PLSFeatures, Z_tr, Z_te); fit once per distinct compression size

    rows = []
    test_preds = {}
    for name in MODEL_NAMES:
        mp = dict(best[name])
        nc = front_end_components(name, mp, N_PLS)
        if nc is None:
            Xi_tr, Xi_te = Sp_tr, Sp_te
        else:
            if nc not in pls_cache:
                f = PLSFeatures(nc).fit(Sp_tr, ds.y[tr])
                pls_cache[nc] = (f, f.transform(Sp_tr), f.transform(Sp_te))
            _, Xi_tr, Xi_te = pls_cache[nc]
            mp = {k: v for k, v in mp.items() if k != "n_components"}
        model = build_model(name, Xi_tr.shape[1], ds.y.shape[1], seed=SEED, fast=FAST, **mp)
        model.fit(Xi_tr, ds.y[tr])
        pred = model.predict(Xi_te)
        test_preds[name] = pred
        reg = regression_metrics(ds.y[te], pred, ds.target_names)
        r2_m, r2_lo, r2_hi = bootstrap_ci(ds.y[te, 0], pred[:, 0], r2_score, seed=SEED,
                                          groups=ds.groups[te])
        clf = classification_metrics(ds.tagsis[te], classification_from_tahini(pred[:, 0], threshold))
        rows.append({"Method": name,
                     "R2_tahini": round(reg["R2_Tahin Oranı"], 4),
                     "R2_CI": f"[{r2_lo:.3f},{r2_hi:.3f}]",
                     "MAE_tahini": round(reg["MAE_Tahin Oranı"], 3),
                     "RPD_tahini": round(reg["RPD_Tahin Oranı"], 3),
                     "R2_mean3": round(reg["R2_mean"], 4),
                     "F1": round(clf["F1"], 4), "Acc": round(clf["Acc"], 4),
                     "n_params": model.n_params()})
        print(f"  {name}: R2_tahini={rows[-1]['R2_tahini']} F1={rows[-1]['F1']}")
    test_df = pd.DataFrame(rows)
    test_df.to_csv(OUT / "Table_holdout_results.csv", index=False)

    # KAN vs each baseline: Nadeau-Bengio corrected resampled t-test on CV folds, Holm-adjusted
    # across the five comparisons (rigor audit 2.1).
    kan = np.array(cv["KAN"]["R2_mean"])
    others = [x for x in MODEL_NAMES if x != "KAN"]
    raw = [(m, float(kan.mean() - np.mean(cv[m]["R2_mean"])),
            corrected_resampled_ttest(kan, np.array(cv[m]["R2_mean"]))[1]) for m in others]
    padj = holm_bonferroni([p for _, _, p in raw])
    stat_rows = [{"Comparison": f"KAN vs {m}", "diff": round(d, 4),
                  "p_corrected": round(p, 4), "p_holm": round(pa, 4), "sig_holm": bool(pa < 0.05)}
                 for (m, d, p), pa in zip(raw, padj)]
    pd.DataFrame(stat_rows).to_csv(OUT / "Table_stats.csv", index=False)

    kan_nc = front_end_components("KAN", best["KAN"], N_PLS)   # KAN's tuned PLS-compression dimension
    plsf = pls_cache[kan_nc][0]                                # its interpretable PLS front-end (loadings)
    _save_artifacts(ds, te, test_preds, plsf, threshold, best, cv, kan_nc)
    _write_results_md(ds, tr, te, threshold, best, cv_rows, test_df, stat_rows)
    print("DONE -> results_phase1/  (review RESULTS_phase1.md; artifacts saved for report.py)")


if __name__ == "__main__":
    main()
