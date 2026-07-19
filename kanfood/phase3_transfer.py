"""Phase 3/4: oils per-food benchmark + shared-chemistry transfer.
(1) Does oils show nonlinearity (KAN/MLP vs PLS)?  (2) Do tahini-KAN and oils-KAN, trained
independently, flag the SAME FTIR bands on a common cm-1 axis -> transferable chemistry.
Run: python -m kanfood.phase3_transfer"""
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from kanfood.data import load_tahini, load_oils, resample_spectra, common_grid
from kanfood.preprocess import Preprocessor, snv
from kanfood.features import PLSFeatures
from kanfood.models import build_model, SPECTRUM_MODELS
from kanfood.metrics import regression_metrics
from kanfood.interpret import fit_kan
from kanfood.bands import BANDS
from kanfood._paths import data_path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PUB = data_path("KANFOOD_PAPER", "results_phase1", "paper")
PUB.mkdir(parents=True, exist_ok=True)
SEED = 42
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "font.size": 10, "font.family": "serif", "axes.titlesize": 11,
                     "axes.labelsize": 10, "legend.fontsize": 8})


def oils_benchmark(n_pls=10):
    oil = load_oils()
    models = ["PLS", "SVM", "RF", "MLP", "KAN"]
    res = {m: [] for m in models}
    gkf = GroupKFold(n_splits=5)
    for tr, va in gkf.split(oil.X, oil.y, oil.groups):
        pp = Preprocessor("snv").fit(oil.X[tr])
        Sp_tr, Sp_va = pp.transform(oil.X[tr]), pp.transform(oil.X[va])
        plsf = PLSFeatures(n_pls).fit(Sp_tr, oil.y[tr])
        Z_tr, Z_va = plsf.transform(Sp_tr), plsf.transform(Sp_va)
        for m in models:
            Xi_tr, Xi_va = (Sp_tr, Sp_va) if m in SPECTRUM_MODELS else (Z_tr, Z_va)
            model = build_model(m, Xi_tr.shape[1], oil.y.shape[1], seed=SEED, fast=False)
            model.fit(Xi_tr, oil.y[tr])
            reg = regression_metrics(oil.y[va], model.predict(Xi_va), oil.target_names)
            res[m].append((reg["R2_mean"], reg["R2_EVOO"]))
    rows = [{"Method": m,
             "R2_mean": round(float(np.mean([r[0] for r in res[m]])), 4),
             "R2_mean_std": round(float(np.std([r[0] for r in res[m]])), 4),
             "R2_EVOO": round(float(np.mean([r[1] for r in res[m]])), 4)} for m in models]
    df = pd.DataFrame(rows)
    df.to_csv(PUB / "Table8_oils_benchmark.csv", index=False)
    return df


def _wave_importance(model, plsf, n_pls, out_idx=0):
    g = np.linspace(-1, 1, 80)
    s = np.zeros(n_pls)
    for k in range(n_pls):
        base = np.zeros((len(g), n_pls))
        base[:, k] = g
        with torch.no_grad():
            out = model(torch.tensor(base, dtype=torch.float32)).numpy()[:, out_idx]
        s[k] = out.max() - out.min()
    w = np.abs(plsf.pls.x_loadings_[:, :n_pls]) @ s
    return w / (w.max() + 1e-12)


def _train_food_kan(X_grid_snv, y, n_pls=12):
    plsf = PLSFeatures(n_pls).fit(X_grid_snv, y)
    Z = plsf.transform(X_grid_snv)
    sx, sy = MinMaxScaler((-1, 1)), StandardScaler()
    model, _ = fit_kan(sx.fit_transform(Z), sy.fit_transform(y), width_hidden=(5,), steps=250, seed=SEED)
    return _wave_importance(model, plsf, n_pls, out_idx=0)


def shared_chemistry():
    tah, oil = load_tahini(), load_oils()
    grid = common_grid(tah, oil, 2.0)
    Xt = snv(resample_spectra(tah.X, tah.wavenumbers, grid))
    Xo = snv(resample_spectra(oil.X, oil.wavenumbers, grid))
    Wt = _train_food_kan(Xt, tah.y)
    Wo = _train_food_kan(Xo, oil.y)
    r = float(pearsonr(Wt, Wo)[0])
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(grid, Wt, lw=1.4, color="#8B5CF6", label="KAN trained on TAHINI")
    ax.plot(grid, Wo, lw=1.4, color="#F59E0B", label="KAN trained on OILS")
    for (lo, hi), assignment, _ in BANDS:
        c = (lo + hi) / 2
        if grid.min() <= c <= grid.max():
            ax.axvline(c, color="0.8", ls=":", lw=0.7, zorder=0)
            ax.text(c, 1.01, assignment.split("(")[0].strip(), rotation=90, fontsize=6,
                    va="bottom", ha="center", color="0.45")
    ax.invert_xaxis()
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("KAN wavenumber importance (norm.)")
    ax.set_title(f"Shared chemistry: independently-trained KANs flag the same FTIR bands "
                 f"(r = {r:.2f})", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.margins(y=0.12)
    fig.savefig(PUB / "Fig13_shared_chemistry.png")
    plt.close(fig)
    pd.DataFrame([{"tahini_vs_oils_importance_correlation": round(r, 4),
                   "grid_points": len(grid),
                   "grid_cm1": f"{grid.min():.0f}-{grid.max():.0f}"}]).to_csv(
        PUB / "Table9_shared_chemistry.csv", index=False)
    return r


def main():
    print("[1/2] Oils per-food benchmark (5-fold GroupKFold by composition)...")
    try:
        df = oils_benchmark()
        print(df.to_string(index=False))
    except Exception as e:
        import traceback; print("  FAILED:", repr(e)); traceback.print_exc()
    print("\n[2/2] Shared-chemistry transfer (tahini-KAN vs oils-KAN bands)...")
    try:
        r = shared_chemistry()
        print(f"  importance-curve correlation tahini<->oils: r = {r:.3f}")
    except Exception as e:
        import traceback; print("  FAILED:", repr(e)); traceback.print_exc()
    print("Phase-3 assets -> results_phase1/paper/")


if __name__ == "__main__":
    main()
