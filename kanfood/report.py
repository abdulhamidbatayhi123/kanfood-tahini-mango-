"""Publication-asset generator. Reads results_phase1/phase1_artifacts.npz + phase1_meta.json
(+ the CSV tables) and writes paper-ready figures and CSV/LaTeX tables to results_phase1/paper/.
Decoupled from training -> rerun anytime to restyle. Run: python -m kanfood.report"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, r2_score, mean_absolute_error

from kanfood.bands import BANDS
from kanfood.metrics import rmse, classification_from_tahini
from kanfood._paths import data_path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

OUT = data_path("KANFOOD_OUT", "results_phase1")
PUB = OUT / "paper"
PUB.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["PLS", "SVM", "RF", "MLP", "CNN", "KAN"]
COLORS = {"PLS": "#3B82F6", "SVM": "#10B981", "RF": "#F59E0B",
          "MLP": "#EF4444", "CNN": "#6B7280", "KAN": "#8B5CF6"}
INTERPRETABLE = {"PLS": "Yes", "SVM": "No", "RF": "Partial", "MLP": "No", "CNN": "No", "KAN": "Yes"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": 1.0,
})


def _load():
    a = np.load(OUT / "phase1_artifacts.npz", allow_pickle=True)
    meta = json.loads((OUT / "phase1_meta.json").read_text(encoding="utf-8"))
    return a, meta


def _parse_ci(s):
    lo, hi = s.strip("[]").split(",")
    return float(lo), float(hi)


def fig_mean_spectra(a):
    wn = a["wavenumbers"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, lab, c in [("mean_tahini", "Pure tahini (sesame)", "#8B5CF6"),
                        ("mean_sunflower", "Pure sunflower paste", "#F59E0B"),
                        ("mean_peanut", "Pure peanut paste", "#10B981")]:
        if np.any(a[key]):
            ax.plot(wn, a[key], lw=1.3, label=lab, color=c)
    ymax = max(a["mean_tahini"].max(), a["mean_sunflower"].max(), a["mean_peanut"].max())
    for (lo, hi), assignment, _ in BANDS:
        c = (lo + hi) / 2
        if wn.min() <= c <= wn.max():
            ax.axvline(c, color="0.7", ls=":", lw=0.7, zorder=0)
            ax.text(c, ymax * 1.02, assignment.split("(")[0].strip(), rotation=90,
                    fontsize=6, va="bottom", ha="center", color="0.4")
    ax.invert_xaxis()
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Absorbance (a.u.)")
    ax.set_title("(a) Mean ATR-FTIR spectra of pure components", loc="left", fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    ax.margins(y=0.14)
    fig.savefig(PUB / "Fig1_mean_spectra.png")
    plt.close(fig)


def _test_metrics(a):
    yt = a["y_test"]
    rows = {}
    for m in MODEL_ORDER:
        p = a[f"pred_{m}"]
        rows[m] = {"R2": r2_score(yt[:, 0], p[:, 0]),
                   "RMSEP": rmse(yt[:, 0], p[:, 0]),
                   "MAE": mean_absolute_error(yt[:, 0], p[:, 0])}
    return rows


def fig_model_comparison(a, holdout_df):
    tm = _test_metrics(a)
    ci = {r["Method"]: _parse_ci(r["R2_CI"]) for _, r in holdout_df.iterrows()}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    # (a) R2 with bootstrap CI
    r2 = [tm[m]["R2"] for m in MODEL_ORDER]
    lo = [tm[m]["R2"] - ci[m][0] for m in MODEL_ORDER]
    hi = [ci[m][1] - tm[m]["R2"] for m in MODEL_ORDER]
    axes[0].bar(MODEL_ORDER, r2, color=[COLORS[m] for m in MODEL_ORDER], edgecolor="black", alpha=0.9)
    axes[0].errorbar(MODEL_ORDER, r2, yerr=[np.clip(lo, 0, None), np.clip(hi, 0, None)],
                     fmt="none", color="black", capsize=4)
    axes[0].set_ylabel("Test R$^2$ (tahini %)")
    axes[0].set_ylim(min(r2) - 0.08, 1.005)
    axes[0].set_title("(a) Accuracy (95% CI)", loc="left", fontweight="bold")
    # (b) RMSEP & MAE
    x = np.arange(len(MODEL_ORDER)); w = 0.38
    axes[1].bar(x - w / 2, [tm[m]["RMSEP"] for m in MODEL_ORDER], w, label="RMSEP", color="#475569")
    axes[1].bar(x + w / 2, [tm[m]["MAE"] for m in MODEL_ORDER], w, label="MAE", color="#94A3B8")
    axes[1].set_xticks(x); axes[1].set_xticklabels(MODEL_ORDER)
    axes[1].set_ylabel("Error (%)"); axes[1].legend(frameon=False)
    axes[1].set_title("(b) Prediction error", loc="left", fontweight="bold")
    # (c) F1
    f1 = [float(holdout_df.set_index("Method").loc[m, "F1"]) for m in MODEL_ORDER]
    axes[2].bar(MODEL_ORDER, f1, color=[COLORS[m] for m in MODEL_ORDER], edgecolor="black", alpha=0.9)
    axes[2].set_ylabel("F1 (adulteration detection)")
    axes[2].set_ylim(min(f1) - 0.1, 1.02)
    axes[2].set_title("(c) Detection F1", loc="left", fontweight="bold")
    fig.suptitle("Model comparison on the group-independent test set", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PUB / "Fig2_model_comparison.png")
    plt.close(fig)


def fig_parity(a):
    yt = a["y_test"]; tag = a["tagsis_test"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, m in zip(axes.ravel(), MODEL_ORDER):
        p = a[f"pred_{m}"][:, 0]
        ax.scatter(yt[tag == 0, 0], p[tag == 0], s=14, alpha=0.5, c="#16A34A", label="Pure", edgecolors="none")
        ax.scatter(yt[tag == 1, 0], p[tag == 1], s=14, alpha=0.5, c=COLORS[m], label="Adulterated", edgecolors="none")
        ax.plot([0, 100], [0, 100], "k--", lw=1)
        ax.set_title(f"{m}: R$^2$={r2_score(yt[:,0], p):.3f}, RMSEP={rmse(yt[:,0], p):.2f}%",
                     fontsize=9, fontweight="bold")
        ax.set_xlim(-5, 105); ax.set_ylim(-5, 105); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("Actual tahini %"); ax.set_ylabel("Predicted %")
    axes[0, 0].legend(loc="lower right", fontsize=7, frameon=False)
    fig.suptitle("Predicted vs actual tahini content (held-out samples)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PUB / "Fig3_parity.png")
    plt.close(fig)


def fig_cv_box(meta):
    cv = meta["cv_r2"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot([cv[m] for m in MODEL_ORDER], tick_labels=MODEL_ORDER, patch_artist=True, widths=0.6)
    for patch, m in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_ylabel("R$^2$ (mean of 3 targets)")
    ax.set_title("5-fold group cross-validation R$^2$ distribution", loc="left", fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(PUB / "Fig4_cv_box.png")
    plt.close(fig)


def fig_confusion(a, meta):
    yt = a["y_test"]; tag = a["tagsis_test"]; thr = meta["threshold"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, m in zip(axes.ravel(), MODEL_ORDER):
        pred_cls = classification_from_tahini(a[f"pred_{m}"][:, 0], thr)
        cm = confusion_matrix(tag, pred_cls, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False, ax=ax,
                    xticklabels=["Pure", "Adult."], yticklabels=["Pure", "Adult."],
                    annot_kws={"size": 12, "weight": "bold"})
        ax.set_title(m, fontweight="bold"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.suptitle(f"Adulteration detection at {thr:.1f}% threshold (test set)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PUB / "Fig5_confusion.png")
    plt.close(fig)


def fig_pls_loadings(a):
    wn = a["wavenumbers"]; L = a["pls_loadings"]   # (P, K)
    fig, ax = plt.subplots(figsize=(9, 5))
    for k in range(min(3, L.shape[1])):
        ax.plot(wn, L[:, k], lw=1.1, label=f"PLS component {k+1}")
    for (lo, hi), assignment, _ in BANDS:
        c = (lo + hi) / 2
        if wn.min() <= c <= wn.max():
            ax.axvline(c, color="0.8", ls=":", lw=0.7, zorder=0)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.invert_xaxis()
    ax.set_xlabel("Wavenumber (cm$^{-1}$)"); ax.set_ylabel("PLS loading")
    ax.set_title("PLS latent variables map back to chemical bands (KAN's interpretable inputs)",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False)
    fig.savefig(PUB / "Fig6_pls_loadings.png")
    plt.close(fig)


def tables(a, meta, cv_df, holdout_df, stats_df):
    tm = _test_metrics(a)
    main = []
    cvi = cv_df.set_index("Method"); hoi = holdout_df.set_index("Method")
    for m in MODEL_ORDER:
        main.append({
            "Method": m,
            "CV R2 (mean+/-std)": f"{cvi.loc[m,'R2_mean']:.3f}+/-{cvi.loc[m,'R2_std']:.3f}",
            "Test R2": f"{tm[m]['R2']:.3f}",
            "Test R2 95% CI": hoi.loc[m, "R2_CI"],
            "RMSEP (%)": f"{tm[m]['RMSEP']:.2f}",
            "MAE (%)": f"{tm[m]['MAE']:.2f}",
            "RPD": f"{hoi.loc[m,'RPD_tahini']:.2f}",
            "F1": f"{hoi.loc[m,'F1']:.3f}",
            "Params": int(hoi.loc[m, "n_params"]) if not pd.isna(hoi.loc[m, "n_params"]) else "-",
            "Interpretable": INTERPRETABLE[m],
        })
    main_df = pd.DataFrame(main)
    tuned_df = pd.DataFrame([{"Method": k, "Selected hyperparameters": v}
                             for k, v in meta["tuned_params"].items()])
    for name, df in [("Table1_main_results", main_df), ("Table2_tuned_params", tuned_df),
                     ("Table3_stats", stats_df)]:
        df.to_csv(PUB / f"{name}.csv", index=False)
        (PUB / f"{name}.tex").write_text(df.to_latex(index=False), encoding="utf-8")
    return main_df


def main():
    if not (OUT / "phase1_artifacts.npz").exists():
        print("No artifacts found. Run `python -m kanfood.run_experiment` first.")
        return
    a, meta = _load()
    cv_df = pd.read_csv(OUT / "Table_cv_summary.csv")
    holdout_df = pd.read_csv(OUT / "Table_holdout_results.csv")
    stats_df = pd.read_csv(OUT / "Table_stats.csv")

    fig_mean_spectra(a)
    fig_model_comparison(a, holdout_df)
    fig_parity(a)
    fig_cv_box(meta)
    fig_confusion(a, meta)
    fig_pls_loadings(a)
    main_df = tables(a, meta, cv_df, holdout_df, stats_df)

    print("Publication assets written to results_phase1/paper/:")
    for f in sorted(PUB.glob("*")):
        print(f"  - {f.name}")
    print("\nMain results table:\n")
    print(main_df.to_string(index=False))


if __name__ == "__main__":
    main()
