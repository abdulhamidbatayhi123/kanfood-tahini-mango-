"""Publication assets for the mango DMC benchmark. Reads results_mango/mango_artifacts.npz +
mango_meta.json (+ CSV tables) and writes paper-ready figures + a main table to results_mango/paper/.
Decoupled from training. Run: python -m kanfood.report_mango"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

from kanfood.metrics import rmse
from kanfood._paths import data_path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

OUT = data_path("KANFOOD_MANGO_OUT", "results_mango")
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
    a = np.load(OUT / "mango_artifacts.npz", allow_pickle=True)
    meta = json.loads((OUT / "mango_meta.json").read_text(encoding="utf-8"))
    return a, meta


def _parse_ci(s):
    lo, hi = s.strip("[]").split(",")
    return float(lo), float(hi)


def _test_metrics(a):
    yt = a["y_test"][:, 0]
    return {m: {"R2": r2_score(yt, a[f"pred_{m}"][:, 0]),
                "RMSEP": rmse(yt, a[f"pred_{m}"][:, 0]),
                "MAE": mean_absolute_error(yt, a[f"pred_{m}"][:, 0])} for m in MODEL_ORDER}


def fig_comparison(a, holdout_df):
    tm = _test_metrics(a)
    ci = {r["Method"]: _parse_ci(r["R2_CI"]) for _, r in holdout_df.iterrows()}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    r2 = [tm[m]["R2"] for m in MODEL_ORDER]
    lo = [max(tm[m]["R2"] - ci[m][0], 0) for m in MODEL_ORDER]
    hi = [max(ci[m][1] - tm[m]["R2"], 0) for m in MODEL_ORDER]
    axes[0].bar(MODEL_ORDER, r2, color=[COLORS[m] for m in MODEL_ORDER], edgecolor="black", alpha=0.9)
    axes[0].errorbar(MODEL_ORDER, r2, yerr=[lo, hi], fmt="none", color="black", capsize=4)
    axes[0].set_ylabel("External-test R$^2$ (DMC)")
    axes[0].set_ylim(min(r2) - 0.06, 1.0)
    axes[0].set_title("(a) Accuracy on held-out Season 4 (95% CI)", loc="left", fontweight="bold")
    x = np.arange(len(MODEL_ORDER)); w = 0.38
    axes[1].bar(x - w / 2, [tm[m]["RMSEP"] for m in MODEL_ORDER], w, label="RMSEP", color="#475569")
    axes[1].bar(x + w / 2, [tm[m]["MAE"] for m in MODEL_ORDER], w, label="MAE", color="#94A3B8")
    axes[1].set_xticks(x); axes[1].set_xticklabels(MODEL_ORDER)
    axes[1].set_ylabel("Error (% DM)"); axes[1].legend(frameon=False)
    axes[1].set_title("(b) Prediction error", loc="left", fontweight="bold")
    fig.suptitle("Mango dry-matter prediction: interseason external validation", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PUB / "FigM2_model_comparison.png")
    plt.close(fig)


def fig_parity(a):
    yt = a["y_test"][:, 0]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    lim = (yt.min() - 1, yt.max() + 1)
    for ax, m in zip(axes.ravel(), MODEL_ORDER):
        p = a[f"pred_{m}"][:, 0]
        ax.scatter(yt, p, s=12, alpha=0.4, c=COLORS[m], edgecolors="none")
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_title(f"{m}: R$^2$={r2_score(yt,p):.3f}, RMSEP={rmse(yt,p):.2f}", fontsize=9, fontweight="bold")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("Actual DMC (%)"); ax.set_ylabel("Predicted DMC (%)")
    fig.suptitle("Predicted vs actual dry-matter content (held-out Season 4)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PUB / "FigM3_parity.png")
    plt.close(fig)


def fig_cv_box(meta):
    cv = meta["cv_r2"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot([cv[m] for m in MODEL_ORDER], tick_labels=MODEL_ORDER, patch_artist=True, widths=0.6)
    for patch, m in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_ylabel("R$^2$ (DMC)")
    ax.set_title("5-fold group cross-validation by population (train)", loc="left", fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(PUB / "FigM4_cv_box.png")
    plt.close(fig)


def main():
    if not (OUT / "mango_artifacts.npz").exists():
        print("No mango artifacts. Run `python -m kanfood.run_mango` first.")
        return
    a, meta = _load()
    cv_df = pd.read_csv(OUT / "Table_mango_cv.csv")
    holdout_df = pd.read_csv(OUT / "Table_mango_holdout.csv")

    fig_comparison(a, holdout_df)
    fig_parity(a)
    fig_cv_box(meta)

    tm = _test_metrics(a)
    cvi = cv_df.set_index("Method"); hoi = holdout_df.set_index("Method")
    rows = [{"Method": m,
             "CV R2 (mean+/-std)": f"{cvi.loc[m,'R2_mean']:.3f}+/-{cvi.loc[m,'R2_std']:.3f}",
             "Test R2": f"{tm[m]['R2']:.3f}", "Test R2 95% CI": hoi.loc[m, "R2_CI"],
             "RMSEP (%)": f"{tm[m]['RMSEP']:.3f}", "MAE (%)": f"{tm[m]['MAE']:.3f}",
             "RPD": f"{hoi.loc[m,'RPD']:.2f}",
             "Params": int(hoi.loc[m, "n_params"]) if not pd.isna(hoi.loc[m, "n_params"]) else "-",
             "Interpretable": INTERPRETABLE[m]} for m in MODEL_ORDER]
    main_df = pd.DataFrame(rows)
    main_df.to_csv(PUB / "TableM1_main_results.csv", index=False)
    (PUB / "TableM1_main_results.tex").write_text(main_df.to_latex(index=False), encoding="utf-8")

    print("Mango publication assets -> results_mango/paper/:")
    for f in sorted(PUB.glob("*")):
        print(f"  - {f.name}")
    print("\nMain results table:\n")
    print(main_df.to_string(index=False))


if __name__ == "__main__":
    main()
