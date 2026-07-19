import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_mean_spectra(ds, out_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for col, lab in [(0, "Pure tahini"), (2, "Pure sunflower"), (1, "Pure peanut")]:
        mask = ds.y[:, col] >= 99.0
        if mask.sum():
            ax.plot(ds.wavenumbers, ds.X[mask].mean(0), lw=1.3, label=f"{lab} (n={int(mask.sum())})")
    ax.invert_xaxis()
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Absorbance")
    ax.set_title("Mean FTIR spectra of pure components")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_cv_box(cv, model_names, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([cv[m]["R2_mean"] for m in model_names], tick_labels=model_names, patch_artist=True)
    ax.set_ylabel("R^2 (mean of 3 targets)")
    ax.set_title("Group-CV R^2 distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
