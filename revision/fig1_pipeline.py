"""Figure 1 (revised): analysis pipeline and validation designs.

Redrawn in the style of the original figure with three corrections for the revision:
  (a) preprocessing (SNV / first-derivative SG) acts per spectrum; the box states what IS fitted on training data;
  (a) PLS regression and the CNN read the preprocessed spectrum, not the PLS scores;
  (b) the validation designs are the nested leave-one-lot-out / leave-one-season-out designs and the
      30 repeated grouped hold-outs of Section 2.3.
Run: python -m revision.fig1_pipeline  -> revision/results/figures/Figure1.png (300 dpi)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "dejavuserif"})

BLUE, GREY, YELLOW, PURPLE, GREEN = "#dbe8fb", "#eef2f7", "#fbf1c7", "#ebe7fb", "#dcf7e6"
EDGE, INK = "#555555", "#111111"
FS = 11


def box(ax, x, y, w, h, text, fc, fs=FS):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=EDGE, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=1.6, color=EDGE,
                                 shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(13.9, 10.6), dpi=300)
    ax.set_xlim(0, 100); ax.set_ylim(0, 76); ax.axis("off")

    # ---------------- panel (a)
    ax.text(0.5, 74.5, "(a)", fontsize=17, fontweight="bold", va="top")
    t = box(ax, 1.5, 58, 19.5, 12, "Tahini, ATR–FTIR\n1554 spectra × 1762 ch.\n55 physical samples\n600–4000 cm$^{-1}$", BLUE)
    m = box(ax, 1.5, 42, 19.5, 12, "Mango, NIR\n11,691 spectra × 103 ch.\n112 populations\n684–990 nm", BLUE)
    pp = box(ax, 25, 49.5, 17, 13, "Preprocessing,\nper spectrum\n(SNV / SG 1st deriv.)", GREY)
    arrow(ax, (21, 64), (25, 57)); arrow(ax, (21, 48), (25, 55))
    fit = box(ax, 24, 40.5, 19, 7.5, "Fitted on training data only:\nPLS projection, scaling,\nhyperparameters, threshold", GREY, fs=8.8)
    full = box(ax, 48, 62, 21, 8, "Preprocessed spectrum\n→ PLS regression · 1-D CNN", YELLOW, fs=10)
    pls = box(ax, 50, 48, 14, 9, "PLS scores\n$x_1 \\ldots x_k$", GREY)
    lat = box(ax, 67, 51.5, 15, 7.5, "Scores\n→ SVR · RF · MLP", YELLOW, fs=10)
    kan = box(ax, 67, 41, 15, 7.5, "Scores\n→ KAN", PURPLE)
    arrow(ax, (42, 59), (48, 66)); arrow(ax, (42, 55), (50, 52.5))
    arrow(ax, (64, 53.5), (67, 55)); arrow(ax, (64, 51), (67, 45))
    acc = box(ax, 84, 55, 15, 15, "Accuracy\n$R^2$ · RMSEP · MAE\nRPD · F1 · LOD\ncluster-bootstrap CI", GREY, fs=10)
    intp = box(ax, 84, 36, 15, 16, "Equation and reading\n• closed-form equation\n• fidelity to the network\n• response curves\n• named bands\n• intrinsic vs SHAP", PURPLE, fs=8.8)
    arrow(ax, (69, 66), (84, 64)); arrow(ax, (82, 55), (84, 60)); arrow(ax, (82, 45), (84, 58)); arrow(ax, (82, 44), (84, 44))

    # ---------------- panel (b)
    ax.text(0.5, 33, "(b)", fontsize=17, fontweight="bold", va="top")
    ax.text(24, 32.3, "Tahini — split by physical sample", fontsize=12.5, fontweight="bold", ha="center", va="top")
    ax.text(74, 32.3, "Mango — split by population", fontsize=12.5, fontweight="bold", ha="center", va="top")
    ax.plot([49.5, 49.5], [1, 30], ls=":", color="#bbbbbb", lw=1.2)
    box(ax, 2, 22, 45, 7, "5 production lots × (1 authentic + 10 blends) = 55 physical samples;\neach blend scanned 14–17 times; each authentic sample 150–151 times", BLUE, fs=9)
    box(ax, 3, 13, 42, 6.5, "Primary: grouped hold-out, 38 / 17 samples (1146 / 408 spectra)\n30 repeated grouped 70/30 hold-outs, selection nested in each", GREEN, fs=9.6)
    box(ax, 3, 2, 42, 8.5, "Nested leave-one-lot-out (5 folds):\nlot and all its blends withheld; preprocessing, compression and\nhyperparameters selected by an inner leave-one-lot-out loop", GREEN, fs=9.6)
    arrow(ax, (24, 22), (24, 19.5)); arrow(ax, (24, 13), (24, 10.5))
    box(ax, 53, 22, 44, 7, "4 harvest seasons, 2 regions, 10 cultivars, 112 populations;\nreference dry matter by oven drying", BLUE, fs=9.6)
    box(ax, 53, 13, 44, 6.5, "Primary: seasons 1–3 calibration (94 populations, n = 10,243);\nseason 4 external test (18 populations, n = 1448), predicted once", GREEN, fs=9.6)
    box(ax, 53, 2, 44, 8.5, "Nested leave-one-season-out (4 folds):\nseason withheld; selection by an inner leave-one-season-out loop;\nno population shared between the two sides of any split", GREEN, fs=9.6)
    arrow(ax, (75, 22), (75, 19.5)); arrow(ax, (75, 13), (75, 10.5))

    fig.savefig(OUT / "Figure1.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(OUT / "Figure1.png")


if __name__ == "__main__":
    main()
