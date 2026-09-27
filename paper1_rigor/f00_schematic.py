"""Figure 1: the analysis pipeline and the leakage-free splitting protocol.

Text fit is checked programmatically rather than by eye -- in the companion paper's revision two
boxes overflowed and were only caught by comparing rendered text extents against their containers.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from paper1_rigor.figstyle import COL2, panel, save, check_no_title

BOX = dict(boxstyle="round,pad=0.30,rounding_size=0.06", linewidth=0.8)


def box(ax, x, y, w, h, text, fc, ec="0.35", fs=6.6, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=fc, edgecolor=ec, **BOX))
    t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                linespacing=1.25, fontweight=weight, zorder=3)
    return (x, y, w, h, t)


def arrow(ax, x1, y1, x2, y2, style="-|>", color="0.35", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=7,
                                 linewidth=0.8, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=1))


def check_fit(fig, ax, boxes, margin=0.98):
    """Assert every label's rendered extent sits inside its box (data coordinates)."""
    fig.canvas.draw()
    inv = ax.transData.inverted()
    bad = []
    for (x, y, w, h, t) in boxes:
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        if (x1 - x0) > w * margin or (y1 - y0) > h * margin:
            bad.append((t.get_text().replace("\n", " / "),
                        round((x1 - x0) / w, 2), round((y1 - y0) / h, 2)))
    assert not bad, "text overflows its box (width, height ratios): " + repr(bad)


BLUE, VIOL, GREY, GREEN, SAND = "#DBEAFE", "#EDE9FE", "#F1F5F9", "#DCFCE7", "#FEF3C7"


def fig_pipeline():
    fig, axes = plt.subplots(2, 1, figsize=(COL2, 5.2),
                             gridspec_kw=dict(height_ratios=[1.0, 1.0]))

    # ---------------------------------------------------------------- (a) pipeline
    ax = axes[0]
    ax.set_xlim(-2, 102); ax.set_ylim(2, 50); ax.axis("off")
    b = []
    b.append(box(ax, 0, 33, 21, 13,
                 "Tahini, ATR–FTIR\n1554 spectra × 1762 ch.\n55 physical samples\n600–4000 cm$^{-1}$", BLUE))
    b.append(box(ax, 0, 12, 21, 13,
                 "Mango, NIR\n11,691 spectra × 103 ch.\n112 populations\n684–990 nm", BLUE))
    b.append(box(ax, 23, 23, 17, 12,
                 "Preprocessing\nfitted on the\ntraining split only\n(SNV / SG 1st deriv.)", GREY))
    b.append(box(ax, 43, 23, 14, 12,
                 "PLS latent\nscores\n$x_1 \\ldots x_n$\n(train only)", GREY))
    b.append(box(ax, 60, 37, 17, 10,
                 "Full spectrum\n→  PLS  ·  1-D CNN", SAND))
    b.append(box(ax, 60, 23, 17, 10, "Latent scores\n→  SVR  ·  RF  ·  MLP", SAND))
    b.append(box(ax, 60, 9, 17, 10, "Latent scores\n→  KAN", VIOL))
    b.append(box(ax, 80, 30, 19, 12,
                 "Accuracy\n$R^2$ · RMSEP · MAE\nRPD · F1 · LOD\ncluster-bootstrap CI", GREY))
    b.append(box(ax, 80, 5, 19, 18,
                 "Interpretability\n• closed-form equation\n• response curves vs\n   the linear effect\n"
                 "• loadings → bands\n• intrinsic vs SHAP", VIOL))
    for y in (39.5, 18.5):
        arrow(ax, 21, y, 23, 29)
    arrow(ax, 40, 29, 43, 29)
    for y in (42, 28, 14):
        arrow(ax, 57, 29, 60, y)
    for y in (42, 28, 14):
        arrow(ax, 77, y, 80, 36)
    arrow(ax, 77, 14, 80, 14)
    panel(ax, "a", dx=0.0, dy=0.93)
    check_fit(fig, ax, b)

    # ------------------------------------------------- (b) leakage-free splitting protocol
    ax = axes[1]
    ax.set_xlim(-2, 102); ax.set_ylim(-1, 45); ax.axis("off")
    b = []
    ax.text(23, 41, "Tahini — split by physical sample", ha="center", fontsize=7, fontweight="bold")
    b.append(box(ax, 0, 24, 26, 12,
                 "5 tahini lots × (1 authentic +\n10 blends) = 55 physical samples;\n"
                 "each scanned 14–17 times\n(authentic lots 150–151 times)", BLUE))
    b.append(box(ax, 0, 12, 26, 9,
                 "grouped hold-out by sample\n38 train / 17 test samples\n(1146 / 408 spectra)", GREEN))
    b.append(box(ax, 0, 1, 26, 8,
                 "also: 10 independent grouped\nsplits; leave-one-lot-out", GREEN))
    arrow(ax, 13, 24, 13, 21); arrow(ax, 13, 12, 13, 9)

    ax.text(74, 41, "Mango — published across-season split", ha="center", fontsize=7, fontweight="bold")
    b.append(box(ax, 50, 24, 24, 12,
                 "4 harvest seasons, 2 regions,\n10 cultivars, 112 populations;\n"
                 "reference dry matter by\noven drying", BLUE))
    b.append(box(ax, 50, 12, 24, 9,
                 "seasons 1–3 → calibration\n(94 populations, n = 10,243)", GREEN))
    # The arrow between these two boxes could be read as data flowing into the test set, which is
    # the opposite of the protocol. The box says what the test set is, where there is room for it.
    b.append(box(ax, 76, 12, 24, 9,
                 "season 4 → external test\n(18 populations, n = 1448)\n"
                 "predicted once, never fitted on", GREEN))
    b.append(box(ax, 50, 1, 50, 8,
                 "no population in common between the two sides;\n"
                 "also: grouped CV by population, leave-one-season-out", GREEN))
    arrow(ax, 62, 24, 62, 21)
    arrow(ax, 74, 16.5, 76, 16.5)
    arrow(ax, 62, 12, 62, 9)
    ax.plot([38, 38], [0, 38], color="0.85", lw=0.8, ls=":")
    panel(ax, "b", dx=0.0, dy=0.90)
    check_fit(fig, ax, b)

    fig.tight_layout()
    check_no_title(fig)
    save(fig, "Figure_01_pipeline")


if __name__ == "__main__":
    fig_pipeline()
