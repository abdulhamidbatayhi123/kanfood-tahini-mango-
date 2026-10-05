"""Publication figure style for the Foods submission.

Rules enforced here, each of them a lesson from the companion paper's revision:
  * no title inside the image -- MDPI puts the description in the caption, and an argumentative
    title baked into a figure reads as advocacy rather than evidence;
  * panel letters (a), (b), ... in bold at the top-left of every panel, inside the image frame;
  * no internal project labels anywhere;
  * a single colour per model across every figure, so the reader learns the mapping once;
  * vector PDF and 600 dpi PNG written together from the same call, so the two can never drift
    apart (in the previous draft the PDFs were four weeks staler than the PNGs and disagreed with
    the tables).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = Path(__file__).resolve().parents[1] / "submission_foods" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# One colour per model, used everywhere.
MODEL_COLOR = {
    "PLS": "#2563EB",   # blue
    "SVM": "#059669",   # green
    "RF":  "#D97706",   # amber
    "MLP": "#DC2626",   # red
    "CNN": "#6B7280",   # grey
    "KAN": "#7C3AED",   # violet
}
MODEL_ORDER = ["PLS", "SVM", "RF", "MLP", "CNN", "KAN"]
ACCENT = "#7C3AED"
NEUTRAL = "#94A3B8"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "font.size": 9,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.3,
    "pdf.fonttype": 42,      # embed TrueType so the PDF is editable/searchable
    "ps.fonttype": 42,
})

# MDPI single-column ~ 8.6 cm, double-column ~ 17.8 cm
COL1, COL2 = 3.39, 7.01   # inches


PANEL_GID = "panel-letter"


def panel(ax, letter, dx=-0.02, dy=1.04):
    """Bold panel letter at the top-left of an axis, inside the image frame.

    The letter sits just *outside* the axis frame by design, which is where MDPI's own figures
    put it; it is tagged so that `check_inside_axes` does not flag the one artist whose position
    this function fixes and guarantees.
    """
    t = ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=9.5,
                fontweight="bold", va="bottom", ha="left")
    t.set_gid(PANEL_GID)
    return t


def save(fig, name):
    """Write the PDF and the PNG from one call so they can never disagree."""
    pdf, png = FIGDIR / f"{name}.pdf", FIGDIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"  -> {pdf.name} + {png.name}")
    return pdf


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def check_no_overlap(fig, ignore=("",)):
    """Guard: no two text annotations inside the same axis may overlap.

    Two 'not applicable' labels printed on top of one another in the first rendered proof of
    Figure 3 and were only caught by looking at the PDF. Overlap is cheap to test, so it is
    tested rather than eyeballed. Tick labels and axis labels are excluded -- matplotlib lays
    those out itself -- and only `Text` artists added by the figure code are compared.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    problems = []
    for ax in fig.axes:
        skip = set(ax.get_xticklabels()) | set(ax.get_yticklabels())
        skip |= {ax.xaxis.label, ax.yaxis.label, ax.title}
        texts = [t for t in ax.texts
                 if t not in skip and t.get_text().strip() not in ignore and t.get_visible()]
        boxes = [(t, t.get_window_extent(renderer=r)) for t in texts]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i][1], boxes[j][1]
                if a.overlaps(b):
                    problems.append(f"{boxes[i][0].get_text()!r} overlaps "
                                    f"{boxes[j][0].get_text()!r}")
    assert not problems, "overlapping text in a publication figure: " + "; ".join(problems)


def check_inside_axes(fig, tol=0.02):
    """Guard: every annotation added by the figure code stays inside its axis.

    A label that spills past the frame is the two-dimensional version of a legend wider than its
    box, which the companion paper's revision had to fix by hand.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    problems = []
    for ax in fig.axes:
        skip = set(ax.get_xticklabels()) | set(ax.get_yticklabels())
        skip |= {ax.xaxis.label, ax.yaxis.label, ax.title}
        ab = ax.get_window_extent(renderer=r)
        pad_x, pad_y = ab.width * tol, ab.height * tol
        for t in ax.texts:
            if (t in skip or not t.get_visible() or not t.get_text().strip()
                    or t.get_gid() == PANEL_GID):
                continue
            tb = t.get_window_extent(renderer=r)
            if (tb.x0 < ab.x0 - pad_x or tb.x1 > ab.x1 + pad_x
                    or tb.y0 < ab.y0 - pad_y or tb.y1 > ab.y1 + pad_y):
                problems.append(repr(t.get_text()))
    assert not problems, "text outside its axis frame: " + "; ".join(problems)


def check_no_title(fig):
    """Guard: assert no axes title or suptitle survived into a publication figure."""
    bad = [a.get_title() for a in fig.axes if a.get_title()]
    if fig._suptitle is not None and fig._suptitle.get_text():
        bad.append(fig._suptitle.get_text())
    assert not bad, f"figure carries an internal title (put it in the caption instead): {bad}"


# The analysis code calls the support-vector model "SVM"; the article calls it SVR (support vector
# regression), and a figure must use the article's name. Every model label a reader sees goes
# through mlabel().
MODEL_LABEL = {"SVM": "SVR"}


def mlabel(m):
    return MODEL_LABEL.get(m, m)


def signed(v, fmt):
    """A number with a true minus sign (U+2212), as in the article's text and tables."""
    return format(v, fmt).replace("-", "−")


def check_legend_clear(fig, samples=40):
    """Guard: no plotted line, curve or marker passes underneath a legend.

    In the first-round figures two legends sat on the spectra they described, which the eye
    forgives on screen and a reviewer does not on paper. Every line is sampled densely along its
    segments (so a straight line crossing the legend box is caught even with no vertex inside)
    and every scatter offset is tested against the legend's window extent.
    """
    import numpy as _np
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    problems = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None or not leg.get_visible():
            continue
        bb = leg.get_window_extent(renderer=r)

        def inside(pts):
            return _np.any((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1)
                           & (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1))

        for ln in ax.get_lines():
            # Faint reference lines (band guides, the zero line) may pass behind an opaque
            # legend box; they are tagged gid="guide" and the legend is drawn with a white face.
            if not ln.get_visible() or ln.get_gid() == "guide":
                continue
            xy = _np.asarray(ln.get_xydata(), float)
            xy = ln.get_transform().transform(xy[_np.isfinite(xy).all(1)])
            if len(xy) > 1:
                t = _np.linspace(0, 1, samples)[:, None]
                xy = _np.concatenate([a + t * (b - a) for a, b in zip(xy[:-1], xy[1:])])
            if len(xy) and inside(xy):
                problems.append(f"axis {fig.axes.index(ax)}: line {ln.get_label()!r}")
        for c in ax.collections:
            off = _np.asarray(c.get_offsets(), float)
            if len(off) and inside(c.get_offset_transform().transform(off)):
                problems.append(f"axis {fig.axes.index(ax)}: markers {c.get_label()!r}")
        for t in ax.texts:
            if t.get_visible() and t.get_text().strip() and t.get_gid() != PANEL_GID                     and t.get_window_extent(renderer=r).overlaps(bb):
                problems.append(f"axis {fig.axes.index(ax)}: label {t.get_text()!r}")
    assert not problems, "data underneath a legend: " + "; ".join(problems)


# A legend that sits over faint reference lines hides them behind a white face rather than letting
# dotted guides run through its text.
OPAQUE_LEGEND = dict(frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)
