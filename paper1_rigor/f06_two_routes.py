"""Figure 5: the two routes from a trained network to a printed equation.

Section 3.3's argument is that the accuracy lost in printing a Kolmogorov-Arnold network belongs to
the *conversion* rather than to the architecture, and that an already-elementary edge basis removes
the conversion rather than improving it. That is the paper's central claim and it was argued only in
prose; a reader had to hold four steps and their failure modes in mind to see why the two routes are
not variants of one procedure but different in kind.

The figure carries no result of its own -- every number annotated on it is reported in Table 6 and
Section 3.3 -- so it adds explanation without adding evidence, which is what a schematic should do.

Text fit is checked programmatically rather than by eye, as in `f00_schematic`: two boxes in the
companion paper's revision overflowed and were caught only by comparing rendered text extents
against their containers.

Output: `submission_foods/figures/Figure_05_two_routes.{pdf,png}`.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from paper1_rigor.figstyle import COL2, check_no_title, save

BOX = dict(boxstyle="round,pad=0.30,rounding_size=0.06", linewidth=0.8)
BLUE, VIOL, GREY, GREEN, SAND = "#DBEAFE", "#EDE9FE", "#F1F5F9", "#DCFCE7", "#FEF3C7"


def box(ax, x, y, w, h, text, fc, ec="0.35", fs=6.6, weight="normal", style="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=fc, edgecolor=ec, **BOX))
    t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                linespacing=1.3, fontweight=weight, fontstyle=style, zorder=3)
    return (x, y, w, h, t)


def arrow(ax, x1, y1, x2, y2, color="0.35", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
                                 linewidth=0.9, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=1))


def check_fit(fig, ax, boxes, margin=0.98):
    """Assert every label's rendered extent sits inside its box, in data coordinates."""
    fig.canvas.draw()
    inv = ax.transData.inverted()
    bad = []
    for (x, y, w, h, t) in boxes:
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        if (x1 - x0) > w * margin or (y1 - y0) > h * margin:
            bad.append((t.get_text().replace("\n", " / ")[:40],
                        round((x1 - x0) / w, 2), round((y1 - y0) / h, 2)))
    assert not bad, "text overflows its box (width, height ratios): " + repr(bad)


def main():
    fig, ax = plt.subplots(figsize=(COL2, 3.5))
    ax.set_xlim(-1, 101)
    ax.set_ylim(-1.5, 58)
    ax.axis("off")
    b = []

    # ---------------------------------------------------------- route A: the conversion
    ax.text(0, 54.5, "Route A — convert a B-spline network", fontsize=7.6, fontweight="bold",
            ha="left", va="center")
    b.append(box(ax, 0, 38, 19, 11,
                 "trained KAN\nwith cubic\nB-spline edges", VIOL))
    b.append(box(ax, 22, 38, 15, 11, "prune", GREY))
    b.append(box(ax, 40, 38, 24, 11,
                 "replace each edge with\nthe best-fitting\nelementary form", SAND))
    b.append(box(ax, 67, 38, 15, 11, "refit\njointly", SAND))
    b.append(box(ax, 85, 38, 15, 11, "printed\nequation", BLUE))
    for x1, x2 in ((19, 22), (37, 40), (64, 67), (82, 85)):
        arrow(ax, x1, 43.5, x2, 43.5)

    b.append(box(ax, 22, 27.5, 78, 8.0,
                 "an approximation of the network: fidelity 0.78–0.98 on tahini and 0.20–0.95 on "
                 "mango,\ndiffering between runs of the same extraction, and one tahini "
                 "extraction in ten unusable", "#FEE2E2", ec="#B91C1C", fs=6.3))

    # ---------------------------------------------------------- route B: no conversion
    ax.text(0, 21.5, "Route B — print a network whose edges are already elementary",
            fontsize=7.6, fontweight="bold", ha="left", va="center")
    b.append(box(ax, 0, 6, 19, 11,
                 "trained KAN with\nChebyshev, power,\nGaussian or\nFourier edges", VIOL))
    b.append(box(ax, 22, 6, 15, 11, "prune", GREY))
    b.append(box(ax, 40, 6, 24, 11,
                 "no substitution step:\nthe network already\n$is$ a closed form", GREEN))
    b.append(box(ax, 85, 6, 15, 11, "printed\nequation", BLUE))
    arrow(ax, 19, 11.5, 22, 11.5)
    arrow(ax, 37, 11.5, 40, 11.5)
    arrow(ax, 64, 11.5, 85, 11.5)
    ax.text(74.5, 12.6, "print", fontsize=6.4, ha="center", va="bottom", color="0.25")

    b.append(box(ax, 22, 0.0, 78, 4.6,
                 "a change of notation: fidelity above 0.9999 in every cell, and repeated runs "
                 "return the identical equation", GREEN, ec="#15803D", fs=6.3))

    check_fit(fig, ax, b)
    check_no_title(fig)
    save(fig, "Figure_05_two_routes")


if __name__ == "__main__":
    main()
