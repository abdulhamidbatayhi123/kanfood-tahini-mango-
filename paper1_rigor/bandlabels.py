"""The vibrational band annotations used in every figure, in one place.

Two reasons this is shared rather than repeated. First, the same band must carry the same label
wherever it appears; the spectra figure and the chemistry figure previously wrote "CH$_2$ bending"
and "CH2 bending" for the same band. Second, bands closer together than a rotated label is wide
cannot each be labelled: three separate labels across the 2850-3010 cm-1 C-H stretching region
collided however they were staggered. Such bands are grouped under one heading here, and each
still gets its own guide line. The individual assignments belong in the text and in Table 1.

Sources: mid-infrared, Guillen and Cabo (1997) and Rohman and Che Man (2010); short-wave
near-infrared, Subedi and Walsh (2011).
"""
import numpy as np

# (positions in cm-1 or nm, label)
MIR_ANNOT = [((3010, 2920, 2860), "C–H stretch"),
             ((1745,), r"$\nu$(C=O)"),
             ((1652,), r"$\nu$(C=C)"),
             ((1465, 1385), "CH$_2$ / CH$_3$ bend"),
             ((1165, 1105), r"$\nu$(C–O)"),
             ((720,), r"$\rho$(CH$_2$)")]

NIR_ANNOT = [((735,), "O–H 3rd ot."),
             ((770,), "O–H/C–H 3rd ot."),
             ((840,), "C–H 3rd ot."),
             ((915,), r"C–H 3rd ot. (CH$_2$)"),
             ((970,), "O–H 2nd ot.")]


def annotate(ax, annot, headroom=0.55, labels=True, fontsize=5.5):
    """Guide lines at every band, and one staggered label per group.

    `labels=False` draws the guide lines only, for panels that sit beside a labelled one.
    `headroom` reserves space above the data so the labels do not sit on the curves.
    """
    y0, y1 = ax.get_ylim()
    if labels:
        y1 = y0 + (y1 - y0) * (1 + headroom)
        ax.set_ylim(y0, y1)
    lo_lim, hi_lim = sorted(ax.get_xlim())
    levels = [0.995, 0.845, 0.695]
    shown = 0
    for positions, label in annot:
        inside = [p for p in positions if lo_lim <= p <= hi_lim]
        if not inside:
            continue
        for p in inside:
            ax.axvline(p, color="0.85", ls=":", lw=0.6, zorder=0, gid="guide")
        if labels:
            ax.text(float(np.mean(inside)), y0 + levels[shown % len(levels)] * (y1 - y0),
                    label, rotation=90, fontsize=fontsize, va="top", ha="center", color="0.40")
        shown += 1
