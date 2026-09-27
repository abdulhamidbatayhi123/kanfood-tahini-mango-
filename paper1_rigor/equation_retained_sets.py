"""Which bands the extracted equation actually keeps, and how far that set moves with the seed.

Table 7 reports the retained count as a median over five seeds. A median hides two things a reader
implementing the equation needs: the spread of the count, and whether it is the *same* bands each
time. It is not. This module writes both out of the same raw file Table 7 is built from.

Output: `results_rigor/band_equation_retained_sets.csv`.
"""
import re
import sys
from functools import reduce

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

from paper1_rigor.common import ROOT, save


def main():
    d = pd.read_csv(ROOT / "results_rigor" / "r13_band_equation_raw_tahini.csv")
    rows = []
    for (stage, method, k, model), g in d[d.stage.str.startswith("external")].groupby(
            ["stage", "method", "k", "model"]):
        if "equation" not in model:
            continue
        sets = []
        for _, r in g.iterrows():
            bands = str(r["bands"]).split(";")
            idx = {int(i) for i in re.findall(r"x_(\d+)", str(r["equation"]))}
            sets.append({bands[i - 1] for i in idx if 0 < i <= len(bands)})
        if not sets:
            continue
        union = sorted(reduce(set.union, sets), key=lambda s: -int(s))
        common = sorted(reduce(set.intersection, sets), key=lambda s: -int(s))
        rows.append({
            "Dataset": "tahini", "Band set": stage, "Selection": method, "Bands offered": int(k),
            "Model": model, "Seeds": len(sets),
            "Bands retained, median": int(pd.Series([len(s) for s in sets]).median()),
            "Bands retained, min": min(len(s) for s in sets),
            "Bands retained, max": max(len(s) for s in sets),
            "Terms, min": int(g.n_terms.min()), "Terms, max": int(g.n_terms.max()),
            "Bands in at least one equation": len(union),
            "Bands in every equation": len(common),
            "The bands in every equation": ";".join(common) if common else "—"})
    df = pd.DataFrame(rows)
    save(df, "band_equation_retained_sets.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
