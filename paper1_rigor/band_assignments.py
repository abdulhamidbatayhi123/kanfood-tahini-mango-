"""The selected bands, with their vibrational assignments where a tabulated one exists.

An equation written in named wavenumbers is only worth more than an equation written in latent
variables if the names mean something. This table is what lets a reader check that: every channel
the selection retained, in order, with the assignment given by the cited band tables — and, where
a channel falls outside every tabulated range, the word "unassigned" rather than a plausible guess.

Selection is recomputed here through the same code path the experiments use, so the table cannot
drift away from the bands the equation was actually fitted on.
"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

from kanfood.bands import BANDS, assign_band as assign_ftir
from kanfood.preprocess import Preprocessor
from paper1_rigor.common import TAHINI, MANGO, tahini_split, mango_split, save
from paper1_rigor.nir_bands import NIR_BANDS, assign_band as assign_nir
from paper1_rigor.r13_band_equation import score_vector, select

# Assignments outside the two tabulated lists are NOT invented. Where a retained channel falls in
# a region that the cited sources describe but do not tabulate as a discrete band, the region is
# named and marked as such; everything else is left unassigned.
REGIONS_MIR = [((2950, 2975), "C–H stretch region (CH3 asymmetric)", "Lipid acyl chains"),
               ((1180, 1320), "C–O stretch / CH2 wagging region", "Esters, acyl chains"),
               ((960, 1000), "=C–H out-of-plane bend region", "Unsaturation (trans)")]


def describe(nm, assign, regions):
    a = assign(float(nm))
    if a["assignment"] != "Unassigned":
        return a["assignment"], a["component"], "tabulated band"
    for (lo, hi), name, comp in regions:
        if lo <= nm <= hi:
            return name, comp, "region, not a tabulated band"
    return "unassigned", "—", "no tabulated assignment"


def run(label, splitter, cfg, k, min_sep, assign, regions, unit):
    ds, tr, te = splitter()
    pp = Preprocessor(cfg["preprocess"]).fit(ds.X[tr])
    idx = select(score_vector(pp.transform(ds.X[tr]), ds.y[tr], "vip"), ds.wavenumbers, k, min_sep)
    rows = []
    for rank, i in enumerate(idx, 1):
        nm = float(ds.wavenumbers[i])
        a, comp, basis = describe(nm, assign, regions)
        rows.append({"Dataset": label, "Order": rank, "Position": f"{nm:.0f}", "Unit": unit,
                     "Assignment": a, "Associated with": comp,
                     "Source of the assignment": basis})
    return rows


def main():
    # Tahini is tabulated at k = 16, the selection Section 3.4 lists and the one its headline
    # comparison is fitted on, so that every band the article names can be looked up here. It was
    # tabulated at k = 12 until 2026-08-27, which left four of the sixteen named in the text
    # (3022, 1472, 1196 and 974 cm-1) with no row, and described the twelve as "the bands the
    # equation retains" when they are the bands it was *offered*: the additive equation prunes to
    # six of sixteen, or eight of twelve. The last column marks the eight-band selection that
    # Equation (3) is fitted on and retains in full, which is a strict subset of these sixteen.
    rows = run("tahini", tahini_split, TAHINI, 16, 15.0, assign_ftir, REGIONS_MIR, "cm⁻¹")
    eq3 = {r["Position"] for r in
           run("tahini", tahini_split, TAHINI, 8, 15.0, assign_ftir, REGIONS_MIR, "cm⁻¹")}
    for r in rows:
        r["In Equation (3)"] = "yes" if r["Position"] in eq3 else "—"
    mango = run("mango", mango_split, MANGO, 8, 9.0, assign_nir, [], "nm")
    for r in mango:
        r["In Equation (3)"] = "—"          # Equation (2) is in latent variables, not in bands
    rows += mango
    from pathlib import Path
    df = pd.DataFrame(rows)
    save(df, "band_assignments.csv")
    out = (Path(__file__).resolve().parents[1] / "submission_foods" / "tables"
           / "Table_12_band_assignments.csv")
    df.to_csv(out, index=False)
    print(f"  -> tables/{out.name}")
    print(df.to_string(index=False))
    n_unassigned = int((df["Assignment"] == "unassigned").sum())
    print(f"\n{len(df)} bands, {n_unassigned} with no tabulated assignment "
          f"(reported as such, not guessed)")


if __name__ == "__main__":
    main()
