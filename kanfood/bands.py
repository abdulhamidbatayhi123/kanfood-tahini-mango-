# FTIR (mid-IR) assignments in WAVENUMBER cm-1 for edible fats/oils/pastes.
# Refs: Guillen & Cabo (1997); Rohman & Che Man (2010); Vlachos et al. (2006).
BANDS = [
    ((3000, 3020), "=C-H stretch (cis C=C)", "Unsaturated fatty acids"),
    ((2900, 2940), "C-H asymmetric stretch (CH2)", "Lipid acyl chains"),
    ((2840, 2880), "C-H symmetric stretch (CH2/CH3)", "Lipid acyl chains"),
    ((1740, 1750), "C=O ester stretch", "Triglycerides"),
    ((1640, 1665), "C=C stretch", "Unsaturation"),
    ((1455, 1475), "CH2 bending (scissor)", "Lipids"),
    ((1370, 1400), "CH3 symmetric bend", "Lipids"),
    ((1150, 1180), "C-O stretch", "Esters"),
    ((1090, 1120), "C-O stretch", "Esters/glycerol"),
    ((715, 725), "CH2 rocking", "Long acyl chains"),
]


def assign_band(wavenumber: float) -> dict:
    for (lo, hi), assignment, component in BANDS:
        if lo <= wavenumber <= hi:
            return {"assignment": assignment, "component": component}
    return {"assignment": "Unassigned", "component": "-"}
