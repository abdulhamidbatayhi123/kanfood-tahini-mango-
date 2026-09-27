"""Short-wave near-infrared band assignments for intact mango fruit, 684-990 nm.

Assignments follow the short-wave NIR literature for intact banana and mango fruit
(Subedi and Walsh, Postharvest Biol. Technol. 2011, 62, 238-245) and the standard overtone
progression for O-H and C-H stretching. Only bands that fall inside the modelled 684-990 nm
window are listed; the dominant feature for dry matter is the second O-H overtone near 970 nm,
because water and dry matter are complementary fractions of the fruit.
"""

NIR_BANDS = [
    ((720, 750), "O–H 3rd overtone", "Water"),
    ((760, 780), "O–H 3rd overtone / C–H 3rd overtone", "Water, carbohydrate"),
    ((830, 850), "C–H 3rd overtone", "Sugars, starch"),
    ((900, 930), "C–H 3rd overtone (CH₂/CH₃)", "Carbohydrate"),
    ((955, 985), "O–H 2nd overtone", "Water (complement of dry matter)"),
]


def assign_band(nm: float) -> dict:
    for (lo, hi), assignment, component in NIR_BANDS:
        if lo <= nm <= hi:
            return {"assignment": assignment, "component": component}
    return {"assignment": "Unassigned", "component": "-"}
