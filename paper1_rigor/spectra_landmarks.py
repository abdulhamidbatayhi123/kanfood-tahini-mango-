"""Tabulate the difference spectra of Figure 2b so the prose that describes them is traceable.

Section 3.1 says which adulterant is the more absorbing where, and where the largest excursions
are.  Until 2026-08-27 those statements were read off the figure by eye and two of them were wrong:
the C-H stretching direction was reversed (peanut is the more absorbing at 2920 and 2851 cm-1, not
sunflower), and the ester carbonyl and CH2 scissoring bands, named as carrying "the largest
excursions", carry two of the smallest.  This module writes the numbers out of the same arrays the
figure is drawn from, so the sentence and the panel cannot disagree again.

Output: `results_rigor/spectra_difference_landmarks.csv`.
"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

from paper1_rigor.common import ROOT, save

P1 = ROOT / "results_phase1"

# The bands Section 3.1 names, with the wavenumber it names them at.
BANDS = [(3010, "cis =C-H stretch"), (2962, "CH3 asymmetric stretch"),
         (2920, "CH2 asymmetric stretch"), (2851, "CH2 symmetric stretch"),
         (1745, "ester C=O stretch"), (1465, "CH2 scissoring"),
         (1105, "C-O stretch"), (989, "=C-H out-of-plane bend"), (720, "CH2 rocking")]

# The windows the prose summarises.
WINDOWS = [(600, 1200, "fingerprint below 1200"), (2800, 3050, "C-H stretching region"),
           (2900, 2940, "CH2 asymmetric band"), (2840, 2875, "CH2 symmetric band"),
           (1090, 1180, "C-O envelope")]


def main():
    a = np.load(P1 / "phase1_artifacts.npz", allow_pickle=True)
    w = np.asarray(a["wavenumbers"], dtype=float)
    d = {"sunflower": np.asarray(a["mean_sunflower"]) - np.asarray(a["mean_tahini"]),
         "peanut": np.asarray(a["mean_peanut"]) - np.asarray(a["mean_tahini"])}

    rows = []
    for nominal, name in BANDS:
        i = int(np.argmin(np.abs(w - nominal)))
        rows.append({"Kind": "band", "Feature": name, "Nominal (cm-1)": nominal,
                     "Channel (cm-1)": round(float(w[i]), 2),
                     "Sunflower - tahini": round(float(d["sunflower"][i]), 5),
                     "Peanut - tahini": round(float(d["peanut"][i]), 5),
                     "Channels": 1})
    for lo, hi, name in WINDOWS:
        m = (w >= lo) & (w <= hi)
        rows.append({"Kind": "window mean", "Feature": name, "Nominal (cm-1)": f"{lo}-{hi}",
                     "Channel (cm-1)": "—",
                     "Sunflower - tahini": round(float(d["sunflower"][m].mean()), 5),
                     "Peanut - tahini": round(float(d["peanut"][m].mean()), 5),
                     "Channels": int(m.sum())})
    for who, v in d.items():
        for rank, j in enumerate(np.argsort(-np.abs(v))[:3], 1):
            rows.append({"Kind": f"largest |difference|, {who}", "Feature": f"rank {rank}",
                         "Nominal (cm-1)": "—", "Channel (cm-1)": round(float(w[j]), 2),
                         "Sunflower - tahini": round(float(d["sunflower"][j]), 5),
                         "Peanut - tahini": round(float(d["peanut"][j]), 5), "Channels": 1})

    # The sign claim of Section 3.1, counted rather than eyeballed.
    m = (w >= 600) & (w <= 1200)
    frac_s = 100.0 * float((d["sunflower"][m] > 0).mean())
    frac_p = 100.0 * float((d["peanut"][m] < 0).mean())
    rows.append({"Kind": "sign agreement below 1200 (% of channels)", "Feature":
                 "sunflower positive / peanut negative", "Nominal (cm-1)": "600-1200",
                 "Channel (cm-1)": "—", "Sunflower - tahini": round(frac_s, 1),
                 "Peanut - tahini": round(frac_p, 1), "Channels": int(m.sum())})

    df = pd.DataFrame(rows)
    save(df, "spectra_difference_landmarks.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
