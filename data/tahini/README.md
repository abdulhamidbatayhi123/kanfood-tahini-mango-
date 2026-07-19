# Tahini adulteration FTIR dataset

Open dataset accompanying the paper *"Interpretability without sacrificing accuracy: Kolmogorov–Arnold
networks with closed-form calibration equations for food adulteration and quality analysis by vibrational
spectroscopy."* Released for full reproducibility and transparency.

**License:** Creative Commons Attribution 4.0 International (CC-BY-4.0) — see `../../LICENSE-DATA`.
If you use this dataset, please cite the paper and this repository (see `../../CITATION.cff`).

## Overview

Mid-infrared (MIR) FTIR spectra of **tahini (sesame paste)** adulterated with **sunflower paste** and
**peanut paste**. The task is quantitative: predict the composition (and, as a binary label, whether a
sample is adulterated).

| Property | Value |
|----------|-------|
| File | `Tahin_Aycicek1.xlsx` (single sheet) |
| Rows | 1,554 spectra |
| Spectral channels | 1,762 |
| Spectral range | ≈ 600–4000 cm⁻¹ (mid-infrared), stored high→low; the ester C=O band at 1743 cm⁻¹ confirms the MIR axis |
| Instrument | Bruker Tensor 27 (Germany) with ATR accessory |
| Resolution | 4 cm⁻¹, 16 co-added scans per spectrum |
| Physical samples | 55 — 5 pure tahini, 25 tahini+sunflower-paste, 25 tahini+peanut-paste |
| Replicates | each physical sample scanned 14–151 times (≈15 per adulteration level; ≈150 per authentic sample) |
| Adulteration levels | 4 %–100 % (component fractions sum to 100 %) |
| Provenance | tahini from reliable suppliers in Istanbul, Türkiye; sunflower/peanut pastes from a local supermarket; measured in the authors' laboratory |

## Column dictionary

The spreadsheet has five metadata columns; **every other column is a wavenumber** whose header is the
value in cm⁻¹ and whose cells are absorbance (a.u.).

| Column (original Turkish) | Meaning | Type / units |
|---------------------------|---------|--------------|
| `isim` | Physical-sample identifier | string — **the grouping key: all replicate scans of one sample share it. Always split by `isim` to avoid replicate leakage.** |
| `Tağşiş Var mı` | "Is it adulterated?" — binary adulteration label | integer, 1 = adulterated, 0 = pure |
| `Tahin Oranı` | Tahini fraction | % (0–100) |
| `Ayçiçek ezmesi oranı` | Sunflower-paste fraction | % (0–100) |
| `Fıstık Ezmesi Oranı` | Peanut-paste fraction | % (0–100) |
| *(numeric headers, e.g. `1743.2`)* | Wavenumber | cm⁻¹; cell = absorbance (a.u.) |

The three composition columns sum to 100 % for each spectrum.

## Loading

The `kanfood` package reads this file directly:

```python
from kanfood.data import load_tahini
ds = load_tahini()          # finds data/tahini/Tahin_Aycicek1.xlsx (or the repo-root copy)
# ds.X: (1554, 1762) absorbance, columns sorted ascending in cm⁻¹
# ds.y: (1554, 3) composition [tahini, peanut, sunflower] %, summing to 100
# ds.groups: (1554,) physical-sample id (isim)  — use for leakage-free grouped splits
# ds.tagsis: (1554,) 1=adulterated, 0=pure
# ds.wavenumbers: (1762,) ascending cm⁻¹ axis
```

Override the location with the `KANFOOD_TAHINI_PATH` environment variable if you keep the file elsewhere.

## Important: leakage-free use

Because each physical sample is scanned many times, a naive random train/test split places replicate scans
of the *same* sample on both sides and inflates every metric. **Always split by `isim` (grouped split).**
The `kanfood` pipeline does this by default (`kanfood/split.py`).
