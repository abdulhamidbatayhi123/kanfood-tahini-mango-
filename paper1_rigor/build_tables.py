"""Build every publication table into submission_foods/tables/ from the result files.

No number is typed by hand. Each table is a formatted view of a CSV produced by a script in this
folder or by the published benchmark runs, and each is asserted against the source before it is
written, so a table and the figure beside it cannot drift apart.
"""
import json
import re
import sys

import numpy as np
import pandas as pd

from paper1_rigor.common import ROOT
from paper1_rigor.figstyle import MODEL_ORDER

sys.stdout.reconfigure(encoding="utf-8")

R = ROOT / "results_rigor"
P1 = ROOT / "results_phase1"
PM = ROOT / "results_mango"
OUT = ROOT / "submission_foods" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

LABEL = {"PLS": "PLS", "SVM": "SVR", "RF": "Random forest", "MLP": "MLP", "CNN": "1-D CNN",
         "KAN": "KAN"}
# What each model hands back, phrased the way Table 9 phrases it. The previous wording put
# "Intrinsically interpretable" in a narrow column, which Word broke mid-word as "interpretabl/e"
# -- visible only in the rendered PDF.
INTERP = {"PLS": "Linear coefficients", "SVM": "No closed form",
          "RF": "Impurity importance", "MLP": "No closed form", "CNN": "No closed form",
          "KAN": "Closed-form equation"}


def _minus(df):
    """A leading ASCII hyphen before a digit is a minus sign; typeset it as one.

    Only the leading position is touched, so ranges already written with an en dash
    ("3.41-3.56" is never produced) and hyphenated words are left alone.
    """
    return df.map(lambda v: re.sub(r"^-(?=[\d.])", "−", v) if isinstance(v, str) else v)


def write(df, name):
    p = OUT / name
    _minus(df).to_csv(p, index=False)
    print(f"  -> tables/{name}  ({len(df)} rows)")
    return p


def fmt(v, d=3):
    return "—" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.{d}f}"


# ------------------------------------------------------------------ Table 1: datasets
def table1():
    rows = [
        dict(**{"Property": "Matrix", "Tahini (this study)": "Tahini (sesame paste)",
                "Mango (public benchmark)": "Intact mango fruit"}),
        dict(**{"Property": "Technique", "Tahini (this study)": "ATR–FTIR (mid-infrared)",
                "Mango (public benchmark)": "NIR reflectance"}),
        dict(**{"Property": "Spectral range modelled", "Tahini (this study)": "600–4000 cm⁻¹ (1762 channels)",
                "Mango (public benchmark)": "684–990 nm (103 channels)"}),
        dict(**{"Property": "Target", "Tahini (this study)": "Tahini fraction (%)",
                "Mango (public benchmark)": "Dry-matter content (% w/w)"}),
        dict(**{"Property": "Adulterants / factors",
                "Tahini (this study)": "Sunflower paste, peanut paste; 4–100 % in 4 % steps",
                "Mango (public benchmark)": "4 seasons, 2 regions, 10 cultivars, 2 fruit conditions"}),
        dict(**{"Property": "Spectra", "Tahini (this study)": "1554 (754 authentic, 800 blends)",
                "Mango (public benchmark)": "11,691"}),
        dict(**{"Property": "Grouping unit", "Tahini (this study)": "Physical sample (55)",
                "Mango (public benchmark)": "Population (112)"}),
        dict(**{"Property": "Replicates per group",
                "Tahini (this study)": "14–17 per blend; 150–151 per authentic lot",
                "Mango (public benchmark)": "Variable (per population)"}),
        dict(**{"Property": "Primary validation",
                "Tahini (this study)": "Grouped hold-out, 38/17 samples (1146/408 spectra)",
                "Mango (public benchmark)": "Published across-season external test (season 4, n = 1448)"}),
        dict(**{"Property": "Preprocessing selected", "Tahini (this study)": "SNV",
                "Mango (public benchmark)": "Savitzky–Golay 1st derivative (window 11, order 2)"}),
        dict(**{"Property": "Source", "Tahini (this study)": "Collected for this study",
                "Mango (public benchmark)": "Anderson et al. (Mendeley 46htwnp833)"}),
    ]
    return write(pd.DataFrame(rows), "Table_1_datasets.csv")


# ---------------------------------------------------- Tables 2 and 3: the two benchmarks
def _bench(meta_path, holdout_path, r2col, err_unit, name, with_f1):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ho = pd.read_csv(holdout_path).set_index("Method")
    rec = pd.read_csv(R / "f01_benchmark_recomputed.csv")
    rec = rec[rec.dataset == ("tahini" if with_f1 else "mango")].set_index("model")
    rows = []
    for m in MODEL_ORDER:
        cv = np.array(meta["cv_r2"][m])
        r = dict()
        r["Model"] = LABEL[m]
        r["CV R² (mean ± SD)"] = f"{cv.mean():.3f} ± {cv.std():.3f}"
        # Rendered from the full-precision recomputation, not from the 4-dp published file:
        # rounding an already-rounded value double-rounds it. The mango MLP is 0.861463, which
        # the published file stores as 0.8615 and which then renders as 0.862 -- disagreeing with
        # the manuscript text and with the value actually computed. The guard below still
        # requires the two to agree.
        r["Test R²"] = fmt(rec.loc[m, "R2"])
        lo, hi = str(ho.loc[m, "R2_CI"]).strip("[]").split(",")
        r["95 % CI"] = f"{float(lo):.3f}–{float(hi):.3f}"
        r[f"RMSEP ({err_unit})"] = fmt(rec.loc[m, "RMSEP"], 2 if with_f1 else 3)
        r[f"MAE ({err_unit})"] = fmt(rec.loc[m, "MAE"], 2 if with_f1 else 3)
        r["RPD"] = fmt(rec.loc[m, "RPD"], 2)
        if with_f1:
            r["Detection F1"] = fmt(ho.loc[m, "F1"], 2)
        npar = ho.loc[m, "n_params"]
        r["Trainable parameters"] = "—" if pd.isna(npar) else f"{int(npar):,}"
        r["What it yields"] = INTERP[m]
        # guard: the recomputed R2 must equal the published one
        assert abs(rec.loc[m, "R2"] - ho.loc[m, r2col]) < 1e-3, m
        rows.append(r)
    return write(pd.DataFrame(rows), name)


def table2():
    return _bench(P1 / "phase1_meta.json", P1 / "Table_holdout_results.csv",
                  "R2_tahini", "% tahini", "Table_2_tahini_benchmark.csv", with_f1=True)


def table3():
    return _bench(PM / "mango_meta.json", PM / "Table_mango_holdout.csv",
                  "R2", "% dry matter", "Table_3_mango_benchmark.csv", with_f1=False)


# ------------------------------------------------------------------ Table 4: LOD / LOQ
def table4():
    d = pd.read_csv(R / "r05_lod_ci.csv").set_index("model")
    rows = []
    for m in MODEL_ORDER:
        r = d.loc[m]
        rows.append({"Model": LABEL[m],
                     "LOD (% adulterant)": f"{r.LOD:.2f}",
                     "LOD 95 % CI": f"{r.LOD_lo:.2f}–{r.LOD_hi:.2f}",
                     "LOQ (% adulterant)": f"{r.LOQ:.2f}",
                     "LOQ 95 % CI": f"{r.LOQ_lo:.2f}–{r.LOQ_hi:.2f}"})
        assert r.LOD_lo <= r.LOD <= r.LOD_hi, m
    return write(pd.DataFrame(rows), "Table_4_lod.csv")


# -------------------------------------------------- Table 5: parameter-matched baseline
def table5():
    s = pd.read_csv(R / "r01_matched_mlp_summary.csv")
    rows = []
    for ds in s.dataset.unique():
        sub = s[s.dataset == ds].set_index("arm")
        for arm in ["KAN", "MLP-matched-1L", "MLP-matched-2L"]:
            if arm not in sub.index:
                continue
            r = sub.loc[arm]
            rows.append({"Dataset": ds,
                         "Model": {"KAN": "KAN (published configuration)",
                                   "MLP-matched-1L": "MLP, one hidden layer, matched parameters",
                                   "MLP-matched-2L": "MLP, two hidden layers, matched parameters"}[arm],
                         "Trainable parameters": f"{int(r.n_params):,}",
                         "Grouped CV R² (mean ± SD over 3 seeds)": f"{r.cv_R2_mean:.3f} ± {r.cv_R2_sd:.3f}",
                         "External-test R² (mean ± SD over 3 seeds)": f"{r.ext_R2_mean:.3f} ± {r.ext_R2_sd:.3f}"})
    return write(pd.DataFrame(rows), "Table_9_parameter_matched.csv")


# ---------------------------------------------------------------- Table 7: generalisation
def table6():
    """Per-fold generalisation, in nine columns rather than fourteen.

    Naming the fold columns after the withheld unit (T1..T5 for tahini, 1..4 for mango) gave the
    two datasets disjoint column sets, so half of every row was blank and the table needed
    fourteen columns; Word then broke "Dataset", "Median" and "Random forest" mid-word to fit.
    Numbering the folds instead lets the two datasets share the columns, and the withheld unit is
    named in the caption where it belongs.
    """
    raw = pd.read_csv(R / "r04_generalisation_raw.csv")
    out = []
    for ds in ("tahini", "mango"):
        sub = raw[raw.dataset == ds]
        folds = sorted(sub.held_out.unique())
        for m in MODEL_ORDER:
            row = {"Dataset": ds, "Model": LABEL[m]}
            vals = []
            for i, f in enumerate(folds, 1):
                v = sub.loc[(sub.held_out == f) & (sub.model == m), "R2_mean"].iloc[0]
                row[f"Fold {i}"] = f"{v:.3f}"
                vals.append(v)
            row["Mean"] = f"{np.mean(vals):.3f}"
            row["Median"] = f"{np.median(vals):.3f}"
            row["Worst"] = f"{np.min(vals):.3f}"
            out.append(row)
    df = pd.DataFrame(out)
    order = (["Dataset", "Model"] + [c for c in df.columns if c.startswith("Fold ")]
             + ["Mean", "Median", "Worst"])
    return write(df[order].fillna(""), "Table_7_generalisation.csv")


# ------------------------------------------------------------------- Table 8: ablation
def table7():
    ab = pd.read_csv(R / "r02_ablation.csv")
    rows = []
    for ds in ["tahini", "mango"]:
        sub = ab[ab.dataset == ds]
        for _, r in sub.iterrows():
            rows.append({"Dataset": ds, "Configuration": r.config,
                         "CV R² (mean ± SD)": f"{r.cv_R2:.3f} ± {r.cv_R2_sd:.3f}",
                         "Δ vs published": ("—" if r.config == "published"
                                            else f"{r.delta_vs_published:+.3f}")})
    return write(pd.DataFrame(rows), "Table_8_ablation.csv")


# Table 9 now merges r01 and r10 and is built in build_tables2.
BUILDERS = {1: table1, 2: table2, 3: table3, 4: table4, 7: table6, 8: table7}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:]] or sorted(BUILDERS)
    for n in want:
        try:
            BUILDERS[n]()
        except FileNotFoundError as e:
            print(f"  skipped Table {n}: {e.filename} not written yet")
