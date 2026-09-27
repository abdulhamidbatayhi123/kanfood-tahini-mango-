"""Tables 5, 6 and 10-16, built from the newer experiments.

Same discipline as `build_tables.py`: no number is typed by hand, every table is a formatted view
of a CSV in `results_rigor/`, and a table whose source file does not exist yet is skipped with a
message rather than filled in.

Numbering across the two builders:
    1 datasets · 2 tahini benchmark · 3 mango benchmark · 4 detection limits
    5 adulterant identification · 6 split robustness · 7 generalisation · 8 ablation
    9 parameter-matched / equal capacity · 10 extraction protocol and equation stability
    11 band equation · 12 edge basis · 13 equal-interpretability budget
    14 additional glass-box baselines · 15 attribution agreement · 16 preprocessing search
    17 detection limit recomputed out of fold  (PROVISIONAL number -- see ledger section F)
"""
import re
import sys

import numpy as np
import pandas as pd

from paper1_rigor.common import ROOT
from paper1_rigor.figstyle import MODEL_ORDER

sys.stdout.reconfigure(encoding="utf-8")

R = ROOT / "results_rigor"
OUT = ROOT / "submission_foods" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
LABEL = {"PLS": "PLS", "SVM": "SVR", "RF": "Random forest", "MLP": "MLP", "CNN": "1-D CNN",
         "KAN": "KAN"}


def _minus(df):
    """A leading ASCII hyphen before a digit is a minus sign; typeset it as one.

    Only the leading position is touched, so ranges already written with an en dash
    ("3.41-3.56" is never produced) and hyphenated words are left alone.
    """
    return df.map(lambda v: re.sub(r"^-(?=[\d.])", "−", v) if isinstance(v, str) else v)


def write(df, name):
    _minus(df).to_csv(OUT / name, index=False)
    print(f"  -> tables/{name}  ({len(df)} rows)")
    return df


def need(*names):
    missing = [n for n in names if not (R / n).exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return [pd.read_csv(R / n) for n in names]


def ms(v, d=3):
    v = pd.Series(v).dropna()
    return "—" if v.empty else f"{v.mean():.{d}f} ± {v.std():.{d}f}"


# ------------------------------------------------------- Table 5: adulterant identification
def table5():
    a, = need("r09_adulterant_type.csv")
    rows = []
    for m in MODEL_ORDER:
        r = a[a.model == m].iloc[0]
        rows.append({
            "Model": LABEL[m],
            "R² peanut fraction": f"{r.R2_peanut:.3f}",
            "R² sunflower fraction": f"{r.R2_sunflower:.3f}",
            "MAE peanut (pp)": f"{r.MAE_peanut:.2f}",
            "MAE sunflower (pp)": f"{r.MAE_sunflower:.2f}",
            "Adulterant predicted on authentic (pp)":
                (f"{r.bias_peanut_authentic + r.bias_sunflower_authentic:+.2f}"
                 if "bias_peanut_authentic" in r else "—"),
            "Identification accuracy [95 % CI]":
                f"{r.identification_accuracy:.3f} "
                f"[{r.identification_acc_lo:.3f}, {r.identification_acc_hi:.3f}]",
            "Macro-F1": f"{r.identification_macroF1:.3f}"})
    return write(pd.DataFrame(rows), "Table_5_adulterant_identification.csv")


# --------------------------------------------------------- Table 6: robustness across splits
def table6():
    s, = need("r07_robustness_summary.csv")
    rows = []
    for m in MODEL_ORDER:
        r = s[s.model == m].iloc[0]
        rows.append({"Model": LABEL[m],
                     "Mean R² over ten grouped hold-outs": f"{r.R2_mean:.3f}",
                     "SD": f"{r.R2_sd:.3f}",
                     "Minimum": f"{r.R2_min:.3f}", "Maximum": f"{r.R2_max:.3f}"})
    return write(pd.DataFrame(rows), "Table_6_split_robustness.csv")


# ------------------------------- Table 10: extraction protocol and stability of the equation
# `r12` names its protocols in full ("P3 full snap + post-symbolic refit"), which in a nine-column
# table wraps to six lines per row and triples the table's height. The caption carries the meaning.
PROTOCOL_LABEL = {"P0 published (mini-batch snap, no refit)": "P0 published",
                  "P1 full-calibration snap only": "P1 snap only",
                  "P2 post-symbolic refit only": "P2 refit only",
                  "P3 full snap + post-symbolic refit": "P3 corrected"}


def table10():
    raw, = need("r12_extraction_protocol_raw.csv")
    ext = raw[raw.stage == "external"]
    rows = []
    for ds in ["tahini", "mango"]:
        for p in dict.fromkeys(ext[ext.dataset == ds].protocol):
            v = ext[(ext.dataset == ds) & (ext.protocol == p)]
            sym = v.r2_symbolic.dropna()
            # Nine columns wrapped every row to two lines in the rendered article. The seed count
            # is 10 in every row and moves to the caption; the rest are shortened there too.
            rows.append({
                "Dataset": ds,
                "Protocol": PROTOCOL_LABEL.get(p, p),
                "Network R² (med.)": f"{v.r2_spline.median():.3f}",
                # Median first: a single failed extraction moves the mean by more than the effect
                # being reported, so the mean alone would misrepresent both protocols.
                "Equation R² (med.)": f"{sym.median():.3f}",
                "Equation R² (mean ± SD)": ms(sym),
                "Equation R² range": f"{sym.min():.3f} to {sym.max():.3f}",
                "Runs below 0.5": int((sym < 0.5).sum()),
                "Fidelity (med.)": f"{v.r2_fidelity.median():.3f}",
                "Terms (med.)": f"{v.n_terms.median():.0f}"})
    write(pd.DataFrame(rows), "Table_10_extraction_protocol.csv")

    # Prefer the combined file; fall back to the per-dataset checkpoint while the mango arm is
    # still running. A `r03_stability_runs.csv` without a `protocol` column predates the rewritten
    # r03 and must never be used, so it is treated as absent rather than as data.
    runs = agr = None
    try:
        runs, agr = need("r03_stability_runs.csv", "r03_stability_agreement.csv")
        if "protocol" not in runs.columns:
            print("  Table 10b: r03_stability_runs.csv is from the superseded script, ignoring it")
            runs = agr = None
    except FileNotFoundError:
        pass
    if runs is None:
        try:
            runs, agr = need("r03_stability_runs_tahini.csv",
                             "r03_stability_agreement_tahini.csv")
            print("  Table 10b: using the tahini checkpoint; the mango arm is still running")
        except FileNotFoundError as e:
            print(f"  skipped Table 10b: {e}")
            return
    rows = []
    for ds in ["tahini", "mango"]:
        for proto in dict.fromkeys(runs[runs.dataset == ds].protocol):
            for src, label in [("seed", "10 random seeds"), ("repeat", "5 repeats, fixed seed"),
                               ("fold", "5 grouped CV folds"),
                               ("preprocess", "3 alternative preprocessings")]:
                v = runs[(runs.dataset == ds) & (runs.protocol == proto) &
                         (runs.source == src)]
                if not len(v):
                    continue
                a = agr[(agr.dataset == ds) & (agr.protocol == proto) & (agr.source == src)]
                sym = v.r2_symbolic.dropna()
                rows.append({
                    "Dataset": ds, "Protocol": proto, "Source of variation": label,
                    "Runs": len(v),
                    "Spline network R²": ms(v.r2_spline),
                    "Equation R²": ms(sym),
                    "Equation R² range": f"{sym.min():.3f}–{sym.max():.3f}" if len(sym) else "—",
                    "Median retained variables": f"{v.n_vars.median():.0f}",
                    "Mean pairwise top-3 overlap": (f"{a.top3_overlap.iloc[0]:.2f}"
                                                    if len(a) else "—"),
                    "Mean pairwise Spearman ρ": (f"{a.spearman.iloc[0]:.2f}" if len(a) else "—")})
    return write(pd.DataFrame(rows), "Table_10b_equation_stability.csv")


# ------------------------------------------------------------------ Table 11: band equation
def table11():
    import glob as _g
    # r13 can be run per dataset, so the summary may be split; take whichever files exist.
    files = sorted(_g.glob(str(R / "r13_band_equation_summary*.csv")))
    if not files:
        raise FileNotFoundError("r13_band_equation_summary*.csv")
    s = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    rows = []
    for _, r in s.sort_values(["dataset", "stage", "R2_mean"], ascending=[True, True, False]
                              ).iterrows():
        # Nine columns made every row wrap to two lines and pushed one row onto a page of its
        # own. The dataset column was "tahini" in all 36 rows and the selection column repeated
        # the band-set column, so both are folded into the caption and into one label.
        rows.append({
            "Band set": {"external": "inner CV (VIP)",
                         "external_anchored": "anchored"}.get(r.stage, r.stage),
            "Offered": int(r.k), "Model": r.model,
            "External R² (mean ± SD, 5 seeds)":
                f"{r.R2_mean:.4f} ± {0.0 if np.isnan(r.R2_sd) else r.R2_sd:.4f}",
            # The source columns are terms_median and vars_median; say so, because the retained
            # set varies from seed to seed and the article now quotes the spread.
            "Bands used": ("—" if np.isnan(r.vars_median) else f"{r.vars_median:.0f}"),
            "Terms": ("—" if np.isnan(r.terms_median) else f"{r.terms_median:.0f}"),
            "Fidelity (mean)": ("—" if np.isnan(r.fidelity_mean) else f"{r.fidelity_mean:.4f}")})
    return write(pd.DataFrame(rows), "Table_11_band_equation.csv")


# ------------------------------------------------------------------- Table 12: the edge basis
# Every arm in Table 12 shares the architecture the caption names, so carrying "1 hidden layer" in
# the cell only forces the column to wrap; and "pykan B-spline + auto_symbolic" has no break
# opportunity, so Word split it mid-word as "auto_symbol / ic". Short labels, defined in the caption.
BASIS_LABEL = {"cheby 1 hidden layer": "Chebyshev", "poly 1 hidden layer": "Powers",
               "rbf 1 hidden layer": "Gaussian", "fourier 1 hidden layer": "Fourier",
               "bspline 1 hidden layer": "B-spline (control)",
               "pykan B-spline + auto_symbolic": "B-spline, converted"}


def table12():
    # The two arms were run by separate scripts and save per-dataset finals: `r19` writes the
    # tahini summary, `r19b` the mango one. Take whichever exist, so the table is complete when
    # both have landed and honest about what is present when only one has.
    try:
        s, = need("r19_basis_external_summary.csv")
    except FileNotFoundError:
        frames = []
        for n in ("r19_basis_external_tahini.csv", "r19b_basis_external_mango.csv"):
            if (R / n).exists():
                frames.append(pd.read_csv(R / n))
        if not frames:
            raise FileNotFoundError("r19_basis_external_tahini.csv")
        e = pd.concat(frames, ignore_index=True)
        # r19 labels the arm in a `model` column; r19b labels it in `basis`. Normalise.
        if "model" not in e.columns:
            e["model"] = e["basis"]
        e["model"] = e["model"].fillna(e.get("basis"))
        s = (e.groupby(["dataset", "representation", "model"])
             .agg(R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
                  fidelity_mean=("fidelity", "mean"), fidelity_min=("fidelity", "min"),
                  terms_median=("n_terms", "median"), vars_median=("n_vars", "median"),
                  params_median=("n_params", "median")).reset_index())
        s = s.assign(_o=s.dataset.map({"tahini": 0, "mango": 1}).fillna(2)).sort_values(
            ["_o", "representation", "model"]).drop(columns="_o")
    rows = []
    for _, r in s.iterrows():
        exact = (not np.isnan(r.fidelity_mean)) and r.fidelity_mean > 0.999
        rows.append({
            # "pykan B-spline + auto_symbolic" and "Approximated" have no break opportunity and
            # Word splits them mid-word ("auto_symbol / ic"). Short labels, defined in the caption.
            "Dataset": r.dataset, "Inputs": r.representation,
            "Edge basis / method": BASIS_LABEL.get(str(r.model), str(r.model)),
            "External R² (mean ± SD)": f"{r.R2_mean:.3f} ± "
                                       f"{0.0 if np.isnan(r.R2_sd) else r.R2_sd:.3f}",
            "Equation fidelity to the network":
                ("—" if np.isnan(r.fidelity_mean) else f"{r.fidelity_mean:.6f}"),
            "Worst-case fidelity":
                ("—" if np.isnan(r.fidelity_min) else f"{r.fidelity_min:.6f}"),
            # NOT the string "None": pandas treats it as a missing-value token, so it would be
            # read back as NaN when the .docx is built and printed as "nan" in the table.
            "Closed form": "Exact" if exact else ("Approximate"
                                                  if not np.isnan(r.fidelity_mean)
                                                  else "No closed form"),
            "Median terms": ("—" if np.isnan(r.terms_median) else f"{r.terms_median:.0f}"),
            "Parameters": f"{int(r.params_median):,}" if not np.isnan(r.params_median) else "—"})
    return write(pd.DataFrame(rows), "Table_12_edge_basis.csv")


# ------------------------------------------------- Table 13: equal-interpretability budget
def table13():
    s, = need("r11_equal_interpretability_summary.csv")
    rows = []
    for ds in ["tahini", "mango"]:
        sub = s[s.dataset == ds]
        for n in sorted(sub.n_latent.unique()):
            v = sub[sub.n_latent == n]
            def g(model, proto):
                q = v[(v.model == model) & (v.protocol == proto)]
                return f"{q.R2_mean.iloc[0]:.3f}" if len(q) else "—"
            terms = v[(v.model == "KAN equation") & (v.protocol == "corrected")]
            rows.append({
                "Dataset": ds, "Latent variables": int(n),
                "PLS R²": g("PLS", "-"),
                "KAN spline network R²": g("KAN spline", "corrected"),
                "KAN equation R² (published extraction)": g("KAN equation", "published"),
                "KAN equation R² (corrected extraction)": g("KAN equation", "corrected"),
                "Terms in the equation": (f"{terms.terms_median.iloc[0]:.0f}"
                                          if len(terms) and not np.isnan(
                                              terms.terms_median.iloc[0]) else "—"),
                "Numbers a reader must be given (PLS)":
                    f"{v[v.model == 'PLS'].disclosure_median.iloc[0]:,.0f}"
                    if len(v[v.model == "PLS"]) else "—"})
    return write(pd.DataFrame(rows), "Table_13a_equal_interpretability.csv")


# ----------------------------------------------------- Table 14: additional glass-box baselines
def table14():
    # The full-spectrum arm of r14 is deliberately out of scope (see r14_glassbox_omitted.csv and
    # the module docstring), so the final summary may never be written. Fall back to the progress
    # file the latent arm writes per dataset, and then to the per-model checkpoints, so the table
    # reports whatever has actually been computed rather than nothing.
    import glob as _glob
    try:
        s, = need("r14_glassbox_baselines_summary.csv")
    except FileNotFoundError:
        frames = []
        for n in ("r14_glassbox_baselines_raw_progress.csv",):
            if (R / n).exists():
                frames.append(pd.read_csv(R / n))
        for f in sorted(_glob.glob(str(R / "r14_glassbox_partial_*.csv"))):
            frames.append(pd.read_csv(f))
        if not frames:
            raise FileNotFoundError("no r14 output yet")
        raw = (pd.concat(frames, ignore_index=True)
               .drop_duplicates(subset=["dataset", "representation", "model", "seed"]))
        # The column and the caption both say "median", so take the median. It was a mean until
        # 2026-08-27, which differed from the median by more than a rounding step in eight of the
        # fourteen rows (tahini spline-MLP 251.0 against 239.3, tahini SR 98.0 against 114.4).
        s = (raw.groupby(["dataset", "representation", "model"])
             .agg(R2_mean=("R2", "mean"), R2_sd=("R2", "std"), seconds=("seconds", "median"))
             .reset_index())
        print(f"  Table 14: assembled from {len(raw)} checkpointed runs "
              f"({', '.join(sorted(raw.dataset.unique()))}); the full-spectrum arm is out of scope")
    CLASS = {"EBM": "Additive glass-box", "EBM+int": "Additive glass-box with interactions",
             "GAM": "Additive glass-box", "XGBoost": "Opaque ensemble",
             "spline-MLP": "Opaque (spline activations on nodes)", "MLP(ref)": "Opaque",
             "SR (gplearn)": "Closed-form (genetic programming)"}
    # Tahini first, to match every other table and the text, and the readable model names the
    # text and Figure 15 use rather than the identifiers the experiment writes.
    s = s.assign(_o=s.dataset.map({"tahini": 0, "mango": 1}).fillna(2)).sort_values(
        ["_o", "representation", "R2_mean"], ascending=[True, True, False])
    rows = []
    for _, r in s.iterrows():
        rows.append({"Dataset": r.dataset, "Inputs": r.representation,
                     "Model": _arm(r.model),
                     "Transparency": CLASS.get(r.model, "—"),
                     "External R² (mean ± SD, 3 seeds)":
                         f"{r.R2_mean:.3f} ± {0.0 if np.isnan(r.R2_sd) else r.R2_sd:.3f}",
                     "Median fit time (s)": f"{r.seconds:.0f}"})
    return write(pd.DataFrame(rows), "Table_14_glassbox_baselines.csv")


# ---------------------------------------------------------------- Table 15: attribution
def table15():
    # Prefer the combined files; fall back to the tahini checkpoint while the mango arm runs, as
    # `table10` and `table16` do. The tahini arm is complete inside the checkpoint.
    try:
        conv, agr = need("r15_shap_convergence.csv", "r15_wavenumber_agreement.csv")
    except FileNotFoundError:
        conv, agr = need("r15_shap_convergence_tahini.csv", "r15_wavenumber_agreement_tahini.csv")
        print("  Table 15: using the tahini checkpoint; the mango arm is still running")
    rows = []
    for _, r in conv.iterrows():
        rows.append({"Dataset": r.dataset, "Model": r.model, "Axis": "PLS latent variables",
                     "SHAP background": int(r.n_bg), "SHAP evaluation rows": int(r.n_eval),
                     "SHAP samples": int(r.nsamples),
                     "ρ permutation vs SHAP": f"{r.rho_perm_shap:.3f}",
                     "ρ KAN intrinsic vs SHAP":
                         (f"{r.rho_kanintrinsic_shap:.3f}"
                          if "rho_kanintrinsic_shap" in r and not pd.isna(
                              r.rho_kanintrinsic_shap) else "—"),
                     # Section S11 quotes this one; without the column its 0.857 matched only the
                     # perceptron's intrinsic-vs-SHAP cell, which is a different quantity.
                     "ρ KAN intrinsic vs permutation":
                         (f"{r.rho_kanintrinsic_perm:.3f}"
                          if "rho_kanintrinsic_perm" in r and not pd.isna(
                              r.rho_kanintrinsic_perm) else "—")})
    write(pd.DataFrame(rows), "Table_15_attribution_latent.csv")
    rows = [{"Dataset": r.dataset, "Comparison": f"{r.a} vs {r.b}",
             "Spearman ρ": f"{r.spearman:.3f}", "Top-5 window overlap": f"{r.top5_overlap:.2f}"}
            for _, r in agr.iterrows()]
    return write(pd.DataFrame(rows), "Table_15b_attribution_wavenumber.csv")


# ------------------------------------------------------------ Table 16: preprocessing search
def table16():
    # Prefer the combined file; fall back to the tahini checkpoint while the mango arm runs, the
    # way `table10` does. The tahini external stage is complete inside the checkpoint, so the
    # tahini half of this table is final either way.
    try:
        e, = need("r17_preprocessing_external.csv")
    except FileNotFoundError:
        raw, = need("r17_preprocessing_raw_tahini.csv")
        e = raw[raw.stage == "s3_external"]
        print("  Table 16: using the tahini checkpoint; the mango arm is still running")
    rows = []
    for _, r in e.iterrows():
        rows.append({"Dataset": r.dataset, "Configuration": r.config,
                     "Preprocessing": r.method, "SG window": int(r.window),
                     # `r17` writes "KAN nc=8"; the count already has its own column.
                     "Model": str(r.model).split(" nc=")[0], "Latent variables": int(r.nc),
                     "External R² (mean ± SD)":
                         f"{r.R2:.4f} ± {0.0 if np.isnan(r.R2_sd) else r.R2_sd:.4f}"})
    return write(pd.DataFrame(rows), "Table_16_preprocessing_search.csv")


# ------------------------------------- Table 17: detection limit from pooled out-of-fold predictions
def table17():
    """The hold-out estimate and the out-of-fold estimate side by side.

    They are not alternatives to choose between: the hold-out figure is what the published split
    supports and the out-of-fold figure is what five blank lots support. Printing only the second
    would hide that the first was the one reported; printing only the first would leave the
    single-blank-lot weakness unquantified. The ratio column is the size of that weakness.
    """
    o, h = need("r08_lod_oof.csv", "r05_lod_ci.csv")
    o, h = o.set_index("model"), h.set_index("model")
    rows = []
    for m in MODEL_ORDER:
        if m not in o.index:
            continue
        a, b = h.loc[m], o.loc[m]
        rows.append({"Model": LABEL[m],
                     "Out-of-fold R²": f"{b.oof_R2:.3f}",
                     "LOD, hold-out (%)": f"{a.LOD:.2f}",
                     "LOD 95 % CI, hold-out": f"{a.LOD_lo:.2f}–{a.LOD_hi:.2f}",
                     "LOD, out-of-fold (%)": f"{b.LOD:.2f}",
                     "LOD 95 % CI, out-of-fold": f"{b.LOD_lo:.2f}–{b.LOD_hi:.2f}",
                     "Ratio": f"{b.LOD / a.LOD:.1f}×",
                     "LOQ, out-of-fold (%)": f"{b.LOQ:.2f}",
                     # semicolons with no space let Word break a line inside a number ("9.5 9"); spaces
                     # give it somewhere legal to wrap
                     "LOD per repeat (%)": str(b.LOD_per_repeat).replace(";", ", ")})
        assert b.LOD_lo <= b.LOD <= b.LOD_hi, m
    return write(pd.DataFrame(rows), "Table_13_lod_out_of_fold.csv")


def sci(v):
    """A spread as display text, e.g. '1.6 x 10^-3'.

    Written as text rather than as a float because a numeric cell is re-parsed when the CSV is read
    back for the .docx and comes out in whatever format pandas chooses -- which is how trailing
    zeros were lost from another table earlier in this project.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    m = v / (10.0 ** e)
    sup = str(abs(e)).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    mant = f"{m:.0f}" if abs(m - round(m)) < 0.05 else f"{m:.1f}"
    return f"{mant} × 10⁻{sup}" if e < 0 else f"{mant} × 10{sup}"


# ------------------------------- Table 18: how reproducible is one run of this pipeline?
def table18():
    """The three spreads of Section 2.12, side by side, plus the volatile configuration.

    Reported as the RANGE over the runs rather than a standard deviation, because with two or three
    runs a range is the honest summary and an SD implies a distribution that has not been sampled.
    The first block is `r20` at the published configuration; the second is `r22` at the
    inner-CV-selected configuration, which is the one at which two of our own runs disagreed.
    """
    a, = need("r20_reproducibility_summary.csv")
    SRC = {"repeats at one seed, one thread count": "Repeated runs in one process",
           "thread count, seed fixed": "Number of compute threads",
           "seed, thread count fixed": "Random seed"}
    HELD = {"repeats at one seed, one thread count": "seed, threads, split, configuration",
            "thread count, seed fixed": "seed, split, configuration",
            "seed, thread count fixed": "threads, split, configuration"}
    rows = []
    for src, label in SRC.items():
        r = {"Source of variation": label, "Held fixed": HELD[src]}
        for arch in ("KAN", "MLP"):
            for har, hl in (("published", "published recipe"), ("convergence", "convergence harness")):
                sub = a[(a.source == src) & (a.architecture == arch) & (a.harness == har)]
                v = float(sub["range"].max()) if len(sub) else float("nan")
                r[f"{arch}, {hl}"] = sci(v)
        rows.append(r)
    # second block: the configuration at which r10 and r10b disagreed
    try:
        b, = need("r22_repro_selected_config_summary.csv")
        b = b[b.source == "thread count, seed fixed"]
    except FileNotFoundError:
        b, = need("r22_repro_selected_config_partial.csv")
        b = (b.groupby(["harness", "seed"])["R2"].agg(lambda g: g.max() - g.min())
              .reset_index().rename(columns={"R2": "range"}))
    r = {"Source of variation": "Number of compute threads",
         "Held fixed": "seed, split; inner-CV-selected configuration"}
    for har, hl in (("published", "published recipe"), ("convergence", "convergence harness")):
        v = b[b.harness == har]["range"]
        r[f"KAN, {hl}"] = sci(None if v.empty else float(v.max()))
        r[f"MLP, {hl}"] = "not run"
    rows.append(r)
    return write(pd.DataFrame(rows), "Table_18_reproducibility.csv")


# ---------------------------- Table 19: the closed form under a distribution shift (r23)
ROUTE_LABEL = {"B-spline + corrected conversion": "converted",
               "OLS on the same inputs": "OLS",
               "elementary basis (cheby)": "exact (Chebyshev)",
               "elementary basis (poly)": "exact (powers)"}


def table19():
    """Network, equation and their agreement on material the network never saw.

    The column that carries the result is not either accuracy but the gap between them, and the
    fidelity beside it: an exact route cannot drift, a converted one can, and the question is how
    much it does when the inputs move away from the calibration set.
    """
    try:
        d, = need("r23_shift_raw.csv")
    except FileNotFoundError:
        try:
            d, = need("r23_shift_tahini.csv")
        except FileNotFoundError:
            d, = need("r23_shift_tahini_partial.csv")
    # The design is leave-one-group-out, so the fold is the unit: average over seeds inside a fold
    # first, then over folds, which is what `r24` tests and what Figure 6c plots. Until 2026-08-27
    # this took a flat mean over all runs, and because two of the fifteen tahini latent-score
    # conversions returned no equation, that row's gap read -0.001 against Table S14's -0.008 --
    # the equation averaged over 13 runs and the network over 15. The "worst" columns are likewise
    # fold values now; they were run minima, which put 0.695 (one seed on lot T1) in a column a
    # reader would compare against the article's 0.844 for that lot.
    n_all = d.groupby(["dataset", "representation", "route"])["R2_network"].size()
    n_eq = d.groupby(["dataset", "representation", "route"])["R2_equation"].count()
    # Pair strictly: a run that returned no equation contributes its network score to nothing
    # either, so the network and equation columns of a fold are over the same seeds. Dropping the
    # rows rather than relying on pandas' skipna is the whole point -- skipna gave the two columns
    # different denominators inside the two affected tahini folds.
    paired = d.dropna(subset=["R2_equation"])
    fold = (paired.groupby(["dataset", "representation", "route", "held_out"])
            .agg(net=("R2_network", "mean"), eqn=("R2_equation", "mean"),
                 fid=("fidelity", "mean")).reset_index())
    g = (fold.groupby(["dataset", "representation", "route"])
         .agg(folds=("held_out", "nunique"),
              net=("net", "mean"), eqn=("eqn", "mean"),
              eq_worst=("eqn", "min"), fid=("fid", "mean"),
              fid_worst=("fid", "min")).reset_index())
    rows = []
    for _, r in g.iterrows():
        key = (r.dataset, r.representation, r.route)
        if int(n_all[key]) != int(n_eq[key]):
            print(f"  Table 19: {r.dataset}/{r.representation}/{r.route} returned an equation in "
                  f"{int(n_eq[key])} of {int(n_all[key])} runs")
        rows.append({
            # `r23` names the routes in full; in a ten-column table those wrap to three lines
            # each and treble the table's height. The caption defines the short forms, which are
            # also the words the prose uses.
            "Dataset": r.dataset, "Inputs": r.representation,
            "Route": ROUTE_LABEL.get(r.route, r.route),
            "Folds": int(r.folds),
            "Network R² (mean)": f"{r.net:.3f}",
            "Equation R² (mean)": f"{r.eqn:.3f}",
            "Equation − network": f"{r.eqn - r.net:+.3f}",
            "Worst fold, equation R²": f"{r.eq_worst:.3f}",
            "Fidelity (mean)": f"{r.fid:.6f}",
            "Fidelity (worst fold)": f"{r.fid_worst:.6f}"})
    return write(pd.DataFrame(rows), "Table_19_equation_under_shift.csv")


# --------------------------------- Tables 20, 21: the corrected tests for the later comparisons
# Section 2.7 promises a Nadeau-Bengio corrected resampled t-test with Holm adjustment for every
# model comparison, and Sections 3.2 and 3.3 use one. The sections written afterwards reported
# means and standard deviations only; `r24` supplies the missing tests and these two tables are
# its formatted view. The variance correction differs by design and the `Design` column says which
# was used. The column itself is deliberately NOT printed -- at nine columns this table broke words
# mid-syllable in the rendered supplement -- so the design is carried by the family name and by the
# caption instead, which Section S12 now says.
FAMILY_LABEL = {"F1": "Edge basis", "F2": "Equation under shift", "F3": "Glass-box rivals"}
# The experiment CSVs name arms the way the code does; the manuscript names them the way a reader
# does. Anything not listed keeps its own name.
ARM_LABEL = {"cheby 1 hidden layer": "Chebyshev basis", "poly 1 hidden layer": "power basis",
             "rbf 1 hidden layer": "Gaussian basis", "fourier 1 hidden layer": "Fourier basis",
             "bspline 1 hidden layer": "B-spline network (control)",
             "pykan B-spline + auto_symbolic": "corrected pykan conversion",
             "SVM": "SVR",
             "MLP(ref)": "reference perceptron", "SR (gplearn)": "symbolic regression",
             "EBM": "boosting machine", "EBM+int": "boosting machine + interactions",
             "GAM": "spline additive model", "XGBoost": "gradient boosting",
             "spline-MLP": "spline-activation MLP"}


# The full arm names are correct but too long for a table this wide; the caption carries the
# basis, and every row of the glass-box family shares it.
SHORT = {"KAN equation (Chebyshev basis)": "KAN equation",
         "B-spline network (control)": "B-spline control",
         "corrected pykan conversion": "pykan conversion"}


def _short(name):
    return SHORT.get(name, name)


def _arm(name):
    if name.startswith("KAN equation (") and name.endswith(")"):
        return f"KAN equation ({_arm(name[len('KAN equation ('):-1])})"
    return ARM_LABEL.get(name, name)


def table20():
    d, = need("r24_stats_new_comparisons.csv")
    rows = []
    for _, r in d.iterrows():
        code, rest = r.family.split(" ", 1)
        setting = rest.strip("()").split("(")[-1].replace("R2", "R²")
        setting = setting.replace("PLS scores", "latent scores").replace(
            "selected bands", "named bands")
        rows.append({
            "Family and setting": f"{FAMILY_LABEL.get(code, code)}: {setting}",
            "Model A": _short(_arm(r.model_a)), "Model B": _short(_arm(r.model_b)),
            "n": int(r.n_folds),
            "Mean difference (A − B)": f"{r.mean_diff:+.4f}".replace("-", "−"),
            "SD of difference": f"{r.sd_diff:.4f}",
            "Holm-adjusted p": ("<0.001" if r.p_holm < 0.001 else f"{r.p_holm:.3f}"),
            "Significant at 0.05": "Yes" if r.significant_005 else "No"})
    return write(pd.DataFrame(rows), "Table_20_corrected_tests.csv")


def table21():
    d, = need("r24_lod_pairwise.csv")
    rows = [{"Model A": _arm(r.model_a), "Model B": _arm(r.model_b),
             "LOD A (%)": f"{r.LOD_a:.2f}", "LOD B (%)": f"{r.LOD_b:.2f}",
             "Difference (percentage points)": f"{r['diff']:+.2f}".replace("-", "−"),
             "95 % CI of the difference":
                 f"{r.diff_lo:+.2f} to {r.diff_hi:+.2f}".replace("-", "−"),
             "Holm-adjusted p": ("<0.001" if r.p_holm < 0.001 else f"{r.p_holm:.3f}"),
             "Significant at 0.05": "Yes" if r.significant_005 else "No"}
            for _, r in d.iterrows()]
    return write(pd.DataFrame(rows), "Table_21_lod_pairwise.csv")


# ------------------------ Table 22: is the conversion reproducible, and does the exact route share it
def table22():
    try:
        d, = need("r26_conversion_determinism.csv")
    except FileNotFoundError:
        d, = need("r26_conversion_determinism_partial.csv")
        print("  Table 22: using the checkpoint; the run has not finished")
    rows = []
    for (route, seed, threads), g in d.groupby(["route", "seed", "threads"]):
        lo, hi = g.R2_equation.min(), g.R2_equation.max()
        flo, fhi = g.fidelity.min(), g.fidelity.max()
        rows.append({
            "Route": "Corrected conversion" if route.startswith("B-spline") else "Elementary basis",
            "Seed": int(seed), "Compute threads": int(threads), "Runs": len(g),
            "Network R²": f"{g.R2_network.iloc[0]:.6f}",
            "Equation R², lowest": f"{lo:.6f}", "Equation R², highest": f"{hi:.6f}",
            "Range": f"{hi - lo:.6f}",
            "Fidelity, lowest": f"{flo:.6f}", "Fidelity, highest": f"{fhi:.6f}"})
    out = pd.DataFrame(rows).sort_values(["Seed", "Route", "Compute threads"])
    return write(out, "Table_22_conversion_determinism.csv")


# ------------------- Table 23: how well fidelity works as a filter, pooled over every extraction
def table23():
    d, = need("r27_fidelity_filter_operating.csv")
    rows = []
    for _, r in d.iterrows():
        rows.append({
            "Failure defined as": r.failure,
            "Fidelity threshold": f"{r.threshold:.2f}",
            "Equations kept": int(r.kept),
            "Failures kept": int(r.failures_kept),
            "Failures caught": int(r.failures_caught),
            "Sound equations discarded": int(r.good_discarded),
            "Worst shortfall still kept": f"{r.worst_shortfall_kept:.3f}"})
    return write(pd.DataFrame(rows), "Table_23_fidelity_filter.csv")


BUILDERS = {5: table5, 6: table6, 10: table10, 11: table11, 12: table12, 13: table13,
            14: table14, 15: table15, 16: table16, 17: table17, 18: table18, 19: table19,
            20: table20, 21: table21, 22: table22, 23: table23}

# ---------------------------------- Table 9: parameter-matched and equal-capacity comparison
def table9():
    """All three forms of the comparison in one table, in the order they were run.

    Reporting only the fair one would hide that the confounded one was run and was misleading;
    reporting only the confounded one would be indefensible. The third block is the one Section S7
    and Section 3.6 actually quote -- it separates the three factors the second block changed
    together (which configuration, how it was trained, how it was scored) -- and the caption
    promised it long before the builder emitted it, which is how its absence was found.

    Columns are kept short: a `Protocol` column carrying "Equal parameters, equal search, both
    trained to convergence" wrapped to four lines in the rendered supplement.
    """
    DS = {"mango (accuracy KAN)": "mango, accuracy", "mango (compact KAN)": "mango, compact"}

    def row(block, ds, model, params, updates, cv, ext, config="—", harness="—"):
        # "Configuration" has no break opportunity and Word split it as "Configuratio/n" in the
        # rendered supplement; the caption carries the full word.
        return {"Block": block, "Dataset": DS.get(ds, ds), "Model": model, "Config.": config,
                "Harness": harness, "Parameters": f"{int(params):,}", "Updates (median)": updates,
                "Grouped CV R²": cv, "External-test R²": ext}

    a, = need("r01_matched_mlp_summary.csv")
    rows = []
    NAME = {"KAN": "KAN", "MLP-matched-1L": "MLP (1 layer)",
            "MLP-matched-2L": "MLP (2 layers)"}
    for ds in a.dataset.unique():
        sub = a[a.dataset == ds].set_index("arm")
        for arm in ["KAN", "MLP-matched-1L", "MLP-matched-2L"]:
            if arm not in sub.index:
                continue
            r = sub.loc[arm]
            rows.append(row("1", ds, NAME[arm], r.n_params, "—",
                            f"{r.cv_R2_mean:.3f} ± {r.cv_R2_sd:.3f}",
                            f"{r.ext_R2_mean:.3f} ± {r.ext_R2_sd:.3f}"))
    try:
        b, = need("r10_fair_capacity_summary.csv")
    except FileNotFoundError as e:
        print(f"  Table 9: fair-capacity block not written yet ({e})")
        return write(pd.DataFrame(rows), "Table_9_parameter_matched.csv")
    for _, r in b.iterrows():
        rows.append(row("2", r.dataset,
                        {"KAN": "KAN", "MLP": "MLP (matched)"}[r.architecture],
                        r.n_params, f"{int(r.updates_median):,}", "—",
                        f"{r.ext_R2_mean:.3f} ± {r.ext_R2_sd:.3f}"))
    try:
        c, = need("r10b_config_vs_harness_summary.csv")
    except FileNotFoundError as e:
        print(f"  Table 9: the factor-separation block is not written yet ({e})")
        return write(pd.DataFrame(rows), "Table_9_parameter_matched.csv")
    CFG = {"published config": "published", "inner-CV selected": "inner-CV"}
    # "published recipe" here collided with "published" in the Config. column, so a reader saw
    # "published published recipe" run together; name the rule instead of the recipe.
    HAR = {"published recipe": "fixed steps", "convergence harness": "early stopping"}
    for _, r in c.iterrows():
        rows.append(row("3", r.dataset,
                        {"KAN": "KAN", "MLP": "MLP (matched)"}[r.architecture],
                        r.params, f"{int(r.updates_median):,}", "—",
                        f"{r.ext_R2_mean:.3f} ± {r.ext_R2_sd:.3f}",
                        config=CFG.get(r.configuration, r.configuration),
                        harness=HAR.get(r.harness, r.harness)))
    return write(pd.DataFrame(rows), "Table_9_parameter_matched.csv")


BUILDERS[9] = table9


if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:]] or sorted(BUILDERS)
    for n in want:
        try:
            BUILDERS[n]()
        except FileNotFoundError as e:
            print(f"  skipped Table {n}: {e} not written yet")
