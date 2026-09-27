"""Corrected significance tests for the comparisons introduced in Sections 3.12-3.17.

Section 2.7 states that every model comparison in this paper is tested with the Nadeau-Bengio
corrected resampled t-test and that each family of such tests is adjusted with Holm's step-down
procedure, and Sections 3.2 and 3.3 do exactly that. The sections written afterwards -- the edge
basis, the equation under distribution shift, the glass-box rivals and the out-of-fold detection
limit -- reported means and standard deviations only. This script supplies the missing tests so
that the same standard applies throughout. The primitives are imported from `paper1_rigor.stats`,
which imports `kanfood.metrics`; nothing here re-implements a test.

**The variance correction, and why it differs by family.** Nadeau and Bengio's correction inflates
the naive paired variance by (1/k + rho), with rho = n_test/n_train, because resampled training
partitions overlap. The four families below are three different designs and each gets the rho its
design implies:

* `r19`/`r19b` (edge basis) and `r14` (glass-box rivals) repeat a fit on ONE fixed external
  partition, varying only the training seed. rho is then the ratio of that partition,
  n_test/n_train, which is the repeated-subsampling case the correction was derived for. Two
  honest caveats travel with it and are stated in the manuscript: the test asks whether a
  difference exceeds run-to-run variation, not whether it would survive a different test set,
  because the test rows never change; and with three seeds (`r14`) the residual degrees of freedom
  are two, so the family has almost no power and a null result there is weak evidence of nothing.
* `r23` (equation under shift) is a leave-one-group-out design -- one production lot or one harvest
  season withheld -- so rho is the mean n_test/n_train over its folds, computed from the CSV.
* `r08` (detection limit) yields one number per model, not a per-fold vector, so a t-test does not
  apply. It is tested instead by the paired cluster bootstrap that already produced its intervals:
  the 2000 draws of `r08_lod_cv.py` are regenerated identically, every model is scored on the same
  draw, and the difference between two models is read off draw by draw. The draws are verified
  against the published point estimates and interval endpoints before any difference is reported.

Outputs
-------
`r24_stats_new_comparisons.csv` families 1-3, one row per comparison, Holm-adjusted within family
`r24_lod_pairwise.csv`          family 4, paired bootstrap differences between detection limits
`r24_omitted.csv`               comparisons whose input CSV does not exist yet, recorded rather
                                than silently dropped
"""
import numpy as np
import pandas as pd

from paper1_rigor.common import OUT, SEED, tahini_split, mango_split, save
from paper1_rigor.stats import paired_table
from kanfood.metrics import holm_bonferroni

# `r08_lod_cv` fixes both of these; they are imported rather than repeated so that a change there
# cannot silently desynchronise the bootstrap reproduced below.
from paper1_rigor.r08_lod_cv import N_BOOT, MODELS as LOD_MODELS, lod_loq

R19_SEEDS_SHARED_WITH_R14 = [42, 7, 2024]
_omitted = []


def omit(family, what, why):
    _omitted.append(dict(family=family, comparison=what, reason=why))
    print(f"  [omitted] {family}: {what} -- {why}", flush=True)


def read(name):
    p = OUT / name
    return pd.read_csv(p) if p.exists() else None


# --------------------------------------------------------------------------------------------
# rho for the two fixed external partitions, taken from the published splits themselves
# --------------------------------------------------------------------------------------------
def fixed_split_rho():
    rho = {}
    for label, fn in [("tahini", tahini_split), ("mango", mango_split)]:
        _, tr, te = fn()
        rho[label] = len(te) / len(tr)
        print(f"  {label}: n_train {len(tr)}  n_test {len(te)}  rho {rho[label]:.4f}", flush=True)
    return rho


def logo_rho(df):
    """Mean n_test/n_train over a leave-one-group-out design, from the fold sizes in the CSV."""
    n = df.groupby("held_out")["n_test"].first()
    total = int(n.sum())
    return float(np.mean([k / (total - k) for k in n]))


# --------------------------------------------------------------------------------------------
# family 1 -- the elementary edge basis against the corrected conversion (r19, r19b)
# --------------------------------------------------------------------------------------------
CONVERSION = "pykan B-spline + auto_symbolic"


def selected_basis(inner, representation):
    """The arm inner cross-validation selected, recovered the way `r19_basis_ablation.run` does.

    Read from the CSV rather than copied from the prose, so that the test is guaranteed to be
    about the arm the experiment actually chose.
    """
    g = inner[(inner["stage"] == "inner_basis") & (inner["representation"] == representation)]
    if g.empty:
        return None
    agg = g.groupby(["basis", "arch", "degree"])["R2"].mean().reset_index()
    best = agg.sort_values("R2", ascending=False).iloc[0]
    return f"{best['basis']} {best['arch']}"


def family_basis(label, inner, ext, rho):
    out = []
    for rep in sorted(ext["representation"].unique()):
        sel = selected_basis(inner, rep)
        e = ext[(ext["representation"] == rep) & (ext["stage"] == "external")]
        if sel is None or sel not in set(e["model"]):
            omit(f"basis/{label}/{rep}", "selected arm", "no inner-CV rows for this representation")
            continue
        seeds = sorted(e[e["model"] == sel]["seed"].unique())
        for quantity in ("R2", "fidelity"):
            scores, missing = {}, []
            for m in sorted(e["model"].unique()):
                s = e[e["model"] == m].set_index("seed")[quantity].reindex(seeds)
                if s.isna().any():
                    missing.append(m)
                else:
                    scores[m] = s.to_numpy(float)
            for m in missing:
                omit(f"basis/{label}/{rep}/{quantity}", m, "missing or non-finite at some seed")
            if sel not in scores or CONVERSION not in scores:
                omit(f"basis/{label}/{rep}/{quantity}", f"{sel} vs {CONVERSION}",
                     "one side absent after the finiteness filter")
                continue
            t = paired_table(scores, reference=sel, rho=rho,
                             label=f"{label} | {rep} | {quantity}")
            t.insert(0, "family", f"F1 edge basis ({label}, {rep}, {quantity})")
            t["design"] = "fixed external partition, repeated seeds"
            t["rho"] = rho
            out.append(t)
    return out


# --------------------------------------------------------------------------------------------
# family 2 -- the printed equation against the network it came from, under shift (r23)
# --------------------------------------------------------------------------------------------
ROUTE_CONV = "B-spline + corrected conversion"
ROUTE_OLS = "OLS on the same inputs"

PAIRS_SHIFT = [
    ("converted equation", "B-spline network"),
    ("exact equation", "elementary network"),
    ("exact equation", "converted equation"),
    ("exact equation", "OLS"),
    ("converted equation", "OLS"),
    ("elementary network", "B-spline network"),
]


def family_shift(raw):
    out = []
    rho_all = {ds: logo_rho(g) for ds, g in raw.groupby("dataset")}
    for (ds, rep), g in raw.groupby(["dataset", "representation"]):
        routes = set(g["route"])
        elem = next((r for r in routes if r.startswith("elementary basis")), None)
        if elem is None or ROUTE_CONV not in routes:
            omit(f"shift/{ds}/{rep}", "elementary vs converted", "a route is missing from r23")
            continue
        folds = sorted(g["held_out"].unique())

        def per_fold(route, column):
            s = g[g["route"] == route].groupby("held_out")[column].mean().reindex(folds)
            return None if s.isna().any() else s.to_numpy(float)

        scores = {"converted equation": per_fold(ROUTE_CONV, "R2_equation"),
                  "B-spline network": per_fold(ROUTE_CONV, "R2_network"),
                  "exact equation": per_fold(elem, "R2_equation"),
                  "elementary network": per_fold(elem, "R2_network")}
        if ROUTE_OLS in routes:
            scores["OLS"] = per_fold(ROUTE_OLS, "R2_equation")
        scores = {k: v for k, v in scores.items() if v is not None}
        pairs = [p for p in PAIRS_SHIFT if p[0] in scores and p[1] in scores]
        for p in PAIRS_SHIFT:
            if p not in pairs:
                omit(f"shift/{ds}/{rep}", " vs ".join(p), "one side has no complete fold vector")
        if pairs:
            t = paired_table(scores, pairs=pairs, rho=rho_all[ds], label=f"{ds} | {rep} | R2")
            t.insert(0, "family", f"F2 equation under shift ({ds}, {rep}, R2)")
            t["design"] = "leave-one-lot/season-out"
            t["rho"] = rho_all[ds]
            out.append(t)

        fid = {"exact equation": per_fold(elem, "fidelity"),
               "converted equation": per_fold(ROUTE_CONV, "fidelity")}
        if all(v is not None for v in fid.values()):
            t = paired_table(fid, pairs=[("exact equation", "converted equation")],
                             rho=rho_all[ds], label=f"{ds} | {rep} | fidelity")
            t.insert(0, "family", f"F2 equation under shift ({ds}, {rep}, fidelity)")
            t["design"] = "leave-one-lot/season-out"
            t["rho"] = rho_all[ds]
            out.append(t)
    return out


# --------------------------------------------------------------------------------------------
# family 3 -- the printed equation against the other glass boxes (r14 against r19/r19b)
# --------------------------------------------------------------------------------------------
def family_glassbox(r14, ext_by_dataset, inner_by_dataset, rho):
    out = []
    for (ds, rep), g in r14.groupby(["dataset", "representation"]):
        ext, inner = ext_by_dataset.get(ds), inner_by_dataset.get(ds)
        if ext is None or inner is None:
            omit(f"glassbox/{ds}/{rep}", "KAN equation reference",
                 "the edge-basis experiment for this dataset has not produced its CSV")
            continue
        sel = selected_basis(inner, rep)
        e = ext[(ext["representation"] == rep) & (ext["stage"] == "external")
                & (ext["model"] == sel)]
        seeds = [s for s in R19_SEEDS_SHARED_WITH_R14
                 if s in set(e["seed"]) and s in set(g["seed"])]
        if sel is None or len(seeds) < 2:
            omit(f"glassbox/{ds}/{rep}", "KAN equation reference",
                 f"only {len(seeds)} seed(s) shared between r14 and the edge-basis run")
            continue
        name = f"KAN equation ({sel})"
        scores = {name: e.set_index("seed")["R2"].reindex(seeds).to_numpy(float)}
        for m in sorted(g["model"].unique()):
            s = g[g["model"] == m].set_index("seed")["R2"].reindex(seeds)
            if s.isna().any():
                omit(f"glassbox/{ds}/{rep}", m, "missing at a shared seed")
            else:
                scores[m] = s.to_numpy(float)
        t = paired_table(scores, reference=name, rho=rho[ds], label=f"{ds} | {rep} | R2")
        t.insert(0, "family", f"F3 glass-box rivals ({ds}, {rep}, R2)")
        t["design"] = "fixed external partition, repeated seeds"
        t["rho"] = rho[ds]
        t["n_seeds_shared"] = len(seeds)
        out.append(t)
    return out


# --------------------------------------------------------------------------------------------
# family 4 -- the out-of-fold detection limits, by paired cluster bootstrap (r08)
# --------------------------------------------------------------------------------------------
def family_lod():
    npz = OUT / "r08_oof_predictions.npz"
    published = read("r08_lod_oof.csv")
    if not npz.exists() or published is None:
        omit("lod", "all pairs", "r08 predictions or summary absent")
        return None
    # `allow_pickle` because `ds.groups` is an array of Python strings; the file is written by
    # `r08_lod_cv.out_of_fold` in this repository and is not external input.
    z = np.load(npz, allow_pickle=True)
    y, groups, tagsis = z["y"], z["groups"].astype(str), z["tagsis"]

    # The draw sequence of `r08_lod_cv.main`, regenerated exactly so that every model is scored
    # on the same resample and the difference between two models can be read draw by draw.
    adult_true = 100 - y[:, 0]
    blank = tagsis == 0
    lots = np.array([g.split("_")[0] for g in groups])
    blank_lots = np.unique(lots[blank])
    samples = np.unique(groups)
    rng = np.random.RandomState(SEED)
    idx_lot = {l: np.where(blank & (lots == l))[0] for l in blank_lots}
    idx_smp = {s: np.where(groups == s)[0] for s in samples}
    draws = []
    for _ in range(N_BOOT):
        b = np.concatenate([idx_lot[l] for l in rng.choice(blank_lots, len(blank_lots), True)])
        a = np.concatenate([idx_smp[s] for s in rng.choice(samples, len(samples), True)])
        draws.append((b, a))

    boot, point = {}, {}
    for m in LOD_MODELS:
        p = 100 - np.nanmean(z[f"pred_{m}"][:, :, 0], axis=0)
        point[m] = lod_loq(adult_true, p, blank)[0]
        vals = np.empty(N_BOOT)
        for i, (b, a) in enumerate(draws):
            sigma = float(np.std(p[b]))
            slope = float(np.polyfit(adult_true[a], p[a], 1)[0])
            vals[i] = 3.3 * sigma / abs(slope if abs(slope) > 1e-6 else 1.0)
        boot[m] = vals

    # Guard (rule 3): a new computation that disagrees with the published one is wrong until
    # proven otherwise. Point estimate and both interval endpoints must reproduce r08's CSV.
    ref = published.set_index("model")
    for m in LOD_MODELS:
        lo, hi = np.percentile(boot[m], 2.5), np.percentile(boot[m], 97.5)
        for got, want, what in [(point[m], ref.loc[m, "LOD"], "LOD"),
                                (lo, ref.loc[m, "LOD_lo"], "LOD_lo"),
                                (hi, ref.loc[m, "LOD_hi"], "LOD_hi")]:
            assert abs(got - float(want)) < 1e-9, \
                f"{m} {what}: regenerated {got!r} does not reproduce r08's {want!r}"
    print(f"  bootstrap reproduces all {len(LOD_MODELS)} published LODs and both endpoints",
          flush=True)

    rows = []
    for i, a in enumerate(LOD_MODELS):
        for b in LOD_MODELS[i + 1:]:
            d = boot[a] - boot[b]
            frac_le = float(np.mean(d <= 0))
            p = min(1.0, 2 * min(frac_le, 1 - frac_le))
            rows.append(dict(family="F4 out-of-fold detection limit (tahini)",
                             model_a=a, model_b=b, n_boot=N_BOOT,
                             LOD_a=point[a], LOD_b=point[b],
                             diff=point[a] - point[b],
                             diff_mean=float(d.mean()),
                             diff_lo=float(np.percentile(d, 2.5)),
                             diff_hi=float(np.percentile(d, 97.5)),
                             p_raw=max(p, 1.0 / N_BOOT)))
    for row, adj in zip(rows, holm_bonferroni([r["p_raw"] for r in rows])):
        row["p_holm"] = float(adj)
        row["significant_005"] = bool(adj < 0.05)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
def main():
    print("=== rho from the published partitions ===", flush=True)
    rho = fixed_split_rho()

    tables = []
    print("\n=== F1: the elementary edge basis (r19, r19b) ===", flush=True)
    for label, inner_name, ext_name in [
            ("tahini", "r19_basis_inner_tahini.csv", "r19_basis_external_tahini.csv"),
            ("mango", "r19b_basis_inner_mango.csv", "r19b_basis_external_mango.csv")]:
        inner, ext = read(inner_name), read(ext_name)
        if inner is None or ext is None:
            omit(f"basis/{label}", "all comparisons", f"{ext_name} does not exist yet")
            continue
        tables += family_basis(label, inner, ext, rho[label])

    print("\n=== F2: the printed equation under a distribution shift (r23) ===", flush=True)
    raw = read("r23_shift_raw.csv")
    if raw is None:
        omit("shift", "all comparisons", "r23_shift_raw.csv does not exist")
    else:
        tables += family_shift(raw)

    print("\n=== F3: the other glass boxes (r14) ===", flush=True)
    r14 = read("r14_glassbox_baselines_raw.csv")
    if r14 is None:
        r14 = read("r14_glassbox_baselines_raw_progress.csv")
        if r14 is not None:
            print("  using the checkpointed progress file; the final r14 CSV does not exist yet",
                  flush=True)
    if r14 is None:
        omit("glassbox", "all comparisons", "no r14 CSV")
    else:
        ext_by = {d: read(n) for d, n in [("tahini", "r19_basis_external_tahini.csv"),
                                          ("mango", "r19b_basis_external_mango.csv")]}
        inner_by = {d: read(n) for d, n in [("tahini", "r19_basis_inner_tahini.csv"),
                                            ("mango", "r19b_basis_inner_mango.csv")]}
        tables += family_glassbox(r14, ext_by, inner_by, rho)

    print("\n=== F4: the out-of-fold detection limit (r08) ===", flush=True)
    lod = family_lod()

    if tables:
        df = pd.concat(tables, ignore_index=True)
        cols = ["family", "comparison", "model_a", "model_b", "n_folds", "design", "rho",
                "mean_a", "mean_b", "mean_diff", "t", "p_raw", "p_holm", "significant_005"]
        df = df[[c for c in cols if c in df] + [c for c in df if c not in cols]]
        save(df, "r24_stats_new_comparisons.csv")
        sig = df[df["significant_005"]]
        print(f"\n{len(df)} tests, {len(sig)} significant after Holm within family", flush=True)
        show = ["family", "model_a", "model_b", "n_folds", "mean_diff", "p_holm"]
        print(df[show].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    if lod is not None:
        save(lod, "r24_lod_pairwise.csv")
        print(f"\ndetection limit: {int(lod['significant_005'].sum())} of {len(lod)} pairs "
              f"differ after Holm", flush=True)
        print(lod[["model_a", "model_b", "diff", "diff_lo", "diff_hi", "p_holm"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    if _omitted:
        save(pd.DataFrame(_omitted), "r24_omitted.csv")


if __name__ == "__main__":
    main()
