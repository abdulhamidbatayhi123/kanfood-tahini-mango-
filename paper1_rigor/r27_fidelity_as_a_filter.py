"""Does fidelity actually work as a filter? Pooled over every extraction this project ran.

The paper's practical recommendation is that a printed equation should be checked against the
network it came from, and that runs failing that check should be discarded. That recommendation is
currently supported by anecdote -- the two unusable runs at seed 777 had fidelity -0.956 and -0.246,
and the usable ones 0.947 to 0.958. A reader entitled to act on it deserves the operating
characteristic, and it costs no new computation: every extraction this project has run already
recorded the accuracy of the network, the accuracy of the printed equation and the agreement between
them.

This script pools those triples from five experiments -- the extraction-protocol comparison, the
equation-stability sweep, the probe test, the determinism test and the distribution-shift test --
and asks the only question a practitioner has: **if I discard every equation whose fidelity falls
below a threshold, what do I throw away and what do I keep by mistake?**

An equation is called `unusable` if it falls more than `TOL` R2 below the network it was extracted
from. That is the practitioner's definition rather than an absolute one: a formula that reproduces a
poor model faithfully is a faithful formula, and the question here is only whether the printed
object represents the fitted one.

Outputs
-------
`r27_fidelity_filter_runs.csv`     one row per pooled extraction, with its source experiment
`r27_fidelity_filter_operating.csv` the confusion counts at a ladder of fidelity thresholds
"""
import sys

import numpy as np
import pandas as pd

from paper1_rigor.common import OUT, save

sys.stdout.reconfigure(encoding="utf-8")

TOL = 0.05           # "unusable" = the equation falls this far below its own network
THRESHOLDS = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]


def _read(name, **cols):
    p = OUT / name
    if not p.exists():
        print(f"  [skip] {name} does not exist")
        return None
    d = pd.read_csv(p)
    missing = [c for c in cols.values() if c not in d.columns]
    if missing:
        print(f"  [skip] {name}: no column {missing}")
        return None
    out = pd.DataFrame({k: d[v] for k, v in cols.items()})
    return out.dropna(subset=["network", "equation", "fidelity"])


def collect():
    frames = []

    d = _read("r12_extraction_protocol_raw.csv", network="r2_spline", equation="r2_symbolic",
              fidelity="r2_fidelity", stage="stage", route="protocol", dataset="dataset")
    if d is not None:
        d = d[d.stage == "external"].drop(columns="stage")
        frames.append(d.assign(experiment="extraction protocol"))

    d = _read("r03_stability_runs.csv", network="r2_spline", equation="r2_symbolic",
              fidelity="r2_fidelity", route="protocol", dataset="dataset")
    if d is not None:
        frames.append(d.assign(experiment="equation stability"))

    d = _read("r25_probe_effect.csv", network="r2_spline", equation="r2_symbolic",
              fidelity="r2_fidelity")
    if d is not None:
        frames.append(d.assign(experiment="probe test", route="P3 corrected", dataset="tahini"))

    for n in ("r26_conversion_determinism.csv", "r26_conversion_determinism_partial.csv"):
        d = _read(n, network="R2_network", equation="R2_equation", fidelity="fidelity",
                  route="route")
        if d is not None:
            frames.append(d.assign(experiment="determinism test", dataset="tahini"))
            break

    d = _read("r23_shift_raw.csv", network="R2_network", equation="R2_equation",
              fidelity="fidelity", route="route", dataset="dataset")
    if d is not None:
        frames.append(d.assign(experiment="distribution shift"))

    # The edge-basis experiments (`r19`, `r19b`) are NOT pooled here, and the omission is a
    # property of their output rather than a choice: until 2026-08-27 they recorded only the
    # equation's accuracy per run and not the network's, so the shortfall this analysis is built
    # on cannot be computed for them. They are now patched to record `R2_network`; a future run
    # can be added here. Section S9 says which experiments are pooled rather than claiming all of
    # them, and reports the one catastrophic extraction they contain separately.
    if not frames:
        raise SystemExit("no extraction runs found")
    return pd.concat(frames, ignore_index=True)


def main():
    runs = collect()
    runs["shortfall"] = runs.network - runs.equation
    runs["unusable"] = runs.shortfall > TOL
    # The exact route is included deliberately: a filter that only ever sees failures cannot be
    # shown to have a false-positive rate, and the elementary-basis runs are what supply the
    # negative class at high fidelity.
    save(runs, "r27_fidelity_filter_runs.csv")

    n, bad = len(runs), int(runs.unusable.sum())
    print(f"\n{n} pooled extractions from {runs.experiment.nunique()} experiments; "
          f"{bad} fall more than {TOL} R2 below their own network")
    print(runs.groupby("experiment").agg(runs=("unusable", "size"),
                                         unusable=("unusable", "sum")).to_string())

    # Two definitions of failure, because they give opposite answers and that difference is the
    # finding. "Catastrophic" is the threshold the paper already uses for an unusable equation;
    # "material" is any shortfall a practitioner would care about.
    DEFS = {"catastrophic (equation R² < 0.5)": runs.equation < 0.5,
            f"material (falls > {TOL} R² below its network)": runs.shortfall > TOL}
    rows = []
    for label, bad_mask in DEFS.items():
        for t in THRESHOLDS:
            keep = runs.fidelity >= t
            rows.append(dict(failure=label, threshold=t, failures=int(bad_mask.sum()),
                             sound=int((~bad_mask).sum()), total=len(runs),
                             kept=int(keep.sum()), discarded=int((~keep).sum()),
                             failures_kept=int((keep & bad_mask).sum()),
                             failures_caught=int((~keep & bad_mask).sum()),
                             good_discarded=int((~keep & ~bad_mask).sum()),
                             worst_shortfall_kept=float(runs.loc[keep, "shortfall"].max())
                             if keep.any() else float("nan")))
    op = pd.DataFrame(rows)
    save(op, "r27_fidelity_filter_operating.csv")
    print()
    print(op.round(4).to_string(index=False))

    # The numbers the manuscript quotes: for each definition of failure, the lowest threshold that
    # catches every one, and what catching them costs.
    for label, bad_mask in DEFS.items():
        sub = op[op.failure == label]
        clean = sub[sub.failures_kept == 0]
        nfail, ngood = int(bad_mask.sum()), int((~bad_mask).sum())
        print(f"\n{label}: {nfail} of {n} runs")
        if len(clean):
            t = clean.threshold.min()
            r = clean[clean.threshold == t].iloc[0]
            print(f"  lowest fidelity threshold catching all of them: {t}, "
                  f"discarding {int(r.good_discarded)} of the {ngood} sound runs")
        else:
            print("  no threshold on this ladder catches all of them")
        if nfail:
            print(f"  their fidelities: {sorted(runs.loc[bad_mask, 'fidelity'].round(3))[:10]}"
                  f"{' …' if nfail > 10 else ''}")
            print(f"  highest fidelity among them: {runs.loc[bad_mask, 'fidelity'].max():.3f}; "
                  f"lowest among the sound runs: {runs.loc[~bad_mask, 'fidelity'].min():.3f}")


if __name__ == "__main__":
    main()
