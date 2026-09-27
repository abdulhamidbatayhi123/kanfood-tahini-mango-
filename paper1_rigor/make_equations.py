"""Produce the equations the manuscript displays, from the model to editable Word maths.

Nothing here is transcribed. For each dataset the extraction is re-run at a chosen seed, the
resulting sympy expression is converted to LaTeX, the LaTeX is converted to OMML through Word's
own MathML transform, and the OMML is read back and checked coefficient by coefficient against
the LaTeX before it is written. The plain-text expression, its LaTeX, the OMML and the R2 it
achieved are all written next to each other, so a reader of the repository can see that the
equation in the paper is the equation the model produced.

Which seed. The symbolic conversion is not reproducible run to run, so quoting one equation
without saying which one is misleading. The default is the seed whose external R2 is the MEDIAN
of the seeds `r12` evaluated -- the representative run, not the best one -- and the seed and the
whole distribution are recorded alongside.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

from paper1_rigor.common import (TAHINI, MANGO, ROOT, tahini_split, mango_split, project, save)
from paper1_rigor.equation import extract
from paper1_rigor.equation_omml import write as write_omml

OUT = ROOT / "submission_foods" / "equations"
OUT.mkdir(parents=True, exist_ok=True)
R = ROOT / "results_rigor"

PROTOCOLS = {
    "published": dict(legacy=True),
    "corrected": dict(full_batch_symbolic=True, refit_steps=200),
}


SEEDS = [42, 7, 2024, 100, 13, 999, 123, 50, 777, 888]


def choose_seed_here(Z_tr, y_tr, Z_te, y_te1, normalise, protocol):
    """Run every seed IN THIS PROCESS and return the one whose equation is the median.

    Taking the seed from `r12`'s table instead would be wrong in a way this study documents: the
    extraction is not reproducible between processes (Section 3.10), so a seed that was the median
    there is not necessarily representative here, and the equation printed in the paper must be
    the one whose accuracy the paper quotes. Both come from this single run.
    """
    scores = {}
    for s in SEEDS:
        r = extract(Z_tr, y_tr, Z_te, y_te1, normalise=normalise, seed=s, **PROTOCOLS[protocol])
        r.pop("model", None)
        scores[s] = r["r2_symbolic"]
        print(f"    seed {s:<5} equation R2 {r['r2_symbolic']:.4f}", flush=True)
    ok = {s: v for s, v in scores.items() if np.isfinite(v)}
    order = sorted(ok, key=ok.get)
    return order[len(order) // 2], np.array([scores[s] for s in SEEDS], dtype=float)


def fmt_latex(latex, lhs, dp=3):
    """Pad coefficients to `dp` decimals so the printed equation reads consistently."""
    import re

    def pad(m):
        v = float(m.group(0))
        return f"{v:.{dp}f}" if abs(v) < 1e6 else m.group(0)

    body = re.sub(r"(?<![\w.])\d+\.\d+", pad, latex)
    return f"{lhs} \\approx {body}"


def build(label, splitter, cfg, nc_key, normalise, lhs, protocol, seed, number):
    ds, tr, te = splitter()
    Z_tr, Z_te, *_ = project(ds, tr, te, cfg["preprocess"], cfg[nc_key])
    r = extract(Z_tr, ds.y[tr], Z_te, ds.y[te, 0], normalise=normalise, seed=seed,
                **PROTOCOLS[protocol])
    r.pop("model", None)
    latex = fmt_latex(r["latex"], lhs)
    name = f"eq{number}_{label}_{protocol}"
    write_omml(latex, name)
    (OUT / f"{name}.txt").write_text(
        f"{label}: closed-form equation, {protocol} extraction protocol, seed {seed}\n"
        f"external R2 (equation) = {r['r2_symbolic']:.4f}\n"
        f"external R2 (spline network it came from) = {r['r2_spline']:.4f}\n"
        f"fidelity of the equation to that network = {r['r2_fidelity']:.4f}\n"
        f"terms = {r['n_terms']}, variables retained = {r['n_vars']}\n\n"
        f"{r['equation']}\n\nLaTeX:\n{latex}\n", encoding="utf-8")
    print(f"  {label:7s} {protocol:9s} seed {seed:<5} equation R2 {r['r2_symbolic']:.4f}  "
          f"spline {r['r2_spline']:.4f}  fidelity {r['r2_fidelity']:.4f}  {r['n_terms']} terms")
    return dict(dataset=label, protocol=protocol, seed=seed, number=number,
                r2_symbolic=r["r2_symbolic"], r2_spline=r["r2_spline"],
                r2_fidelity=r["r2_fidelity"], n_terms=r["n_terms"], n_vars=r["n_vars"],
                equation=r["equation"], latex=latex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tahini-seed", type=int, default=None)
    ap.add_argument("--mango-seed", type=int, default=None)
    ap.add_argument("--protocol", default="corrected", choices=list(PROTOCOLS))
    a = ap.parse_args()

    rows = []
    for label, splitter, cfg, nc_key, normalise, lhs, arg, number in [
            ("tahini", tahini_split, TAHINI, "kan_nc", True,
             r"\hat{y}_{\mathrm{tahini}}", a.tahini_seed, 1),
            ("mango", mango_split, MANGO, "kan_eq_nc", False,
             r"\hat{y}_{\mathrm{DM}}", a.mango_seed, 2)]:
        print(f"\n  {label}: choosing the representative seed in this run "
              f"({a.protocol} protocol)")
        ds, tr, te = splitter()
        Z_tr, Z_te, *_ = project(ds, tr, te, cfg["preprocess"], cfg[nc_key])
        if arg is not None:
            seed, dist = arg, np.array([])
        else:
            seed, dist = choose_seed_here(Z_tr, ds.y[tr], Z_te, ds.y[te, 0], normalise, a.protocol)
        if dist.size:
            print(f"  -> {len(dist)} seeds in this run, equation R2 "
                  f"{np.nanmin(dist):.3f}-{np.nanmax(dist):.3f}; median seed {seed}")
        row = build(label, splitter, cfg, nc_key, normalise, lhs, a.protocol, seed, number)
        row["run_seeds"] = ";".join(f"{v:.4f}" for v in dist) if dist.size else ""
        rows.append(row)
    save(pd.DataFrame(rows), f"equations_displayed_{a.protocol}.csv")


if __name__ == "__main__":
    main()
