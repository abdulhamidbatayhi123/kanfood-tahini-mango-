"""The mango arm of `r19`, on a grid reduced to the question that arm has to answer.

Why this is separate
--------------------
`r19`'s selection grid is five bases x two depths x three degrees x three inner folds x two input
representations: 180 fits. On tahini's 1146 rows that costs about two hours. On mango's 10,243 rows,
with the training budget scaled by dataset size as `basis_kan.fit` now requires, the same grid
projects to the better part of a day, and in practice it made no measurable progress in three and a
half hours while sharing a machine with the other experiments.

That grid is larger than the question needs. What §3.12 asks of mango is whether an *already
elementary* edge basis costs accuracy on the non-linear matrix, and whether the printed equation is
exact there too. The variable under test is the basis. Depth and degree are nuisance parameters, and
tahini selected the same values for both of its input representations -- one hidden layer, degree
four -- so they are fixed here at that choice rather than re-searched.

What this costs, stated plainly: the mango basis is selected on mango's own training folds, but its
depth and degree are inherited from tahini rather than chosen on mango. If mango's optimum lay at a
different depth or degree, this arm would not find it, and the mango numbers are therefore a
slightly conservative estimate of what an elementary basis can do here. The external comparison --
every basis at the selected configuration, against the B-spline control and the corrected pykan
conversion, with fidelity measured on the printed expression -- is unchanged.

Everything else is `r19` unmodified: the same `run`, the same `fit_score`, the same selection
inside the training partition, the same external test used once.
"""
import sys
import time

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

from paper1_rigor import r19_basis_ablation as R
from paper1_rigor.common import MANGO, mango_split, save

# The reduction. Bases stay complete -- that is the variable under test, and Fourier has to remain
# so the section can show that an exact basis is not automatically an adequate one.
R.ARCHS = [a for a in R.ARCHS if a[0] == "1 hidden layer"]
R.DEGREES = [4]


def main():
    t0 = time.time()
    print("=== mango edge-basis arm, reduced grid ===", flush=True)
    print(f"  bases {R.BASES}", flush=True)
    print(f"  arch  {[a[0] for a in R.ARCHS]}  (fixed from tahini's selection)", flush=True)
    print(f"  degree {R.DEGREES}            (fixed from tahini's selection)", flush=True)
    print(f"  {len(R.BASES) * len(R.ARCHS) * len(R.DEGREES)} arms x {R.INNER_FOLDS} inner folds "
          f"x 2 representations", flush=True)
    # `R.run` checkpoints each representation as it finishes, to
    # `r19_basis_{inner,external}_mango_partial.csv`; without that this script wrote nothing until
    # the very end and a killed session discarded the whole run, which is what happened once.
    inner, ext = R.run("mango", mango_split, MANGO, "kan_eq_nc", False, 9.0)
    save(inner, "r19b_basis_inner_mango.csv")
    save(ext, "r19b_basis_external_mango.csv")
    summ = (ext.groupby(["dataset", "representation", "model"])
            .agg(R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
                 fidelity_mean=("fidelity", "mean"), fidelity_min=("fidelity", "min"),
                 terms_median=("n_terms", "median"), vars_median=("n_vars", "median"),
                 params_median=("n_params", "median"))
            .reset_index())
    save(summ, "r19b_basis_external_summary_mango.csv")
    print("\n", summ.to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
