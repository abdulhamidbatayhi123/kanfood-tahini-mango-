"""One place for the inferential statistics, so every table in the manuscript uses the same rules.

The rules themselves are the companion paper's, because its reviewers asked for them and because
two papers by the same authors must not use different standards:

* Differences between models across cross-validation folds are tested with the **Nadeau-Bengio
  corrected resampled t-test**, never a plain paired t-test. The correction exists because the
  training sets of overlapping folds share most of their rows, which makes the naive variance
  estimate far too small.
* Families of such tests are adjusted with **Holm's step-down procedure**.
* Interval estimates on a held-out set are **cluster bootstraps** that resample whole physical
  samples or whole populations, never individual scans, because replicate scans of one sample are
  not independent observations.

All three primitives already exist in `kanfood.metrics` and are imported, not re-implemented.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from kanfood.metrics import (bootstrap_ci, corrected_resampled_ttest, holm_bonferroni, rmse, rpd)

__all__ = ["describe", "paired_table", "ci_table", "rmse", "rpd", "r2_score"]


def describe(values, name="value"):
    v = np.asarray([x for x in np.ravel(values) if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {f"{name}_n": 0}
    return {f"{name}_n": int(v.size), f"{name}_mean": float(v.mean()),
            f"{name}_sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            f"{name}_median": float(np.median(v)), f"{name}_min": float(v.min()),
            f"{name}_max": float(v.max())}


def paired_table(scores, reference=None, rho=None, label="", pairs=None):
    """Corrected paired tests over a family of model comparisons, Holm-adjusted together.

    `scores` maps a model name to its per-fold scores; every model must have the same folds in
    the same order. With `reference` given, only that model is compared against the others (the
    family is then k-1 tests); with `pairs` given, exactly those (a, b) comparisons are made;
    otherwise every pair is tested. `rho` overrides the Nadeau-Bengio variance-inflation ratio
    n_test/n_train, which defaults to 1/(k-1) for k-fold CV -- pass the real ratio when the
    design is not plain k-fold, e.g. a leave-one-group-out with unequal folds.

    `pairs` exists because several of the comparisons this project needs are neither
    all-against-all nor all-against-one: Section 3.13 asks whether each printed equation differs
    from *its own* network, which is one nominated comparison per route and not a full family.
    Testing the full family instead would spend the Holm correction on comparisons nothing in the
    manuscript claims.
    """
    names = list(scores)
    lengths = {len(np.ravel(v)) for v in scores.values()}
    if len(lengths) != 1:
        raise ValueError(f"models have different fold counts: "
                         f"{ {n: len(np.ravel(scores[n])) for n in names} }")
    if pairs is not None:
        if reference is not None:
            raise ValueError("give `pairs` or `reference`, not both")
        unknown = {n for p in pairs for n in p} - set(names)
        if unknown:
            raise ValueError(f"pairs name models that are not in `scores`: {sorted(unknown)}")
        pairs = list(pairs)
    else:
        pairs = ([(reference, b) for b in names if b != reference] if reference
                 else list(itertools.combinations(names, 2)))
    rows = []
    for a, b in pairs:
        sa, sb = np.ravel(scores[a]).astype(float), np.ravel(scores[b]).astype(float)
        d = sa - sb
        if d.std(ddof=1) < 1e-10 * max(1.0, abs(d.mean())):
            # A numerically constant difference makes the t statistic explode rather than
            # vanish; report it as untestable instead of printing a spurious p < 0.001.
            t, p = float("nan"), 1.0
        else:
            t, p = corrected_resampled_ttest(sa, sb, rho=rho)
        rows.append(dict(comparison=label, model_a=a, model_b=b, n_folds=len(sa),
                         mean_a=float(np.mean(sa)), mean_b=float(np.mean(sb)),
                         sd_a=float(sa.std(ddof=1)), sd_b=float(sb.std(ddof=1)),
                         mean_diff=float(np.mean(sa - sb)),
                         sd_diff=float(d.std(ddof=1)), t=t, p_raw=p))
    if rows:
        for row, adj in zip(rows, holm_bonferroni([r["p_raw"] for r in rows])):
            row["p_holm"] = float(adj)
            row["significant_005"] = bool(adj < 0.05)
    return pd.DataFrame(rows)


def ci_table(y_true, preds, groups, n_boot=2000, seed=42, conf=0.95, label=""):
    """Cluster-bootstrap point estimate and interval for R2, RMSEP and RPD, per model.

    `preds` maps a model name to its predictions on the same held-out rows. Resampling is over
    whole `groups`, so the interval reflects the number of independent physical samples rather
    than the number of scans.
    """
    rows = []
    y_true = np.asarray(y_true, float)
    for name, p in preds.items():
        p = np.asarray(p, float)
        row = dict(dataset=label, model=name, n_test=len(y_true),
                   n_groups=int(len(np.unique(groups))),
                   R2=float(r2_score(y_true, p)), RMSEP=rmse(y_true, p), RPD=rpd(y_true, p))
        for metric, fn in [("R2", lambda a, b: r2_score(a, b)),
                           ("RMSEP", lambda a, b: rmse(a, b)),
                           ("RPD", lambda a, b: rpd(a, b))]:
            _, lo, hi = bootstrap_ci(y_true, p, fn, n_boot=n_boot, seed=seed, conf=conf,
                                     groups=groups)
            row[f"{metric}_lo"], row[f"{metric}_hi"] = lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_ci(point, lo, hi, dp=3):
    return f"{point:.{dp}f} [{lo:.{dp}f}, {hi:.{dp}f}]"
