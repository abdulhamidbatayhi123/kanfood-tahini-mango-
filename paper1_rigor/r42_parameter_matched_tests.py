"""Paired tests for the twelve-cell parameter-matched comparison (Table S6, block 3).

Round 2 of the *Foods* review (reviewer 1, comment 1): "The parameter-matched MLP has a higher mean
than that of the KAN, but no significance test was performed. Please add a simple paired test or
soften the conclusion to 'the two are comparable in accuracy, and no KAN advantage was found.'"

Each KAN cell of block 3 (two configurations x two training rules, on tahini and on the two mango
capacities) is compared with the parameter-matched MLP trained by the same rule, paired by seed:
the same five seeds (42, 7, 2024, 100, 13), the same training partition and the same external test
set. Reported per comparison:

* the mean difference (MLP - KAN) and its 95 % t-interval;
* a two-sided paired t-test (4 degrees of freedom) -- the simple test the reviewer asks for. It asks
  whether the difference exceeds seed-to-seed variation on this one test set, not whether it would
  hold on new material, which a fixed partition cannot answer;
* the same test with the Nadeau-Bengio variance correction at rho = n_test/n_train, the convention
  of the seed-repeated families of Table S14, as a deliberately conservative sensitivity check;
* two one-sided tests of equivalence (TOST) at margins of +/-0.01 and +/-0.02 R2, on the plain paired
  variance and on the corrected variance (the latter is the convention of Section 2.5 and Section S23).

Holm adjustment is applied over the twelve comparisons, separately for each kind of test. Input:
`r10b_config_vs_harness_raw.csv` (the per-seed results behind Table S6, block 3).
"""
import numpy as np
import pandas as pd
from scipy import stats

from kanfood.metrics import holm_bonferroni
from paper1_rigor.common import OUT, save

RHO = {"tahini": 408 / 1146, "mango (accuracy KAN)": 1448 / 10243,
       "mango (compact KAN)": 1448 / 10243}
HARNESS = {"published recipe": "fixed steps", "convergence harness": "early stopping"}
CONFIG = {"published config": "published", "inner-CV selected": "inner-CV"}


def tost_p(d, margin, rho=0.0):
    n, m = len(d), d.mean()
    se = np.sqrt(d.var(ddof=1) * (1 / n + rho))   # rho = 0: plain paired standard error
    return max(stats.t.sf((m + margin) / se, n - 1), stats.t.cdf((m - margin) / se, n - 1))


def main():
    raw = pd.read_csv(OUT / "r10b_config_vs_harness_raw.csv")
    rows = []
    for ds in ["tahini", "mango (accuracy KAN)", "mango (compact KAN)"]:
        for h in ["published recipe", "convergence harness"]:
            mlp = raw[(raw.dataset == ds) & (raw.architecture == "MLP") & (raw.harness == h)] \
                .set_index("seed").external_R2
            for cfg in ["published config", "inner-CV selected"]:
                kan = raw[(raw.dataset == ds) & (raw.architecture == "KAN") & (raw.harness == h)
                          & (raw.configuration == cfg)].set_index("seed").external_R2
                seeds = sorted(set(mlp.index) & set(kan.index))
                a, b = mlp[seeds].to_numpy(), kan[seeds].to_numpy()
                d, n = a - b, len(seeds)
                sd = d.std(ddof=1)
                half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
                t_c = d.mean() / np.sqrt(sd ** 2 * (1 / n + RHO[ds]))
                rows.append(dict(
                    dataset=ds.replace(" KAN", ""), harness=HARNESS[h], kan_configuration=CONFIG[cfg],
                    n_seeds=n, MLP_mean=a.mean(), KAN_mean=b.mean(), diff_MLP_minus_KAN=d.mean(),
                    ci95_lo=d.mean() - half, ci95_hi=d.mean() + half,
                    seeds_MLP_higher=int((d > 0).sum()),
                    p_paired=stats.ttest_rel(a, b).pvalue,
                    p_corrected=2 * stats.t.sf(abs(t_c), n - 1),
                    p_tost_001=tost_p(d, 0.01), p_tost_002=tost_p(d, 0.02),
                    p_tost_001_corrected=tost_p(d, 0.01, RHO[ds]),
                    p_tost_002_corrected=tost_p(d, 0.02, RHO[ds])))
    df = pd.DataFrame(rows)
    for col in ["p_paired", "p_corrected", "p_tost_001", "p_tost_002", "p_tost_001_corrected",
                "p_tost_002_corrected"]:
        df[col + "_holm"] = holm_bonferroni(df[col].tolist())
    save(df, "r42_parameter_matched_tests.csv")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
