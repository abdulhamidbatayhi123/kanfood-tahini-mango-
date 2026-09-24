"""R14 -- Why does the cloud reproduction select a different mango MLP than Table 3?

Scores every mango MLP configuration of the published grid on the published tuning folds (3-fold grouped,
seed 42, SG1, training seasons 1-3) exactly as kanfood.tune.tune_model does, and reports the margin between
the configuration selected here, (256,128)/24, and the published one, (128,64)/16.
Run: python -m revision.r14_mango_mlp_tie
"""
import numpy as np
import pandas as pd

from revision.common import RESULTS, load, primary_split, _subset, inner_score, CFG
from kanfood.split import stratified_group_kfold
from kanfood.run_mango import MANGO_GRIDS

ds, meta = load("mango")
tr, te = primary_split("mango", ds, meta)
sub = _subset(ds, tr)
folds = stratified_group_kfold(sub.groups, sub.tagsis, 3, 42)
rows = []
for p in MANGO_GRIDS["MLP"]:
    sc = [inner_score(sub, a, b, "MLP", p, "sg1", CFG["mango"]["n_pls"], seed=42) for a, b in folds]
    rows.append({"hidden": str(p["hidden"]), "n_components": p["n_components"], "mean": np.mean(sc), "folds": np.round(sc, 4).tolist()})
    print(rows[-1], flush=True)
df = pd.DataFrame(rows).sort_values("mean", ascending=False)
df.to_csv(RESULTS / "r14" / "mlp_tie.csv", index=False)
print(df.to_string(index=False))
