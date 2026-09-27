"""Intrinsic KAN importance versus two post-hoc attributions (permutation and SHAP).

The manuscript's interpretability claim is that the KAN reports intrinsically the same drivers a
black box would only reveal through an external explainer. That is a quantitative claim and is
tested here as a rank correlation, not asserted.
"""
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from kanfood.preprocess import Preprocessor
from kanfood.features import PLSFeatures
from kanfood.interpret import fit_kan
from paper1_rigor.common import TAHINI, MANGO, SEED, tahini_split, mango_split, save

N_SHAP_BG, N_SHAP_EVAL = 100, 300


def build(ds, tr, te, preprocess, nc, width):
    pp = Preprocessor(preprocess).fit(ds.X[tr])
    S_tr, S_te = pp.transform(ds.X[tr]), pp.transform(ds.X[te])
    pf = PLSFeatures(nc).fit(S_tr, ds.y[tr])
    Z_tr, Z_te = pf.transform(S_tr), pf.transform(S_te)
    sx, sy = MinMaxScaler((-1, 1)), StandardScaler()
    Zs, ys = sx.fit_transform(Z_tr), sy.fit_transform(ds.y[tr])
    m, _ = fit_kan(Zs, ys, width_hidden=width, steps=250, seed=SEED)

    def predict(Z):
        with torch.no_grad():
            out = m(torch.tensor(sx.transform(Z), dtype=torch.float32)).numpy()
        return sy.inverse_transform(out)[:, 0]

    return m, predict, Z_tr, Z_te, nc


def intrinsic(m, nc):
    g = np.linspace(-1, 1, 80)
    s = np.zeros(nc)
    for k in range(nc):
        base = np.zeros((len(g), nc)); base[:, k] = g
        with torch.no_grad():
            out = m(torch.tensor(base, dtype=torch.float32)).numpy()[:, 0]
        s[k] = out.max() - out.min()
    return s / s.sum()


def permutation(predict, Z_te, y_te, nc, repeats=10):
    base = r2_score(y_te, predict(Z_te))
    rng = np.random.RandomState(SEED)
    p = np.zeros(nc)
    for k in range(nc):
        drops = []
        for _ in range(repeats):
            Zp = Z_te.copy(); Zp[:, k] = rng.permutation(Zp[:, k])
            drops.append(base - r2_score(y_te, predict(Zp)))
        p[k] = max(np.mean(drops), 0.0)
    return p / (p.sum() + 1e-12)


def shap_importance(predict, Z_tr, Z_te):
    import shap
    rng = np.random.RandomState(SEED)
    bg = Z_tr[rng.choice(len(Z_tr), min(N_SHAP_BG, len(Z_tr)), replace=False)]
    ev = Z_te[rng.choice(len(Z_te), min(N_SHAP_EVAL, len(Z_te)), replace=False)]
    expl = shap.KernelExplainer(predict, shap.kmeans(bg, 25))
    vals = expl.shap_values(ev, nsamples=200, silent=True)
    v = np.abs(np.asarray(vals)).mean(axis=0).ravel()
    return v / (v.sum() + 1e-12)


def run(label, ds, tr, te, preprocess, nc, width):
    print(f"\n=== {label}: intrinsic vs post-hoc importance (nc={nc}) ===")
    m, predict, Z_tr, Z_te, nc = build(ds, tr, te, preprocess, nc, width)
    y_te = ds.y[te, 0]
    ii = intrinsic(m, nc)
    pi = permutation(predict, Z_te, y_te, nc)
    si = shap_importance(predict, Z_tr, Z_te)
    rho_perm = spearmanr(ii, pi).statistic
    rho_shap = spearmanr(ii, si).statistic
    rho_ps = spearmanr(pi, si).statistic
    print(f"  Spearman rho  intrinsic vs permutation = {rho_perm:.3f}")
    print(f"  Spearman rho  intrinsic vs SHAP        = {rho_shap:.3f}")
    print(f"  Spearman rho  permutation vs SHAP      = {rho_ps:.3f}")
    per_comp = pd.DataFrame({"dataset": label,
                             "component": [f"x{i+1}" for i in range(nc)],
                             "KAN_intrinsic": ii, "permutation": pi, "SHAP": si})
    summary = dict(dataset=label, n_components=nc,
                   rho_intrinsic_vs_permutation=rho_perm,
                   rho_intrinsic_vs_SHAP=rho_shap,
                   rho_permutation_vs_SHAP=rho_ps,
                   shap_background=N_SHAP_BG, shap_eval=min(N_SHAP_EVAL, len(Z_te)))
    return per_comp, summary


def main():
    ds, tr, te = tahini_split()
    c1, s1 = run("tahini", ds, tr, te, TAHINI["preprocess"], TAHINI["kan_nc"], TAHINI["kan_width"])
    dm, mtr, mte = mango_split()
    c2, s2 = run("mango", dm, mtr, mte, MANGO["preprocess"], MANGO["kan_eq_nc"], MANGO["kan_eq_width"])
    save(pd.concat([c1, c2], ignore_index=True), "r06_importance_per_component.csv")
    save(pd.DataFrame([s1, s2]), "r06_importance_agreement.csv")


if __name__ == "__main__":
    main()
