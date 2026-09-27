"""What the models say drives the calibration, on the axis where the chemistry lives.

Three weaknesses of the first attribution analysis (`r06`) are addressed here.

1. BUDGET. `r06` ran `shap.KernelExplainer` with a 100-sample background and 300 evaluation
   points at 200 samples each, and reported a single Spearman rho -- 0.76 on tahini, which is
   below the 0.86-0.90 the companion paper reported. A number that low has to be shown not to be
   a sampling artefact before it is printed, so rho is recomputed over a ladder of budgets and
   the trend is reported with it.

2. LIKE-FOR-LIKE. The claim being made is that the KAN reports intrinsically what a black box
   yields only through an external explainer. That is a comparison, so SHAP is computed for the
   MLP and the CNN as well, not only for the KAN.

3. THE AXIS. Agreement measured on PLS scores is agreement about latent variables, which are
   not chemistry. Attributions are therefore also reported on the WAVENUMBER axis two ways:
     * by propagating latent attributions through the PLS loadings, |W| @ s, which is exact for
       the projection and needs no extra model evaluations;
     * by BLOCK permutation of contiguous spectral windows, which is model-agnostic and works
       for models that never see latent scores at all (PLS on the full spectrum, the CNN). This
       puts every model on one common axis, so the chemistry agreement is measured rather than
       asserted, and the leading windows can be matched against published assignments.
"""
import time
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from kanfood.bands import assign_band as assign_ftir
from kanfood.features import PLSFeatures
from kanfood.interpret import fit_kan
from kanfood.metrics import normalize_to_100
from kanfood.models import build_model
from kanfood.preprocess import Preprocessor
from paper1_rigor.common import ckpt, TAHINI, MANGO, SEED, tahini_split, mango_split, save
from paper1_rigor.nir_bands import assign_band as assign_nir

warnings.filterwarnings("ignore")

BUDGETS = [(50, 150, 100), (100, 300, 200), (200, 600, 400)]   # (background, eval rows, nsamples)
N_BLOCKS = 40
PERM_REPEATS = 10


# ------------------------------------------------------------------ attribution estimators
def intrinsic_kan(m, n_in, n_points=80):
    """Range of each input's marginal response, read straight off the trained edge functions."""
    g = np.linspace(-1, 1, n_points)
    s = np.zeros(n_in)
    for j in range(n_in):
        base = np.zeros((len(g), n_in))
        base[:, j] = g
        with torch.no_grad():
            out = m(torch.tensor(base, dtype=torch.float32)).numpy()[:, 0]
        s[j] = out.max() - out.min()
    return s / (s.sum() + 1e-12)


def permutation_importance(predict, A, y, repeats=PERM_REPEATS, seed=SEED):
    base = r2_score(y, predict(A))
    rng = np.random.RandomState(seed)
    v = np.zeros(A.shape[1])
    for j in range(A.shape[1]):
        drops = []
        for _ in range(repeats):
            Ap = A.copy()
            Ap[:, j] = rng.permutation(Ap[:, j])
            drops.append(base - r2_score(y, predict(Ap)))
        v[j] = max(float(np.mean(drops)), 0.0)
    return v / (v.sum() + 1e-12)


def block_permutation(predict, S, y, blocks, repeats=PERM_REPEATS, seed=SEED):
    """Permute whole contiguous spectral windows, jointly, so that correlated neighbouring
    channels cannot substitute for one another -- the standard failure of per-channel
    permutation on spectra."""
    base = r2_score(y, predict(S))
    rng = np.random.RandomState(seed)
    v = np.zeros(len(blocks))
    for bi, idx in enumerate(blocks):
        drops = []
        for _ in range(repeats):
            Sp = S.copy()
            order = rng.permutation(len(Sp))
            Sp[:, idx] = Sp[order][:, idx]        # permute the window as a block, row-wise
            drops.append(base - r2_score(y, predict(Sp)))
        v[bi] = max(float(np.mean(drops)), 0.0)
    return v / (v.sum() + 1e-12)


def shap_importance(predict, A_bg, A_ev, n_bg, n_ev, nsamples, seed=SEED):
    import shap
    rng = np.random.RandomState(seed)
    bg = A_bg[rng.choice(len(A_bg), min(n_bg, len(A_bg)), replace=False)]
    ev = A_ev[rng.choice(len(A_ev), min(n_ev, len(A_ev)), replace=False)]
    expl = shap.KernelExplainer(predict, shap.kmeans(bg, min(25, len(bg))))
    vals = expl.shap_values(ev, nsamples=nsamples, silent=True)
    v = np.abs(np.asarray(vals)).mean(axis=0).ravel()
    return v / (v.sum() + 1e-12)


# --------------------------------------------------------------------------- model wrappers
def _post(p, normalise):
    p = np.asarray(p).reshape(len(p), -1)
    return (normalize_to_100(p) if normalise else p)[:, 0]


def build_all(ds, tr, te, cfg, normalise):
    """Every model in the benchmark, on the representation the published pipeline gives it."""
    pp = Preprocessor(cfg["preprocess"]).fit(ds.X[tr])
    S_tr, S_te = pp.transform(ds.X[tr]), pp.transform(ds.X[te])
    y_tr = ds.y[tr]

    out = {"S_tr": S_tr, "S_te": S_te, "spectral": {}, "latent": {}}
    cache = {}
    specs = [("PLS", None, dict(n_components=cfg["pls_nc"])),
             ("SVM", cfg["pls_nc"], dict(C=100.0, gamma=0.05)),
             ("RF", cfg["pls_nc"], dict(n_estimators=300, max_depth=10)),
             ("MLP", cfg["mlp_nc"], dict(hidden=cfg["mlp_hidden"])),
             ("CNN", None, dict(channels=(16, 32), lr=1e-3))]
    for name, nc, kw in specs:
        if nc is None:
            A_tr, A_te = S_tr, S_te
        else:
            if nc not in cache:
                pf = PLSFeatures(nc).fit(S_tr, y_tr)
                cache[nc] = (pf.transform(S_tr), pf.transform(S_te), pf)
            A_tr, A_te, _ = cache[nc]
        mdl = build_model(name, A_tr.shape[1], y_tr.shape[1], seed=SEED, **kw).fit(A_tr, y_tr)
        out["spectral"][name] = (lambda m, pf_=(None if nc is None else cache[nc][2]):
                                 (lambda S: _post(m.predict(S if pf_ is None else pf_.transform(S)),
                                                  normalise)))(mdl)
        if nc is not None:
            out["latent"][name] = (A_tr, A_te,
                                   (lambda m: (lambda A: _post(m.predict(A), normalise)))(mdl))

    # the KAN, built exactly as the published equation pipeline builds it
    nck = cfg["kan_nc"]
    if nck not in cache:
        pf = PLSFeatures(nck).fit(S_tr, y_tr)
        cache[nck] = (pf.transform(S_tr), pf.transform(S_te), pf)
    Z_tr, Z_te, pf_k = cache[nck]
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    sx, sy = MinMaxScaler((-1, 1)), StandardScaler()
    Zs, ys = sx.fit_transform(Z_tr), sy.fit_transform(y_tr)
    m_kan, _ = fit_kan(Zs, ys, width_hidden=cfg["kan_width"], steps=250, seed=SEED)

    def kan_latent(Z):
        with torch.no_grad():
            o = m_kan(torch.tensor(sx.transform(Z), dtype=torch.float32)).numpy()
        return _post(sy.inverse_transform(o), normalise)

    out["latent"]["KAN"] = (Z_tr, Z_te, kan_latent)
    out["spectral"]["KAN"] = lambda S: kan_latent(pf_k.transform(S))
    out["kan_model"], out["kan_nc"], out["pls_kan"] = m_kan, nck, pf_k
    return out


def blocks_for(axis, n_blocks=N_BLOCKS):
    edges = np.linspace(0, len(axis), n_blocks + 1).astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(n_blocks) if edges[i + 1] > edges[i]]


def run(label, splitter, cfg, normalise, assign):
    ds, tr, te = splitter()
    y_te = ds.y[te, 0]
    axis = ds.wavenumbers
    print(f"\n=== {label}: attribution ===", flush=True)
    B = build_all(ds, tr, te, cfg, normalise)

    # ---- latent axis: intrinsic vs permutation vs SHAP, at three budgets ------------------
    lat_rows, conv_rows = [], []
    ii = intrinsic_kan(B["kan_model"], B["kan_nc"])
    for name, (A_tr, A_te, pred) in B["latent"].items():
        pi = permutation_importance(pred, A_te, y_te)
        for n_bg, n_ev, nsamp in BUDGETS:
            t0 = time.time()
            si = shap_importance(pred, A_tr, A_te, n_bg, n_ev, nsamp)
            row = dict(dataset=label, model=name, n_bg=n_bg, n_eval=n_ev, nsamples=nsamp,
                       rho_perm_shap=float(spearmanr(pi, si).statistic),
                       seconds=round(time.time() - t0, 1))
            if A_te.shape[1] == len(ii):
                row["rho_kanintrinsic_shap"] = float(spearmanr(ii, si).statistic)
                row["rho_kanintrinsic_perm"] = float(spearmanr(ii, pi).statistic)
            conv_rows.append(row)
            print(f"  latent {name:4s} bg={n_bg:<4} ev={n_ev:<4} ns={nsamp:<4} "
                  f"rho(perm,SHAP)={row['rho_perm_shap']:.3f}  ({row['seconds']:.0f}s)",
                  flush=True)
        for j in range(A_te.shape[1]):
            lat_rows.append(dict(dataset=label, model=name, component=f"x{j+1}",
                                 permutation=pi[j], SHAP=si[j],
                                 KAN_intrinsic=(ii[j] if j < len(ii) else np.nan)))

    # ---- wavenumber axis ------------------------------------------------------------------
    blocks = blocks_for(axis)
    centres = np.array([axis[b].mean() for b in blocks])
    spec_rows, imp = [], {}
    for name, pred in B["spectral"].items():
        t0 = time.time()
        v = block_permutation(pred, B["S_te"], y_te, blocks)
        imp[name] = v
        print(f"  spectral {name:4s} block permutation done ({time.time() - t0:.0f}s)", flush=True)

    # the KAN's intrinsic latent importance, propagated through the PLS loadings it was given
    W = np.abs(np.asarray(B["pls_kan"].pls.x_loadings_))          # (p, n_components)
    proj = W @ ii[:W.shape[1]]
    imp["KAN intrinsic (projected)"] = proj / (proj.sum() + 1e-12)
    proj_blocks = np.array([imp["KAN intrinsic (projected)"][b].sum() for b in blocks])
    imp["KAN intrinsic (projected)"] = proj_blocks / proj_blocks.sum()

    for bi, c in enumerate(centres):
        a = assign(float(c))
        spec_rows.append(dict(dataset=label, centre=float(c), assignment=a["assignment"],
                              component=a["component"],
                              **{k: float(v[bi]) for k, v in imp.items()}))

    names = list(imp)
    agree = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            agree.append(dict(dataset=label, a=names[i], b=names[j],
                              spearman=float(spearmanr(imp[names[i]], imp[names[j]]).statistic),
                              top5_overlap=len(set(np.argsort(imp[names[i]])[-5:]) &
                                               set(np.argsort(imp[names[j]])[-5:])) / 5.0))
    print("  wavenumber-axis agreement with the KAN's intrinsic explanation:", flush=True)
    for r in agree:
        if r["a"] == "KAN intrinsic (projected)" or r["b"] == "KAN intrinsic (projected)":
            other = r["b"] if r["a"] == "KAN intrinsic (projected)" else r["a"]
            print(f"    vs {other:28s} rho {r['spearman']:.3f}  top-5 overlap "
                  f"{r['top5_overlap']:.2f}", flush=True)
    return conv_rows, lat_rows, spec_rows, agree


def _done(*names):
    """Reload a completed arm from its checkpoints instead of recomputing it.

    The tahini arm costs the better part of an hour and its four checkpoints are written together,
    so either all four exist and the arm finished or none does. Re-running it after a killed
    session would spend that hour reproducing numbers already on disk.
    """
    from paper1_rigor.common import OUT
    paths = [OUT / n for n in names]
    if not all(p.exists() for p in paths):
        return None
    print(f"  [resume] {names[0]} and its three companions exist; skipping that arm", flush=True)
    return [pd.read_csv(p).to_dict("records") for p in paths]


def main():
    t0 = time.time()
    TAHINI_CKPT = ("r15_shap_convergence_tahini.csv", "r15_latent_importance_tahini.csv",
                   "r15_wavenumber_importance_tahini.csv", "r15_wavenumber_agreement_tahini.csv")
    done = _done(*TAHINI_CKPT)
    if done is not None:
        c1, l1, s1, a1 = done
    else:
        c1, l1, s1, a1 = run("tahini", tahini_split, TAHINI, True, assign_ftir)
        for obj, name in zip((c1, l1, s1, a1), TAHINI_CKPT):
            ckpt(obj, name)
    c2, l2, s2, a2 = run("mango", mango_split, MANGO, False, assign_nir)
    save(pd.DataFrame(c1 + c2), "r15_shap_convergence.csv")
    save(pd.DataFrame(l1 + l2), "r15_latent_importance.csv")
    save(pd.DataFrame(s1 + s2), "r15_wavenumber_importance.csv")
    save(pd.DataFrame(a1 + a2), "r15_wavenumber_agreement.csv")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
