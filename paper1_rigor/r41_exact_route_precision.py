"""Why the fidelity of the exact route is not exactly 1, and what the input clipping does.

Round 2 of the *Foods* review (reviewer 1, comment 2): "The fidelity of the exact route is not
strictly equal to 1, and the basis functions may become unstable during extrapolation. Please
explain the floating-point error in one sentence and add an input-domain restriction or clipping
rule."

Three measurements, on the network that Equation (3) is printed from (tahini, eight VIP-selected
bands, additive power basis of degree 4, seed 42 -- the configuration recorded in
`band_equation_displayed_tahini_poly.csv`):

1. Fidelity of the printed equation as published (coefficients rounded to three significant
   figures, coefficients below 1e-3 of the largest omitted) against the same network printed at
   full double precision with nothing omitted. The difference isolates what the rounding costs;
   what remains at full precision is the single-precision (float32) arithmetic of the network.
2. Initialisation. `make_band_equation.py`, which produced Equation (3), constructs the network
   before `basis_kan.fit` sets the seed, so the published network's initial weights were not seeded
   and it cannot be regenerated bit for bit. The decomposition above is therefore done on a refit
   with the initialisation seeded too, and five further refits with initialisation seeds 1-5 show
   how far the external R2, the fidelity as printed and the equation length move with the
   initialisation alone.
3. Clipping. Every input is min-max scaled to [-1, 1] on the training partition and new inputs are
   clipped to that interval before the network or the equation is evaluated, so a polynomial edge
   function is never evaluated outside the range it was fitted on. This module reports how often
   that rule is active: the share of input values clipped on the external test set and on every
   withheld production lot (leave-one-lot-out, bands and scaler refitted on the other four lots).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from kanfood.metrics import normalize_to_100
from kanfood.preprocess import Preprocessor
from paper1_rigor import basis_kan as BK
from paper1_rigor.common import TAHINI, tahini_split, save
from paper1_rigor.r13_band_equation import score_vector, select

BASIS, DEGREE, K, SEED, LAMB, PRUNE, MIN_SEP = "poly", 4, 8, 42, 1e-4, 0.05, 15.0
INIT_SEEDS = [1, 2, 3, 4, 5]


def fit_on(ds, tr, te):
    pp = Preprocessor(TAHINI["preprocess"]).fit(ds.X[tr])
    S_tr, S_te = pp.transform(ds.X[tr]), pp.transform(ds.X[te])
    idx = select(score_vector(S_tr, ds.y[tr], "vip"), ds.wavenumbers, K, MIN_SEP)
    names = [f"A_{ds.wavenumbers[i]:.0f}" for i in idx]
    A_tr, A_te = S_tr[:, idx], S_te[:, idx]
    sx = MinMaxScaler((-1, 1)).fit(A_tr)
    raw_te = sx.transform(A_te)
    clipped_share = float(np.mean((raw_te < -1) | (raw_te > 1)))
    return names, sx, A_tr, A_te, raw_te, clipped_share


def spectra_clipped(ds, te, raw_te):
    """How many whole spectra the rule touches: a spectrum counts if any of its bands is clipped.
    Authentic spectra (tahini fraction 100%) are counted separately, with the lot they come from."""
    hit = ((raw_te < -1) | (raw_te > 1)).any(axis=1)
    auth = np.isclose(ds.y[te, 0], 100.0)
    lots = sorted({str(g).split("_")[0] for g in np.asarray(ds.groups)[te][auth]})
    return dict(n_spectra=int(len(te)), n_spectra_clipped=int(hit.sum()),
                n_authentic=int(auth.sum()), n_authentic_clipped=int((hit & auth).sum()),
                authentic_lots=";".join(lots))


def main():
    rows = []
    ds, tr, te = tahini_split()
    names, sx, A_tr, A_te, raw_te, clip_ext = fit_on(ds, tr, te)
    sy = StandardScaler().fit(ds.y[tr])
    Xtr, Xte = sx.transform(A_tr), np.clip(raw_te, -1, 1)
    # Seed the weight initialisation too: BK.fit seeds training, but the network is constructed
    # before it, so without this the initial weights (and the refit) differ from run to run.
    import torch
    torch.manual_seed(SEED); np.random.seed(SEED)
    m = BK.AdditiveKAN(K, basis=BASIS, degree=DEGREE, n_out=ds.y.shape[1])
    BK.fit(m, Xtr, sy.transform(ds.y[tr]), lamb=LAMB, prune_frac=PRUNE, seed=SEED)
    net = BK.predict(m, Xte)[:, 0]
    pred = normalize_to_100(np.asarray(sy.inverse_transform(BK.predict(m, Xte))))[:, 0]
    r2 = float(r2_score(ds.y[te, 0], pred))

    for label, rt, drop in (("published (3 significant figures, small terms omitted)", 3, 1e-3),
                            ("full double precision, nothing omitted", 17, 0.0)):
        ev = BK.evaluate(BK.equation(m, var_names=names, round_to=rt, drop_below=drop), Xte,
                         var_names=names)
        fid = float(r2_score(net, ev))
        rows.append(dict(analysis="fidelity", setting=label, external_R2_network=r2,
                         fidelity=fid, one_minus_fidelity=1.0 - fid,
                         max_abs_diff_scaled=float(np.max(np.abs(net - ev)))))
        print(f"  {label:58s} fidelity {fid:.10f}   1-fid {1-fid:.2e}   "
              f"max|diff| {np.max(np.abs(net - ev)):.2e}")

    # How much the initialisation alone moves the published configuration (training seed fixed).
    for init_seed in INIT_SEEDS:
        torch.manual_seed(init_seed); np.random.seed(init_seed)
        mi = BK.AdditiveKAN(K, basis=BASIS, degree=DEGREE, n_out=ds.y.shape[1])
        BK.fit(mi, Xtr, sy.transform(ds.y[tr]), lamb=LAMB, prune_frac=PRUNE, seed=SEED)
        net_i = BK.predict(mi, Xte)
        pred_i = normalize_to_100(np.asarray(sy.inverse_transform(net_i)))[:, 0]
        pr_i = BK.equation(mi, var_names=names)
        fid_i = float(r2_score(net_i[:, 0], BK.evaluate(pr_i, Xte, var_names=names)))
        rows.append(dict(analysis="initialisation", setting=f"initialisation seed {init_seed}",
                         external_R2_network=float(r2_score(ds.y[te, 0], pred_i)), fidelity=fid_i,
                         one_minus_fidelity=1.0 - fid_i, n_terms=BK.n_terms(pr_i),
                         bands_retained=len(BK.retained(mi))))
        print(f"  init seed {init_seed}: R2 {rows[-1]['external_R2_network']:.4f}  fidelity "
              f"{fid_i:.8f}  terms {rows[-1]['n_terms']}  bands {rows[-1]['bands_retained']}")

    # The constants a reader needs to evaluate Equation (3) from a spectrum: the training range of
    # each band (the min-max scaling to [-1, 1], outside which inputs are clipped) and the mean and
    # SD of the training tahini fraction (the equation predicts it in standardised units).
    for nm, lo, hi in zip(names, sx.data_min_, sx.data_max_):
        rows.append(dict(analysis="scaling", setting=f"{nm} (SNV absorbance)", train_min=float(lo),
                         train_max=float(hi)))
    rows.append(dict(analysis="scaling", setting="tahini fraction (%)", train_mean=float(sy.mean_[0]),
                     train_sd=float(sy.scale_[0])))
    rows.append(dict(analysis="clipping", setting="external test set (17 samples)",
                     share_clipped=clip_ext, **spectra_clipped(ds, te, raw_te)))
    print(f"  clipped input values, external test: {100*clip_ext:.2f} %")
    lots = np.array([g.split("_")[0] for g in ds.groups])
    for lot in sorted(np.unique(lots)):
        te_l, tr_l = np.where(lots == lot)[0], np.where(lots != lot)[0]
        _, _, _, _, _, share = fit_on(ds, tr_l, te_l)
        rows.append(dict(analysis="clipping", setting=f"withheld lot {lot}", share_clipped=share))
        print(f"  clipped input values, withheld lot {lot}: {100*share:.2f} %")
    save(pd.DataFrame(rows), "r41_exact_route_precision.csv")


if __name__ == "__main__":
    main()
