"""The canonical closed-form-equation chain, in one place, with the published chain as a guard.

Why this module exists
----------------------
Three separate scripts previously re-implemented the "train -> prune -> auto_symbolic -> predict"
chain, and one of them (`r03_stability.py`) got it wrong in a way that produced impossible numbers.
Everything that needs an equation now calls `extract()` here, and `extract()` in `legacy` mode is
asserted to reproduce the published values before any new mode is trusted:

    tahini  spline 0.9834 / symbolic 0.9479   (results_phase1/paper/kan_equation.txt)
    mango   spline 0.8185 / symbolic 0.8233   (results_mango/paper/mango_equation.txt)

Two bugs in the previous re-implementation, both root-caused and both avoided here
---------------------------------------------------------------------------------
1. `MultKAN.prune()` calls `attribute()`, which scores nodes from `self.acts` -- the activations
   recorded by the **most recent forward pass**. Running synthetic probe inputs (an importance
   sweep with all inputs but one held at zero) through the model before pruning therefore prunes
   the network on the basis of activations that never occur in the data. Measured on tahini seed
   42: identical chain, symbolic R2 = 0.9480; with a probe sweep inserted before `prune()`,
   symbolic R2 = -0.0029. `sensitivity()` below is therefore documented as destructive and every
   caller is required to run it *after* extraction, never between the fit and the prune.
2. `auto_symbolic()` calls `set_mode(l, i, j, 's')`, which sets `act_fun[l].mask = 0` and
   `symbolic_fun[l].mask = 1`. Counting retained edges from the spline mask afterwards therefore
   always returns zero. `retained_inputs()` reads the symbolic mask when the model has been
   symbolised and the spline mask when it has not.

Two levers the published chain was missing
------------------------------------------
A. `auto_symbolic` snaps each edge by fitting `c*f(a*x+b)+d` to `self.acts[l][:, i]` and
   `self.spline_postacts[l][:, j, i]` -- the activations of the **last forward pass**. Inside
   `MultKAN.fit(..., batch=128)` the last forward pass is a random 128-row mini-batch drawn with
   `np.random.choice`, so by default every edge is snapped against 128 rows rather than the whole
   calibration set, and the result depends on the NumPy RNG history of the whole process. That is
   why the published mango value (0.8233) reproduces exactly only when the four `complexity_table`
   fits run first, and gives 0.8422 when the extraction is run on its own. Passing the full
   training set through the model immediately before `auto_symbolic` removes both problems.
   `full_batch_symbolic` controls it; `legacy` mode switches it off.
B. pykan's symbolic edges are `c * f(a*x + b) + d` with `a, b, c, d` trainable
(`Symbolic_KANLayer.affine` is an `nn.Parameter`). The documented workflow is
train -> prune -> auto_symbolic -> **train again**, so that those affine parameters are fitted
jointly rather than edge-by-edge. The published chain stopped before that final step. Measured on
tahini seed 42, adding a 200-step post-symbolic refit moves the equation from R2 0.9479 to 0.9858.
`refit_steps` exposes it; `legacy` mode sets it to 0.
"""
import copy
import re
import numpy as np
import sympy
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from kan import KAN
from kanfood.metrics import normalize_to_100
from kanfood.interpret import fit_kan, SYM_LIB, CKPT_DIR
from paper1_rigor.common import silent, SEED

__all__ = ["extract", "sensitivity", "retained_inputs", "count_terms", "SYM_LIB"]

# Published reference values, for the guard in `verify_reference()`.
REFERENCE = {"tahini": (0.9834, 0.9479), "mango": (0.8185, 0.8233)}


def _t(A):
    return torch.tensor(np.asarray(A), dtype=torch.float32)


def _build(n_in, width, n_out, grid, k, seed):
    """`width` may contain ints (addition nodes) or [n_add, n_mult] pairs (MultKAN layers)."""
    torch.manual_seed(seed)
    return silent(lambda: KAN(width=[n_in] + list(width) + [n_out], grid=grid, k=k, seed=seed,
                              device="cpu", auto_save=False, grid_range=[-1, 1],
                              ckpt_path=str(CKPT_DIR)))


def count_terms(expr_str):
    """Additive terms in the printed expression, and the distinct input variables it uses."""
    if not expr_str or expr_str.startswith("FAILED"):
        return np.nan, np.nan
    terms = len([s for s in re.split(r"(?<![eE])\s[+-]\s", " " + expr_str.strip()) if s.strip()])
    return terms, len(set(re.findall(r"x_\{?(\d+)", expr_str)))


def retained_inputs(model, n_in):
    """Input variables that still reach the first hidden layer.

    After `auto_symbolic` the spline mask is zeroed by construction, so the symbolic mask is the
    one that carries the information. Before symbolisation the spline mask is the live one.
    """
    sym = model.symbolic_fun[0].mask.detach().cpu().numpy()          # (out_dim, in_dim)
    spl = model.act_fun[0].mask.detach().cpu().numpy().reshape(n_in, -1)   # (in_dim, out_dim)
    live = np.abs(sym).sum(axis=0) + np.abs(spl).sum(axis=1)
    return [int(i + 1) for i in range(n_in) if live[i] > 0]


def sensitivity(model, n_in, n_points=80):
    """Intrinsic importance: the range of each input's marginal response.

    DESTRUCTIVE -- this runs synthetic forward passes and therefore overwrites `model.acts`, which
    `prune()` reads. Call it only on a model that will not be pruned afterwards.
    """
    g = np.linspace(-1, 1, n_points)
    s = np.zeros(n_in)
    for j in range(n_in):
        base = np.zeros((len(g), n_in))
        base[:, j] = g
        with torch.no_grad():
            out = model(_t(base)).numpy()[:, 0]
        s[j] = float(out.max() - out.min())
    return s


def extract(Z_tr, y_tr, Z_te, y_te_primary, *, normalise, seed=SEED, width=(3,), grid=5, k=3,
            steps=250, prune_steps=60, refit_steps=200, weight_simple=0.8, r2_threshold=0.0,
            lib=SYM_LIB, lamb=1e-3, legacy=False, print_units=None,
            full_batch_symbolic=True, want_sensitivity=False,
            opt="Adam", lr=0.005, refine_grids=(), refine_steps=150):
    """Fit a KAN, prune it, convert it to a closed form, and score all three stages.

    `legacy=True` reproduces the published chain exactly (no post-symbolic refit).
    `print_units` may be `(sx, sy)` scalers so the printed formula is in original units.

    Training levers, all switched off by default so the published chain is unchanged:

    `opt` / `lr`
        pykan's own default is LBFGS at lr=1.0; the published pipeline overrode it with Adam at
        an Adam-scale learning rate and the two have never been compared on these data. LBFGS is
        run full-batch (pykan's `batch=-1`), Adam at batch 128 as published.
    `refine_grids`
        pykan's documented route to a fine grid is to train coarse and then call `refine()`,
        not to initialise at the fine grid. Each entry is a new grid size; the model is refined
        to it and trained for `refine_steps` further steps. `refine()` initialises the new
        spline coefficients from `self.cache_data`, which after `fit(batch=128)` holds only the
        last random mini-batch -- the whole calibration set is pushed through first so the new
        knots are placed on the real data range.

    The post-symbolic refit always uses Adam. It adjusts four affine parameters per edge, and a
    full-batch quasi-Newton step on that small, badly scaled problem is unstable in practice.

    Returns a dict with, on the held-out data:
        r2_spline    the pruned/unpruned spline network
        r2_symbolic  the closed-form equation
        r2_fidelity  the equation scored against the SPLINE NETWORK's predictions, not against y
                     (how faithfully the equation reproduces the model it came from)
        equation, latex, n_terms, n_vars, retained
    """
    if legacy:
        refit_steps = 0
        full_batch_symbolic = False
    n_in, n_out = Z_tr.shape[1], y_tr.shape[1]
    sx, sy = MinMaxScaler((-1, 1)), StandardScaler()
    Zs, ys = sx.fit_transform(Z_tr), sy.fit_transform(y_tr)
    dsd = {"train_input": _t(Zs), "train_label": _t(ys),
           "test_input": _t(Zs), "test_label": _t(ys)}

    def pred(m, Z):
        with torch.no_grad():
            out = m(_t(sx.transform(Z))).numpy()
        out = sy.inverse_transform(out)
        out = normalize_to_100(out) if normalise else out
        return out[:, 0]

    def score(a, b):
        try:
            if not np.all(np.isfinite(b)):
                return float("nan")
            return float(r2_score(a, b))
        except Exception:
            return float("nan")

    # --- stage 1: fit. The published configuration goes through kanfood.interpret.fit_kan so that
    #     `legacy` is bit-for-bit the published chain; anything else is built here.
    published_cfg = (tuple(width) == (3,) and grid == 5 and k == 3 and lamb == 1e-3
                     and opt == "Adam" and lr == 0.005 and not refine_grids)
    batch = -1 if opt == "LBFGS" else 128
    if published_cfg:
        m, dsd = fit_kan(Zs, ys, width_hidden=(3,), grid=grid, steps=steps, seed=seed)
    else:
        m = _build(n_in, width, n_out, grid, k, seed)
        silent(lambda: m.fit(dsd, opt=opt, steps=steps, lr=lr, lamb=lamb, batch=batch))
        for g in refine_grids:
            silent(lambda: m.get_act(dsd["train_input"]))   # place the new knots on ALL the data
            m = silent(lambda: m.refine(g))
            m.auto_save = False                             # refine() returns an auto-saving copy
            silent(lambda: m.fit(dsd, opt=opt, steps=refine_steps, lr=lr, lamb=lamb, batch=batch))

    out = dict(seed=seed, width=str(width), grid=grid, k=k, steps=steps, refit_steps=refit_steps,
               weight_simple=weight_simple, r2_threshold=r2_threshold,
               full_batch_symbolic=full_batch_symbolic, n_in=n_in, opt=opt, lr=lr,
               refine_grids=";".join(map(str, refine_grids)),
               final_grid=(refine_grids[-1] if refine_grids else grid),
               n_params=int(sum(p.numel() for p in m.parameters())))

    # The published chain scores the spline network BEFORE pruning; keep that definition so the
    # numbers are comparable with results_phase1/paper/kan_equation.txt.
    spline_pre = pred(m, Z_te)

    # Intrinsic importance of the trained (unpruned) network. It is computed on a DEEP COPY,
    # because the synthetic probe inputs overwrite `acts`, which `prune()` reads a few lines
    # below -- the exact defect that produced the impossible numbers in the previous
    # `r03_stability.py` (module docstring, item 1).
    if want_sensitivity:
        out_sens = sensitivity(copy.deepcopy(m), n_in)
    else:
        out_sens = None

    # --- stage 2: prune. Nothing may run a forward pass on synthetic data before this point.
    try:
        m = silent(lambda: m.prune())
        if prune_steps:
            silent(lambda: m.fit(dsd, opt=opt, steps=prune_steps, lr=lr, lamb=lamb, batch=batch))
    except Exception as e:
        out.update(r2_spline=score(y_te_primary, spline_pre), r2_spline_pruned=float("nan"),
                   r2_symbolic=float("nan"), r2_fidelity=float("nan"),
                   equation=f"FAILED at prune: {type(e).__name__}: {e}", latex="",
                   n_terms=np.nan, n_vars=np.nan, retained="")
        return out

    spline_te = pred(m, Z_te)                       # after pruning, for the fidelity comparison
    out["r2_spline"] = score(y_te_primary, spline_pre)
    out["r2_spline_pruned"] = score(y_te_primary, spline_te)

    # --- stage 3: symbolic conversion, then the post-symbolic refit of the affine parameters.
    try:
        if full_batch_symbolic:
            # Snap every edge against the WHOLE calibration set rather than the last random
            # mini-batch (see note A in the module docstring).
            with torch.no_grad():
                m(dsd["train_input"])
        silent(lambda: m.auto_symbolic(lib=lib, weight_simple=weight_simple,
                                       r2_threshold=r2_threshold))
        if refit_steps:
            silent(lambda: m.fit(dsd, opt="Adam", steps=refit_steps, lr=0.005, lamb=0.0, batch=128))
        symb_te = pred(m, Z_te)
        out["r2_symbolic"] = score(y_te_primary, symb_te)
        out["r2_fidelity"] = score(spline_te, symb_te)
        out["retained"] = ";".join(map(str, retained_inputs(m, n_in)))
        vars_ = [sympy.Symbol(f"x_{i+1}") for i in range(n_in)]
        kw = {}
        if print_units is not None:
            psx, psy = print_units
            kw = dict(normalizer=[psx[0], psx[1]], output_normalizer=[psy[0], psy[1]])
        expr = silent(lambda: m.symbolic_formula(var=vars_, **kw)[0][0])
        try:
            from kan.utils import ex_round
            expr = ex_round(expr, 3)
        except Exception:
            pass
        out["equation"], out["latex"] = str(expr), sympy.latex(expr)
    except Exception as e:
        out.update(r2_symbolic=float("nan"), r2_fidelity=float("nan"), retained="",
                   equation=f"FAILED at symbolic: {type(e).__name__}: {e}", latex="")

    out["n_terms"], out["n_vars"] = count_terms(out.get("equation", ""))
    out["sensitivity"] = out_sens
    out["model"] = m
    return out


SPLINE_TOL = 2e-3      # the spline stage is deterministic and must match exactly
SYMBOLIC_TOL = 0.03    # the legacy symbolic stage is not -- see note A; measured spread ~0.02


def verify_reference(verbose=True):
    """Guard: this module must reproduce the published pipeline before any new mode is trusted.

    Two parts, because the two stages have different reproducibility properties.

    * The SPLINE stage is deterministic. Reproducing it exactly proves that the data, the split,
      the preprocessing, the PLS compression, the architecture and the fit are identical to the
      pipeline that produced the published tables. This is a hard requirement.
    * The LEGACY SYMBOLIC stage is not deterministic across execution contexts, because
      `auto_symbolic` snaps each edge against the activations of the last forward pass, which
      inside `MultKAN.fit(batch=128)` is a random 128-row mini-batch drawn from the NumPy global
      RNG (module docstring, note A). Measured values for the published tahini configuration:
      0.9479 (published), 0.9480, 0.9536. For mango: 0.8233 (published, reproduced exactly when
      the published script's four-point complexity sweep runs first) and 0.8422 (standalone).
      The guard therefore allows a tolerance here and the manuscript reports a distribution.

    `full_batch_symbolic=True` removes the mini-batch dependence and is checked for determinism
    separately by `verify_determinism()`.
    """
    from paper1_rigor.common import TAHINI, MANGO, tahini_split, mango_split, project
    hard_ok = True
    for label, splitter, cfg, nc, normalise in [
            ("tahini", tahini_split, TAHINI, "kan_nc", True),
            ("mango", mango_split, MANGO, "kan_eq_nc", False)]:
        ds, tr, te = splitter()
        Z_tr, Z_te, *_ = project(ds, tr, te, cfg["preprocess"], cfg[nc])
        r = extract(Z_tr, ds.y[tr], Z_te, ds.y[te, 0], normalise=normalise, legacy=True)
        exp_sp, exp_sy = REFERENCE[label]
        sp_ok = abs(r["r2_spline"] - exp_sp) < SPLINE_TOL
        sy_ok = abs(r["r2_symbolic"] - exp_sy) < SYMBOLIC_TOL
        hard_ok &= sp_ok
        if verbose:
            print(f"  {label:7s} spline   {r['r2_spline']:.4f}  (published {exp_sp})  "
                  f"-> {'OK' if sp_ok else 'MISMATCH -- HARD FAILURE'}")
            print(f"  {label:7s} symbolic {r['r2_symbolic']:.4f}  (published {exp_sy})  "
                  f"-> {'within documented spread' if sy_ok else 'OUTSIDE documented spread'}")
    return hard_ok


def verify_determinism(verbose=True):
    """`full_batch_symbolic=True` should make the symbolic stage reproducible run to run."""
    from paper1_rigor.common import TAHINI, tahini_split, project
    ds, tr, te = tahini_split()
    Z_tr, Z_te, *_ = project(ds, tr, te, TAHINI["preprocess"], TAHINI["kan_nc"])
    vals = []
    for _ in range(2):
        r = extract(Z_tr, ds.y[tr], Z_te, ds.y[te, 0], normalise=True,
                    full_batch_symbolic=True, refit_steps=200)
        vals.append(r["r2_symbolic"])
    same = abs(vals[0] - vals[1]) < 1e-6
    if verbose:
        print(f"  full-batch symbolic, two identical calls: {vals[0]:.6f} / {vals[1]:.6f}  "
              f"-> {'deterministic' if same else 'NOT deterministic'}")
    return same


if __name__ == "__main__":
    print("Verifying paper1_rigor.equation against the published pipeline ...")
    ok = verify_reference()
    print()
    print("Checking determinism of the proposed extraction protocol ...")
    det = verify_determinism()
    print()
    print("GUARD PASSED" if ok else "HARD FAILURE -- the spline stage does not reproduce")
