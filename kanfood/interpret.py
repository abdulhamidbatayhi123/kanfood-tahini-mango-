"""Phase 2: KAN interpretability on tahini.
Produces (into results_phase1/paper/): the symbolic equation (txt + LaTeX), an accuracy-vs-complexity
table, KAN response curves overlaid on the linear PLS effect (where they differ = KAN's added value),
and PLS-component-to-chemistry loadings. Run: python -m kanfood.interpret"""
import sys
import os
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import sympy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from kan import KAN
from kanfood.data import load_tahini
from kanfood.split import group_holdout_split
from kanfood.preprocess import Preprocessor
from kanfood.features import PLSFeatures
from kanfood.metrics import normalize_to_100
from kanfood.bands import BANDS
from kanfood._paths import data_path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

OUT = data_path("KANFOOD_OUT", "results_phase1")
PUB = OUT / "paper"
PUB.mkdir(parents=True, exist_ok=True)
# pykan's prune() spawns an auto_save=True child that logs/saves to ckpt_path; point it at an existing
# scratch dir (absolute, CWD-independent) so equation extraction doesn't crash on a missing './model'.
CKPT_DIR = OUT / "_kan_ckpt"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
SYM_LIB = ['x', 'x^2', 'x^3', 'exp', 'log', 'sqrt', 'tanh', 'sin', '1/x']

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": 1.0,
})


def silent(fn):
    with open(os.devnull, "w") as dn, redirect_stdout(dn), redirect_stderr(dn):
        return fn()


def canonical_kan_n_components(default=8):
    """KAN's PLS-compression dimension selected by the benchmark (results_phase1/phase1_meta.json),
    so the equation and response curves are read off the SAME model that was benchmarked.
    Falls back to `default` if the benchmark has not been run yet."""
    meta_path = OUT / "phase1_meta.json"
    if meta_path.exists():
        try:
            return int(json.loads(meta_path.read_text(encoding="utf-8")).get("kan_n_components", default))
        except Exception:
            pass
    return default


def get_split():
    ds = load_tahini()
    tr, te = group_holdout_split(ds.groups, 0.30, SEED)
    pp = Preprocessor("snv").fit(ds.X[tr])
    return ds, tr, te, pp.transform(ds.X[tr]), pp.transform(ds.X[te])


def fit_kan(Zs, ys, width_hidden=(3,), grid=5, steps=200, seed=SEED):
    torch.manual_seed(seed)
    t = lambda A: torch.tensor(A, dtype=torch.float32)
    dsd = {"train_input": t(Zs), "train_label": t(ys), "test_input": t(Zs), "test_label": t(ys)}
    m = KAN(width=[Zs.shape[1]] + list(width_hidden) + [ys.shape[1]], grid=grid, k=3,
            seed=seed, device="cpu", auto_save=False, grid_range=[-1, 1], ckpt_path=str(CKPT_DIR))
    silent(lambda: m.fit(dsd, opt="Adam", steps=steps, lr=0.005, lamb=0.001, batch=128))
    return m, dsd


def kan_pred(m, sx, sy, Z):
    with torch.no_grad():
        out = m(torch.tensor(sx.transform(Z), dtype=torch.float32)).numpy()
    return normalize_to_100(sy.inverse_transform(out))


def complexity_table(Sp_tr, Sp_te, ytr, yte):
    rows = []
    for n in (5, 8, 12):
        plsf = PLSFeatures(n).fit(Sp_tr, ytr)
        Ztr, Zte = plsf.transform(Sp_tr), plsf.transform(Sp_te)
        sx, sy = MinMaxScaler((-1, 1)), StandardScaler()
        Zs, ys = sx.fit_transform(Ztr), sy.fit_transform(ytr)
        m, _ = fit_kan(Zs, ys, width_hidden=(3,), steps=200)
        r2 = r2_score(yte[:, 0], kan_pred(m, sx, sy, Zte)[:, 0])
        rows.append({"n_PLS_components": n, "KAN_test_R2_tahini": round(r2, 4)})
    df = pd.DataFrame(rows)
    df.to_csv(PUB / "Table4_kan_complexity.csv", index=False)
    return df


def extract_equation(Ztr, ytr, yte_tahini, Zte, n):
    sx, sy = MinMaxScaler((-1, 1)), StandardScaler()
    Zs, ys = sx.fit_transform(Ztr), sy.fit_transform(ytr)
    m, dsd = fit_kan(Zs, ys, width_hidden=(3,), steps=250)
    r2_spline = r2_score(yte_tahini, kan_pred(m, sx, sy, Zte)[:, 0])
    eq_text, eq_latex, r2_symb = "(extraction failed)", "", float("nan")
    try:
        m = silent(lambda: m.prune())
        silent(lambda: m.fit(dsd, opt="Adam", steps=60, lr=0.005, lamb=0.001, batch=128))
        silent(lambda: m.auto_symbolic(lib=SYM_LIB))
        r2_symb = r2_score(yte_tahini, kan_pred(m, sx, sy, Zte)[:, 0])
        vars_ = [sympy.Symbol(f"x_{i+1}") for i in range(n)]
        expr = m.symbolic_formula(var=vars_)[0][0]
        try:
            from kan.utils import ex_round
            expr = ex_round(expr, 3)
        except Exception:
            pass
        eq_text, eq_latex = str(expr), sympy.latex(expr)
    except Exception as e:
        eq_text = f"symbolic extraction failed: {e!r}"
    (PUB / "kan_equation.txt").write_text(
        f"KAN tahini equation (standardized output) as a function of PLS scores x_1..x_{n}\n"
        f"spline-KAN test R2 (tahini) = {r2_spline:.4f}\n"
        f"symbolic-KAN test R2 (tahini) = {r2_symb:.4f}\n\n"
        f"{eq_text}\n\nLaTeX:\n{eq_latex}\n", encoding="utf-8")
    return r2_spline, r2_symb, eq_text


def fig_response_vs_pls(Ztr, ytr, n):
    """KAN marginal response per PLS component vs the linear PLS effect (centered, standardized)."""
    sx, sy = MinMaxScaler((-1, 1)), StandardScaler()
    Zs, ys = sx.fit_transform(Ztr), sy.fit_transform(ytr)
    m, _ = fit_kan(Zs, ys, width_hidden=(3,), steps=250)
    lin = LinearRegression().fit(Zs, ys[:, 0])   # linear effect in standardized output units
    g = np.linspace(-1, 1, 120)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for i in range(n):
        base = np.zeros((len(g), Zs.shape[1]))
        base[:, i] = g
        with torch.no_grad():
            kan_out = m(torch.tensor(base, dtype=torch.float32)).numpy()[:, 0]
        kan_c = kan_out - kan_out[len(g) // 2]                 # center at x=0
        lin_c = lin.coef_[i] * g                               # linear (PLS-style) effect, already centered
        ax = axes[i]
        ax.plot(g, lin_c, color="#3B82F6", ls="--", lw=1.5, label="Linear (PLS) effect")
        ax.plot(g, kan_c, color="#8B5CF6", lw=2.0, label="KAN learned function")
        dev = float(np.max(np.abs(kan_c - lin_c)))
        ax.set_title(f"PLS comp. $x_{{{i+1}}}$  (nonlin. dev={dev:.2f})", fontsize=9)
        ax.axhline(0, color="0.7", lw=0.6); ax.set_xlabel(f"$x_{{{i+1}}}$ (scaled)")
        ax.set_ylabel("Δ standardized tahini")
        ax.grid(alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    axes[0].legend(loc="best", fontsize=7, frameon=False)
    fig.suptitle("KAN learned response per latent variable vs the linear PLS effect "
                 "(divergence = nonlinearity only KAN captures)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PUB / "Fig7_kan_response_vs_pls.png")
    plt.close(fig)


def fig_component_chemistry(Sp_tr, ytr, n, top=3):
    plsf = PLSFeatures(n).fit(Sp_tr, ytr)
    load = plsf.pls.x_loadings_              # (P, n)
    ds = load_tahini()
    wn = ds.wavenumbers
    fig, ax = plt.subplots(figsize=(9, 5))
    for k in range(min(top, n)):
        ax.plot(wn, load[:, k], lw=1.1, label=f"PLS component $x_{{{k+1}}}$")
    for (lo, hi), assignment, _ in BANDS:
        c = (lo + hi) / 2
        if wn.min() <= c <= wn.max():
            ax.axvline(c, color="0.8", ls=":", lw=0.7, zorder=0)
            ax.text(c, ax.get_ylim()[1], assignment.split("(")[0].strip(), rotation=90,
                    fontsize=6, va="top", ha="center", color="0.45")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.invert_xaxis()
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("PLS loading")
    ax.set_title("The equation's variables are chemistry: PLS components vs FTIR bands",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False)
    fig.savefig(PUB / "Fig8_component_chemistry.png")
    plt.close(fig)


def main():
    ds, tr, te, Sp_tr, Sp_te = get_split()
    ytr, yte = ds.y[tr], ds.y[te]

    print("Accuracy vs equation complexity:")
    print(complexity_table(Sp_tr, Sp_te, ytr, yte).to_string(index=False))

    N = canonical_kan_n_components()   # the benchmark's KAN compression -> one canonical config
    print(f"\nCanonical KAN compression (from benchmark): n_components={N}")
    plsf = PLSFeatures(N).fit(Sp_tr, ytr)
    Ztr, Zte = plsf.transform(Sp_tr), plsf.transform(Sp_te)
    r2s, r2sym, eq = extract_equation(Ztr, ytr, yte[:, 0], Zte, N)
    print(f"\nHeadline equation at N={N}: spline R2={r2s:.4f}, symbolic R2={r2sym:.4f}")
    print(eq)

    fig_response_vs_pls(Ztr, ytr, N)
    fig_component_chemistry(Sp_tr, ytr, N)
    print("\nPhase-2 assets -> results_phase1/paper/: kan_equation.txt, Table4_kan_complexity.csv, "
          "Fig7_kan_response_vs_pls.png, Fig8_component_chemistry.png")


if __name__ == "__main__":
    main()
