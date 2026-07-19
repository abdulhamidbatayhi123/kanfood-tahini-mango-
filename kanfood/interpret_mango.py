"""Interpretability for the mango DMC KAN: symbolic equation (read off the SAME compact (3,) KAN family
as tahini), accuracy-vs-complexity, and learned response curves vs the linear PLS effect. Evaluated on the
held-out Season-4 external set. Writes to results_mango/paper/. Run: python -m kanfood.interpret_mango"""
import sys
import json
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

from kanfood.data import load_mango
from kanfood.preprocess import Preprocessor
from kanfood.features import PLSFeatures
from kanfood.interpret import fit_kan, silent, SYM_LIB
from kanfood._paths import data_path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

OUT = data_path("KANFOOD_MANGO_OUT", "results_mango")
PUB = OUT / "paper"
PUB.mkdir(parents=True, exist_ok=True)
SEED = 42
WL_LO, WL_HI, PREPROCESS = 684.0, 990.0, "sg1"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": 1.0,
})


def canonical_kan_n_components(default=16):
    mp = OUT / "mango_meta.json"
    if mp.exists():
        try:
            return int(json.loads(mp.read_text(encoding="utf-8")).get("kan_n_components", default))
        except Exception:
            pass
    return default


def kan_pred(m, sx, sy, Z):
    """Single-target prediction (NO composition normalization, unlike the tahini helper)."""
    with torch.no_grad():
        out = m(torch.tensor(sx.transform(Z), dtype=torch.float32)).numpy()
    return sy.inverse_transform(out)[:, 0]


def get_split():
    ds = load_mango()
    keep = (ds.wavenumbers >= WL_LO) & (ds.wavenumbers <= WL_HI)
    X, wn = ds.X[:, keep], ds.wavenumbers[keep]
    tr = np.isin(ds.sets, ["Cal", "Tuning"])
    te = ds.sets == "Val Ext"
    pp = Preprocessor(PREPROCESS).fit(X[tr])
    return wn, pp.transform(X[tr]), ds.y[tr], pp.transform(X[te]), ds.y[te]


def complexity_table(Sp_tr, ytr, Sp_te, yte):
    rows = []
    for n in (8, 12, 16, 20):
        plsf = PLSFeatures(n).fit(Sp_tr, ytr)
        Ztr, Zte = plsf.transform(Sp_tr), plsf.transform(Sp_te)
        sx, sy = MinMaxScaler((-1, 1)), StandardScaler()
        Zs, ys = sx.fit_transform(Ztr), sy.fit_transform(ytr)
        m, _ = fit_kan(Zs, ys, width_hidden=(3,), steps=200)
        rows.append({"n_PLS_components": n, "KAN_test_R2_DMC": round(r2_score(yte[:, 0], kan_pred(m, sx, sy, Zte)), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(PUB / "TableM4_kan_complexity.csv", index=False)
    return df


def extract_equation(Sp_tr, ytr, Sp_te, yte, n):
    plsf = PLSFeatures(n).fit(Sp_tr, ytr)
    Ztr, Zte = plsf.transform(Sp_tr), plsf.transform(Sp_te)
    sx, sy = MinMaxScaler((-1, 1)), StandardScaler()
    Zs, ys = sx.fit_transform(Ztr), sy.fit_transform(ytr)
    m, dsd = fit_kan(Zs, ys, width_hidden=(3,), steps=250)
    r2_spline = r2_score(yte[:, 0], kan_pred(m, sx, sy, Zte))
    eq_text, eq_latex, r2_symb = "(extraction failed)", "", float("nan")
    try:
        m = silent(lambda: m.prune())
        silent(lambda: m.fit(dsd, opt="Adam", steps=60, lr=0.005, lamb=0.001, batch=128))
        silent(lambda: m.auto_symbolic(lib=SYM_LIB))
        r2_symb = r2_score(yte[:, 0], kan_pred(m, sx, sy, Zte))
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
    (PUB / "mango_equation.txt").write_text(
        f"KAN mango dry-matter equation (standardized output) as a function of PLS scores x_1..x_{n}\n"
        f"spline-KAN external-test R2 (DMC) = {r2_spline:.4f}\n"
        f"symbolic-KAN external-test R2 (DMC) = {r2_symb:.4f}\n\n{eq_text}\n\nLaTeX:\n{eq_latex}\n",
        encoding="utf-8")
    return r2_spline, r2_symb, eq_text


def fig_response(Sp_tr, ytr, n):
    plsf = PLSFeatures(n).fit(Sp_tr, ytr)
    Ztr = plsf.transform(Sp_tr)
    sx, sy = MinMaxScaler((-1, 1)), StandardScaler()
    Zs, ys = sx.fit_transform(Ztr), sy.fit_transform(ytr)
    m, _ = fit_kan(Zs, ys, width_hidden=(3,), steps=250)
    lin = LinearRegression().fit(Zs, ys[:, 0])
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
        kan_c = kan_out - kan_out[len(g) // 2]
        ax = axes[i]
        ax.plot(g, lin.coef_[i] * g, color="#3B82F6", ls="--", lw=1.5, label="Linear (PLS) effect")
        ax.plot(g, kan_c, color="#8B5CF6", lw=2.0, label="KAN learned function")
        ax.set_title(f"PLS comp. $x_{{{i+1}}}$  (dev={float(np.max(np.abs(kan_c - lin.coef_[i]*g))):.2f})", fontsize=9)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_xlabel(f"$x_{{{i+1}}}$ (scaled)")
        ax.set_ylabel("Δ standardized DMC")
        ax.grid(alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    axes[0].legend(loc="best", fontsize=7, frameon=False)
    fig.suptitle("Mango DMC: KAN learned response per latent variable vs the linear PLS effect",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(PUB / "FigM7_kan_response_vs_pls.png")
    plt.close(fig)


# The benchmark's accuracy-optimal KAN is a 2-layer (16,8) net at nc=24 (external R2~0.86); for a READABLE
# closed-form equation we use the compact, symbolically-extractable (3,) net at a small compression. This
# makes the accuracy<->interpretability trade-off explicit (see TableM4 complexity sweep).
EQUATION_NC = 12


def main():
    wn, Sp_tr, ytr, Sp_te, yte = get_split()
    print("Accuracy vs equation complexity (DMC, external test):")
    print(complexity_table(Sp_tr, ytr, Sp_te, yte).to_string(index=False))
    N = EQUATION_NC
    print(f"\nCompact interpretable KAN for the equation: n_components={N} "
          f"(accuracy-optimal config is (16,8) nc={canonical_kan_n_components()})")
    r2s, r2sym, eq = extract_equation(Sp_tr, ytr, Sp_te, yte, N)
    print(f"Headline equation at N={N}: spline R2={r2s:.4f}, symbolic R2={r2sym:.4f}")
    print(eq)
    fig_response(Sp_tr, ytr, N)
    print("\nMango interpretability assets -> results_mango/paper/: mango_equation.txt, "
          "TableM4_kan_complexity.csv, FigM7_kan_response_vs_pls.png")


if __name__ == "__main__":
    main()
