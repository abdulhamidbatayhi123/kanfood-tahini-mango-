"""Kolmogorov-Arnold layers whose edge basis is already elementary, so the equation is exact.

The problem this addresses
--------------------------
A B-spline KAN is converted to a closed form by `auto_symbolic`, which *replaces* each learned
edge function with the best-fitting elementary function from a small library. That step is an
approximation, and it is where the published pipeline loses accuracy (tahini: spline 0.983 ->
equation 0.948). `paper1_rigor.equation` shows most of that loss is procedural rather than
architectural, but a residual gap remains by construction: a cubic B-spline is not an elementary
function, so *something* has to be approximated.

If the edge basis is chosen to be elementary in the first place, nothing has to be approximated.
A Chebyshev, Fourier or Gaussian basis is a finite sum of elementary terms, so the trained network
IS a closed-form expression and the equation reproduces the model to machine precision. The price
is a longer expression -- a degree-d polynomial per retained variable instead of one snapped
function -- and that trade is exactly what the experiment reports.

Design
------
* `phi(x)` maps each input to `n_basis` elementary features. Every basis here is defined on
  [-1, 1] and inputs are scaled to that range, matching the convention pykan uses for its grid.
* `AdditiveKAN` is a depth-1 KAN: y = sum_i f_i(x_i) + b. It is exactly a generalised additive
  model in the chosen basis, and its closed form is a sum of univariate expressions.
* `TwoLayerKAN` adds one hidden layer of Kolmogorov-Arnold form: h_j = sum_i f_ij(x_i), then
  y = sum_j g_j(h_j) + b. Still closed form, but nested, so the printed expression is longer.
* Sparsity is an L1 penalty on the edge coefficient blocks, followed by hard pruning of edges
  whose whole coefficient block is negligible and a short refit. Pruning is what keeps the
  printed equation readable, and the number of surviving edges is reported rather than assumed.

Nothing here replaces the B-spline KAN of the main pipeline; it is an ablation over the edge
basis, run under the same protocol, so that the choice of basis is shown to be a choice.
"""
import numpy as np
import sympy
import torch
import torch.nn as nn

BASES = ("cheby", "fourier", "rbf", "poly", "bspline")


# --------------------------------------------------------------------------- basis functions
def _cheby(x, d):
    """Chebyshev polynomials T_0..T_d of the first kind, by the standard recurrence."""
    out = [torch.ones_like(x), x]
    for _ in range(2, d + 1):
        out.append(2 * x * out[-1] - out[-2])
    return torch.stack(out[:d + 1], dim=-1)


def _fourier(x, d):
    """1, sin(pi x), cos(pi x), ..., sin(m pi x), cos(m pi x), truncated to d+1 terms."""
    feats = [torch.ones_like(x)]
    m = 1
    while len(feats) < d + 1:
        feats.append(torch.sin(m * np.pi * x))
        if len(feats) < d + 1:
            feats.append(torch.cos(m * np.pi * x))
        m += 1
    return torch.stack(feats, dim=-1)


def _rbf(x, d, centres, h):
    return torch.exp(-((x.unsqueeze(-1) - centres) / h) ** 2)


def _poly(x, d):
    return torch.stack([x ** k for k in range(d + 1)], dim=-1)


class Basis(nn.Module):
    """Elementary feature map applied to every input independently."""

    def __init__(self, kind="cheby", degree=6):
        super().__init__()
        self.kind, self.degree = kind, degree
        self.n_basis = degree + 1
        if kind == "rbf":
            self.register_buffer("centres", torch.linspace(-1, 1, self.n_basis))
            self.h = 2.0 / max(self.degree, 1)
        elif kind == "bspline":
            from kan.spline import B_batch
            self._B = B_batch
            self.k = 3
            g = torch.linspace(-1, 1, degree + 1)
            step = 2.0 / degree
            ext = torch.cat([g[0] - step * torch.arange(self.k, 0, -1), g,
                             g[-1] + step * torch.arange(1, self.k + 1)])
            self.register_buffer("grid1", ext[None, :])
            self.n_basis = degree + self.k

    def forward(self, x):                       # x: (batch, n_in) -> (batch, n_in, n_basis)
        x = torch.clamp(x, -1.0, 1.0)
        if self.kind == "cheby":
            return _cheby(x, self.degree)
        if self.kind == "fourier":
            return _fourier(x, self.degree)
        if self.kind == "poly":
            return _poly(x, self.degree)
        if self.kind == "rbf":
            return _rbf(x, self.degree, self.centres, self.h)
        if self.kind == "bspline":
            b, n = x.shape
            grid = self.grid1.repeat(n, 1)
            return self._B(x, grid, k=self.k)
        raise ValueError(self.kind)

    # ---- the same feature map, symbolically -------------------------------------------
    def sym(self, s):
        d = self.degree
        if self.kind == "cheby":
            out = [sympy.Integer(1), s]
            for _ in range(2, d + 1):
                out.append(sympy.expand(2 * s * out[-1] - out[-2]))
            return out[:d + 1]
        if self.kind == "fourier":
            feats, m = [sympy.Integer(1)], 1
            while len(feats) < d + 1:
                feats.append(sympy.sin(m * sympy.pi * s))
                if len(feats) < d + 1:
                    feats.append(sympy.cos(m * sympy.pi * s))
                m += 1
            return feats
        if self.kind == "poly":
            return [s ** k for k in range(d + 1)]
        if self.kind == "rbf":
            return [sympy.exp(-((s - float(c)) / self.h) ** 2) for c in self.centres]
        raise ValueError(f"{self.kind} has no elementary closed form (that is the point of the "
                         f"comparison)")


# --------------------------------------------------------------------------------- networks
class AdditiveKAN(nn.Module):
    """y = sum_i f_i(x_i) + b, with f_i expanded in the chosen elementary basis."""

    def __init__(self, n_in, basis="cheby", degree=6, n_out=1):
        super().__init__()
        self.phi = Basis(basis, degree)
        self.n_in, self.n_out = n_in, n_out
        self.W = nn.Parameter(torch.randn(n_in, self.phi.n_basis, n_out) * 0.05)
        self.b = nn.Parameter(torch.zeros(n_out))
        self.register_buffer("mask", torch.ones(n_in))

    def forward(self, x):
        B = self.phi(x) * self.mask[None, :, None]
        return torch.einsum("bic,ico->bo", B, self.W) + self.b

    def edge_norms(self):
        return self.W.detach().abs().sum(dim=(1, 2)) * self.mask.detach()

    def l1(self):
        return self.W.abs().sum()


class TwoLayerKAN(nn.Module):
    """h_j = sum_i f_ij(x_i);  y = sum_j g_j(h_j) + b. One Kolmogorov-Arnold hidden layer."""

    def __init__(self, n_in, hidden=3, basis="cheby", degree=6, n_out=1):
        super().__init__()
        self.phi = Basis(basis, degree)
        self.n_in, self.hidden, self.n_out = n_in, hidden, n_out
        self.W1 = nn.Parameter(torch.randn(n_in, self.phi.n_basis, hidden) * 0.05)
        self.W2 = nn.Parameter(torch.randn(hidden, self.phi.n_basis, n_out) * 0.05)
        self.b = nn.Parameter(torch.zeros(n_out))
        self.register_buffer("mask", torch.ones(n_in))

    def forward(self, x):
        B = self.phi(x) * self.mask[None, :, None]
        h = torch.einsum("bic,ich->bh", B, self.W1)
        h = torch.tanh(h)                      # keeps the hidden activations inside the basis range
        return torch.einsum("bjc,jco->bo", self.phi(h), self.W2) + self.b

    def edge_norms(self):
        return self.W1.detach().abs().sum(dim=(1, 2)) * self.mask.detach()

    def l1(self):
        return self.W1.abs().sum() + self.W2.abs().sum()


# ---------------------------------------------------------------------------------- training
def fit(model, X, y, steps=None, lr=5e-3, lamb=1e-4, batch=256, seed=42, prune_frac=0.0,
        refit_steps=None, epochs=180, refit_epochs=72):
    """Adam with an L1 penalty on the edge coefficients, then optional hard pruning and a refit.

    `prune_frac` removes the inputs whose entire coefficient block contributes least, as a
    fraction of the total edge norm; it is what keeps the printed equation short.

    The budget is set in EPOCHS, not in mini-batch steps. A fixed step count gives a large dataset
    proportionally less training: 800 steps at batch 256 is 178 epochs of the 1146-row tahini
    training partition but only 20 epochs of the 10,243-row mango one, and the mango equation was
    correspondingly undertrained (external R2 0.27 at 800 steps against 0.80 at 12,000). Passing
    `steps` explicitly still overrides, for the cases where a fixed count is what is wanted.
    """
    n_batches = max(1, int(np.ceil(len(X) / batch)))
    if steps is None:
        steps = epochs * n_batches
    if refit_steps is None:
        refit_steps = refit_epochs * n_batches
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)

    def _train(n_steps):
        for _ in range(n_steps):
            idx = torch.randint(0, len(Xt), (min(batch, len(Xt)),), generator=g)
            opt.zero_grad()
            (lossf(model(Xt[idx]), yt[idx]) + lamb * model.l1()).backward()
            opt.step()

    _train(steps)
    if prune_frac > 0:
        e = model.edge_norms()
        order = torch.argsort(e)
        cum = torch.cumsum(e[order], 0) / (e.sum() + 1e-12)
        drop = order[cum < prune_frac]
        if len(drop) < model.n_in:                 # never prune every input away
            with torch.no_grad():
                model.mask[drop] = 0.0
            _train(refit_steps)
    return model


def predict(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


# ------------------------------------------------------------------------- exact closed form
def equation(model, var_names=None, round_to=3, drop_below=1e-3):
    """The trained network as an exact sympy expression in the scaled inputs.

    Exact means exact: no function is substituted for another, so evaluating this expression
    reproduces `predict` to floating-point precision. `drop_below` removes coefficients that are
    negligible relative to the largest one; that IS an approximation and the caller is expected
    to report the resulting fidelity rather than assume it.
    """
    names = var_names or [f"x_{i+1}" for i in range(model.n_in)]
    xs = [sympy.Symbol(n) for n in names]
    W1 = (model.W if isinstance(model, AdditiveKAN) else model.W1).detach().numpy()
    mask = model.mask.detach().numpy()
    scale = np.abs(W1).max() + 1e-12

    def sig(c, n):
        """Round to n significant digits, so a small coefficient is shortened, not erased."""
        c = float(c)
        if c == 0.0 or not np.isfinite(c):
            return 0.0
        return float(f"%.{n}g" % c)

    def edge(sym_feats, w):
        # Coefficients are rounded here, at construction. Rounding the finished expression with
        # `sympy.N` instead is what makes a nested two-layer Gaussian expression hang.
        terms = [sympy.Float(sig(c, round_to)) * f for c, f in zip(w, sym_feats)
                 if abs(float(c)) > drop_below * scale]
        return sympy.Add(*terms) if terms else sympy.Integer(0)

    if isinstance(model, AdditiveKAN):
        # rounded like every other coefficient: an unrounded bias prints all 15 digits
        # of its float next to coefficients given to three, which looks like an error
        # and is one.
        expr = sympy.Float(sig(model.b.detach().numpy()[0], round_to))
        for i, s in enumerate(xs):
            if mask[i] == 0:
                continue
            expr = expr + edge(model.phi.sym(s), W1[i, :, 0])
        parts = []
    else:
        # A two-layer network is printed as a small SYSTEM -- one line per hidden unit, then the
        # output line in terms of those units -- rather than as one substituted expression.
        # Substituting the hidden expressions in makes a Gaussian or Fourier basis explode
        # combinatorially (minutes of sympy time for an unreadable result); the system form is
        # the same model, is what the KAN 2.0 literature prints, and stays readable.
        parts, hsyms = [], []
        for j in range(model.hidden):
            h = sympy.Integer(0)
            for i, s in enumerate(xs):
                if mask[i] == 0:
                    continue
                h = h + edge(model.phi.sym(s), W1[i, :, j])
            hj = sympy.Symbol(f"h_{j+1}")
            parts.append((hj, sympy.tanh(h)))
            hsyms.append(hj)
        W2 = model.W2.detach().numpy()
        # rounded like every other coefficient: an unrounded bias prints all 15 digits
        # of its float next to coefficients given to three, which looks like an error
        # and is one.
        expr = sympy.Float(sig(model.b.detach().numpy()[0], round_to))
        for j, hj in enumerate(hsyms):
            expr = expr + edge(model.phi.sym(hj), W2[j, :, 0])
    # No `nsimplify` and no unconditional `expand`: on a nested two-layer expression with float
    # coefficients both are combinatorial and will hang. Collecting an additive model's powers is
    # cheap and makes the printed polynomial readable; the nested form is left as written.
    if isinstance(model, AdditiveKAN) and model.phi.kind in ("cheby", "poly"):
        expr = sympy.expand(expr)
        # `expand` folds every edge's degree-zero term into the intercept, so the intercept is a
        # SUM of rounded coefficients and prints with more digits than any of them. Round it once
        # more, after the fold. Only the constant is touched, and only in this cheap branch.
        const, rest = expr.as_coeff_Add()
        expr = sympy.Float(sig(const, round_to)) + rest
    return (expr, parts)


def evaluate(printed, X, var_names=None):
    """Evaluate a printed equation (or equation system) numerically, exactly as written.

    This is how fidelity is measured: the expression that will appear in the paper is compiled
    and run on the held-out rows, so what is checked is the printed object, not the network.
    """
    expr, parts = printed
    names = var_names or [f"x_{i+1}" for i in range(X.shape[1])]
    xs = [sympy.Symbol(n) for n in names]
    cols = [np.asarray(X[:, i], dtype=float) for i in range(X.shape[1])]
    env, syms, vals = {}, list(xs), list(cols)
    for hj, rhs in parts:
        f = sympy.lambdify(syms, rhs, "numpy")
        v = np.broadcast_to(np.asarray(f(*vals), dtype=float), (len(X),))
        env[hj] = v
        syms, vals = syms + [hj], vals + [v]
    f = sympy.lambdify(syms, expr, "numpy")
    return np.broadcast_to(np.asarray(f(*vals), dtype=float), (len(X),))


def as_text(printed):
    expr, parts = printed
    lines = [f"{h} = {rhs}" for h, rhs in parts]
    lines.append(f"y = {expr}")
    return "\n".join(lines)


def n_terms(printed):
    """Additive terms a reader has to read: the output line plus every hidden line."""
    expr, parts = printed
    total = 0
    for e in [expr] + [rhs for _, rhs in parts]:
        e = sympy.sympify(e)
        total += len(e.args) if e.is_Add else 1
    return total


def retained(model):
    return [i + 1 for i, m in enumerate(model.mask.detach().numpy()) if m > 0]
