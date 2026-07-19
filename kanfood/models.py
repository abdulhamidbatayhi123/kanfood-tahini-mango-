import numpy as np
import torch
import torch.nn as nn
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from kan import KAN
from kanfood.metrics import normalize_to_100

MODEL_NAMES = ["PLS", "SVM", "RF", "MLP", "CNN", "KAN"]
# Models that consume the FULL spectrum directly; all others consume PLS scores (see validate.py).
SPECTRUM_MODELS = {"PLS", "CNN"}


def front_end_components(name, params, default):
    """PLS-score front-end size for a model.

    Spectrum models (PLS, CNN) consume the full preprocessed spectrum, so they have no PLS
    compression front-end (-> None); any `n_components` in their params is the model's own.
    Score models (SVM/RF/MLP/KAN) are fed PLS latent scores; the compression dimension is a
    per-model tunable -- a tuned `n_components` if present, else the shared `default`."""
    if name in SPECTRUM_MODELS:
        return None
    return params.get("n_components", default)


class _Base:
    # Predictions are renormalized to a composition summing to 100 only for multi-target data
    # (e.g. tahini's 3 fractions). Single-target regression (e.g. mango dry-matter %) must NOT be
    # renormalized -- that would collapse every prediction to the constant 100. Set at fit time.
    _norm = True

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def _post(self, p):
        p = np.asarray(p)
        p = p.reshape(len(p), -1)            # guarantee 2-D (n, n_targets)
        return normalize_to_100(p) if self._norm else p

    def n_params(self):
        return None


class PLSModel(_Base):
    def __init__(self, n_components=12, **_):
        self.n_components = n_components

    def fit(self, X, y):
        self._norm = y.shape[1] > 1
        self.m = PLSRegression(n_components=min(self.n_components, X.shape[1], X.shape[0]))
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self._post(self.m.predict(X))

    def n_params(self):
        return int(self.m.n_components * (self.m.x_loadings_.shape[0] + self.m.y_loadings_.shape[0]))


class SVMModel(_Base):
    def __init__(self, C=100.0, gamma="scale", **_):
        self.C = C
        self.gamma = gamma

    def fit(self, X, y):
        self._norm = y.shape[1] > 1
        self.sx, self.sy = StandardScaler(), StandardScaler()
        Xs = self.sx.fit_transform(X)
        ys = self.sy.fit_transform(y)
        self.m = MultiOutputRegressor(SVR(kernel="rbf", C=self.C, gamma=self.gamma)).fit(Xs, ys)
        return self

    def predict(self, X):
        p = self.sy.inverse_transform(self.m.predict(self.sx.transform(X)))
        return self._post(p)


class RFModel(_Base):
    def __init__(self, n_estimators=300, max_depth=None, seed=42, **_):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, X, y):
        self._norm = y.shape[1] > 1
        yt = y if self._norm else y.ravel()          # avoid sklearn column-vector warning for 1 target
        self.m = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                       random_state=self.seed, n_jobs=-1).fit(X, yt)
        return self

    def predict(self, X):
        return self._post(self.m.predict(X))


class _Torch(_Base):
    def __init__(self, seed=42, fast=False):
        self.seed = seed
        self.fast = fast

    def _prep(self, X, y):
        torch.manual_seed(self.seed)
        self._norm = y.shape[1] > 1
        self.sx, self.sy = StandardScaler(), StandardScaler()
        Xs = self.sx.fit_transform(X)
        ys = self.sy.fit_transform(y)
        return (torch.tensor(Xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32))

    def _train(self, net, Xt, yt, epochs, batch, lr, wd=0.0):
        """Mini-batch Adam training (proper convergence, reproducible)."""
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        lossf = nn.MSELoss()
        n = Xt.shape[0]
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(epochs):
            net.train()
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch):
                b = perm[i:i + batch]
                if b.numel() < 2:        # BatchNorm needs >1 sample
                    continue
                opt.zero_grad()
                loss = lossf(net(Xt[b]), yt[b])
                loss.backward()
                opt.step()

    def _predict_scaled(self, net, X):
        net.eval()
        with torch.no_grad():
            out = net(torch.tensor(self.sx.transform(X), dtype=torch.float32)).numpy()
        return self._post(self.sy.inverse_transform(out))


class MLPModel(_Torch):
    def __init__(self, hidden=(128, 64), lr=1e-3, wd=1e-4, epochs=200, batch=64, seed=42, fast=False, **_):
        super().__init__(seed, fast)
        self.hidden = hidden
        self.lr = lr
        self.wd = wd
        self.epochs = 3 if fast else epochs
        self.batch = batch

    def fit(self, X, y):
        Xt, yt = self._prep(X, y)
        layers, prev = [], X.shape[1]
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, y.shape[1]))
        self.net = nn.Sequential(*layers)
        self._train(self.net, Xt, yt, self.epochs, self.batch, self.lr, self.wd)
        return self

    def predict(self, X):
        return self._predict_scaled(self.net, X)

    def n_params(self):
        return int(sum(p.numel() for p in self.net.parameters()))


class CNN1DModel(_Torch):
    """1-D CNN on the full spectrum (the standard end-to-end deep baseline), trained with mini-batches."""

    def __init__(self, channels=(16, 32), lr=1e-3, epochs=150, batch=64, seed=42, fast=False, **_):
        super().__init__(seed, fast)
        self.channels = channels
        self.lr = lr
        self.epochs = 3 if fast else epochs
        self.batch = batch

    def fit(self, X, y):
        Xt, yt = self._prep(X, y)
        c1, c2 = self.channels

        class Net(nn.Module):
            def __init__(self, n_out):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(1, c1, 7, stride=2, padding=3), nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(c1, c2, 5, stride=2, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                    nn.AdaptiveAvgPool1d(8))
                self.head = nn.Sequential(nn.Flatten(), nn.Linear(c2 * 8, 64), nn.ReLU(),
                                          nn.Linear(64, n_out))

            def forward(self, x):
                return self.head(self.conv(x.unsqueeze(1)))

        self.net = Net(y.shape[1])
        self._train(self.net, Xt, yt, self.epochs, self.batch, self.lr)
        return self

    def predict(self, X):
        return self._predict_scaled(self.net, X)

    def n_params(self):
        return int(sum(p.numel() for p in self.net.parameters()))


class KANModel(_Torch):
    """KAN on decorrelated PLS-score inputs, MinMax-scaled into the [-1,1] B-spline grid.
    Root-caused fix (Phase 1.5) for KAN instability: collinear inputs + grid extrapolation."""

    def __init__(self, width_hidden=(10, 5), grid=5, lamb=0.001, steps=300, seed=42, fast=False, **_):
        super().__init__(seed, fast)
        self.width_hidden = width_hidden
        self.grid = grid
        self.lamb = lamb
        self.steps = 5 if fast else steps

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self._norm = y.shape[1] > 1
        self.sx = MinMaxScaler(feature_range=(-1, 1))
        self.sy = StandardScaler()
        Xs = self.sx.fit_transform(X)
        ys = self.sy.fit_transform(y)
        ds = {"train_input": torch.tensor(Xs, dtype=torch.float32),
              "train_label": torch.tensor(ys, dtype=torch.float32),
              "test_input": torch.tensor(Xs, dtype=torch.float32),
              "test_label": torch.tensor(ys, dtype=torch.float32)}
        width = [X.shape[1]] + list(self.width_hidden) + [y.shape[1]]
        self.model = KAN(width=width, grid=self.grid, k=3, seed=self.seed,
                         device="cpu", auto_save=False, grid_range=[-1, 1])
        self.model.fit(ds, opt="Adam", steps=self.steps, lr=0.005, batch=128, lamb=self.lamb)
        return self

    def predict(self, X):
        with torch.no_grad():
            out = self.model(torch.tensor(self.sx.transform(X), dtype=torch.float32)).numpy()
        return self._post(self.sy.inverse_transform(out))

    def n_params(self):
        return int(sum(p.numel() for p in self.model.parameters()))


def build_model(name, input_dim, n_targets, seed=42, fast=False, **params):
    if name == "PLS":
        return PLSModel(**params)
    if name == "SVM":
        return SVMModel(**params)
    if name == "RF":
        return RFModel(seed=seed, **params)
    if name == "MLP":
        return MLPModel(seed=seed, fast=fast, **params)
    if name == "CNN":
        return CNN1DModel(seed=seed, fast=fast, **params)
    if name == "KAN":
        return KANModel(seed=seed, fast=fast, **params)
    raise ValueError(name)
