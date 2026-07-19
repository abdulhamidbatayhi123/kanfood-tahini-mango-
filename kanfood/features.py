import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.cross_decomposition import PLSRegression


class MIFeatureSelector:
    """Select top-k wavenumbers by mutual information with a single target. Fit on TRAIN only.
    Retained for interpretability / wavenumber-importance analysis (not the KAN front-end)."""

    def __init__(self, n_features: int = 20, seed: int = 42, n_neighbors: int = 5):
        self.n_features = n_features
        self.seed = seed
        self.n_neighbors = n_neighbors
        self.indices_ = None
        self.scores_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MIFeatureSelector":
        scores = mutual_info_regression(X, y, random_state=self.seed, n_neighbors=self.n_neighbors)
        self.scores_ = scores
        k = min(self.n_features, X.shape[1])
        self.indices_ = np.sort(np.argsort(scores)[-k:])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.indices_]

    def fit_transform(self, X, y) -> np.ndarray:
        return self.fit(X, y).transform(X)


class PLSFeatures:
    """Supervised, decorrelated compression of a spectrum to PLS latent scores. Fit on TRAIN only.

    This is the KAN front-end: the 20 MI wavenumbers are highly collinear (one narrow band), which
    makes the spline optimisation ill-conditioned and KAN unstable across seeds. PLS scores are
    decorrelated and supervised, giving a single, stable, accurate KAN -- while remaining interpretable
    (each component's loadings map back to chemical wavenumbers)."""

    def __init__(self, n_components: int = 12):
        self.n_components = n_components
        self.pls = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSFeatures":
        n = min(self.n_components, X.shape[1], X.shape[0])
        self.pls = PLSRegression(n_components=n)
        self.pls.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pls.transform(X)

    def fit_transform(self, X, y) -> np.ndarray:
        return self.fit(X, y).transform(X)
