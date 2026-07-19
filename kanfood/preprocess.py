import numpy as np
from scipy.signal import savgol_filter


def snv(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd


class Preprocessor:
    """Stateful so MSC reference is learned on train and reused on test (no leakage)."""

    def __init__(self, method: str = "snv", window: int = 11, poly: int = 2):
        self.method = method
        self.window = window
        self.poly = poly
        self.reference_ = None

    def fit(self, X: np.ndarray) -> "Preprocessor":
        if "msc" in self.method:
            self.reference_ = X.mean(axis=0)
        return self

    def _msc(self, X):
        ref = self.reference_
        out = np.empty_like(X)
        for i in range(X.shape[0]):
            slope, intercept = np.polyfit(ref, X[i], 1)
            slope = slope if slope != 0 else 1.0
            out[i] = (X[i] - intercept) / slope
        return out

    def _sg(self, X, deriv):
        if X.shape[1] <= self.window:
            return X
        return savgol_filter(X, self.window, self.poly, deriv=deriv, axis=1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        m = self.method
        if m == "raw":
            return X
        if m == "snv":
            return snv(X)
        if m == "msc":
            return self._msc(X)
        if m == "sg1":
            return self._sg(X, 1)
        if m == "sg2":
            return self._sg(X, 2)
        if m == "snv+sg1":
            return self._sg(snv(X), 1)
        raise ValueError(f"unknown method: {m}")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
