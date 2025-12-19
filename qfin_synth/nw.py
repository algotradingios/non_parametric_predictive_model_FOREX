import numpy as np
from sklearn.neighbors import BallTree

def _gaussian_kernel(dist_over_h2: np.ndarray) -> np.ndarray:
    return np.exp(-dist_over_h2)

class NWKernelRegressor:
    def __init__(self, bandwidth: float = 0.2, leaf_size: int = 40):
        if bandwidth <= 0:
            raise ValueError("bandwidth must be > 0")
        self.bandwidth = float(bandwidth)
        self.leaf_size = int(leaf_size)
        self.tree: BallTree | None = None
        self.y: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[0] != len(y):
            raise ValueError("X must be [n,d] and aligned with y")

        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        self.scale_ = std
        Xn = X / self.scale_
        self.tree = BallTree(Xn, leaf_size=self.leaf_size, metric="euclidean")
        self.y = y
        return self

    def predict(self, X, k: int | None = None):
        if self.tree is None or self.y is None or self.scale_ is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        Xn = X / self.scale_
        h = self.bandwidth

        if k is None:
            rad = 3.0 * h
            inds, dists = self.tree.query_radius(Xn, r=rad, return_distance=True, sort_results=True)
            preds = np.zeros(Xn.shape[0], dtype=float)
            for i, (idx, dist) in enumerate(zip(inds, dists)):
                if len(idx) == 0:
                    _, ind1 = self.tree.query(Xn[i:i+1], k=1, return_distance=True)
                    preds[i] = float(self.y[ind1[0][0]])
                    continue
                w = _gaussian_kernel((dist ** 2) / (2 * h * h))
                preds[i] = float(np.dot(w, self.y[idx]) / (w.sum() + 1e-12))
            return preds
        else:
            k = int(k)
            dist, ind = self.tree.query(Xn, k=k, return_distance=True)
            w = _gaussian_kernel((dist ** 2) / (2 * h * h))
            return (w * self.y[ind]).sum(axis=1) / (w.sum(axis=1) + 1e-12)

class MuSigmaNonParam:
    def __init__(self, h_mu=0.2, h_sigma=0.2):
        self.mu_est = NWKernelRegressor(bandwidth=h_mu)
        self.sg_est = NWKernelRegressor(bandwidth=h_sigma)

    def fit(self, X_state, r_next):
        X = np.asarray(X_state, dtype=float)
        y = np.asarray(r_next, dtype=float).reshape(-1)
        self.mu_est.fit(X, y)
        mu_hat = self.mu_est.predict(X)
        resid2 = (y - mu_hat) ** 2
        self.sg_est.fit(X, np.sqrt(np.maximum(resid2, 1e-12)))
        return self

    def predict_mu(self, X_state):
        return self.mu_est.predict(X_state)

    def predict_sigma(self, X_state):
        return np.maximum(self.sg_est.predict(X_state), 1e-8)
