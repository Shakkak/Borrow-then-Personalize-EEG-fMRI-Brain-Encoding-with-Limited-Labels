"""Closed-form ridge regression with a penalty chosen per output by cross-validation.

One SVD of each training fold gives the predictions for every candidate
alpha at once, so choosing a penalty for each of thousands of outputs
(channels x time points, or vertices) costs little more than one fit. The
penalty is picked on training images only, by k-fold cross-validation, and
the final model is refitted on all training images with each output's own
alpha.
"""
from __future__ import annotations

import numpy as np


def _svd_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Predictions (n_alphas, n_test, n_outputs) for every alpha, with centred data plus the training mean."""
    xm, ym = x_train.mean(0), y_train.mean(0)
    u, s, vt = np.linalg.svd(x_train - xm, full_matrices=False)
    uty = u.T @ (y_train - ym)
    xt = (x_test - xm) @ vt.T
    out = np.empty((len(alphas), len(x_test), y_train.shape[1]), dtype=np.float32)
    for i, a in enumerate(alphas):
        out[i] = xt @ ((s / (s ** 2 + a))[:, None] * uty) + ym
    return out


class RidgePerOutput:
    def __init__(self, alphas, n_folds: int = 5, seed: int = 13):
        self.alphas, self.n_folds, self.seed = np.asarray(alphas, float), n_folds, seed

    def fit(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> "RidgePerOutput":
        """`groups` (one label per row, e.g. the image of each trial) keeps a group's rows in one fold (14e)."""
        n = len(x)
        rng = np.random.default_rng(self.seed)
        if groups is None:
            folds = np.array_split(rng.permutation(n), min(self.n_folds, n))
        else:
            labels, inverse = np.unique(groups, return_inverse=True)
            parts = np.array_split(rng.permutation(len(labels)), min(self.n_folds, len(labels)))
            folds = [np.flatnonzero(np.isin(inverse, g)) for g in parts]
        err = np.zeros((len(self.alphas), y.shape[1]))
        for f in folds:
            tr = np.setdiff1d(np.arange(n), f)
            # one alpha at a time, as _svd_predict computes them (float32 predictions): the same errors without
            # holding every alpha's predictions at once (whole fMRI pools do not fit in memory otherwise)
            xtr, ytr = x[tr], y[tr]
            xm, ym = xtr.mean(0), ytr.mean(0)
            u, s, vt = np.linalg.svd(xtr - xm, full_matrices=False)
            uty = u.T @ (ytr - ym)
            xt = (x[f] - xm) @ vt.T
            del xtr, ytr, u
            for i, a in enumerate(self.alphas):
                pred = (xt @ ((s / (s ** 2 + a))[:, None] * uty) + ym).astype(np.float32)
                err[i] += ((pred - y[f]) ** 2).sum(0)
        self.best_alpha_ = self.alphas[err.argmin(0)]
        # cross-validated R^2 per output at its own alpha, from training images only: used to choose
        # between feature sets without looking at the test images
        ss = ((y - y.mean(0)) ** 2).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.cv_r2_ = np.where(ss > 0, 1 - err.min(0) / ss, np.nan)
        self.alpha_at_edge_ = float(np.mean((self.best_alpha_ == self.alphas.min()) | (self.best_alpha_ == self.alphas.max())))
        xm, ym = x.mean(0), y.mean(0)
        u, s, vt = np.linalg.svd(x - xm, full_matrices=False)
        uty = u.T @ (y - ym)
        coef = np.empty((x.shape[1], y.shape[1]), dtype=np.float32)
        for a in np.unique(self.best_alpha_):
            cols = self.best_alpha_ == a
            coef[:, cols] = vt.T @ ((s / (s ** 2 + a))[:, None] * uty[:, cols])
        self.coef_, self.x_mean_, self.y_mean_ = coef, xm, ym
        return self

    def refit(self, x: np.ndarray, y: np.ndarray) -> "RidgePerOutput":
        """A copy fitted on new data with this model's per-output penalties, without cross-validating again."""
        new = RidgePerOutput(self.alphas, self.n_folds, self.seed)
        new.best_alpha_ = self.best_alpha_
        xm, ym = x.mean(0), y.mean(0)
        u, s, vt = np.linalg.svd(x - xm, full_matrices=False)
        uty = u.T @ (y - ym)
        coef = np.empty((x.shape[1], y.shape[1]), dtype=np.float32)
        for a in np.unique(self.best_alpha_):
            cols = self.best_alpha_ == a
            coef[:, cols] = vt.T @ ((s / (s ** 2 + a))[:, None] * uty[:, cols])
        new.coef_, new.x_mean_, new.y_mean_ = coef, xm, ym
        return new

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean_) @ self.coef_ + self.y_mean_
