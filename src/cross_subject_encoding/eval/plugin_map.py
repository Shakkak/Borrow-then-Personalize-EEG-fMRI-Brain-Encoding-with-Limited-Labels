"""Helpers: the person-specific map on top of any borrowed prediction, and H4b stacked with
own ridge. A base fitted on the person's k labelled images (Gu, converter, own ridge) predicts those same images
too well, so a map fitted on its in-sample predictions would learn the overfit; the map is fitted on the base's
out-of-fold predictions on the k images instead, and applied to the base's test prediction from all k."""
from __future__ import annotations

from typing import Callable

import numpy as np


def folds(k: int, n_folds: int, seed: int) -> list[np.ndarray]:
    return np.array_split(np.random.default_rng(seed).permutation(k), n_folds)


def out_of_fold(fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray], k: int, n_folds: int = 5,
                seed: int = 13) -> np.ndarray:
    """fit_predict(train_rows, held_rows) -> predictions for held_rows (positions within the k labelled images),
    fitted on train_rows only. Returns (k, ...) predictions, each row from a model that never saw it."""
    out = None
    for held in folds(k, n_folds, seed):
        keep = np.setdiff1d(np.arange(k), held)
        pred = np.asarray(fit_predict(keep, held))
        if out is None:
            out = np.empty((k,) + pred.shape[1:], dtype=np.float64)
        out[held] = pred
    return out


def stack_fit(parts: list[np.ndarray], y: np.ndarray, window_index: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Least-squares weights combining several (k, channels, times) predictions into y, one weight per part for
    each channel and window (all time points of the window pooled), on image-centred data.
    Returns (channels, windows, parts)."""
    k, ch, _ = y.shape
    wins = np.unique(window_index)
    yc = y - y.mean(0)
    pc = [p - p.mean(0) for p in parts]
    w = np.zeros((ch, len(wins), len(parts)))
    for c in range(ch):
        for j, win in enumerate(wins):
            idx = window_index == win
            x = np.stack([p[:, c, idx].ravel() for p in pc], 1)
            w[c, j] = np.linalg.solve(x.T @ x + ridge * np.eye(len(parts)), x.T @ yc[:, c, idx].ravel())
    return w


def stack_apply(weights: np.ndarray, parts: list[np.ndarray], window_index: np.ndarray) -> np.ndarray:
    """Combine test predictions (n, channels, times) with `stack_fit` weights."""
    out = np.zeros_like(parts[0], dtype=np.float64)
    for j, win in enumerate(np.unique(window_index)):
        idx = window_index == win
        for i, p in enumerate(parts):
            out[:, :, idx] += weights[:, j, i][None, :, None] * p[:, :, idx]
    return out
