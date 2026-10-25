"""Alignment controls: functional-alignment maps fitted on the same k labelled images as H4b.

Procrustes (hyperalignment-style): an orthogonal channel map per window, aligning others' average to the person.
SRM-style: a shared d-dimensional response learned from the other people (one orthonormal map each, Chen et al. 2015),
and the new person's orthonormal map fitted on the k labelled images. All on image-centred data, as H4b; arrays are
(images, channels, times) and `window_index` gives each time point's window."""
from __future__ import annotations

import numpy as np


def _rows(a: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(images, channels, times) restricted to the window's time points -> (images * times, channels)."""
    return a[:, :, idx].transpose(0, 2, 1).reshape(-1, a.shape[1])


def procrustes(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Orthogonal R minimizing ||a R - y|| (Schoenemann): R = U V' from the SVD of a'y. a, y: (samples, channels)."""
    u, _, vt = np.linalg.svd(a.T @ y)
    return u @ vt


def fit_procrustes_map(avg_k: np.ndarray, y_k: np.ndarray, window_index: np.ndarray) -> dict:
    a, y = avg_k - avg_k.mean(0), y_k - y_k.mean(0)
    return {w: procrustes(_rows(a, window_index == w), _rows(y, window_index == w)) for w in np.unique(window_index)}


def apply_rotation(maps: dict, avg_test: np.ndarray, window_index: np.ndarray) -> np.ndarray:
    """Replace others' average by its rotated version (a pure alignment, no residual to the average)."""
    a = avg_test - avg_test.mean(0)
    out = np.empty_like(a, dtype=np.float64)
    for w, r in maps.items():
        idx = window_index == w
        n, _, t = a[:, :, idx].shape
        out[:, :, idx] = (_rows(a, idx) @ r).reshape(n, t, -1).transpose(0, 2, 1)
    return out


def srm(xs: list[np.ndarray], d: int, iters: int = 10, seed: int = 0) -> tuple[list[np.ndarray], np.ndarray]:
    """Deterministic SRM: X_i ~ S W_i' with W_i (channels, d) orthonormal columns, S (samples, d) shared.
    Alternates S = mean_i X_i W_i and W_i = Procrustes-type update (U V' of X_i' S). Returns (Ws, S)."""
    rng = np.random.default_rng(seed)
    ch = xs[0].shape[1]
    ws = [np.linalg.qr(rng.standard_normal((ch, d)))[0] for _ in xs]
    for _ in range(iters):
        s = np.mean([x @ w for x, w in zip(xs, ws)], 0)
        ws = []
        for x in xs:
            u, _, vt = np.linalg.svd(x.T @ s, full_matrices=False)
            ws.append(u @ vt)
    s = np.mean([x @ w for x, w in zip(xs, ws)], 0)
    return ws, s


def srm_fit(fit_rows: list[np.ndarray], window_index: np.ndarray, d: int, iters: int = 10, seed: int = 0) -> dict:
    """The other people's orthonormal maps per window, from their (images, channels, times) arrays."""
    centre = lambda a: a - a.mean(0)
    return {w: srm([_rows(centre(f), window_index == w) for f in fit_rows], d, iters, seed)[0] for w in np.unique(window_index)}


def srm_apply(maps: dict, src_k: list[np.ndarray], src_test: list[np.ndarray], y_k: np.ndarray,
              window_index: np.ndarray) -> np.ndarray:
    """New person's prediction from the others' maps: shared response on the k and test images (mean of the
    others' projections), the person's orthonormal map from the k images (U V' of Y' S), prediction S_test W'."""
    y = y_k - y_k.mean(0)
    n_test, ch, _ = src_test[0].shape
    out = np.empty((n_test, ch, y.shape[2]))
    centre = lambda a: a - a.mean(0)
    for w, ws in maps.items():
        idx = window_index == w
        s_k = np.mean([_rows(centre(x), idx) @ wq for x, wq in zip(src_k, ws)], 0)
        s_t = np.mean([_rows(centre(x), idx) @ wq for x, wq in zip(src_test, ws)], 0)
        u, _, vt = np.linalg.svd(_rows(y, idx).T @ s_k, full_matrices=False)
        out[:, :, idx] = (s_t @ (u @ vt).T).reshape(n_test, int(idx.sum()), ch).transpose(0, 2, 1)
    return out


def srm_predict(src_k: list[np.ndarray], src_test: list[np.ndarray], y_k: np.ndarray, window_index: np.ndarray,
                d: int, fit_rows: list[np.ndarray] | None = None, iters: int = 10, seed: int = 0) -> np.ndarray:
    """`srm_fit` on fit_rows (default: src_k) followed by `srm_apply`."""
    maps = srm_fit(src_k if fit_rows is None else fit_rows, window_index, d, iters, seed)
    return srm_apply(maps, src_k, src_test, y_k, window_index)
