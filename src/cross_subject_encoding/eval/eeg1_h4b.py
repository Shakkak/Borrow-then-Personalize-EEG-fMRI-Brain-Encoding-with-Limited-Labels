"""H4b: align others' average to a new person with a per-window channel-mixing map fitted on k labelled images.

Label-free H4 is unidentifiable here (whitened single trials, per-output correlation scoring). H4b uses
the k labelled pairs instead: for each 100 ms window, the person's response minus others' average is
regressed on the average's 17 channels at the same time (ridge, shrinking toward no change), so the map is
I + B. About 17 x 17 x 10 numbers, against own ridge's 1,000 x 1,700. Penalty chosen by cross-validation
over images (time points of one image stay together).
"""
from __future__ import annotations

import numpy as np

ALPHAS = tuple(10.0 ** e for e in range(-2, 7))


def _ridge_cv(x: np.ndarray, y: np.ndarray, groups: np.ndarray, alphas=ALPHAS, folds: int = 5, seed: int = 13):
    """Shared-penalty ridge (no intercept; inputs centred) with group-wise CV; returns (coef, alpha)."""
    uniq = np.unique(groups)
    split = np.array_split(np.random.default_rng(seed).permutation(uniq), folds)
    err = np.zeros(len(alphas))
    for held in split:
        test = np.isin(groups, held)
        u, s, vt = np.linalg.svd(x[~test], full_matrices=False)
        uty = u.T @ y[~test]
        for i, a in enumerate(alphas):
            coef = vt.T @ ((s / (s ** 2 + a))[:, None] * uty)
            err[i] += ((x[test] @ coef - y[test]) ** 2).sum()
    best = alphas[int(np.argmin(err))]
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return vt.T @ ((s / (s ** 2 + best))[:, None] * (u.T @ y)), best


def fit_map(avg_k: np.ndarray, y_k: np.ndarray, window_index: np.ndarray, alphas=ALPHAS, folds: int = 5,
            seed: int = 13, shuffle: bool = False) -> tuple[dict, dict]:
    """Per-window correction B_w (channels x channels): y - avg ~ avg_centred @ B_w. shuffle=True pairs the
    person's responses with randomly permuted images (control: no real pairing). Returns (coefs, penalties)."""
    k, ch, _ = avg_k.shape
    a_k = avg_k - avg_k.mean(0)
    y_c = y_k - y_k.mean(0)
    if shuffle:
        y_c = y_c[np.random.default_rng(seed + 1000).permutation(k)]
    r_k = y_c - a_k
    coefs, chosen = {}, {}
    for w in np.unique(window_index):
        idx = np.where(window_index == w)[0]
        x = a_k[:, :, idx].transpose(0, 2, 1).reshape(-1, ch)
        r = r_k[:, :, idx].transpose(0, 2, 1).reshape(-1, ch)
        coefs[int(w)], chosen[int(w)] = _ridge_cv(x, r, np.repeat(np.arange(k), len(idx)), alphas, folds, seed)
        chosen[int(w)] = float(chosen[int(w)])
    return coefs, chosen


def apply_map(coefs: dict, avg_test: np.ndarray, window_index: np.ndarray) -> np.ndarray:
    n, ch, _ = avg_test.shape
    a_t = avg_test - avg_test.mean(0)
    out = avg_test.astype(np.float64).copy()
    for w, coef in coefs.items():
        idx = np.where(window_index == w)[0]
        xt = a_t[:, :, idx].transpose(0, 2, 1).reshape(-1, ch)
        out[:, :, idx] += (xt @ coef).reshape(n, len(idx), ch).transpose(0, 2, 1)
    return out


def mean_map(maps: list[dict]) -> dict:
    return {w: np.mean([m[w] for m in maps], 0) for w in maps[0]}


def channel_map_predict(avg_k: np.ndarray, y_k: np.ndarray, avg_test: np.ndarray, window_index: np.ndarray,
                        alphas=ALPHAS, folds: int = 5, seed: int = 13) -> tuple[np.ndarray, dict]:
    """avg_k, y_k: (k, channels, times) others' average and the person's responses on the k labelled images;
    avg_test: (n, channels, times). window_index: (times,) window id per time point.
    Returns the aligned test prediction (n, channels, times) and the chosen penalty per window."""
    coefs, chosen = fit_map(avg_k, y_k, window_index, alphas, folds, seed)
    return apply_map(coefs, avg_test, window_index), chosen
