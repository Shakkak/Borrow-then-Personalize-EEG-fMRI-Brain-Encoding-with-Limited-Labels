"""Fit the personal channel map with a prior learned from other people's maps.

H4b's ridge pulls each window's correction B (channels x channels) towards zero, i.e. the map towards no
change. Here the pull goes towards the donors' mean map m, along the donors' covariance S of vec(B):

    minimise ||R - X B||^2 + alpha (b - m)' P (b - m),  b = vec(B) (column-stacked),  P = inverse of
    Sigma = (1 - gamma) S / (tr S / d) + gamma I,  d = channels^2.

Writing b = m + L u with Sigma = L L' turns this into ordinary ridge in u, solved from 17 x 17 Gram
matrices only, so cross-validation over (alpha, gamma) is cheap. gamma = 1 with m = 0 is exactly H4b
(same folds, same penalty grid); gamma = 1 with the donors' mean is a mean-map prior.
"""
from __future__ import annotations

import numpy as np

from cross_subject_encoding.eval.eeg1_h4b import ALPHAS

GAMMAS = (0.05, 0.2, 0.5, 1.0)


def prior_from_maps(maps: list[dict]) -> dict:
    """Donor maps (each {window: B}) -> {window: (mean B, covariance of vec(B))}; vec is column-stacked."""
    out = {}
    for w in maps[0]:
        v = np.stack([m[w].reshape(-1, order="F") for m in maps])
        out[w] = (v.mean(0).reshape(maps[0][w].shape, order="F"), np.cov(v, rowvar=False) if len(v) > 1
                  else np.zeros((v.shape[1], v.shape[1])))
    return out


def _factor(cov: np.ndarray, gamma: float) -> np.ndarray:
    d = cov.shape[0]
    tr = np.trace(cov)
    sigma = gamma * np.eye(d) + ((1 - gamma) * cov * d / tr if tr > 0 else 0.0)
    e, v = np.linalg.eigh((sigma + sigma.T) / 2)
    return v * np.sqrt(np.clip(e, 0.0, None))


def _solve(lmat, g, c, m, alphas):
    """Ridge in u for each alpha; g = X'X, c = X'R (both ch x ch). Returns the list of vec(B)."""
    ch = g.shape[0]
    kg = np.kron(np.eye(ch), g)
    a = lmat.T @ kg @ lmat
    e, q = np.linalg.eigh((a + a.T) / 2)
    rhs = q.T @ (lmat.T @ (c.reshape(-1, order="F") - kg @ m))
    return [m + lmat @ (q @ (rhs / (e + al))) for al in alphas]


def fit_map_prior(avg_k: np.ndarray, y_k: np.ndarray, window_index: np.ndarray, prior: dict | None,
                  alphas=ALPHAS, gammas=GAMMAS, folds: int = 5, seed: int = 13) -> tuple[dict, dict]:
    """Like eeg1_h4b.fit_map, with the penalty centred on prior[w][0] and shaped by prior[w][1].
    prior=None means zero mean and identity covariance (H4b). Returns (coefs, {window: (alpha, gamma)})."""
    k, ch, _ = avg_k.shape
    a_k = avg_k - avg_k.mean(0)
    r_k = (y_k - y_k.mean(0)) - a_k
    uniq = np.arange(k)
    split = np.array_split(np.random.default_rng(seed).permutation(uniq), folds)
    coefs, chosen = {}, {}
    for w in np.unique(window_index):
        idx = np.where(window_index == w)[0]
        x = a_k[:, :, idx].transpose(0, 2, 1)  # (k, times, ch)
        r = r_k[:, :, idx].transpose(0, 2, 1)
        if prior is None:
            m, cov, grid = np.zeros(ch * ch), np.zeros((ch * ch, ch * ch)), (1.0,)
        else:
            m, cov, grid = prior[int(w)][0].reshape(-1, order="F"), prior[int(w)][1], gammas
        gx = np.einsum("ntc,ntd->ncd", x, x)  # per-image Gram pieces, summed per fold
        cx = np.einsum("ntc,ntd->ncd", x, r)
        g_all, c_all = gx.sum(0), cx.sum(0)
        err = np.zeros((len(grid), len(alphas)))
        factors = [_factor(cov, gm) for gm in grid]
        for held in split:
            g_te, c_te = gx[held].sum(0), cx[held].sum(0)
            g_tr, c_tr = g_all - g_te, c_all - c_te
            kg_te = np.kron(np.eye(ch), g_te)
            vc_te = c_te.reshape(-1, order="F")
            for i, lmat in enumerate(factors):
                for j, b in enumerate(_solve(lmat, g_tr, c_tr, m, alphas)):
                    err[i, j] += b @ kg_te @ b - 2 * b @ vc_te  # held-out squared error minus a constant
        i, j = np.unravel_index(int(np.argmin(err)), err.shape)
        b = _solve(factors[i], g_all, c_all, m, (alphas[j],))[0]
        coefs[int(w)] = b.reshape(ch, ch, order="F")
        chosen[int(w)] = (float(alphas[j]), float(grid[i]))
    return coefs, chosen


# A prior from the electrode layout instead of from donors. The 17 occipito-parietal channels on a flat
# 10-10 grid, in grid steps (x: left to right, y: back of the head negative). Approximate by design: the prior only
# needs which electrodes are neighbours and roughly how far apart.
LAYOUT_10_10 = {"p7": (-4, 0), "p5": (-3, 0), "p3": (-2, 0), "p1": (-1, 0), "pz": (0, 0), "p2": (1, 0), "p4": (2, 0),
                "p6": (3, 0), "p8": (4, 0), "po7": (-3, -1), "po3": (-1, -1), "poz": (0, -1), "po4": (1, -1),
                "po8": (3, -1), "o1": (-1, -2), "oz": (0, -2), "o2": (1, -2)}


def electrode_kernel(ch_names, sigma: float = 1.0) -> np.ndarray:
    """Gaussian kernel over the layout, channels matched by name (case-insensitive); unknown names raise KeyError."""
    xy = np.array([LAYOUT_10_10[str(c).strip().lower()] for c in ch_names], float)
    d2 = ((xy[:, None] - xy[None]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * sigma ** 2))


def graph_prior(ch_names, windows, sigma: float = 1.0) -> dict:
    """Zero-mean prior for fit_map_prior that makes the map's correction smooth over neighbouring electrodes, on both
    its input and output channels: cov(vec B) = K kron K (column-stacked vec), K = electrode_kernel."""
    k = electrode_kernel(ch_names, sigma)
    n = len(k)
    return {int(w): (np.zeros((n, n)), np.kron(k, k)) for w in np.unique(windows)}
