"""Building blocks of the NSD confirmation (P-7). Each is the fMRI form of a piece already used on EEG:
common vertices (EEG: channels by name), a per-output stack applied as fitted with the person's labelled mean as
intercept (EEG: per channel and window), identification in fixed galleries (EEG: the 200 test images), and the
family's one-sided Bonferroni bounds (P-6)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def common_vertices(masks: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """masks: per person, a boolean fsaverage mask selecting that person's challenge vertices (in fsaverage order).
    Returns the fsaverage indices present in every mask and, per person, the columns of their challenge array that hold
    those vertices, in the same order."""
    common = np.where(np.logical_and.reduce([np.asarray(m, bool) for m in masks]))[0]
    cols = [np.searchsorted(np.where(np.asarray(m, bool))[0], common) for m in masks]
    return common, cols


def stack_fit(parts_oof: list[np.ndarray], y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """One least-squares weight per part for every output, on image-centred out-of-fold predictions (k, outputs).
    Returns (parts, outputs)."""
    p = np.stack([a - a.mean(0) for a in parts_oof])  # (parts, k, outputs)
    yc = y - y.mean(0)
    g = np.einsum("ikv,jkv->vij", p, p) + ridge * np.eye(len(parts_oof))[None]
    b = np.einsum("ikv,kv->vi", p, yc)
    return np.linalg.solve(g, b[..., None])[..., 0].T


def stack_apply(w: np.ndarray, parts_oof: list[np.ndarray], parts_test: list[np.ndarray], y_mean: np.ndarray) -> np.ndarray:
    """Applied as fitted: each part centred on its out-of-fold mean over the k images, weighted, plus the
    person's mean response over the k images."""
    out = np.zeros_like(parts_test[0], dtype=np.float64)
    for wi, oof, test in zip(w, parts_oof, parts_test):
        out += wi * (test - oof.mean(0))
    return out + y_mean


def with_labelled_mean(pred_test: np.ndarray, pred_k: np.ndarray, y_k: np.ndarray) -> np.ndarray:
    """The prediction's own mean over the labelled images replaced by the person's mean response over them."""
    return pred_test - pred_k.mean(0) + y_k.mean(0)


def galleries(n_images: int, n_galleries: int = 4, size: int = 200, seed: int = 346) -> list[np.ndarray]:
    order = np.random.default_rng(seed).permutation(n_images)
    if n_galleries * size > n_images:
        raise ValueError("galleries do not fit")
    return [np.sort(order[g * size:(g + 1) * size]) for g in range(n_galleries)]


def identify_top1(pred: np.ndarray, meas: np.ndarray, centred: bool = False) -> float:
    """Percent of images whose measured response correlates best with their own prediction (Pearson over outputs)."""
    p, m = np.asarray(pred, np.float64), np.asarray(meas, np.float64)
    if centred:
        p, m = p - p.mean(0), m - m.mean(0)
    p = (p - p.mean(1, keepdims=True)) / (p.std(1, keepdims=True) + 1e-30)
    m = (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-30)
    r = m @ p.T / p.shape[1]
    return float((r.argmax(1) == np.arange(len(m))).mean() * 100)


def encoding_r(pred: np.ndarray, meas: np.ndarray) -> float:
    """Mean over outputs of the correlation across images."""
    p, m = pred - pred.mean(0), meas - meas.mean(0)
    r = (p * m).sum(0) / np.sqrt((p ** 2).sum(0) * (m ** 2).sum(0) + 1e-30)
    return float(np.mean(r))


def verdict(diff: np.ndarray, margin: float, n_claims: int = 3, alpha: float = 0.05) -> dict:
    """One-sided bounds at 1 - alpha / n_claims (P-6's rule): supported if the lower bound exceeds the margin,
    rejected if the upper bound is below it, otherwise inconclusive."""
    x = np.asarray(diff, float)
    se = x.std(ddof=1) / np.sqrt(len(x))
    t = stats.t.ppf(1 - alpha / n_claims, len(x) - 1)
    lo, hi = x.mean() - t * se, x.mean() + t * se
    v = "supported" if lo > margin else ("rejected" if hi < margin else "inconclusive")
    return {"mean": float(x.mean()), "lower": float(lo), "upper": float(hi), "margin": float(margin), "n": int(len(x)),
            "positive": int((x > 0).sum()), "verdict": v}
