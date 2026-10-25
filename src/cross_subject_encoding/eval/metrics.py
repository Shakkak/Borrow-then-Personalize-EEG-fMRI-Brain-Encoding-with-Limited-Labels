"""Scores for encoding models, the same on both tracks.

- `pearson_per_output`: correlation across test images between predicted
  and measured response, per output (channel x time point, or vertex).
- `noise_normalized`: r squared over the noise ceiling, per output, then
  averaged per region or 100 ms window. The Algonauts 2023 metric is this
  mean over all vertices. On EEG the ceiling is the reliability of
  the averaged test responses from notebook 01, so the score is r^2 over
  that reliability.
- `identification`: for each test image, correlate its measured response
  with the predicted response of every candidate image, and record whether
  the right one ranks first or in the top five. Chance top-1 is 1 / n.
"""
from __future__ import annotations

import numpy as np


def pearson_per_output(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """(n_images, n_outputs) arrays in, one correlation per output out; NaN for constant columns."""
    constant = (np.ptp(y_true, 0) == 0) | (np.ptp(y_pred, 0) == 0)
    t, p = y_true - y_true.mean(0), y_pred - y_pred.mean(0)
    den = np.sqrt((t ** 2).sum(0) * (p ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(~constant & (den > 0), (t * p).sum(0) / den, np.nan)


def noise_normalized(r: np.ndarray, ceiling_r2: np.ndarray, min_ceiling: float = 1e-3) -> np.ndarray:
    """r^2 / ceiling per output; outputs with a ceiling below `min_ceiling` are NaN rather than exploding."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ceiling_r2 > min_ceiling, np.sign(r) * r ** 2 / ceiling_r2, np.nan)


def window_means(values: np.ndarray, times_s: np.ndarray, window_ms: int) -> dict[int, float]:
    """Mean of a (channels, time) array over channels and over consecutive windows of `window_ms`."""
    edges = np.arange(times_s.min() * 1000, times_s.max() * 1000 + window_ms, window_ms)
    out = {}
    for lo in edges[:-1]:
        sel = (times_s * 1000 >= lo) & (times_s * 1000 < lo + window_ms)
        if sel.any():
            out[int(lo)] = float(np.nanmean(values[..., sel]))
    return out


def identification(y_true: np.ndarray, y_pred: np.ndarray, gallery_pred: np.ndarray | None = None) -> dict[str, float]:
    """Top-1 and top-5 accuracy of picking each image's prediction from its measured response.

    gallery_pred adds distractor predictions (extra rows) to the candidate set.
    """
    cands = y_pred if gallery_pred is None else np.concatenate([y_pred, gallery_pred])
    z = lambda a: (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-8)
    sim = z(y_true) @ z(cands).T / y_true.shape[1]  # (n_test, n_candidates)
    rank = (sim > sim[np.arange(len(y_true)), np.arange(len(y_true))][:, None]).sum(1)
    hub = np.bincount(sim.argmax(1), minlength=len(cands)) / len(y_true)  # share of queries each candidate wins
    return {"top1": float((rank == 0).mean()), "top5": float((rank < 5).mean()), "chance_top1": 1 / len(cands),
            "max_hub_share": float(hub.max())}


def time_curve(r: np.ndarray, n_channels: int) -> np.ndarray:
    """Per-output values (channels x time, flattened channel-major) to the channel mean at each time point."""
    return np.nanmean(r.reshape(n_channels, -1), 0)
