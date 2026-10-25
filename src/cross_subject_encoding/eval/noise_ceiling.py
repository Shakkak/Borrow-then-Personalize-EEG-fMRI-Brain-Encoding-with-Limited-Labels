"""Split-half noise ceiling for the EEG test set.

The 80 test repetitions are split in half at random many times. For each
split, the two half-averages are correlated across the 200 test images, at
every channel and time point, and Spearman-Brown predicts the reliability of
an average over more repetitions:

    r_n = n * r_1 / (1 + (n - 1) * r_1),   with r_1 the single-trial reliability.

`reliability` is the predicted correlation between two independent n-trial
averages. The highest correlation a model can reach with an n-trial average
is its square root, returned as `max_r`, which is the ceiling that scores are
divided by. Both are given for 80 repetitions (the full test
average) and 4 (what a new subject's single session would give).
"""
from __future__ import annotations

import numpy as np


def _corr_over_images(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation across axis 0 (images) at every remaining index."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(0) / den, np.nan)


def split_half(test: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Mean split-half correlation, shape (channels, time). test: (images, reps, channels, time)."""
    rng = np.random.default_rng(seed)
    n_rep = test.shape[1]
    acc = np.zeros(test.shape[2:])
    for _ in range(n_splits):
        perm = rng.permutation(n_rep)
        a = test[:, perm[: n_rep // 2]].mean(1)
        b = test[:, perm[n_rep // 2:]].mean(1)
        acc += _corr_over_images(a, b)
    return acc / n_splits


def spearman_brown(r: np.ndarray, factor: float) -> np.ndarray:
    """Reliability of an average over `factor` times as many trials as the one r was measured on."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return factor * r / (1 + (factor - 1) * r)


def noise_ceiling(test: np.ndarray, n_splits: int, seed: int, reps_scored=(80, 4)) -> dict[str, np.ndarray]:
    """Reliability and max achievable correlation per channel and time, for each number of repetitions."""
    half = test.shape[1] // 2
    r_half = split_half(test, n_splits, seed)
    r_single = spearman_brown(r_half, 1 / half)  # step down from half-size averages to one trial
    # A negative reliability means no reliable signal. Left negative, Spearman-Brown can step it up past 1
    # (r_n = n r / (1 + (n - 1) r) with 1 + (n - 1) r < 0), so it is set to zero, the usual convention.
    r_single = np.clip(r_single, 0, None)
    out = {"r_split_half": r_half, "r_single_trial": r_single}
    for n in reps_scored:
        rel = spearman_brown(r_single, n)
        out[f"reliability_{n}"] = rel
        out[f"max_r_{n}"] = np.sqrt(np.clip(rel, 0, 1))
    return out


def noise_floor(max_r: np.ndarray, times_s: np.ndarray, q: float = 0.95) -> np.ndarray:
    """Per subject, the q-quantile of max_r over channels and time points before image onset.

    max_r: (subjects, channels, time). Clipping negative reliabilities to zero and then taking a square root
    makes max_r positive on pure noise: about 0.127 on average for 200 images and 80 repetitions, and
    notebook 01 measured 0.130 before onset. Values at or below this floor are not evidence of signal.
    """
    pre = times_s < 0
    return np.nanquantile(max_r[..., pre].reshape(max_r.shape[0], -1), q, axis=1)


def reliable(max_r: np.ndarray, times_s: np.ndarray, q: float = 0.95) -> np.ndarray:
    """Boolean mask (subjects, channels, time): max_r above that subject's pre-onset noise floor.

    Noise-normalized scores are reported only where this holds, so a ceiling that is mostly
    clipping bias never divides a score.
    """
    return max_r > noise_floor(max_r, times_s, q)[:, None, None]
