"""The THINGS-EEG2 paper's correlation, noise ceiling and intervals, reproduced from its released code.

Source: `05_synthetic_data_analyses/correlation.py` and `correlation_stats.py`
of github.com/gifale95/eeg_encoding. Per person, 100 times: draw 40 of the
80 test repetitions at random (sklearn `resample` without replacement, the
global NumPy generator seeded with 20200220 at the start of each person's
run), correlate the prediction with the average of the other 40 at every
channel and time point, and average the 100 correlations. The noise
ceiling's lower bound correlates the two halves, its upper bound one half
with all 80. Across people: average over channels, then over people, with
a 95% interval from 10,000 resamples of people at each time point (the
generator seeded again with 20200220).
"""
from __future__ import annotations

import numpy as np

from cross_subject_encoding.eval.metrics import pearson_per_output

PAPER_SEED = 20200220


def paper_correlation(synt: dict[str, np.ndarray], bio_test: np.ndarray, n_iter: int = 100,
                      seed: int = PAPER_SEED) -> dict:
    """synt: {name: (images, channels, time)}; bio_test: (images, repetitions, channels, time)."""
    from sklearn.utils import resample

    np.random.seed(seed)
    n_img, n_rep, n_ch, n_t = bio_test.shape
    flat = lambda a: a.reshape(n_img, -1)
    corr = {k: np.zeros((n_iter, n_ch * n_t)) for k in synt}
    low, up = np.zeros((n_iter, n_ch * n_t)), np.zeros((n_iter, n_ch * n_t))
    avg_all = flat(bio_test.mean(1))
    for i in range(n_iter):
        idx = resample(np.arange(0, n_rep), replace=False, n_samples=int(n_rep / 2))
        half_1 = flat(np.delete(bio_test, idx, 1).mean(1))
        half_2 = flat(bio_test[:, idx].mean(1))
        for k, s in synt.items():
            corr[k][i] = pearson_per_output(half_1, flat(s))
        low[i] = pearson_per_output(half_1, half_2)
        up[i] = pearson_per_output(half_1, avg_all)
    shape = (n_ch, n_t)
    return {"correlation": {k: v.mean(0).reshape(shape) for k, v in corr.items()},
            "noise_ceiling_low": low.mean(0).reshape(shape), "noise_ceiling_up": up.mean(0).reshape(shape)}


def paper_stats(per_person: np.ndarray, n_iter: int = 10000, seed: int = PAPER_SEED, alpha: float = 0.05) -> dict:
    """per_person: (people, channels, time). The channel-and-people mean over time, its peak, and the intervals.

    alpha = 0.05 gives the paper's 95% intervals; a smaller alpha widens them (for example 0.05 / 5 when
    a hypothesis is decided over five label budgets)."""
    np.random.seed(seed)
    by_person = per_person.mean(1)  # (people, time): averaged over channels first
    mean = by_person.mean(0)
    n = by_person.shape[0]
    lo, hi = np.zeros(by_person.shape[1]), np.zeros(by_person.shape[1])
    for t in range(by_person.shape[1]):
        # sklearn's resample(x) draws np.random.randint(0, n, n) from the global generator once per call;
        # drawing all n_iter rows at once gives the same numbers in the same order, much faster
        idx = np.random.randint(0, n, size=(n_iter, n))
        dist = by_person[idx, t].mean(1)
        lo[t], hi[t] = np.percentile(dist, 100 * alpha / 2), np.percentile(dist, 100 * (1 - alpha / 2))
    peak = int(np.argmax(mean))
    return {"curve": mean, "ci_lower": lo, "ci_upper": hi, "peak_index": peak, "peak": float(mean[peak]),
            "peak_lo": float(lo[peak]), "peak_hi": float(hi[peak])}
