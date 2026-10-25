"""Frozen EEG1 confirmation decision (protocol P-1).

Per target p and comparator c, d_pc = mean over k in {250, 500} and 100 <= t < 200 ms of
score(core) - score(c), where every score is already averaged over draws 13-15 (and core seeds for the core).
Supported: all five Bonferroni one-sided lower bounds exceed delta. Rejected: any Bonferroni one-sided
upper bound lies below delta. Otherwise inconclusive. Fewer than MIN_TARGETS complete targets: insufficient.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

COMPARATORS = ("avg_sources", "gu_window", "linear_converter", "own_ridge", "own_ridge_oc")
MIN_TARGETS = 20


def decide(d: np.ndarray, delta: float, alpha: float = 0.05, min_targets: int = MIN_TARGETS,
           comparators=COMPARATORS) -> dict:
    """d: (targets, comparators) paired differences; rows with any NaN are excluded and counted."""
    d = np.asarray(d, dtype=float)
    if d.ndim != 2 or d.shape[1] != len(comparators):
        raise ValueError("d must be (targets, comparators)")
    if not np.isfinite(delta):
        raise ValueError("delta must be finite")
    complete = np.isfinite(d).all(1)
    d = d[complete]
    n = len(d)
    out = {"n_targets": int(n), "n_excluded": int((~complete).sum()), "delta": float(delta), "alpha": alpha,
           "bonferroni_m": len(comparators), "per_comparator": {}}
    if n < min_targets:
        out["verdict"] = "insufficient"
        return out
    t = stats.t.ppf(1 - alpha / len(comparators), n - 1)
    lows, ups = [], []
    for j, name in enumerate(comparators):
        mean, se = d[:, j].mean(), d[:, j].std(ddof=1) / np.sqrt(n)
        low, up = mean - t * se, mean + t * se
        lows.append(low)
        ups.append(up)
        out["per_comparator"][name] = {"mean": float(mean), "sd": float(d[:, j].std(ddof=1)), "lower": float(low),
                                       "upper": float(up), "positive_targets": int((d[:, j] > 0).sum())}
    out["t_critical"] = float(t)
    out["verdict"] = ("supported" if min(lows) > delta else "rejected" if min(ups) < delta else "inconclusive")
    return out


def sign_flip_p(d: np.ndarray, n_flips: int = 100_000, seed: int = 13) -> float:
    """Descriptive one-sided sign-flip p for mean(d) > 0 (Monte Carlo); not part of the decision."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n_flips, len(d)))
    null = (signs * d).mean(1)
    return float((1 + (null >= d.mean()).sum()) / (1 + n_flips))


def endpoint(scores: dict[tuple[str, str, int], float], targets: list[str], budgets=(250, 500),
             comparators=COMPARATORS, candidate: str = "prior_both") -> np.ndarray:
    """scores[(target, method, k)] = window-mean score already averaged over draws (and core seeds).

    Returns d (targets, comparators); NaN where any needed score is missing.
    """
    d = np.full((len(targets), len(comparators)), np.nan)
    for i, person in enumerate(targets):
        for j, name in enumerate(comparators):
            gains = [scores.get((person, candidate, k), np.nan) - scores.get((person, name, k), np.nan)
                     for k in budgets]
            d[i, j] = np.mean(gains)
    return d
