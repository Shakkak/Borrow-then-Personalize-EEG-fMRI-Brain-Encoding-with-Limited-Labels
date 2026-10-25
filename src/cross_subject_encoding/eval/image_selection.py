"""Choose which k images a new person labels, from image-only information.

Both policies are greedy k-centre (farthest-point) selection with a seeded random first image, in one of two
spaces: the paper's KPCA features (first components, each scaled to unit variance) or others' average
prediction (first principal components, unscaled). Neither looks at the new person's EEG. k-centre tends to
pick outlying images, so `spread` reports how far the chosen images sit from the pool's centre against a
random set of the same size (KPCA components are unscaled).
"""
from __future__ import annotations

import numpy as np


def feature_space(z: np.ndarray, n_components: int = 100) -> np.ndarray:
    f = np.asarray(z[:, :n_components], np.float64)
    return (f - f.mean(0)) / np.where(f.std(0) > 0, f.std(0), 1.0)


def prediction_space(pred: np.ndarray, n_components: int = 50) -> np.ndarray:
    p = np.asarray(pred, np.float64).reshape(len(pred), -1)
    p = p - p.mean(0)
    u, s, _ = np.linalg.svd(p, full_matrices=False)
    return u[:, :n_components] * s[:n_components]


def k_center(space: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Greedy farthest-point order of k image indices; nested (the first 50 of k = 100 are the k = 50 set)."""
    n = len(space)
    chosen = [int(np.random.default_rng(seed).integers(n))]
    dist = ((space - space[chosen[0]]) ** 2).sum(1)
    for _ in range(k - 1):
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, ((space - space[nxt]) ** 2).sum(1))
    return np.asarray(chosen)


def spread(space: np.ndarray, idx: np.ndarray, reference: np.ndarray) -> float:
    """Median distance from the pool's centre of the chosen images, divided by the same for a reference set."""
    centre = space.mean(0)
    d = lambda i: float(np.median(np.sqrt(((space[i] - centre) ** 2).sum(1))))
    return d(idx) / d(reference)
