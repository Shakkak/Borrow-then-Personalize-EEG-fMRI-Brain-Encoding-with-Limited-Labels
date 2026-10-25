"""Hold-one-subject-out folds and nested label budgets.

For each held-out subject and seed, one random order of that subject's
training images is drawn. The k-subset is its first k images, so the
50-image set always sits inside the 100-image set (nested), and a larger k
only adds images. Only the first max(k) positions are stored; k = None means
every training image.
"""
from __future__ import annotations

import numpy as np


def nested_subsets(n_images: int, k_grid: list, subjects: list[str], seeds: list[int]) -> dict:
    k_max = max(k for k in k_grid if k is not None)
    out = {}
    for i, s in enumerate(subjects):
        out[s] = {}
        for seed in seeds:
            # the seed and the subject's position together fix the order, so reruns reproduce it
            rng = np.random.default_rng([seed, i])
            out[s][str(seed)] = rng.permutation(n_images)[:k_max].tolist()
    return {"k_grid": k_grid, "k_max_stored": k_max, "n_images": n_images, "orders": out}


def subset(subsets: dict, subject: str, seed: int, k) -> np.ndarray:
    """Image indices of the k-subset (None: all images)."""
    if k is None:
        return np.arange(subsets["n_images"])
    if k > subsets["k_max_stored"]:
        raise ValueError(f"k={k} is above the stored maximum {subsets['k_max_stored']}")
    return np.asarray(subsets["orders"][subject][str(seed)][:k])
