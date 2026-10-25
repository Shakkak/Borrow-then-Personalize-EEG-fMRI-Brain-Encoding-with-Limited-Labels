"""A low-rank personal map for many outputs. The EEG map (H4b) mixes 17 channels per 100 ms window;
fMRI has about 39,000 vertices and no time, so a full vertex-by-vertex map is out of reach. Here the borrowed average
prediction is reduced to its top K principal directions (computed from predictions only, never from the person's
responses), and ridge maps those K scores to the person's residual (response minus borrowed prediction) at every
output, so the correction has rank at most K. With K equal to the number of outputs and identity basis this is a
ridge map from the whole borrowed prediction to the residual, H4b's form without windows.
"""
from __future__ import annotations

import numpy as np

from cross_subject_encoding.models.ridge import RidgePerOutput

ALPHAS = [10.0 ** e for e in range(-2, 9)]


def prediction_basis(avg_pred: np.ndarray, k_components: int) -> tuple[np.ndarray, np.ndarray]:
    """(mean, U): the mean and top principal directions (outputs, K) of borrowed predictions over many images
    (n_images, outputs). Images only: no response of the target enters."""
    mean = avg_pred.mean(0)
    xc = np.asarray(avg_pred - mean, np.float64)
    if len(xc) < xc.shape[1]:  # fMRI: fewer images than vertices; the image-by-image Gram gives the same directions
        e, u = np.linalg.eigh(xc @ xc.T)
        top = np.argsort(e)[::-1][:k_components]
        v = xc.T @ u[:, top]
        return mean, v / np.linalg.norm(v, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    return mean, vt[:k_components].T


def fit_lowrank_map(avg_k: np.ndarray, y_k: np.ndarray, mean: np.ndarray, basis: np.ndarray, alphas=ALPHAS,
                    folds: int = 5, seed: int = 13, groups: np.ndarray | None = None) -> RidgePerOutput:
    """Ridge from the K component scores of the borrowed prediction to the residual y - avg, alpha per output by
    cross-validation over the k labelled images (rows of one group, e.g. one image's trials, kept in one fold)."""
    scores = (avg_k - mean) @ basis
    return RidgePerOutput(alphas, min(folds, len(avg_k)), seed).fit(scores, y_k - avg_k, groups)


def apply_lowrank_map(model: RidgePerOutput, avg: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return avg + model.predict((avg - mean) @ basis)
