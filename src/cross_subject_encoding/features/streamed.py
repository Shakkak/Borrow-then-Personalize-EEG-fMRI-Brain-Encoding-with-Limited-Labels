"""The paper's feature pipeline (standardize, degree-4 polynomial kernel, kernel PCA) for image sets too large to hold
(NSD: 70,566 images). Same arithmetic as `paper_linearizing.standardize_stats`, `gram`,
`poly_kernel` and `kernel_pca`, rearranged: the fit images' standardized features stay on the device, and every other
image is extracted in chunks and reduced to its kernel row against the fit images at once, so the full feature
matrix (70,566 x 205,672) is never stored.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from cross_subject_encoding.models import paper_linearizing as pl


def _gram_rows(z_query: torch.Tensor, z_fit: torch.Tensor, chunk: int = 8192) -> np.ndarray:
    """z_query @ z_fit.T, each block of features multiplied in float32 and summed in float64 (as `pl.gram`)."""
    acc = torch.zeros((len(z_query), len(z_fit)), dtype=torch.float64, device=z_fit.device)
    for a in range(0, z_fit.shape[1], chunk):
        acc += (z_query[:, a:a + chunk].float() @ z_fit[:, a:a + chunk].float().T).double()
    return acc.cpu().numpy()


def streamed_kpca(extract_rows: Callable[[np.ndarray], np.ndarray], n_total: int, fit_idx: np.ndarray, device: str,
                  n_components: int = pl.N_COMPONENTS, chunk: int = 2048, progress=None) -> tuple[np.ndarray, dict]:
    """Kernel-PCA features (n_total, n_components) for rows 0..n_total-1. extract_rows(indices) returns those rows' raw
    features (len, d) in the given order. The kernel PCA is fitted on fit_idx rows, as the paper fits it on training
    images; standardization statistics come from the same rows."""
    fit_idx = np.asarray(fit_idx)
    x_fit = extract_rows(fit_idx)
    d = x_fit.shape[1]
    mean, scale = pl.standardize_stats(x_fit, np.ones(len(x_fit), bool))
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    scale_t = torch.as_tensor(scale, dtype=torch.float32, device=device)
    z_fit = torch.empty((len(x_fit), d), dtype=torch.float16, device=device)
    for a in range(0, len(x_fit), chunk):
        z_fit[a:a + chunk] = ((torch.as_tensor(x_fit[a:a + chunk], device=device).float() - mean_t) / scale_t).half()
    del x_fit
    # the fit kernel: standardized values kept in float16 on the device; products in float32, sums in float64
    k_fit = np.concatenate([_gram_rows(z_fit[a:a + chunk], z_fit) for a in range(0, len(z_fit), chunk)])
    k_fit = pl.poly_kernel((k_fit + k_fit.T) / 2, d)
    rest = np.setdiff1d(np.arange(n_total), fit_idx)

    def query_blocks():
        for start in range(0, len(rest), chunk):
            rows = rest[start:start + chunk]
            z = (torch.as_tensor(extract_rows(rows), device=device).float() - mean_t) / scale_t
            yield start, pl.poly_kernel(_gram_rows(z, z_fit), d)
            if progress:
                progress(start + len(rows), len(rest))

    z_train, z_rest, _ = pl.kernel_pca_batched(k_fit, query_blocks(), len(rest), n_components)
    out = np.empty((n_total, n_components), np.float32)
    out[fit_idx], out[rest] = z_train, z_rest
    return out, {"raw_features": int(d), "fit_images": int(len(fit_idx)), "other_images": int(len(rest))}
