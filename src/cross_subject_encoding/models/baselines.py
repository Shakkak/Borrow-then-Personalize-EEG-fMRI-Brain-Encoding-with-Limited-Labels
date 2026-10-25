"""The baselines for a held-out subject with k labelled images.

- B0, B1: per-subject ridge on image features.
- B2: Gu et al.'s linear ensemble: the held-out subject's
  response as a weighted sum of the other subjects' predictions, weights
  and intercept by ordinary least squares on the k labelled images.
  `gu_by_group` gives each 100 ms window one set of weights, fitted on the
  window mean as Gu et al. fit region means, or on all its outputs pooled
  (both adapted); `gu_per_output` gives every output its own
  weights (adapted).
- B3: a linear prediction converter (not EEG2EEG): the best
  source subject's prediction mapped onto the held-out subject's outputs by
  ridge, fitted on the k labelled images.
- `ols_oof`: out-of-fold source predictions for the k images.
"""
from __future__ import annotations

import numpy as np

from cross_subject_encoding.models.ridge import RidgePerOutput


def gu_ensemble(source_preds_train: list[np.ndarray], y_train: np.ndarray,
                source_preds_test: list[np.ndarray], groups: np.ndarray) -> np.ndarray:
    """Group-mean predictions (n_test, n_groups) from the other subjects' predictions.

    source_preds_*: one (n_images, n_outputs) array per source subject; groups: the region or group index
    of each output. Fitting uses region means, as Gu et al. do.
    """
    ug = np.unique(groups)
    mean_by = lambda a: np.stack([a[:, groups == g].mean(1) for g in ug], 1)
    y_g = mean_by(y_train)
    src_tr = np.stack([mean_by(p) for p in source_preds_train], -1)  # (n_train, n_groups, n_sources)
    src_te = np.stack([mean_by(p) for p in source_preds_test], -1)
    out = np.empty((src_te.shape[0], len(ug)), dtype=np.float32)
    for j in range(len(ug)):
        a = np.column_stack([src_tr[:, j], np.ones(len(src_tr))])
        w, *_ = np.linalg.lstsq(a, y_g[:, j], rcond=None)
        out[:, j] = np.column_stack([src_te[:, j], np.ones(len(src_te))]) @ w
    return out


def best_source(source_preds_train: list[np.ndarray], y_train: np.ndarray) -> int:
    """Index of the source subject whose predictions correlate best with the held-out subject's k responses."""
    def mean_r(p):
        t, q = y_train - y_train.mean(0), p - p.mean(0)
        den = np.sqrt((t ** 2).sum(0) * (q ** 2).sum(0)) + 1e-12
        return float(np.nanmean((t * q).sum(0) / den))
    return int(np.argmax([mean_r(p) for p in source_preds_train]))


def eeg2eeg_converter(source_pred_train: np.ndarray, y_train: np.ndarray, source_pred_test: np.ndarray,
                      alphas, seed: int) -> np.ndarray:
    """Map one source subject's predictions onto the held-out subject's outputs with per-output ridge."""
    return RidgePerOutput(alphas, n_folds=min(5, len(y_train)), seed=seed).fit(source_pred_train, y_train).predict(source_pred_test)


def gu_per_output(source_train: np.ndarray, y_train: np.ndarray, source_test: np.ndarray) -> np.ndarray:
    """Gu et al.'s ensemble with every output as its own group, solved for all outputs at once.

    source_train (n_sources, k, n_outputs), y_train (k, n_outputs), source_test (n_sources, n_test, n_outputs).
    EEG has no regions, so each channel and time point gets its own weights and intercept. The
    normal equations are solved per output; a tiny ridge (1e-6 of the trace) only guards against an exactly
    singular system, since the sources' predictions come from the same image features and are collinear.
    """
    s, k, o = source_train.shape
    a = np.concatenate([source_train, np.ones((1, k, o), source_train.dtype)]).transpose(2, 1, 0)  # (o, k, s+1)
    ata = a.transpose(0, 2, 1) @ a  # (o, s+1, s+1)
    ata += 1e-6 * np.trace(ata, axis1=1, axis2=2)[:, None, None] / (s + 1) * np.eye(s + 1)
    aty = a.transpose(0, 2, 1) @ y_train.T[:, :, None]  # (o, s+1, 1)
    w = np.linalg.solve(ata, aty)[..., 0]  # (o, s+1)
    b = np.concatenate([source_test, np.ones((1,) + source_test.shape[1:], source_test.dtype)])  # (s+1, m, o)
    return np.einsum("smo,os->mo", b, w).astype(np.float32)


def ols_oof(x: np.ndarray, y: np.ndarray, n_folds: int = 10, seed: int = 13, block: int = 4096) -> np.ndarray:
    """Out-of-fold predictions of ordinary least squares with an intercept (the paper's estimator, cross-fitted).

    Every image is predicted by a model fitted without its fold. Notebook 03 gave Gu et al.'s ensemble and
    the converter the source people's predictions on images those sources were fitted on; Gu et al.'s
    sources had never seen the new person's training images (NSD people view different images), so
    out-of-fold predictions are the faithful version on THINGS-EEG2, where everyone sees the same images. The normal equations are built once and each fold's rows subtracted, in float64; the outputs
    are processed in column blocks to bound memory.
    """
    x1 = np.concatenate([np.ones((len(x), 1)), x], 1).astype(np.float64)
    folds = np.array_split(np.random.default_rng(seed).permutation(len(x)), n_folds)
    xtx = x1.T @ x1
    inv = [np.linalg.inv(xtx - x1[f].T @ x1[f]) for f in folds]
    out = np.empty(y.shape, dtype=np.float32)
    for c in range(0, y.shape[1], block):
        yc = np.asarray(y[:, c:c + block], dtype=np.float64)
        xty = x1.T @ yc
        for f, a in zip(folds, inv):
            out[f, c:c + block] = x1[f] @ (a @ (xty - x1[f].T @ yc[f]))
    return out


def window_groups(times_s: np.ndarray, n_channels: int, window_ms: int) -> np.ndarray:
    """Group index of every output (channel-major, channels x time) by its time window, all channels together."""
    ms = np.round((np.asarray(times_s) - times_s[0]) * 1000).astype(int)
    return np.tile(ms // window_ms, n_channels)


def gu_by_group(source_train: np.ndarray, y_train: np.ndarray, source_test: np.ndarray, groups: np.ndarray,
                fit: str = "mean") -> np.ndarray:
    """Gu et al.'s linear ensemble with one set of weights per group of outputs, applied to every output in it.

    source_train (n_sources, k, n_outputs), y_train (k, n_outputs), source_test (n_sources, n_test, n_outputs),
    groups (n_outputs,). EEG has no regions, so a group is all channels in one time window.
    fit="mean": as Gu et al.'s `ensemble.py`, sklearn's `LinearRegression` maps the sources' group-mean
    predictions to the person's group-mean response on the k images. fit="pooled": the same number of
    weights, fitted on every output of the group at once (each output centred over the k images), because
    EEG channels in a window have mixed signs and their mean can cancel the signal, which an fMRI region
    mean does not. Each output keeps the group's weights and gets its own intercept, matching the k images'
    mean, which does not change any correlation over test images.
    """
    from sklearn.linear_model import LinearRegression

    out = np.empty(source_test.shape[1:], dtype=np.float32)
    for g in np.unique(groups):
        cols = groups == g
        src, y = source_train[:, :, cols], y_train[:, cols]
        if fit == "mean":
            w = LinearRegression().fit(src.mean(2).T, y.mean(1)).coef_
        elif fit == "pooled":
            a = (src - src.mean(1, keepdims=True)).reshape(len(src), -1).T
            w = LinearRegression(fit_intercept=False).fit(a, (y - y.mean(0)).ravel()).coef_
        else:
            raise ValueError(fit)
        fitted = np.einsum("s,sko->ko", w, src)
        out[:, cols] = np.einsum("s,smo->mo", w, source_test[:, :, cols]) + (y - fitted).mean(0)
    return out


def ols_without_rows(x1tx1: np.ndarray, x1ty: np.ndarray, x1_rows: np.ndarray, y_rows: np.ndarray) -> np.ndarray:
    """OLS coefficients refitted without some rows, from the full normal equations (x1 includes the intercept column).

    Used to give a source model that has never seen the new person's k images while one model predicts
    both those images and the test images, Gu et al.'s condition without cross-fitting.
    """
    return np.linalg.solve(x1tx1 - x1_rows.T @ x1_rows, x1ty - x1_rows.T @ y_rows)
