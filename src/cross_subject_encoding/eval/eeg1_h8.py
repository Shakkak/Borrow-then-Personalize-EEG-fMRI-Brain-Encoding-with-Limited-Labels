"""H8 (protocol P-2): borrowing versus own-data methods when the new person's whitening uses only m trials.

The new person's unwhitened epochs are whitened with a matrix estimated from m of their training trials
(m = k: only the k labelled ones; m = "all": every training trial, the 08a reference). Sources stay fully
whitened. Each condition is scored in its own coordinates (P-2).
"""
from __future__ import annotations

import numpy as np

from ..data.things_eeg1 import mvnn_whitener
from ..eval.paper_correlation import paper_correlation
from ..models.baselines import gu_by_group
from ..models.ridge import RidgePerOutput

BORROWING = ("avg_sources", "gu_window")
OWN = ("own_ridge", "own_ridge_oc")
CONTRASTS = tuple((b, o) for b in BORROWING for o in OWN)


def whiten(raw_train: np.ndarray, raw_test: np.ndarray, rows) -> tuple[np.ndarray, np.ndarray]:
    """raw_train (images, channels, times), raw_test (images, reps, channels, times); whitener from `rows`."""
    w = mvnn_whitener(raw_train[rows])
    return np.einsum("ij,njt->nit", w, raw_train), np.einsum("ij,nrjt->nrit", w, raw_test)


def evaluate_person(raw_train, raw_test, src_train, src_test, z_train, z_test, orders: dict, budgets, conditions,
                    groups, alphas, folds: int, n_iter: int = 100, others=None) -> dict:
    """Scores {(condition, method, k, draw): curve (channels, times)} for one new person.

    src_train/src_test: the other people's full-data predictions, (sources, images, outputs).
    orders: {draw: image order}; the first k images are the labelled ones. others: rows of src_* to borrow from
    (default all), indexed without copying the full arrays.
    conditions: "k" (whitening from the k labelled trials), an int m (first m of the order), or "all".
    """
    n_test = raw_test.shape[0]
    shape = raw_test.shape[2:]
    others = np.arange(len(src_test)) if others is None else np.asarray(others)
    avg_test = src_test[others].mean(0)
    out = {}
    for condition in conditions:
        for draw, order in orders.items():
            order = np.asarray(order)
            for budget in budgets:
                rows = {"k": order[:budget], "all": np.arange(len(raw_train))}.get(condition)
                if rows is None:
                    rows = order[:int(condition)]
                train, test = whiten(raw_train, raw_test, rows)
                idx = order[:budget]
                target = train[idx].reshape(budget, -1)
                own = RidgePerOutput(alphas, folds, draw).fit(z_train[idx], target)
                own_test = own.predict(z_test)
                oof = np.empty_like(target)
                for fold in np.array_split(np.random.default_rng(draw).permutation(budget), folds):
                    keep = np.setdiff1d(np.arange(budget), fold)
                    oof[fold] = own.refit(z_train[idx[keep]], target[keep]).predict(z_train[idx[fold]])
                predictions = {
                    "avg_sources": avg_test,
                    "gu_window": gu_by_group(src_train[np.ix_(others, idx)], target, src_test[others], groups, "pooled"),
                    "own_ridge": own_test,
                    "own_ridge_oc": RidgePerOutput(alphas, folds, draw).fit(oof, target).predict(own_test),
                }
                predictions = {name: p.reshape(n_test, *shape) for name, p in predictions.items()}
                scored = paper_correlation(predictions, test, n_iter=n_iter)
                for name, curve in scored["correlation"].items():
                    out[(str(condition), name, budget, draw)] = curve
    return out


def contrasts(window_scores: dict, people: list[str], condition: str, budgets) -> np.ndarray:
    """d (people, 4): borrowing minus own, averaged over k; window_scores[(person, condition, method, k)]."""
    d = np.full((len(people), len(CONTRASTS)), np.nan)
    for i, person in enumerate(people):
        for j, (b, o) in enumerate(CONTRASTS):
            d[i, j] = np.mean([window_scores.get((person, condition, b, k), np.nan) -
                               window_scores.get((person, condition, o, k), np.nan) for k in budgets])
    return d
