"""Ablation and interpretation helpers for H4b's per-window channel-mixing maps (09b/09c)."""
from __future__ import annotations

import numpy as np


def window_index(times: np.ndarray, window_ms: int | None) -> np.ndarray:
    """Window id per time point; None = one window over the whole epoch."""
    if window_ms is None:
        return np.zeros(len(times), dtype=int)
    return np.round((np.asarray(times) - times[0]) * 1000).astype(int) // window_ms


def truncate_rank(coefs: dict, rank: int) -> dict:
    """Best rank-r approximation (SVD) of each window's correction map."""
    out = {}
    for w, b in coefs.items():
        u, s, vt = np.linalg.svd(b, full_matrices=False)
        out[w] = (u[:, :rank] * s[:rank]) @ vt[:rank]
    return out


def map_norms(coefs: dict) -> dict:
    """Frobenius norm of each window's correction (0 = no change to others' average)."""
    return {w: float(np.linalg.norm(b)) for w, b in coefs.items()}


def cosine(a: dict, b: dict) -> dict:
    """Cosine similarity of two people's (or two halves') maps, per window."""
    out = {}
    for w in a:
        x, y = a[w].ravel(), b[w].ravel()
        den = np.linalg.norm(x) * np.linalg.norm(y)
        out[w] = float(x @ y / den) if den > 0 else np.nan
    return out


def reproducibility(half_a: dict[str, dict], half_b: dict[str, dict]) -> dict:
    """Within-person similarity (a person's two disjoint halves) versus between-person similarity
    (different people's first halves), per window. Inputs: {person: {window: map}}."""
    people = sorted(half_a)
    windows = sorted(half_a[people[0]])
    within = {w: [cosine(half_a[p], half_b[p])[w] for p in people] for w in windows}
    between = {w: [cosine(half_a[p], half_a[q])[w] for i, p in enumerate(people) for q in people[i + 1:]] for w in windows}
    return {"within": {w: float(np.nanmean(v)) for w, v in within.items()},
            "between": {w: float(np.nanmean(v)) for w, v in between.items()},
            "within_by_person": {p: float(np.nanmean([cosine(half_a[p], half_b[p])[w] for w in windows])) for p in people}}


def leading_component(coef: np.ndarray, reference: np.ndarray | None = None) -> dict:
    """The map's strongest input-to-output pattern. A map B (input channel x output channel) adds x @ B to
    others' average x; B ~ s u v^T means: read the average along spatial pattern u ("from") and add it along
    pattern v ("to"). The sign of (u, v) is arbitrary: it is set so that v correlates positively with
    `reference` if given, else so that v's largest-magnitude entry is positive. Share = s1^2 / sum s^2."""
    u, s, vt = np.linalg.svd(coef)
    u1, v1 = u[:, 0], vt[0]
    flip = (v1 @ reference < 0) if reference is not None else (v1[np.argmax(np.abs(v1))] < 0)
    if flip:
        u1, v1 = -u1, -v1
    return {"from": u1, "to": v1, "s": float(s[0]), "share": float(s[0] ** 2 / np.sum(s ** 2))}


def abs_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """|cos| between two patterns, for sign-free comparison of singular vectors."""
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(abs(a @ b) / den) if den > 0 else float("nan")
