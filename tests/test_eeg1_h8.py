import numpy as np

from cross_subject_encoding.data import things_eeg1 as e1
from cross_subject_encoding.eval import eeg1_h8 as h8
from cross_subject_encoding.models.baselines import window_groups


def world(seed=0, n=60, n_test=8, ch=3, t=20, sources=4):
    g = np.random.default_rng(seed)
    z, zt = g.normal(size=(n, 5)), g.normal(size=(n_test, 5))
    w = g.normal(size=(5, ch * t))
    mix = np.eye(ch) + 0.5 * g.normal(size=(ch, ch))  # correlated channel noise, so whitening matters
    noise = lambda *s: np.einsum("ij,...jt->...it", mix, g.normal(size=s))
    raw = (z @ w).reshape(n, ch, t) + noise(n, ch, t)
    raw_test = (zt @ w).reshape(n_test, 1, ch, t) + noise(n_test, 12, ch, t)
    src = np.stack([(z @ w) + 0.3 * g.normal(size=(n, ch * t)) for _ in range(sources)])
    src_t = np.stack([(zt @ w) + 0.3 * g.normal(size=(n_test, ch * t)) for _ in range(sources)])
    return raw, raw_test, src, src_t, z, zt, ch, t


def test_whitening_from_rows_and_all_matches_mvnn():
    raw, raw_test, *_ = world()
    tr, te = h8.whiten(raw, raw_test, np.arange(len(raw)))
    w = e1.mvnn_whitener(raw)
    assert np.allclose(tr, np.einsum("ij,njt->nit", w, raw)) and te.shape == raw_test.shape
    tr_k, _ = h8.whiten(raw, raw_test, np.arange(10))
    assert not np.allclose(tr_k, tr)


def test_evaluate_person_conditions_and_contrasts():
    raw, raw_test, src, src_t, z, zt, ch, t = world()
    times = np.round(-0.05 + np.arange(t) / 100, 3)
    out = h8.evaluate_person(raw, raw_test, src, src_t, z, zt, {13: np.random.default_rng(1).permutation(60)},
                             (12, 20), ("k", 30, "all"), window_groups(times, ch, 100), [0.1, 10.0, 1000.0], 3, n_iter=5)
    assert len(out) == 3 * 1 * 2 * 4
    assert all(curve.shape == (ch, t) and np.isfinite(curve).all() for curve in out.values())
    ws = {("p", c, m, k): float(v.mean()) for (c, m, k, d), v in out.items()}
    d = h8.contrasts(ws, ["p"], "k", (12, 20))
    assert d.shape == (1, 4) and np.isfinite(d).all()


def test_others_subset_equals_explicit_subset():
    raw, raw_test, src, src_t, z, zt, ch, t = world()
    times = np.round(-0.05 + np.arange(t) / 100, 3)
    args = (z, zt, {13: np.arange(60)}, (12,), ("all",), window_groups(times, ch, 100), [1.0], 3)
    a = h8.evaluate_person(raw, raw_test, src, src_t, *args, n_iter=3, others=[0, 2])
    b = h8.evaluate_person(raw, raw_test, src[[0, 2]], src_t[[0, 2]], *args, n_iter=3)
    assert all(np.allclose(a[key], b[key]) for key in a)
