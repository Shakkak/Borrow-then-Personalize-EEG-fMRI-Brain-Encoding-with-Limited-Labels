"""Alignment controls: Procrustes recovers a planted rotation; SRM-style prediction recovers a shared response."""
import numpy as np

from cross_subject_encoding.eval import alignment_controls as al


def rotation(rng, ch):
    return np.linalg.qr(rng.standard_normal((ch, ch)))[0]


def test_procrustes_recovers_a_planted_rotation_per_window():
    rng = np.random.default_rng(0)
    k, ch, t = 200, 6, 20
    window = np.repeat([0, 1], 10)
    avg = rng.standard_normal((k, ch, t))
    r = [rotation(rng, ch), rotation(rng, ch)]
    y = np.empty_like(avg)
    for w in (0, 1):
        idx = window == w
        y[:, :, idx] = np.einsum("nct,cd->ndt", avg[:, :, idx], r[w])
    maps = al.fit_procrustes_map(avg, y + 3.0, window)
    for w in (0, 1):
        np.testing.assert_allclose(maps[w], r[w], atol=1e-8)
    pred = al.apply_rotation(maps, avg, window)
    np.testing.assert_allclose(pred, y - y.mean(0), atol=1e-8)


def test_srm_style_prediction_recovers_the_shared_response():
    rng = np.random.default_rng(1)
    n_k, n_test, ch, t, d = 300, 60, 8, 10, 3
    window = np.zeros(t, int)
    shared_k, shared_t = rng.standard_normal((n_k, t, d)), rng.standard_normal((n_test, t, d))
    def person():
        w = np.linalg.qr(rng.standard_normal((ch, d)))[0]
        make = lambda s: np.einsum("ntd,cd->nct", s, w)
        return make(shared_k), make(shared_t)
    others = [person() for _ in range(5)]
    new_k, new_t = person()
    pred = al.srm_predict([o[0] for o in others], [o[1] for o in others], new_k + 0.05 * rng.standard_normal(new_k.shape), window, d=d)
    r = np.corrcoef(pred.ravel(), (new_t - new_t.mean(0)).ravel())[0, 1]
    assert r > 0.95
