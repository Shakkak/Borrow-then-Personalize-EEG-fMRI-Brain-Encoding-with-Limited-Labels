import numpy as np

from cross_subject_encoding.eval import eeg1_h4b as h4b
from cross_subject_encoding.eval import h4b_analysis as ha


def test_windows_and_rank():
    times = np.round(-0.2 + np.arange(100) / 100, 3)
    assert ha.window_index(times, 100).max() == 9 and ha.window_index(times, None).max() == 0
    assert ha.window_index(times, 50).max() == 19
    g = np.random.default_rng(0)
    b = {0: g.normal(size=(5, 5))}
    assert np.linalg.matrix_rank(ha.truncate_rank(b, 2)[0]) == 2
    assert np.allclose(ha.truncate_rank(b, 5)[0], b[0])
    assert ha.map_norms({0: np.zeros((3, 3))})[0] == 0


def test_person_maps_reproduce_within_and_differ_between():
    g = np.random.default_rng(1)
    ch, t, k = 5, 20, 400
    win = np.repeat(np.arange(2), t // 2)
    mixes = {p: np.eye(ch) + 0.6 * g.normal(size=(ch, ch)) for p in "abcd"}  # each person their own mix
    half_a, half_b = {}, {}
    for p, mix in mixes.items():
        avg = g.normal(size=(2 * k, ch, t))
        y = np.einsum("ij,njt->nit", mix, avg) + 0.5 * g.normal(size=(2 * k, ch, t))
        half_a[p] = h4b.fit_map(avg[:k], y[:k], win)[0]
        half_b[p] = h4b.fit_map(avg[k:], y[k:], win)[0]
    r = ha.reproducibility(half_a, half_b)
    assert all(r["within"][w] > 0.9 for w in r["within"])
    assert all(r["within"][w] > r["between"][w] + 0.3 for w in r["within"])


def test_leading_component_recovers_planted_pattern_and_sign():
    rng = np.random.default_rng(3)
    u, v = rng.standard_normal(17), rng.standard_normal(17)
    coef = 2.0 * np.outer(u, v) + 0.01 * rng.standard_normal((17, 17))
    c = ha.leading_component(-coef, reference=v)
    assert ha.abs_cosine(c["to"], v) > 0.99 and c["to"] @ v > 0
    assert ha.abs_cosine(c["from"], u) > 0.99 and c["share"] > 0.99
    np.testing.assert_allclose(c["s"] * np.outer(c["from"], c["to"]), -coef, atol=0.2)


def test_abs_cosine_ignores_sign():
    a = np.arange(1.0, 5.0)
    assert ha.abs_cosine(a, -a) == 1.0
