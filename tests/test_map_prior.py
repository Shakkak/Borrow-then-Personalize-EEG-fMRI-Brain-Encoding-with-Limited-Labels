import numpy as np

from cross_subject_encoding.eval import eeg1_h4b as h4b
from cross_subject_encoding.eval import image_selection as sel
from cross_subject_encoding.eval import map_prior as mp


def person(g, avg, mix, noise=1.0):
    return np.einsum("ij,njt->nit", mix, avg) + noise * g.normal(size=avg.shape)


def corr(p, y):
    p, y = p - p.mean(0), y - y.mean(0)
    return np.mean((p * y).sum(0) / np.sqrt((p ** 2).sum(0) * (y ** 2).sum(0)))


def test_no_prior_is_exactly_h4b():
    g = np.random.default_rng(0)
    avg = g.normal(size=(60, 5, 20))
    y = person(g, avg, np.eye(5) + 0.4 * g.normal(size=(5, 5)))
    win = np.repeat(np.arange(2), 10)
    ref, ref_alpha = h4b.fit_map(avg, y, win, seed=13)
    got, chosen = mp.fit_map_prior(avg, y, win, None, seed=13)
    for w in ref:
        assert np.allclose(got[w], ref[w], atol=1e-8)
        assert chosen[w] == (ref_alpha[w], 1.0)


def test_prior_from_donors_helps_at_small_k():
    """People's maps vary along two shared directions; with 20 labelled images the covariance prior should beat
    shrinking towards no change."""
    g = np.random.default_rng(1)
    ch, t = 6, 20
    basis = g.normal(size=(2, ch, ch))
    mix = lambda: np.eye(ch) + np.tensordot(g.normal(size=2), basis, 1) * 0.5
    win = np.zeros(t, int)
    donors = []
    for _ in range(12):
        a = g.normal(size=(400, ch, t))
        donors.append(h4b.fit_map(a, person(g, a, mix()), win)[0])
    prior = mp.prior_from_maps(donors)
    gains = []
    for s in range(8):
        a = g.normal(size=(20 + 200, ch, t))
        y = person(g, a, mix(), noise=2.0)
        base = h4b.apply_map(h4b.fit_map(a[:20], y[:20], win, seed=s)[0], a[20:], win)
        coefs, chosen = mp.fit_map_prior(a[:20], y[:20], win, prior, seed=s)
        gains.append(corr(h4b.apply_map(coefs, a[20:], win), y[20:]) - corr(base, y[20:]))
    assert min(gains) > 0 and np.mean(gains) > 0.003


def test_k_center_is_nested_spread_out_and_seeded():
    g = np.random.default_rng(2)
    space = sel.feature_space(g.normal(size=(500, 30)) * np.arange(1, 31), 20)
    assert np.allclose(space.std(0), 1)
    a, b = sel.k_center(space, 40, 13), sel.k_center(space, 20, 13)
    assert len(set(a)) == 40 and np.array_equal(a[:20], b)
    assert not np.array_equal(a, sel.k_center(space, 40, 14))
    assert sel.spread(space, a, g.choice(500, 40, replace=False)) > 1.0
    assert sel.prediction_space(g.normal(size=(100, 3, 4)), 5).shape == (100, 5)


def test_electrode_kernel_by_name_and_order():
    names = ["O1", "Oz", "O2", "PO7", "PO3", "POz", "PO4", "PO8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8"]
    k = mp.electrode_kernel(names)
    assert k.shape == (17, 17) and np.allclose(k, k.T) and np.linalg.eigvalsh(k).min() > -1e-10
    assert k[0, 1] > k[0, 2] > k[0, 16]  # O1 is nearer Oz than O2, and far from P8
    perm = np.random.default_rng(0).permutation(17)
    assert np.allclose(mp.electrode_kernel([names[i].lower() for i in perm]), k[np.ix_(perm, perm)])
    try:
        mp.electrode_kernel(["O1", "Cz"])
        raise AssertionError("an unknown channel must raise")
    except KeyError:
        pass


def test_graph_prior_helps_a_smooth_map_at_small_k():
    """A person whose map correction is smooth over the layout: with 15 labelled images the graph prior should
    beat shrinking towards no change."""
    g = np.random.default_rng(3)
    names = ["O1", "Oz", "O2", "PO7", "PO3", "POz", "PO4", "PO8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8"]
    k = mp.electrode_kernel(names)
    ch, t = 17, 20
    win = np.zeros(t, int)
    gains = []
    for _ in range(4):
        b = np.linalg.cholesky(k + 1e-9 * np.eye(ch)) @ g.normal(size=(ch, ch)) @ np.linalg.cholesky(k + 1e-9 * np.eye(ch)).T * 0.15
        mix = np.eye(ch) + b.T
        a = g.normal(size=(15, ch, t))
        y = person(g, a, mix)
        a_te = g.normal(size=(300, ch, t))
        y_te = person(g, a_te, mix, noise=0.0)
        plain = h4b.apply_map(h4b.fit_map(a, y, win)[0], a_te, win)
        graph = h4b.apply_map(mp.fit_map_prior(a, y, win, mp.graph_prior(names, win))[0], a_te, win)
        gains.append(corr(graph, y_te) - corr(plain, y_te))
    assert np.mean(gains) > 0
