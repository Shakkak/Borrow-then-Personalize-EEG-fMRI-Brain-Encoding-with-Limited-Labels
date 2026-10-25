import numpy as np

from cross_subject_encoding.eval import nsd_confirm as nc


def test_common_vertices_columns_point_at_the_same_fsaverage_vertex():
    m1 = np.zeros(20, bool); m1[[1, 3, 4, 7, 9, 12]] = True
    m2 = np.zeros(20, bool); m2[[0, 3, 4, 9, 12, 15]] = True
    common, cols = nc.common_vertices([m1, m2])
    assert common.tolist() == [3, 4, 9, 12]
    for m, c in zip((m1, m2), cols):
        assert np.array_equal(np.where(m)[0][c], common)


def test_stack_recovers_planted_weights_and_uses_the_labelled_mean():
    g = np.random.default_rng(0)
    a, b = g.normal(size=(300, 50)), g.normal(size=(300, 50))
    y = 0.7 * a + 0.2 * b + 3.0 + 0.01 * g.normal(size=(300, 50))
    w = nc.stack_fit([a, b], y)
    assert np.allclose(w[0], 0.7, atol=0.01) and np.allclose(w[1], 0.2, atol=0.01)
    out = nc.stack_apply(w, [a, b], [a[:5], b[:5]], y.mean(0))
    assert np.abs(out - y[:5]).max() < 0.1


def test_with_labelled_mean_and_identification():
    g = np.random.default_rng(1)
    pattern = g.normal(size=(200, 300))
    meas = pattern + 5.0 + 0.3 * g.normal(size=pattern.shape)
    pred = pattern - 4.0  # right patterns, wrong shared offset
    fixed = nc.with_labelled_mean(pred, pred[:50], meas[:50])
    assert abs(fixed.mean() - meas.mean()) < 0.2
    assert nc.identify_top1(fixed, meas) > 95 and nc.identify_top1(pred, meas, centred=True) > 95
    assert nc.identify_top1(g.normal(size=pred.shape), meas) < 5


def test_galleries_disjoint_and_verdicts():
    gs = nc.galleries(872)
    assert len(gs) == 4 and len(set(np.concatenate(gs))) == 800
    assert nc.verdict(np.array([0.05, 0.06, 0.04, 0.05, 0.07, 0.05, 0.06, 0.05]), 0.01)["verdict"] == "supported"
    assert nc.verdict(np.array([-0.05, -0.06, -0.04, -0.05, -0.07, -0.05, -0.06, -0.05]), 0.01)["verdict"] == "rejected"
    assert nc.verdict(np.array([0.05, -0.06, 0.04, -0.05, 0.07, -0.05, 0.06, -0.05]), 0.01)["verdict"] == "inconclusive"
