"""Plug-in map helpers: out-of-fold predictions never use their own rows; stacking recovers planted weights."""
import numpy as np

from cross_subject_encoding.eval import plugin_map as pm


def test_out_of_fold_never_sees_its_own_rows():
    seen = []

    def fit_predict(train, held):
        assert not set(train) & set(held)
        seen.extend(held)
        return np.full((len(held), 2), float(len(train)))

    out = pm.out_of_fold(fit_predict, 23, n_folds=5, seed=1)
    assert sorted(seen) == list(range(23)) and out.shape == (23, 2)


def test_stack_recovers_planted_weights_per_channel_and_window():
    rng = np.random.default_rng(0)
    k, ch, t = 400, 3, 20
    window = np.repeat([0, 1], 10)
    a, b = rng.standard_normal((k, ch, t)), rng.standard_normal((k, ch, t))
    true = rng.uniform(0, 1, (ch, 2, 2))
    y = np.zeros((k, ch, t))
    for j in range(2):
        idx = window == j
        y[:, :, idx] = true[:, j, 0][None, :, None] * a[:, :, idx] + true[:, j, 1][None, :, None] * b[:, :, idx]
    w = pm.stack_fit([a, b], y + 5.0, window)
    np.testing.assert_allclose(w, true, atol=1e-6)
    out = pm.stack_apply(w, [a, b], window)
    np.testing.assert_allclose(out, y, atol=1e-6)
