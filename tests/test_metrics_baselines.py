"""Ridge, metrics and the borrowing baselines on a small synthetic problem with a known answer."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cross_subject_encoding.eval.metrics import identification, noise_normalized, pearson_per_output, window_means
from cross_subject_encoding.models.baselines import best_source, eeg2eeg_converter, gu_ensemble
from cross_subject_encoding.models.ridge import RidgePerOutput

rng = np.random.default_rng(0)


def test_ridge_metrics_baselines():
    x = rng.normal(size=(200, 30))
    w = rng.normal(size=(30, 12))
    y = x @ w + 0.5 * rng.normal(size=(200, 12))
    model = RidgePerOutput(np.logspace(-2, 4, 7)).fit(x[:150], y[:150])
    r = pearson_per_output(y[150:], model.predict(x[150:]))
    assert r.shape == (12,) and r.min() > 0.9
    assert np.all(np.isfinite(noise_normalized(r, np.full(12, 0.5))))
    wm = window_means(r, np.linspace(-0.2, 0.79, 12), 100)
    assert len(wm) > 0
    ident = identification(y[150:], model.predict(x[150:]))
    assert all(0 <= v <= 1 for v in ident.values() if np.isscalar(v))
    preds_tr = [y[:150] + rng.normal(0, s, (150, 12)) for s in (0.5, 5.0)]
    preds_te = [y[150:] + rng.normal(0, s, (50, 12)) for s in (0.5, 5.0)]
    assert best_source(preds_tr, y[:150]) == 0
    assert gu_ensemble(preds_tr, y[:150], preds_te, np.repeat(np.arange(3), 4)).shape == (50, 3)
    assert eeg2eeg_converter(preds_tr[0], y[:150], preds_te[0], np.logspace(-2, 4, 7), 0).shape == (50, 12)


