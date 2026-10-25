"""The pre-onset noise floor separates planted signal from the clipping bias."""
import numpy as np

from cross_subject_encoding.eval.noise_ceiling import noise_ceiling, reliable


def test_floor_keeps_signal_and_drops_noise():
    rng = np.random.default_rng(0)
    times = np.linspace(-0.2, 0.79, 20)
    test = rng.normal(size=(200, 80, 3, 20))
    test[:, :, :, times > 0.1] += rng.normal(size=(200, 1, 3, 1)) * 0.5  # image-specific signal after 100 ms
    max_r = noise_ceiling(test, 20, 0, (80,))["max_r_80"][None]
    pre = max_r[..., times < 0]
    assert 0.05 < pre.mean() < 0.25  # the bias: positive on pure noise
    mask = reliable(max_r, times)
    assert mask[..., times > 0.1].mean() > 0.95
    assert mask[..., (times >= 0) & (times <= 0.1)].mean() < 0.2
