import numpy as np

from cross_subject_encoding.eval import eeg1_h4b as h4b


def make(seed=0, k=200, n=50, ch=5, t=20, mix_scale=0.6, noise=0.5):
    g = np.random.default_rng(seed)
    avg = g.normal(size=(k + n, ch, t))
    mix = np.eye(ch) + mix_scale * g.normal(size=(ch, ch))  # the person's channels are a mix of the average's
    y = np.einsum("ij,njt->nit", mix, avg) + noise * g.normal(size=(k + n, ch, t))
    return avg[:k], y[:k], avg[k:], y[k:], np.repeat(np.arange(2), t // 2)


def corr(p, y):
    p, y = p - p.mean(0), y - y.mean(0)
    return np.mean((p * y).sum(0) / np.sqrt((p ** 2).sum(0) * (y ** 2).sum(0)))


def test_recovers_a_channel_mix_and_beats_the_plain_average():
    a_k, y_k, a_t, y_t, win = make()
    pred, chosen = h4b.channel_map_predict(a_k, y_k, a_t, win)
    assert pred.shape == a_t.shape and set(chosen) == {0, 1}
    assert corr(pred, y_t) > corr(a_t, y_t) + 0.05


def test_no_mix_shrinks_to_the_average():
    a_k, y_k, a_t, y_t, win = make(mix_scale=0.0, noise=3.0)
    pred, chosen = h4b.channel_map_predict(a_k, y_k, a_t, win)
    assert abs(corr(pred, y_t) - corr(a_t, y_t)) < 0.02
    assert min(chosen.values()) >= 10.0  # heavy shrinkage when there is nothing to learn


def test_shuffled_pairs_learn_nothing_and_generic_map_transfers_a_shared_mix():
    a_k, y_k, a_t, y_t, win = make(seed=1)
    coefs, _ = h4b.fit_map(a_k, y_k, win, shuffle=True)
    # without real pairing the residual fit learns to discard the average: no gain, typically a collapse
    assert corr(h4b.apply_map(coefs, a_t, win), y_t) < corr(a_t, y_t)
    maps = [h4b.fit_map(*make(seed=1)[:2], win)[0] for _ in range(2)]  # same mix: a generic map should help
    assert corr(h4b.apply_map(h4b.mean_map(maps), a_t, win), y_t) > corr(a_t, y_t) + 0.05
