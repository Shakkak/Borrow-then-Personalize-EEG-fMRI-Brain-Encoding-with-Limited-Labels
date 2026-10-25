import numpy as np

from cross_subject_encoding.eval import lowrank_map as lm


def corr(p, y):
    p, y = p - p.mean(0), y - y.mean(0)
    return float(np.mean((p * y).sum(0) / np.sqrt((p ** 2).sum(0) * (y ** 2).sum(0))))


def test_lowrank_map_recovers_a_personal_low_rank_change():
    """Borrowed predictions live in a 10-dimensional image space over 500 outputs; the person's responses add a
    rank-3 re-mixing of that space. With 150 labelled images the K = 20 map should beat the plain average."""
    g = np.random.default_rng(0)
    n, v, d = 2000, 500, 10
    latent = g.normal(size=(n, d))
    avg = latent @ g.normal(size=(d, v))
    personal = avg + (avg @ g.normal(size=(v, 3)) @ g.normal(size=(3, v))) * 0.002
    y = personal + 0.5 * g.normal(size=(n, v))
    mean, basis = lm.prediction_basis(avg[:1500], 20)
    assert basis.shape == (v, 20) and np.allclose(basis.T @ basis, np.eye(20), atol=1e-8)
    k, test = np.arange(150), np.arange(1500, 2000)
    model = lm.fit_lowrank_map(avg[k], y[k], mean, basis)
    pred = lm.apply_lowrank_map(model, avg[test], mean, basis)
    mse = lambda a: float(((a - personal[test]) ** 2).mean())
    assert mse(pred) < 0.5 * mse(avg[test])  # most of the personal change recovered


def test_no_personal_change_means_no_harm():
    g = np.random.default_rng(1)
    latent = g.normal(size=(1200, 8))
    avg = latent @ g.normal(size=(8, 200))
    y = avg + 0.5 * g.normal(size=avg.shape)
    mean, basis = lm.prediction_basis(avg[:1000], 20)
    model = lm.fit_lowrank_map(avg[:100], y[:100], mean, basis)
    pred = lm.apply_lowrank_map(model, avg[1000:], mean, basis)
    assert corr(pred, avg[1000:]) > 0.98  # cross-validated ridge keeps the correction small


def test_basis_from_gram_equals_svd_basis():
    g = np.random.default_rng(2)
    x = g.normal(size=(60, 400)) @ np.diag(np.linspace(3, 0.1, 400))
    mean, b_gram = lm.prediction_basis(x, 10)  # 60 images < 400 outputs: the Gram route
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    assert np.allclose(np.abs(np.sum(b_gram * vt[:10].T, 0)), 1.0, atol=1e-6)  # same directions up to sign


def test_grouped_folds_keep_repeats_together_and_none_is_unchanged():
    """14e: with `groups`, every image's repeats share a fold; without it the folds (and penalties) are as before."""
    from cross_subject_encoding.models.ridge import RidgePerOutput
    g = np.random.default_rng(3)
    images = np.repeat(np.arange(40), 3)
    x = g.normal(size=(40, 6))[images] + 0.1 * g.normal(size=(120, 6))
    y = x @ g.normal(size=(6, 5)) + g.normal(size=(120, 5))
    a = RidgePerOutput(lm.ALPHAS, 5, 13).fit(x, y)
    b = RidgePerOutput(lm.ALPHAS, 5, 13).fit(x, y, None)
    assert np.array_equal(a.best_alpha_, b.best_alpha_) and np.allclose(a.coef_, b.coef_)
    seen = []
    orig = np.isin
    def spy(el, test, *args, **kw):
        out = orig(el, test, *args, **kw)
        seen.append(out)
        return out
    np.isin = spy
    try:
        RidgePerOutput(lm.ALPHAS, 5, 13).fit(x, y, images)
    finally:
        np.isin = orig
    folds = [np.flatnonzero(m) for m in seen]
    assert len(folds) == 5 and sum(len(f) for f in folds) == 120
    for f in folds:
        assert set(np.flatnonzero(np.isin(images, images[f]))) == set(f)  # no image split across folds
