"""The shared core of the EEG1 confirmation and the adapters that shrink towards the group, on planted data."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cross_subject_encoding.models.shared_core import MultiSubjectEncoder, latent, train_shared_images  # noqa: E402


def test_kpca_mlp_core_transfers():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((300, 50)).astype(np.float32)
    a = rng.standard_normal((50, 8)).astype(np.float32) / 7
    z = np.tanh(x @ a)
    ys = {}
    for p in range(4):
        y = z @ rng.standard_normal((8, 20)).astype(np.float32)
        ys[f"p{p}"] = torch.as_tensor(y + 0.3 * rng.standard_normal(y.shape).astype(np.float32))
    torch.manual_seed(7)
    model = MultiSubjectEncoder(50, {p: 20 for p in ("p0", "p1", "p2")})
    out = train_shared_images(model, torch.as_tensor(x), {p: ys[p] for p in ("p0", "p1", "p2")}, np.arange(200), np.arange(200, 230),
                              epochs=30, lr=3e-3, batch=50, seed=7, log=lambda *_: None)
    from cross_subject_encoding.models.ridge import RidgePerOutput
    lat = latent(model, torch.as_tensor(x))
    k, test = np.arange(60), np.arange(230, 300)
    y3 = ys["p3"].numpy()
    alphas = [10.0 ** e for e in range(-2, 5)]
    core = RidgePerOutput(alphas, 5, 7).fit(lat[k], y3[k]).predict(lat[test])
    raw = RidgePerOutput(alphas, 5, 7).fit(x[k], y3[k]).predict(x[test])
    r = lambda p: float(np.median([np.corrcoef(p[:, j], y3[test, j])[0, 1] for j in range(20)]))
    print(f"KPCA-MLP: held-out {out['best_score']:.3f}; new person with 60 images: core {r(core):.3f}, ridge on inputs {r(raw):.3f}")
    assert r(core) > r(raw)


def test_adapters_shrink_towards_the_group():
    from cross_subject_encoding.models.shared_core import adapter_predictions
    rng = np.random.default_rng(3)
    z = rng.standard_normal((400, 10))
    w_mean = rng.standard_normal((6, 10))
    w_new = w_mean + 0.2 * rng.standard_normal((6, 10))  # the new person is close to the group
    y = z @ w_new.T + 0.8 * rng.standard_normal((400, 6))
    k, test = np.arange(20), np.arange(200, 400)
    p = adapter_predictions(z[k], y[k], z[test], w_mean, np.zeros(6), [10.0 ** e for e in range(-2, 5)], 5, 3)
    truth = z[test] @ w_new.T
    r = {n: float(np.mean([np.corrcoef(v[:, j], truth[:, j])[0, 1] for j in range(6)])) for n, v in p.items()}
    print({n: round(v, 3) for n, v in r.items()})
    assert r["prior_ridge"] > r["fresh"] and r["mean"] > 0.8


def test_prior_feature_ridge_recovers_what_the_core_misses():
    from cross_subject_encoding.models.shared_core import adapter_predictions, prior_feature_ridge
    rng = np.random.default_rng(5)
    x = rng.standard_normal((900, 30))          # the full features
    lat = x[:, :8]                               # the core keeps only 8 shared directions
    w_mean = rng.standard_normal((5, 8))
    own = rng.standard_normal((5, 30)) * 0.5     # the person also uses directions the core dropped
    y = lat @ w_mean.T + x @ own.T + 0.5 * rng.standard_normal((900, 5))
    k, test = np.arange(600), np.arange(700, 900)
    al = [10.0 ** e for e in range(-2, 5)]
    m_k, m_t = lat[k] @ w_mean.T, lat[test] @ w_mean.T
    pr = adapter_predictions(lat[k], y[k], lat[test], w_mean, np.zeros(5), al, 5, 1)["prior_ridge"]
    pk = prior_feature_ridge(m_k, m_t, x[k], x[test], y[k], al, 5, 1)
    truth = y[test]
    r = lambda p: float(np.mean([np.corrcoef(p[:, j], truth[:, j])[0, 1] for j in range(5)]))
    print(f"prior_ridge {r(pr):.3f}, prior_kpca {r(pk):.3f}")
    assert r(pk) > r(pr) + 0.05
