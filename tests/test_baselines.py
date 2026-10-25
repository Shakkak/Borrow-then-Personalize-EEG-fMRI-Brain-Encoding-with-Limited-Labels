"""The self-check and the baselines' helpers, on small planted cases."""
import json

import numpy as np

from cross_subject_encoding.utils.selfcheck import check, headline


def test_selfcheck_statuses(tmp_path):
    rows = [{"measure": "in range", "measured": 0.5, "expected": "0.4 to 0.6", "ok": (0.4, 0.6), "look_again": ""},
            {"measure": "out of range", "measured": 0.9, "expected": "below 0.6", "ok": (None, 0.6), "look_again": "x"},
            {"measure": "callable", "measured": 3, "expected": "odd", "ok": lambda v: v % 2 == 1, "look_again": ""},
            {"measure": "no range", "measured": 7, "expected": "", "ok": None, "look_again": ""},
            {"measure": "missing", "measured": None, "expected": "", "ok": (0, 1), "look_again": ""}]
    got = [r["status"] for r in check("99", rows, tmp_path)]
    assert got == ["OK", "LOOK AGAIN", "OK", "REPORTED", "NOT MEASURED"]
    assert (tmp_path / "99_selfcheck.csv").read_text().count("\n") == 6
    h = json.loads(headline("99", {"a": 0.123456, "b": "text"}, tmp_path).read_text())
    assert h == {"a": 0.1235, "b": "text"}


def test_gu_per_output_matches_the_per_group_loop():
    from cross_subject_encoding.models.baselines import gu_ensemble, gu_per_output

    rng = np.random.default_rng(1)
    src_tr, src_te = rng.normal(size=(3, 40, 6)), rng.normal(size=(3, 10, 6))
    y = 0.7 * src_tr[0] - 0.2 * src_tr[2] + 0.1 + 0.05 * rng.normal(size=(40, 6))
    loop = gu_ensemble(list(src_tr), y, list(src_te), np.arange(6))
    vec = gu_per_output(src_tr, y, src_te)
    assert np.allclose(loop, vec, atol=1e-4)


def test_ridge_refit_keeps_penalties_and_matches_a_fresh_fit():
    from cross_subject_encoding.models.ridge import RidgePerOutput

    rng = np.random.default_rng(3)
    x = rng.normal(size=(120, 10))
    y = x @ rng.normal(size=(10, 4)) + rng.normal(size=(120, 4))
    m = RidgePerOutput([0.1, 1.0, 10.0, 100.0], 5, 13).fit(x, y)
    r = m.refit(x, y)
    assert np.array_equal(r.best_alpha_, m.best_alpha_)
    assert np.allclose(r.predict(x), m.predict(x), atol=1e-4)
    half = m.refit(x[:60], y[:60])  # new data, same penalties
    assert half.coef_.shape == m.coef_.shape and not np.allclose(half.coef_, m.coef_)


def test_ols_oof_matches_a_fit_per_fold():
    from cross_subject_encoding.models.baselines import ols_oof
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(120, 6)), rng.normal(size=(120, 5))
    got = ols_oof(x, y, n_folds=4, seed=3, block=2)
    folds = np.array_split(np.random.default_rng(3).permutation(120), 4)
    want = np.empty_like(y)
    for f in folds:
        keep = np.setdiff1d(np.arange(120), f)
        a = np.column_stack([np.ones(len(keep)), x[keep]])
        beta, *_ = np.linalg.lstsq(a, y[keep], rcond=None)
        want[f] = np.column_stack([np.ones(len(f)), x[f]]) @ beta
    assert np.allclose(got, want, atol=1e-5)


def test_gu_by_group_uses_sklearn_on_group_means():
    from sklearn.linear_model import LinearRegression
    from cross_subject_encoding.models.baselines import gu_by_group, window_groups
    rng = np.random.default_rng(1)
    times = np.round(np.arange(-0.2, 0.8, 0.01), 2)
    groups = window_groups(times, 3, 100)
    assert groups.shape == (300,) and len(np.unique(groups)) == 10 and groups[0] == 0 and groups[99] == 9 and groups[100] == 0
    s_tr, s_te = rng.normal(size=(4, 30, 300)), rng.normal(size=(4, 7, 300))
    y = 0.5 * s_tr[0] - 0.2 * s_tr[2] + rng.normal(size=(30, 300))
    got = gu_by_group(s_tr, y, s_te, groups)
    cols = groups == 3
    w = LinearRegression().fit(s_tr[:, :, cols].mean(2).T, y[:, cols].mean(1)).coef_
    want = np.einsum("s,smo->mo", w, s_te[:, :, cols])
    # each output's intercept only shifts it: the per-output deviation from the test mean is what must match
    assert np.allclose(got[:, cols] - got[:, cols].mean(0), want - want.mean(0), atol=1e-5)
    # pooled: one set of weights from every output of the group, each centred over the k images
    pooled = gu_by_group(s_tr, y, s_te, groups, fit="pooled")
    a = (s_tr[:, :, cols] - s_tr[:, :, cols].mean(1, keepdims=True)).reshape(4, -1).T
    w2, *_ = np.linalg.lstsq(a, (y[:, cols] - y[:, cols].mean(0)).ravel(), rcond=None)
    want2 = np.einsum("s,smo->mo", w2, s_te[:, :, cols])
    assert np.allclose(pooled[:, cols] - pooled[:, cols].mean(0), want2 - want2.mean(0), atol=1e-4)
    assert np.allclose(w2, [0.5, 0, -0.2, 0], atol=0.1)  # the planted weights come back


def test_ols_without_rows_matches_a_refit():
    from cross_subject_encoding.models.baselines import ols_without_rows
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(80, 5)), rng.normal(size=(80, 3))
    x1 = np.column_stack([np.ones(80), x])
    drop = np.arange(10, 25)
    keep = np.setdiff1d(np.arange(80), drop)
    want, *_ = np.linalg.lstsq(x1[keep], y[keep], rcond=None)
    got = ols_without_rows(x1.T @ x1, x1.T @ y, x1[drop], y[drop])
    assert np.allclose(got, want, atol=1e-8)
