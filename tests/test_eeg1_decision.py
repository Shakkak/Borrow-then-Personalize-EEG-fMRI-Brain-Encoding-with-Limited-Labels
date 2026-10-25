import numpy as np

from cross_subject_encoding.eval import eeg1_decision as dec


def make(mean, sd=0.005, n=24, seed=0):
    return np.random.default_rng(seed).normal(mean, sd, size=(n, 5))


def test_verdicts():
    assert dec.decide(make(0.05), 0.0035)["verdict"] == "supported"
    assert dec.decide(make(-0.02), 0.0035)["verdict"] == "rejected"
    d = make(0.05)
    d[:, 0] = make(-0.02)[:, 0]  # one comparator clearly below the margin rejects the all-five claim
    assert dec.decide(d, 0.0035)["verdict"] == "rejected"
    assert dec.decide(make(0.0035, sd=0.02), 0.0035)["verdict"] == "inconclusive"


def test_bonferroni_t_and_bounds():
    out = dec.decide(make(0.01), 0.0035)
    from scipy import stats
    assert np.isclose(out["t_critical"], stats.t.ppf(1 - 0.05 / 5, 23))
    c = out["per_comparator"]["avg_sources"]
    assert c["lower"] < c["mean"] < c["upper"]


def test_missing_targets_excluded_and_minimum():
    d = make(0.05)
    d[0, 2] = np.nan
    out = dec.decide(d, 0.0035)
    assert out["n_targets"] == 23 and out["n_excluded"] == 1 and out["verdict"] == "supported"
    d[:5, 0] = np.nan
    assert dec.decide(d, 0.0035)["verdict"] == "insufficient"


def test_endpoint_averages_budgets_and_flags_missing():
    scores = {("a", "prior_both", 250): 0.3, ("a", "prior_both", 500): 0.4}
    for name in dec.COMPARATORS:
        scores[("a", name, 250)], scores[("a", name, 500)] = 0.2, 0.2
    d = dec.endpoint(scores, ["a", "b"])
    assert np.allclose(d[0], 0.15) and np.isnan(d[1]).all()


def test_sign_flip_descriptive():
    assert dec.sign_flip_p(np.full(24, 0.01)) < 0.001
    assert dec.sign_flip_p(np.full(24, -0.01)) > 0.99
