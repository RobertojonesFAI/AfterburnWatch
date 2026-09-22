import numpy as np
import pandas as pd
import pytest

from afterburn_watch.predict import (
    M1Coefficients,
    evaluate,
    likelihood,
    rainfall_threshold,
    recalibrate,
)


def _data(rng, coefs, n):
    T, F, S = rng.uniform(0, 0.8, n), rng.uniform(0.1, 0.9, n), rng.uniform(0.1, 0.4, n)
    R = rng.uniform(2, 20, n)
    B, Ct, Cf, Cs = coefs
    p = 1 / (1 + np.exp(-(B + R * (Ct * T + Cf * F + Cs * S))))
    return pd.DataFrame({"T": T, "F": F, "S": S, "R": R, "response": (rng.random(n) < p).astype(int)})


def test_staley_coefficients_and_likelihood_formula():
    c = M1Coefficients.staley2017(15)
    assert c.theta.tolist() == [-3.63, 0.41, 0.67, 0.70]
    x = -3.63 + 10 * (0.41 * 0.4 + 0.67 * 0.5 + 0.70 * 0.25)
    assert likelihood(c, 0.4, 0.5, 0.25, 10)[0] == pytest.approx(1 / (1 + np.exp(-x)))
    with pytest.raises(ValueError):
        M1Coefficients.staley2017(45)


def test_rainfall_threshold_inverts_likelihood():
    c = M1Coefficients.staley2017(15)
    T, F, S = np.array([0.4, 0.1]), np.array([0.5, 0.2]), np.array([0.25, 0.3])
    R = rainfall_threshold(c, T, F, S, p=0.5)
    np.testing.assert_allclose(likelihood(c, T, F, S, R), 0.5, rtol=1e-9)
    assert np.isnan(rainfall_threshold(c, [0.0], [0.0], [0.0])[0])


def test_recalibrate_recovers_truth_with_enough_data():
    rng = np.random.default_rng(1)
    truth = (-4.5, 0.25, 1.10, 0.20)
    new, info = recalibrate(M1Coefficients.staley2017(), _data(rng, truth, 8000), prior_strength=0.1)
    assert info["updated"]
    np.testing.assert_allclose(new.theta, truth, atol=0.25)
    assert new.source.startswith("recalibrated_from:")


def test_strong_prior_barely_moves():
    rng = np.random.default_rng(2)
    prior = M1Coefficients.staley2017()
    new, _ = recalibrate(prior, _data(rng, (-4.5, 0.25, 1.10, 0.20), 200), prior_strength=1e6)
    np.testing.assert_allclose(new.theta, prior.theta, atol=0.01)


def test_recalibrate_guards():
    prior = M1Coefficients.staley2017()
    rng = np.random.default_rng(3)
    small = _data(rng, prior.theta, 10)
    same, info = recalibrate(prior, small, min_obs=30)
    assert same is prior and not info["updated"]
    one_class = _data(rng, prior.theta, 50).assign(response=1)
    same, info = recalibrate(prior, one_class)
    assert same is prior and "one response class" in info["reason"]


def test_evaluate_prefers_the_generating_model():
    rng = np.random.default_rng(4)
    truth = (-4.5, 0.25, 1.10, 0.20)
    d = _data(rng, truth, 3000)
    good = evaluate(M1Coefficients(*truth), d)
    published = evaluate(M1Coefficients.staley2017(), d)
    assert good["log_loss"] < published["log_loss"]
    assert set(good) == {"n", "brier", "log_loss", "auc"}
