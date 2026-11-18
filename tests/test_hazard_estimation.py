"""Checks of the estimator used for the logit hazard (scripts/08), on simulated data.

Script 08 fits each cause on all events plus a 10% sample of non-events and adds log(0.10) to the intercept
(King and Zeng 2001). These tests simulate a monthly hazard with a known intercept and slopes and confirm that
the corrected case-control fit recovers them and reproduces the population event rate.
"""
import numpy as np
import pytest
import statsmodels.api as sm

from mval.p1 import ns_basis

Q = 0.10


def _simulate(n=400_000, seed=0):
    rng = np.random.default_rng(seed)
    gap = rng.normal(1.0, 1.5, n)                       # owner's rate advantage, percentage points
    ltv = rng.uniform(0.2, 0.95, n)
    X = np.column_stack([np.ones(n), np.maximum(gap, 0), ltv])
    beta = np.array([-5.6, -0.115, -1.0])               # monthly hazard near 0.2%, odds ratio 0.89 per point of gap
    y = (rng.random(n) < 1 / (1 + np.exp(-X @ beta))).astype(float)
    return X, y, beta, rng


def test_case_control_with_prior_correction_recovers_the_population_hazard():
    X, y, beta, rng = _simulate()
    keep = (y == 1) | (rng.random(len(y)) < Q)
    res = sm.GLM(y[keep], X[keep], family=sm.families.Binomial()).fit()
    b = res.params.copy()
    uncorrected_rate = (1 / (1 + np.exp(-X @ b))).mean()
    b[0] += np.log(Q)
    assert abs(b[0] - beta[0]) < 4 * res.bse[0]
    assert abs(b[1] - beta[1]) < 4 * res.bse[1]
    assert abs(b[2] - beta[2]) < 4 * res.bse[2]
    corrected_rate = (1 / (1 + np.exp(-X @ b))).mean()
    assert corrected_rate == pytest.approx(y.mean(), rel=0.05)
    assert uncorrected_rate > 5 * y.mean()              # without the correction the hazard is overstated ~10x


def test_case_control_matches_the_full_data_fit():
    X, y, _, rng = _simulate(seed=1)
    full = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    keep = (y == 1) | (rng.random(len(y)) < Q)
    cc = sm.GLM(y[keep], X[keep], family=sm.families.Binomial()).fit()
    b = cc.params.copy(); b[0] += np.log(Q)
    assert np.abs(b - full.params).max() < 3 * cc.bse.max()


def test_natural_spline_is_linear_beyond_the_boundary_knots():
    knots = np.array([0.5, 3.0, 7.0, 12.5, 20.0, 30.0])
    x = np.linspace(31, 60, 30)
    B = ns_basis(x, knots)
    assert B.shape == (30, len(knots) - 1)
    second_diff = np.diff(B, n=2, axis=0)
    assert np.abs(second_diff).max() < 1e-8             # every basis function is a straight line past the last knot
    below = ns_basis(np.linspace(0.0, 0.4, 5), knots)
    assert np.abs(below[:, 1:]).max() < 1e-12           # and only the linear term is non-zero before the first
