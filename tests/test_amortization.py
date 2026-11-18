"""Hand-checkable amortization, SMM/CPR and pool-runoff arithmetic."""
import numpy as np
import pandas as pd
import pytest

from mval import amortization as am


def test_smm_cpr_round_trip():
    cpr = np.array([0.0, 0.03, 0.06, 0.25, 0.60])
    assert am.cpr_from_smm(am.smm_from_cpr(cpr)) == pytest.approx(cpr)
    assert am.cpr_from_smm(0.005) == pytest.approx(1 - 0.995 ** 12)


def test_scheduled_balance_endpoints_and_level_payment():
    B0, rate, term = 300_000.0, 6.0, 360
    assert am.scheduled_balance(B0, rate, term, 0) == pytest.approx(B0)
    assert am.scheduled_balance(B0, rate, term, term) == pytest.approx(0.0, abs=1e-6)
    r = rate / 1200
    payment = B0 * r / (1 - (1 + r) ** -term)          # textbook level payment: 1,798.65
    assert payment == pytest.approx(1798.65, abs=0.01)
    first_principal = payment - B0 * r
    assert B0 - am.scheduled_balance(B0, rate, term, 1) == pytest.approx(first_principal)


def test_psa_ramp():
    assert am.psa_cpr(15) == pytest.approx(0.03)
    assert am.psa_cpr(30) == pytest.approx(0.06)
    assert am.psa_cpr(120, speed=2.0) == pytest.approx(0.12)


def _pool(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"balance": rng.uniform(1e5, 5e5, n), "note_rate": rng.uniform(3.0, 7.0, n),
                         "months_remaining": rng.integers(120, 360, n).astype(float)})


def test_pool_projection_without_prepayment_follows_schedule():
    pool = _pool()
    months = pd.DataFrame({"month": [f"m{k}" for k in range(1, 25)]})
    zero = lambda a, k: pd.DataFrame({"h_refi": 0.0, "h_sale": 0.0, "h_default": 0.0}, index=a.index)
    proj = am.pool_projection(pool, zero, months, 24)
    assert proj.prepaid_principal.sum() == pytest.approx(0.0)
    expected_end = sum(am.scheduled_balance(b, r, n, 24) for b, r, n in pool.itertuples(index=False))
    assert proj.end_balance.iloc[-1] == pytest.approx(expected_end, rel=1e-9)
    assert (proj.cpr == 0).all()


def test_pool_projection_recovers_a_constant_cpr_and_conserves_principal():
    pool = _pool(seed=1)
    months = pd.DataFrame({"month": [f"m{k}" for k in range(1, 13)]})
    smm = float(am.smm_from_cpr(0.08))
    flat = lambda a, k: pd.DataFrame({"h_refi": smm / 2, "h_sale": smm / 2, "h_default": 0.0}, index=a.index)
    proj = am.pool_projection(pool, flat, months, 12)
    assert proj.cpr.to_numpy() == pytest.approx(0.08)
    paid = proj.sched_principal.sum() + proj.prepaid_principal.sum() + proj.defaulted_principal.sum()
    assert pool.balance.sum() - proj.end_balance.iloc[-1] == pytest.approx(paid)
    slower = am.pool_projection(pool, lambda a, k: flat(a, k) * 0.5, months, 12)
    assert slower.end_balance.iloc[-1] > proj.end_balance.iloc[-1]      # slower prepayment, more balance left
