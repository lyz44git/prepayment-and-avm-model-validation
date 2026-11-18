"""Level-pay amortization, SMM/CPR conversions and expected pool runoff.

Conventions: rates in percent per year, ``k`` in months since origination, balances in currency units.
``SMM`` is the single monthly mortality (share of the post-scheduled balance prepaid in a month) and
``CPR = 1 - (1 - SMM)^12`` its annualised equivalent; ``MDR`` is the analogous monthly default rate.
``pool_projection`` rolls every active loan forward with its own hazards: expected balance
``B_{t+1} = (B_t - SP_t) * (1 - SMM_t - MDR_t)`` and aggregates to pool cash flows.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ArrayLike = float | np.ndarray | pd.Series


def _monthly_rate(annual_rate_pct: ArrayLike) -> np.ndarray:
    return np.asarray(annual_rate_pct, dtype=float) / 1200.0


def scheduled_balance(B0: ArrayLike, annual_rate_pct: ArrayLike, term_months: ArrayLike, k: ArrayLike) -> np.ndarray:
    """Outstanding balance after ``k`` scheduled level payments (0 at and after maturity)."""
    B0 = np.asarray(B0, dtype=float)
    r = _monthly_rate(annual_rate_pct)
    n = np.asarray(term_months, dtype=float)
    k = np.clip(np.asarray(k, dtype=float), 0.0, None)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        growth_n = np.power(1.0 + r, n)
        growth_k = np.power(1.0 + r, k)
        bal = np.where(r > 0, B0 * (growth_n - growth_k) / (growth_n - 1.0), B0 * (1.0 - k / n))
    return np.where(k >= n, 0.0, np.maximum(bal, 0.0))


def scheduled_principal(B0: ArrayLike, annual_rate_pct: ArrayLike, term_months: ArrayLike, k: ArrayLike) -> np.ndarray:
    """Scheduled principal paid in month ``k`` (``B_{k-1} - B_k``)."""
    k = np.asarray(k, dtype=float)
    return scheduled_balance(B0, annual_rate_pct, term_months, k - 1) - scheduled_balance(B0, annual_rate_pct, term_months, k)


def scheduled_principal_from_balance(balance: ArrayLike, annual_rate_pct: ArrayLike, months_remaining: ArrayLike) -> np.ndarray:
    """Scheduled principal for the next payment given the current balance and remaining term."""
    bal = np.asarray(balance, dtype=float)
    r = _monthly_rate(annual_rate_pct)
    n = np.asarray(months_remaining, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        payment = np.where(r > 0, bal * r / (1.0 - np.power(1.0 + r, -np.maximum(n, 1.0))), bal / np.maximum(n, 1.0))
        principal = payment - bal * r
    return np.where(n <= 1, bal, np.clip(principal, 0.0, bal))


def smm_from_cpr(cpr: ArrayLike) -> np.ndarray:
    """SMM = 1 - (1 - CPR)^(1/12)."""
    return 1.0 - np.power(1.0 - np.asarray(cpr, dtype=float), 1.0 / 12.0)


def cpr_from_smm(smm: ArrayLike) -> np.ndarray:
    """CPR = 1 - (1 - SMM)^12."""
    return 1.0 - np.power(1.0 - np.asarray(smm, dtype=float), 12.0)


def psa_cpr(age_months: ArrayLike, speed: float = 1.0) -> np.ndarray:
    """PSA benchmark: CPR = speed * 0.2% * min(age, 30), i.e. 6% at month 30 for 100% PSA."""
    return speed * 0.002 * np.minimum(np.asarray(age_months, dtype=float), 30.0)


def pool_projection(
    active_loans: pd.DataFrame,
    hazard_fn: Callable[[pd.DataFrame, int], pd.DataFrame],
    scenario_path: pd.DataFrame,
    horizon: int = 36,
) -> pd.DataFrame:
    """Expected pool runoff under loan-level hazards.

    Parameters
    ----------
    active_loans : one row per active loan with ``balance`` (current), ``note_rate`` (pct),
        ``months_remaining`` (term minus age).
    hazard_fn : ``hazard_fn(active_loans, k)`` returns a DataFrame aligned to ``active_loans`` with
        columns ``h_refi, h_sale, h_default`` for projection month ``k`` (1-based).
    scenario_path : the scenario macro path (used for the month labels; must cover ``horizon`` rows).
    horizon : number of months to project.

    Returns
    -------
    DataFrame[month, k, balance, sched_principal, prepaid_principal, defaulted_principal, smm, cpr,
    wal_remaining] where ``balance`` is the pool balance at the start of month ``k`` and
    ``wal_remaining`` the weighted-average life (years) of the remaining pool, horizon-truncated with the
    balance left after the last projected month treated as a bullet at the horizon.
    """
    if len(scenario_path) < horizon:
        raise ValueError("scenario path shorter than horizon")
    bal = active_loans["balance"].to_numpy(dtype=float).copy()
    rate = active_loans["note_rate"].to_numpy(dtype=float)
    remaining = active_loans["months_remaining"].to_numpy(dtype=float).copy()
    rows: list[dict[str, float]] = []
    for k in range(1, horizon + 1):
        start_bal = bal.sum()
        sp = scheduled_principal_from_balance(bal, rate, remaining)
        after_sched = bal - sp
        h = hazard_fn(active_loans, k)
        smm_i = np.clip(h["h_refi"].to_numpy(float) + h["h_sale"].to_numpy(float), 0.0, 1.0)
        mdr_i = np.clip(h["h_default"].to_numpy(float), 0.0, 1.0 - smm_i)
        prepaid = after_sched * smm_i
        defaulted = after_sched * mdr_i
        bal = after_sched - prepaid - defaulted
        remaining = np.maximum(remaining - 1.0, 0.0)
        denom = after_sched.sum()
        smm_pool = prepaid.sum() / denom if denom > 0 else 0.0
        rows.append(
            {
                "month": scenario_path["month"].iloc[k - 1],
                "k": k,
                "balance": start_bal,
                "sched_principal": sp.sum(),
                "prepaid_principal": prepaid.sum(),
                "defaulted_principal": defaulted.sum(),
                "smm": smm_pool,
                "cpr": float(cpr_from_smm(smm_pool)),
                "end_balance": bal.sum(),
            }
        )
    out = pd.DataFrame(rows)
    out["wal_remaining"] = _weighted_average_life(out)
    return out


def _weighted_average_life(proj: pd.DataFrame) -> np.ndarray:
    """WAL (years) of the pool remaining at the start of each month, horizon-truncated with a bullet."""
    cf = (proj["sched_principal"] + proj["prepaid_principal"] + proj["defaulted_principal"]).to_numpy(float)
    horizon = len(proj)
    terminal = float(proj["end_balance"].iloc[-1])
    wal = np.zeros(horizon)
    for i in range(horizon):
        t = np.arange(1, horizon - i + 1) / 12.0
        flows = cf[i:]
        total = flows.sum() + terminal
        wal[i] = (np.dot(t, flows) + t[-1] * terminal) / total if total > 0 else 0.0
    return wal
