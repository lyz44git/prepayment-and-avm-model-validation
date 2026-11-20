"""Unit checks of the credit-loss helpers on simulated inputs (no data needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mval.p3 import mi, era_of, balance_share, macro_monthly, start_survival  # noqa: E402


def test_month_index_and_eras():
    assert mi(2000, 1) == 0 and mi(2026, 9) == 320
    assert list(era_of([2000, 2004, 2005, 2008, 2009, 2013, 2014, 2019, 2020, 2026])) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_balance_share_amortises_to_zero():
    b = balance_share(np.array([6.0, 6.0, 0.0]), np.array([120, 120, 120]), np.array([0, 120, 60]))
    assert b[0] == pytest.approx(1.0) and b[1] == pytest.approx(0.0, abs=1e-12) and b[2] == pytest.approx(0.5)


def test_macro_reverts_in_a_straight_line_after_the_scenario():
    rows = [{"year": y, "q": q, "ur": 5.0, "dur4": 0.0, "bbb_spread": 2.0, "prime": 4.0, "dprime4": 0.0}
            for y in range(2006, 2026) for q in range(1, 5)]
    rows += [{"year": 2026, "q": 1, "ur": 9.0, "dur4": 4.0, "bbb_spread": 4.0, "prime": 3.0, "dprime4": -1.0}]
    m = macro_monthly(pd.DataFrame(rows), last_month=mi(2027, 12), revert_quarters=4)
    lr = m.attrs["long_run"]["ur"]
    end = mi(2026, 3)
    assert m.loc[end, "ur"] == 9.0
    assert m.loc[end + 6, "ur"] == pytest.approx(9.0 + (lr - 9.0) * 0.5)
    assert m.loc[end + 12, "ur"] == pytest.approx(lr) and m.loc[end + 18, "ur"] == pytest.approx(lr)


def _toy_scenario():
    """Flat history, then a shock in the last quarter (2026Q1), monthly table to the end of 2028."""
    rows = [{"year": y, "q": q, "ur": 5.0, "dur4": 0.0, "bbb_spread": 2.0, "prime": 4.0, "dprime4": 0.0}
            for y in range(2006, 2026) for q in range(1, 5)]
    rows += [{"year": 2026, "q": 1, "ur": 9.0, "dur4": 4.0, "bbb_spread": 4.0, "prime": 3.0, "dprime4": -1.0}]
    return pd.DataFrame(rows), mi(2029, 12), mi(2026, 3)


def test_scenario_changes_follow_the_reverted_levels():
    """Finding L16: after the scenario, the 4-quarter changes are the 12-month differences of the reverted levels."""
    q, last, end = _toy_scenario()
    m = macro_monthly(q, last_month=last)
    for chg, lev in (("dur4", "ur"), ("dprime4", "prime")):
        after = m.loc[end + 1:, chg]
        diff = (m[lev] - m[lev].shift(12)).loc[end + 1:]
        assert np.allclose(after, diff)
    assert m.loc[end + 12, "dur4"] == pytest.approx(m.loc[end + 12, "ur"] - 9.0)   # a full year of reversion
    assert m.loc[end + 36, "dur4"] == pytest.approx(0.0)                             # flat level, no change
    old = macro_monthly(q, last_month=last, changes_from_levels=False)
    assert old.loc[:end].equals(m.loc[:end])                                         # history and scenario untouched
    assert old.loc[end + 36, "dur4"] == pytest.approx(old.attrs["long_run"]["dur4"])  # the first runs reverted the change itself


def test_start_survival_compounds_earlier_months_only():
    idx = np.array([0, 0, 0, 1, 1])
    hco = np.array([0.1, 0.2, 0.1, 0.5, 0.6])
    hpf = np.array([0.1, 0.8, 0.1, 0.1, 0.1])
    s, n_over, mx = start_survival(idx, hco, hpf)
    assert np.allclose(s, [1.0, 0.8, 0.0, 1.0, 0.4])       # month 2 of loan 0 keeps S = 0.8 although its own term is 0
    assert n_over == 0 and mx == pytest.approx(1.0)
    s2, n2, _ = start_survival(np.array([0, 0]), np.array([0.7, 0.1]), np.array([0.6, 0.1]))
    assert n2 == 1 and np.allclose(s2, [1.0, 0.0])           # hazards summing above one are counted and clipped
