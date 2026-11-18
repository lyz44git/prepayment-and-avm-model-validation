"""Unit checks of the credit-loss helpers on simulated inputs (no data needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mval.p3 import mi, era_of, balance_share, macro_monthly  # noqa: E402


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
