"""Hand-checkable metric arithmetic."""

from __future__ import annotations

import numpy as np
import pytest

from mval import metrics


def test_ks_hand_built():
    assert metrics.ks_stat([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert metrics.ks_stat([0, 1, 0, 1], [1, 2, 3, 4]) == pytest.approx(0.5)


def test_auc_and_gini():
    y = [0, 0, 1, 1, 0, 1]
    s = [0.1, 0.4, 0.35, 0.8, 0.2, 0.9]
    a = metrics.auc(y, s)
    assert a == pytest.approx(8 / 9)
    assert metrics.gini(a) == pytest.approx(2 * a - 1)
    point, lo, hi = metrics.auc_ci(y, s, n_boot=100, seed=0)
    assert point == pytest.approx(a) and lo <= point <= hi


def test_psi_identical_and_shifted():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    assert metrics.psi(ref, ref) == pytest.approx(0.0)
    assert metrics.psi(ref, ref + 1.0) > 0.25
    assert metrics.psi(ref, rng.normal(size=5000)) < 0.05


def test_lift_table_sums_and_capture():
    rng = np.random.default_rng(1)
    s = rng.random(1000)
    y = (rng.random(1000) < s).astype(int)
    lt = metrics.lift_table(y, s, n_bins=10)
    assert lt["n"].sum() == 1000 and lt["n"].tolist() == [100] * 10
    assert lt["events"].sum() == y.sum()
    assert lt["cum_capture"].iloc[-1] == pytest.approx(1.0)
    assert lt["lift"].iloc[0] > 1 > lt["lift"].iloc[-1]
    assert (lt["rate"] * lt["n"]).sum() == pytest.approx(y.sum())


def test_precision_at_k_and_brier():
    y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    s = [0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    assert metrics.precision_at_k(y, s, k=0.2) == pytest.approx(1.0)
    assert metrics.precision_at_k(y, s, k=0.3) == pytest.approx(2 / 3)
    assert metrics.brier([1, 0], [1.0, 0.0]) == pytest.approx(0.0)
    assert metrics.brier([1, 0], [0.5, 0.5]) == pytest.approx(0.25)


def test_paired_delta_zero_for_identical_scores():
    rng = np.random.default_rng(2)
    s = rng.random(500)
    y = (rng.random(500) < s).astype(int)
    d = metrics.paired_bootstrap_delta_auc(y, s, s, n_boot=50, seed=0)
    assert d["delta"] == pytest.approx(0.0) and d["lo"] == pytest.approx(0.0) and d["hi"] == pytest.approx(0.0)
    better = metrics.paired_bootstrap_delta_auc(y, rng.random(500), s, n_boot=50, seed=0)
    assert better["delta"] > 0 and better["lo"] > 0


def test_calibration_table_shape():
    rng = np.random.default_rng(3)
    s = rng.random(2000)
    y = (rng.random(2000) < s).astype(int)
    ct = metrics.calibration_table(y, s, n_bins=10)
    assert len(ct) == 10 and ct["n"].sum() == 2000
    assert ct["mean_score"].is_monotonic_decreasing


def test_ks_is_tie_aware():
    # all scores tied: the two ECDFs coincide at the only distinct value -> KS = 0, whatever the row order
    assert metrics.ks_stat([0, 1, 0, 1], [5, 5, 5, 5]) == pytest.approx(0.0)
    assert metrics.ks_stat([1, 0, 1, 0], [5, 5, 5, 5]) == pytest.approx(0.0)
    # two tie blocks with the same event rate: still 0 (the positional version reported 0.5)
    assert metrics.ks_stat([0, 1, 0, 1], [1, 1, 2, 2]) == pytest.approx(0.0)
    # two tie blocks that separate the classes perfectly
    assert metrics.ks_stat([0, 0, 1, 1], [1, 1, 2, 2]) == pytest.approx(1.0)
    # coarse integer score with heavy ties: equals scipy's two-sample KS and is invariant to row order
    from scipy.stats import ks_2samp

    rng = np.random.default_rng(0)
    s = rng.integers(1, 20, 3000).astype(float)
    s[rng.random(3000) < 0.4] = 10.0  # imputed-median block
    y = (rng.random(3000) < 0.05 + 0.01 * s).astype(int)
    ks = metrics.ks_stat(y, s)
    assert ks == pytest.approx(ks_2samp(s[y == 1], s[y == 0]).statistic)
    perm = rng.permutation(3000)
    assert metrics.ks_stat(y[perm], s[perm]) == pytest.approx(ks)


def test_tied_scores_split_by_seeded_key_not_row_order():
    # 200 tied scores at the top-decile boundary: the positional rule would select the first rows
    # (all events here); the seeded rule selects a random subset, reproducibly
    s = np.r_[np.full(200, 0.9), np.linspace(0, 0.5, 800)]
    y = np.r_[np.ones(100), np.zeros(100), np.zeros(800)].astype(int)
    p = metrics.precision_at_k(y, s, k=0.1)
    assert 0.2 < p < 0.8
    assert metrics.precision_at_k(y, s, k=0.1) == pytest.approx(p)  # reproducible
    assert metrics.precision_at_k(y, s, k=0.1, seed=1) != pytest.approx(p)  # but a tie-break, not a certainty
    lt = metrics.lift_table(y, s, n_bins=10)
    assert lt["n"].tolist() == [100] * 10 and 0.2 < lt["rate"].iloc[0] < 0.8
    assert lt.equals(metrics.lift_table(y, s, n_bins=10))
