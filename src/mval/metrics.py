"""Discrimination, calibration and stability metrics.

All functions take plain array-likes and return floats or tidy DataFrames. AUC is computed from
the Mann–Whitney rank statistic (ties get average ranks), confidence intervals are percentile
bootstraps over households, and the paired bootstrap for ``ΔAUC`` resamples the *same* rows for
both scores so the difference is not swamped by sampling noise common to both models.

Tied scores (coarse vendor scores, median imputation) are handled explicitly: KS compares the
two ECDFs only at distinct score values, and decile / top-``k`` membership among tied rows is
decided by a seeded random key rather than by input order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _clean(y, s) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(s))
    return y[ok].astype(int), s[ok]


def auc(y, s) -> float:
    """Area under the ROC curve via ranks; ``nan`` if only one class is present."""
    y, s = _clean(y, s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_ci(y, s, n_boot: int = 200, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """``(auc, lo, hi)`` with a percentile bootstrap over observations."""
    y, s = _clean(y, s)
    point = auc(y, s)
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = auc(y[idx], s[idx])
    boots = boots[~np.isnan(boots)]
    if boots.size == 0:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_bootstrap_delta_auc(y, s_a, s_b, n_boot: int = 200, seed: int = 0, alpha: float = 0.05) -> dict[str, float]:
    """``AUC(s_b) - AUC(s_a)`` with a paired bootstrap CI (same resampled rows for both scores)."""
    y = np.asarray(y, dtype=float)
    s_a = np.asarray(s_a, dtype=float)
    s_b = np.asarray(s_b, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(s_a) | np.isnan(s_b))
    y, s_a, s_b = y[ok].astype(int), s_a[ok], s_b[ok]
    delta = auc(y, s_b) - auc(y, s_a)
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = auc(y[idx], s_b[idx]) - auc(y[idx], s_a[idx])
    boots = boots[~np.isnan(boots)]
    return {
        "delta": float(delta),
        "lo": float(np.quantile(boots, alpha / 2)) if boots.size else float("nan"),
        "hi": float(np.quantile(boots, 1 - alpha / 2)) if boots.size else float("nan"),
    }


def ks_stat(y, s) -> float:
    """Kolmogorov–Smirnov distance between the score distributions of events and non-events.

    Tie-aware: the two empirical CDFs are compared only at distinct score values (the last row of
    each tie block), so a coarse or median-imputed score cannot inflate KS through the arbitrary
    order of tied rows. Equivalent to ``scipy.stats.ks_2samp(s[y == 1], s[y == 0]).statistic``.
    """
    y, s = _clean(y, s)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    cum_pos = np.cumsum(y_sorted) / y_sorted.sum()
    cum_neg = np.cumsum(1 - y_sorted) / (len(y_sorted) - y_sorted.sum())
    last_in_block = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    return float(np.max(np.abs(cum_pos[last_in_block] - cum_neg[last_in_block])))


def gini(auc_value: float) -> float:
    """Gini coefficient ``2·AUC − 1``."""
    return 2.0 * auc_value - 1.0


def _order_desc(s: np.ndarray, seed: int = 0) -> np.ndarray:
    """Positions of ``s`` from highest to lowest score.

    Tie rule: rows with equal scores are ordered by a seeded random key rather than by their
    position in the input, so decile membership and top-``k`` selection among tied scores are
    unbiased with respect to the input order and reproducible for a given ``seed``.
    """
    rng = np.random.default_rng(seed)
    tie_key = rng.random(len(s))
    return np.lexsort((tie_key, -s))


def _score_bins(s: np.ndarray, n_bins: int, seed: int = 0) -> np.ndarray:
    """Equal-count bins of a score, 1 = highest scores; ties broken by :func:`_order_desc`."""
    order = _order_desc(s, seed)
    bins = np.empty(len(s), dtype=int)
    bins[order] = np.minimum(np.arange(len(s)) * n_bins // len(s), n_bins - 1) + 1
    return bins


def lift_table(y, s, n_bins: int = 10, seed: int = 0) -> pd.DataFrame:
    """Decile table: ``decile`` (1 = top scores), ``n``, ``events``, ``rate``, ``lift``, ``cum_capture``.

    Equal-count bins; tied scores straddling a bin edge are split by a seeded random key (see
    :func:`_order_desc`), not by input order.
    """
    y, s = _clean(y, s)
    bins = _score_bins(s, n_bins, seed)
    df = pd.DataFrame({"decile": bins, "y": y}).groupby("decile").agg(n=("y", "size"), events=("y", "sum")).reset_index()
    base = y.mean() if len(y) else float("nan")
    df["rate"] = df["events"] / df["n"]
    df["lift"] = df["rate"] / base if base else np.nan
    df["cum_capture"] = df["events"].cumsum() / max(y.sum(), 1)
    return df


def precision_at_k(y, s, k: float = 0.05, seed: int = 0) -> float:
    """Event rate among the top ``k`` share of scores (a fixed mail budget).

    Ties at the selection boundary are broken by a seeded random key (see :func:`_order_desc`).
    """
    y, s = _clean(y, s)
    n_top = max(1, int(np.ceil(k * len(y))))
    top = _order_desc(s, seed)[:n_top]
    return float(y[top].mean())


def psi(ref, cur, n_bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index with quantile bins defined on ``ref``.

    Bins are the unique ``ref`` quantiles (so a degenerate reference collapses gracefully), with
    open outer edges. ``eps`` floors each bin share before taking logs.
    """
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1]))
    edges = np.concatenate([[-np.inf], edges, [np.inf]])
    ref_share = np.histogram(ref, bins=edges)[0] / ref.size
    cur_share = np.histogram(cur, bins=edges)[0] / cur.size
    ref_share = np.clip(ref_share, eps, None)
    cur_share = np.clip(cur_share, eps, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def calibration_table(y, s, n_bins: int = 10, seed: int = 0) -> pd.DataFrame:
    """Reliability table by score bin (1 = highest): ``n``, ``mean_score``, ``event_rate``."""
    y, s = _clean(y, s)
    bins = _score_bins(s, n_bins, seed)
    df = pd.DataFrame({"bin": bins, "y": y, "s": s}).groupby("bin").agg(n=("y", "size"), mean_score=("s", "mean"), event_rate=("y", "mean")).reset_index()
    return df


def brier(y, s) -> float:
    """Mean squared error of probabilistic scores."""
    y, s = _clean(y, s)
    return float(np.mean((s - y) ** 2)) if len(y) else float("nan")
