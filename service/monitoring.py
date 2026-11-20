"""Population Stability Index (PSI) against a stored reference distribution.

PSI = sum_b (a_b - e_b) * ln(a_b / e_b), where e_b and a_b are the reference and current shares of bin b.
Numeric features use the reference deciles as bin edges plus a separate missing-value bin; categorical
features use their reference categories plus an "other" bin. Empty bins are floored at 1e-4.
Thresholds follow common practice and the valuation study's monitoring plan: < 0.10 stable, 0.10-0.25 watch,
>= 0.25 investigate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-4
WATCH, INVESTIGATE = 0.10, 0.25


def numeric_reference(x: pd.Series, n_bins: int = 10) -> dict:
    v = x.to_numpy(dtype=float)
    ok = ~np.isnan(v)
    edges = np.unique(np.quantile(v[ok], np.linspace(0, 1, n_bins + 1)[1:-1])) if ok.any() else np.array([])
    shares = _numeric_shares(v, edges)
    return {"kind": "numeric", "edges": edges.tolist(), "shares": shares.tolist()}


def categorical_reference(x: pd.Series) -> dict:
    vc = x.astype("Int64").value_counts(normalize=True, dropna=False)
    cats = [int(c) for c in vc.index if pd.notna(c)]
    shares = [float(vc[c]) for c in cats]
    # the last bin is "other or missing" in current_shares, so the reference puts its missing share there too
    # (it was set to 0, which made a distribution with missing values drift against itself; the stored reference
    # has no missing categorical values, so no published PSI changes)
    return {"kind": "categorical", "categories": cats, "shares": shares + [max(0.0, 1.0 - sum(shares))]}


def _numeric_shares(v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(v)
    counts = np.bincount(np.searchsorted(edges, v[ok], side="right"), minlength=len(edges) + 1).astype(float)
    counts = np.append(counts, (~ok).sum())          # last bin = missing
    return counts / max(len(v), 1)


def current_shares(x: pd.Series, ref: dict) -> np.ndarray:
    if ref["kind"] == "numeric":
        return _numeric_shares(x.to_numpy(dtype=float), np.asarray(ref["edges"], dtype=float))
    v = x.astype("Int64")
    cats = ref["categories"]
    shares = [float((v == c).sum()) for c in cats]
    shares.append(float(len(v) - sum(shares)))       # other / missing
    return np.asarray(shares) / max(len(v), 1)


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.maximum(np.asarray(expected, dtype=float), EPS)
    a = np.maximum(np.asarray(actual, dtype=float), EPS)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float) -> str:
    return "investigate" if value >= INVESTIGATE else "watch" if value >= WATCH else "stable"


def psi_table(frame: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """PSI for every column of `frame` that has a reference entry."""
    rows = []
    for col, ref in reference["columns"].items():
        if col not in frame:
            continue
        val = psi(ref["shares"], current_shares(frame[col], ref))
        rows.append({"column": col, "psi": val, "status": status(val)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def load_reference(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
