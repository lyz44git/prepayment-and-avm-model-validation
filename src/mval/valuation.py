"""Valuation accuracy statistics (IAAO-style) shared by the vendor audit and the challenger evaluation."""
from __future__ import annotations
import numpy as np, pandas as pd

def accuracy(est, price, prev=None) -> dict:
    est = np.asarray(est, float); price = np.asarray(price, float)
    ok = np.isfinite(est) & (est > 1000) & np.isfinite(price)
    out = {"n": int(len(est)), "coverage": float(ok.mean())}
    if ok.sum() < 30:
        return out
    r = est[ok] / price[ok]; ape = np.abs(r - 1); med = np.median(r)
    w_mean = est[ok].sum() / price[ok].sum()
    out.update({
        "median_ratio": float(med), "MdAPE": float(np.median(ape)), "PPE10": float((ape <= 0.10).mean()),
        "PPE20": float((ape <= 0.20).mean()), "COD": float(np.mean(np.abs(r - med)) / med * 100),
        "PRD": float(r.mean() / w_mean),                      # >1.03 regressive, <0.98 progressive (IAAO)
        "over10": float((r > 1.10).mean()), "under10": float((r < 0.90).mean()),
        "spearman": float(pd.Series(est[ok]).rank().corr(pd.Series(price[ok]).rank())),
        "eq_sale_0.5pct": float((ape < 0.005).mean()),
    })
    if prev is not None:
        pv = np.asarray(prev, float)[ok]
        m = np.isfinite(pv) & (pv > 0)
        out["eq_prev_sale_0.5pct"] = float((np.abs(est[ok][m] / pv[m] - 1) < 0.005).mean()) if m.any() else np.nan
    return out

def table(df: pd.DataFrame, est_cols: dict, by: str | None = None, price="price", prev="price_prev") -> pd.DataFrame:
    rows = []
    groups = [("all", df)] if by is None else list(df.groupby(by, observed=True))
    for g, d in groups:
        for name, col in est_cols.items():
            rows.append({"group": g, "estimator": name, **accuracy(d[col], d[price], d[prev] if prev in d else None)})
    return pd.DataFrame(rows)
