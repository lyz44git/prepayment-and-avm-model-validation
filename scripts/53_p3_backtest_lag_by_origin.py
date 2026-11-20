"""Credit-loss study check (2026-11-19): the 24-month backtest with the macro lag chosen at each origin.

The published backtests (43, 48) use the 12-month charge-off macro lag chosen on the 2018-19 validation months, which an
origin before 2019 could not have known. Here, at each December origin T, the lag of the charge-off macro terms (0, 12
or 24 months) is chosen on information available at T only: the champion logit is fitted on months up to T - 24 and
the lag with the lowest event-weighted log loss on months T - 23 .. T is kept (case-control weights: events 1,
non-events 1 / 0.05). The models are then refitted on data to T with that lag and project 24 months with the realised
macro path, exactly as in scripts/48_p3_backtest_v2.py (payoff hazard at lag 0, era model M1v, recalibrated M1r,
severity). The pass rule is the plan's: within +/-25% of actual charge-off dollars at 6 or more of 8 origins.
Writes reports/p3/r5_backtest_lag_by_origin.csv and reports/p3/r5_backtest_lag_by_origin_summary.json.
"""
import sys, json, time; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from _paths import WORK, REPORTS
from mval.p3 import (ORIGINS, W1, LOG_RATE, SAMPLE_RATE, HOLDOUT_STATES, CreditDesign, macro_quarterly, macro_monthly,
                     month_label, ns_basis, AGE_KNOTS, SECTORS)

R = REPORTS / "p3"; OUT = WORK / "p3"; H = 24; LAGS = (0, 12, 24)
T0 = time.time(); log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
mm = macro_monthly(macro_quarterly(Path("data_external/fed2026"), "baseline"), W1 + 400)
cc = pd.read_parquet(OUT / "cc.parquet"); cc["franchise"] = cc.franchise.astype(bool)
loans = pd.read_parquet(OUT / "loans.parquet"); loans = loans[loans.in_window].copy(); loans["franchise"] = loans.franchise.astype(bool)
loans["censor_m"] = W1


def add_macro(p, L):
    mcol = lambda c, lag: mm[c].reindex(p.m - lag).to_numpy()
    p["dur4"], p["bbb_spread"] = mcol("dur4", L), mcol("bbb_spread", L)
    p["prime"], p["dprime4"] = mcol("prime", 0), mcol("dprime4", 0)
    p["dur4_pif"] = mcol("dur4", 0)
    return p


def fit_logit(p, cause, L, eras=False):
    d = CreditDesign(cause, L if cause == "co" else 0, eras=eras).fit(p)
    X = d.transform(p if cause == "co" else p.assign(dur4=p.dur4_pif)); X.insert(0, "const", 1.0)
    res = sm.Logit(p.y.to_numpy(), X.to_numpy(np.float64)).fit(method="newton", maxiter=60, disp=0)
    b = pd.Series(res.params, index=X.columns); b["const"] += LOG_RATE
    return {"beta": b, "design": d, "shares": p.state.value_counts(normalize=True)}


def pred_logit(m, p, cause):
    q = p if cause == "co" else p.assign(dur4=p.dur4_pif)
    X = m["design"].transform(q)
    st_cols = [c for c in m["design"].columns if c.startswith("st_")]
    ho = ~q.state.isin(m["design"].states).to_numpy()
    if ho.any():
        X.loc[ho, st_cols] = np.array([m["shares"].get(c[3:], 0.0) for c in st_cols], np.float32)
    X.insert(0, "const", 1.0)
    return 1 / (1 + np.exp(-(X.to_numpy(np.float64) @ m["beta"].reindex(X.columns).to_numpy())))


def sev_X(p):
    X = pd.DataFrame(index=p.index)
    B = ns_basis(np.clip(p.age.to_numpy(float), 1, 400), AGE_KNOTS)
    for j in range(B.shape[1]): X[f"age_ns{j}"] = B[:, j]
    X["log_amount"] = np.log(p.gross_approval.clip(1000, 1e8)); X["gty_share"] = p.gty_share
    for g in ["express", "preferred", "lowdoc", "other"]: X[f"proc_{g}"] = (p.proc == g).astype(float)
    for g in ["startup", "new", "unknown"]: X[f"bage_{g}"] = (p.bage == g).astype(float)
    for s in SECTORS + ["na"]: X[f"naics_{s}"] = (p.naics2 == s).astype(float)
    X["dur4"], X["bbb_spread"] = p.dur4, p.bbb_spread
    X.insert(0, "const", 1.0); return X


def rows(cause, lo, hi, L):
    y = f"y_{cause}"
    p = cc[(cc.m >= lo) & (cc.m <= hi) & ~cc.holdout & ((cc[y] == 1) | cc.keep)].copy()
    p["y"] = p[y].astype(int); p["w"] = np.where(p.y == 1, 1.0, 1 / SAMPLE_RATE)
    return add_macro(p, L)


def wlogloss(p, s):
    s = np.clip(s, 1e-12, 1 - 1e-12)
    return float(-(p.w * (p.y * np.log(s) + (1 - p.y) * np.log(1 - s))).sum() / p.w.sum())


def project(book, T, models, L):
    n = np.clip(np.minimum(T + H, book.censor_m.to_numpy()) - T, 0, H)
    idx = np.repeat(np.arange(len(book)), n)
    step = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    p = book.iloc[idx].reset_index(drop=True)
    p["m"] = T + 1 + step; p["age"] = p.m - p.disb_m; p["moy"] = (p.m % 12) + 1
    p = add_macro(p, L)
    hpf = pred_logit(models["pif"], p, "pif"); sev = np.clip(models["sev"].predict(sev_X(p)), 0, 1)
    out = {}
    for name, key in (("M1", "co"), ("M1v", "co_v")):
        hco = pred_logit(models[key], p, "co")
        surv = np.ones(len(book)); e_d = np.zeros(len(book))
        for k in range(H):
            sel = step == k
            if not sel.any(): break
            li = idx[sel]; h1, h2 = hco[sel], hpf[sel]
            e_d[li] += surv[li] * h1 * sev[sel] * book.gross_approval.to_numpy()[li]
            surv[li] *= (1 - h1 - h2).clip(0, 1)
        out[name] = e_d.sum()
    out["M1r"] = out["M1"] * models["factor"]
    return out


res = []
for T in ORIGINS:
    sel = []
    for L in LAGS:                                   # choose the lag on information available at T
        fit, val = rows("co", 0, T - 24, L), rows("co", T - 23, T, L)
        m = fit_logit(fit, "co", L)
        sel.append({"lag": L, "val_logloss_x1e3": 1e3 * wlogloss(val, pred_logit(m, val, "co")), "val_events": int(val.y.sum())})
    sel = pd.DataFrame(sel); L = int(sel.sort_values("val_logloss_x1e3").lag.iloc[0])
    mco, mpf = rows("co", 0, T, L), rows("pif", 0, T, L)
    models = {"co": fit_logit(mco, "co", L), "pif": fit_logit(mpf, "pif", L), "co_v": fit_logit(mco, "co", L, eras=True)}
    tr = mco[mco.m > T - 24]
    models["factor"] = float(np.clip(tr.y.sum() / (tr.w * pred_logit(models["co"], tr, "co")).sum(), 0.25, 4))
    co_fit = loans[(loans.status == "co") & (loans.co_m <= T) & ~loans.state.isin(HOLDOUT_STATES)].copy()
    co_fit["age"] = co_fit.co_m - co_fit.disb_m; co_fit["m"] = co_fit.co_m
    co_fit = add_macro(co_fit, L); s_y = (co_fit.co_amount / co_fit.gross_approval).clip(0, 1)
    models["sev"] = sm.GLM(s_y, sev_X(co_fit), family=sm.families.Binomial()).fit()
    book = loans[(np.maximum(loans.disb_m + 1, 72) <= T) & (loans.end_m > T) & (loans.censor_m > T)].copy()
    act_co = (book.status == "co") & (book.co_m <= T + H) & (book.co_m <= book.censor_m)
    actual_d = float(book.co_amount.where(act_co, 0).sum())
    pred = project(book, T, models, L)
    for name, v in pred.items():
        res.append({"origin": month_label(T), "chosen_lag": L, "model": name, "pred_dollars": v, "actual_dollars": actual_d,
                    "dollar_ratio": v / actual_d, **{f"val_logloss_lag{r.lag}_x1e3": r.val_logloss_x1e3 for r in sel.itertuples()}})
    log(month_label(T), "lag", L, {k: round(v / actual_d, 3) for k, v in pred.items()})
bt = pd.DataFrame(res); bt.to_csv(R / "r5_backtest_lag_by_origin.csv", index=False)

v2 = pd.read_csv(R / "r4_backtest_rolling_v2.csv"); v2 = v2[v2.macro == "realised"]
summary = {"pass_rule": "within +/-25% of actual 24-month charge-off dollars (|ratio - 1| <= 0.25) at >= 6 of 8 origins",
           "lag_choice": "per origin T: lowest event-weighted log loss on T-23..T of the champion fitted to T-24, lags 0/12/24",
           "chosen_lags": {r.origin: int(r.chosen_lag) for r in bt[bt.model == "M1"].itertuples()}}
for name in ("M1", "M1v", "M1r"):
    a = bt[bt.model == name].set_index("origin").dollar_ratio
    b = v2[v2.model == name].set_index("origin").dollar_ratio.reindex(a.index)
    summary[name] = {"within_25pct_lag_by_origin": int(((a - 1).abs() <= 0.25).sum()),
                     "within_25pct_fixed_lag_12": int(((b - 1).abs() <= 0.25).sum()),
                     "mean_abs_log_ratio_lag_by_origin": float(np.abs(np.log(a)).mean()),
                     "mean_abs_log_ratio_fixed_lag_12": float(np.abs(np.log(b)).mean()),
                     "per_origin": {k: {"lag_by_origin": float(a[k]), "fixed_lag_12": float(b[k])} for k in a.index}}
(R / "r5_backtest_lag_by_origin_summary.json").write_text(json.dumps(summary, indent=2))
log(json.dumps({k: (v if not isinstance(v, dict) or "per_origin" not in v else {kk: vv for kk, vv in v.items() if kk != "per_origin"})
                for k, v in summary.items()}, indent=1))
log("done")
