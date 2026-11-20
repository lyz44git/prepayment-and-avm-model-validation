"""Credit-loss study check (2026-11-19): the second-round candidates' backtest with the macro lag chosen at each origin.

Companion to 53 for the candidates of deviation D5 with the relief overlay O1: M1v (approval eras), R2a (M1v scaled by
trailing 24-month actual / predicted), R2b (M1v plus downturn exposure) and R2c (R2b scaled). At each December origin
T the charge-off lag (0, 12 or 24 months) is chosen on information available at T: the champion logit fitted on
months up to T - 24, scored on T - 23 .. T (event-weighted log loss), with the O1 macro path. The candidates are then
refitted on data to T with that lag and project 24 months with the realised macro path, as in
scripts/45_p3_remediation2.py. Pass rule: within +/-25% of actual charge-off dollars at 6 or more of 8 origins.
Writes reports/p3/r6_backtest_lag_by_origin_r2.csv and reports/p3/r6_backtest_lag_by_origin_r2_summary.json.
"""
import sys, json, time; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from _paths import WORK, REPORTS
from mval.p3 import (ORIGINS, W1, LOG_RATE, SAMPLE_RATE, HOLDOUT_STATES, CreditDesign, macro_quarterly, macro_monthly,
                     downturn_exposure, month_label, ns_basis, AGE_KNOTS, SECTORS)

R = REPORTS / "p3"; OUT = WORK / "p3"; FED = Path("data_external/fed2026")
H, WIN, LAGS = 24, 36, (0, 12, 24)
T0 = time.time(); log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
mm = macro_monthly(macro_quarterly(FED, "baseline", relief_overlay=True), W1 + 400)
cc = pd.read_parquet(OUT / "cc.parquet"); cc["franchise"] = cc.franchise.astype(bool)
loans = pd.read_parquet(OUT / "loans.parquet"); loans = loans[loans.in_window].copy(); loans["franchise"] = loans.franchise.astype(bool)


def add_macro(p, L):
    col = lambda c, lag: mm[c].reindex(p.m - lag).to_numpy()
    p["dur4"], p["bbb_spread"] = col("dur4", L), col("bbb_spread", L)
    p["prime"], p["dprime4"], p["dur4_pif"] = col("prime", 0), col("dprime4", 0), col("dur4", 0)
    p["dx"] = downturn_exposure(mm.dur4, p.m.to_numpy(), p.disb_m.to_numpy(), lag=L, window=WIN)
    return p


def rows(cause, lo, hi, L):
    y = f"y_{cause}"
    p = cc[(cc.m >= lo) & (cc.m <= hi) & ~cc.holdout & ((cc[y] == 1) | cc.keep)].copy()
    p["y"] = p[y].astype(int); p["w"] = np.where(p.y == 1, 1.0, 1 / SAMPLE_RATE)
    return add_macro(p, L)


def fit(p, cause, L, eras=False, downturn=False):
    d = CreditDesign(cause, L if cause == "co" else 0, eras=eras, downturn=downturn).fit(p)
    X = d.transform(p if cause == "co" else p.assign(dur4=p.dur4_pif)); X.insert(0, "const", 1.0)
    r = sm.Logit(p.y.to_numpy(), X.to_numpy(np.float64)).fit(method="newton", maxiter=80, disp=0)
    b = pd.Series(r.params, index=X.columns); b["const"] += LOG_RATE
    return {"beta": b, "design": d, "shares": p.state.value_counts(normalize=True)}


def pred(m, p, cause):
    q = p if cause == "co" else p.assign(dur4=p.dur4_pif)
    X = m["design"].transform(q)
    st = [c for c in m["design"].columns if c.startswith("st_")]
    ho = ~q.state.isin(m["design"].states).to_numpy()
    if ho.any():
        X.loc[ho, st] = np.array([m["shares"].get(c[3:], 0.0) for c in st], np.float32)
    X.insert(0, "const", 1.0)
    return 1 / (1 + np.exp(-(X.to_numpy(np.float64) @ m["beta"].reindex(X.columns).to_numpy())))


def factor(m, p, T):
    tr = p[p.m > T - 24]
    return float(np.clip(tr.y.sum() / (tr.w * pred(m, tr, "co")).sum(), 0.25, 4))


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


def wlogloss(p, s):
    s = np.clip(s, 1e-12, 1 - 1e-12)
    return float(-(p.w * (p.y * np.log(s) + (1 - p.y) * np.log(1 - s))).sum() / p.w.sum())


def project(book, T, mods, L):
    n = np.clip(np.minimum(T + H, W1) - T, 0, H) * np.ones(len(book), int)
    idx = np.repeat(np.arange(len(book)), n); step = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    p = book.iloc[idx].reset_index(drop=True)
    p["m"] = T + 1 + step; p["age"] = p.m - p.disb_m; p["moy"] = (p.m % 12) + 1
    p = add_macro(p, L)
    hp = pred(mods["pif"], p, "pif"); sv = np.clip(mods["sev"].predict(sev_X(p)), 0, 1); amt = book.gross_approval.to_numpy()
    out = {}
    for key in ("v", "vd"):
        hc = pred(mods[key], p, "co"); surv = np.ones(len(book)); e_d = np.zeros(len(book))
        for k in range(H):
            sel = step == k
            if not sel.any(): break
            li = idx[sel]; e_d[li] += surv[li] * hc[sel] * sv[sel] * amt[li]
            surv[li] *= np.clip(1 - hc[sel] - hp[sel], 0, 1)
        out[key] = e_d.sum()
    return {"M1v": out["v"], "R2a": out["v"] * mods["f_v"], "R2b": out["vd"], "R2c": out["vd"] * mods["f_vd"]}


res = []
for T in ORIGINS:
    sel = []
    for L in LAGS:
        f_, v_ = rows("co", 0, T - 24, L), rows("co", T - 23, T, L)
        sel.append((wlogloss(v_, pred(fit(f_, "co", L), v_, "co")), L))
    L = min(sel)[1]
    fco, fpf = rows("co", 0, T, L), rows("pif", 0, T, L)
    mods = {"pif": fit(fpf, "pif", L), "v": fit(fco, "co", L, eras=True), "vd": fit(fco, "co", L, eras=True, downturn=True)}
    mods["f_v"], mods["f_vd"] = factor(mods["v"], fco, T), factor(mods["vd"], fco, T)
    cf = loans[(loans.status == "co") & (loans.co_m <= T) & ~loans.state.isin(HOLDOUT_STATES)].copy()
    cf["age"] = cf.co_m - cf.disb_m; cf["m"] = cf.co_m; cf = add_macro(cf, L)
    mods["sev"] = sm.GLM((cf.co_amount / cf.gross_approval).clip(0, 1), sev_X(cf), family=sm.families.Binomial()).fit()
    book = loans[(np.maximum(loans.disb_m + 1, 72) <= T) & (loans.end_m > T)].copy()
    act = (book.status == "co") & (book.co_m <= T + H); ad = float(book.co_amount.where(act, 0).sum())
    for name, v in project(book, T, mods, L).items():
        res.append({"origin": month_label(T), "chosen_lag": L, "model": name, "pred_dollars": v, "actual_dollars": ad, "dollar_ratio": v / ad})
    log(month_label(T), "lag", L, {r["model"]: round(r["dollar_ratio"], 3) for r in res[-4:]})
bt = pd.DataFrame(res); bt.to_csv(R / "r6_backtest_lag_by_origin_r2.csv", index=False)
pub = pd.read_csv(R / "r2_backtest.csv"); pub = pub[pub.overlay & (pub.macro == "realised")]
summary = {"chosen_lags": {r.origin: int(r.chosen_lag) for r in bt[bt.model == "M1v"].itertuples()}}
for name in ("M1v", "R2a", "R2b", "R2c"):
    a = bt[bt.model == name].set_index("origin").dollar_ratio; b = pub[pub.model == name].set_index("origin").dollar_ratio.reindex(a.index)
    summary[name] = {"within_25pct_lag_by_origin": int(((a - 1).abs() <= 0.25).sum()), "within_25pct_fixed_lag_12": int(((b - 1).abs() <= 0.25).sum()),
                     "mean_abs_log_ratio_lag_by_origin": float(np.abs(np.log(a)).mean()), "mean_abs_log_ratio_fixed_lag_12": float(np.abs(np.log(b)).mean()),
                     "per_origin": {k: {"lag_by_origin": float(a[k]), "fixed_lag_12": float(b[k])} for k in a.index}}
(R / "r6_backtest_lag_by_origin_r2_summary.json").write_text(json.dumps(summary, indent=2))
log(json.dumps({k: (v if k == "chosen_lags" else {kk: vv for kk, vv in v.items() if kk != "per_origin"}) for k, v in summary.items()}, indent=1))
log("done")
