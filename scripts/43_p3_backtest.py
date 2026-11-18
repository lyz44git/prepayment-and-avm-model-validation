"""Credit-loss study, step 4: rolling-origin backtest of 24-month charge-off forecasts.

At each origin T (December 2009, 2011, ..., 2023): refit M0 and M1 (charge-off and payoff) and the S1 severity
model on data up to T (hold-out states excluded from the fits), take the book at risk at T, and project charge-off
counts and dollars for months T+1..T+24, (a) with the realised macro path and (b) with macro held at its value at
T. Actuals come from the loan table. Also records the population stability index of the book's 24-month
charge-off probability between consecutive origins. Writes reports/p3/backtest_rolling*.csv.
"""
import sys, json, pickle, time; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from _paths import WORK, REPORTS
from mval.p3 import (ORIGINS, W1, LOG_RATE, SAMPLE_RATE, HOLDOUT_STATES, CreditDesign, macro_quarterly, macro_monthly,
                     plan_sha, month_label, ns_basis, AGE_KNOTS, SECTORS)

ROOT = Path(".").resolve(); R = REPORTS / "p3"; OUT = WORK / "p3"
assert plan_sha(ROOT) == json.loads((R / "plan_hash.json").read_text())["sha256"]
L = json.loads((R / "champion.json").read_text())["lag_months"]
H = 24
T0 = time.time(); log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
mm = macro_monthly(macro_quarterly(Path("data_external/fed2026"), "baseline"), W1 + 400)
cc = pd.read_parquet(OUT / "cc.parquet"); cc["franchise"] = cc.franchise.astype(bool)
loans = pd.read_parquet(OUT / "loans.parquet"); loans = loans[loans.in_window].copy(); loans["franchise"] = loans.franchise.astype(bool)
loans["censor_m"] = W1                       # censored only at the data date (deviation D1)

def add_macro(p, frozen_at=None):
    mcol = lambda c, lag: mm[c].reindex((p.m - lag) if frozen_at is None else np.minimum(p.m - lag, frozen_at)).to_numpy()
    p["dur4"], p["bbb_spread"] = mcol("dur4", L), mcol("bbb_spread", L)
    p["prime"], p["dprime4"] = mcol("prime", 0), mcol("dprime4", 0)
    p["dur4_pif"] = mcol("dur4", 0)
    return p

def fit_logit(p, cause, eras=False):
    d = CreditDesign(cause, L if cause == "co" else 0, eras=eras).fit(p)
    X = d.transform(p); X.insert(0, "const", 1.0)
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

def lookup(p):
    k = pd.DataFrame({"a": np.minimum(p.age // 12, 20), "g": p.proc, "y": p.y, "w": p.w})
    return k.groupby(["a", "g"]).apply(lambda d: (d.y * d.w).sum() / d.w.sum(), include_groups=False), \
           k.groupby(["a"]).apply(lambda d: (d.y * d.w).sum() / d.w.sum(), include_groups=False)

def pred_lookup(m, p):
    a = np.minimum(p.age // 12, 20)
    v = m[0].reindex(pd.MultiIndex.from_arrays([a, p.proc])).to_numpy()
    v2 = m[1].reindex(range(21)).ffill().bfill().reindex(a).to_numpy()
    return np.where(np.isnan(v), v2, v)

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

def train_rows(cause, T):
    y = f"y_{cause}"
    p = cc[(cc.m <= T) & ~cc.holdout & ((cc[y] == 1) | cc.keep)].copy()
    p["y"] = p[y].astype(int); p["w"] = np.where(p.y == 1, 1.0, 1 / SAMPLE_RATE)
    return add_macro(p)

def project(book, T, models, frozen):
    """Expand the book to months T+1..T+H (stopping at each loan's censoring month) and accumulate expected values."""
    n = np.clip(np.minimum(T + H, book.censor_m.to_numpy()) - T, 0, H)
    idx = np.repeat(np.arange(len(book)), n)
    step = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    p = book.iloc[idx].reset_index(drop=True)
    p["m"] = T + 1 + step; p["age"] = p.m - p.disb_m; p["moy"] = (p.m % 12) + 1
    p = add_macro(p, T if frozen else None)
    out = {}
    for name in ("M1", "M0", "M1v"):
        if name == "M1":
            hco, hpf = pred_logit(models["co"], p, "co"), pred_logit(models["pif"], p, "pif")
        elif name == "M1v":                       # deviation D4: approval-era effects in the charge-off hazard
            hco, hpf = pred_logit(models["co_v"], p, "co"), pred_logit(models["pif"], p, "pif")
        else:
            hco, hpf = pred_lookup(models["co0"], p), pred_lookup(models["pif0"], p)
        sev = np.clip(models["sev"].predict(sev_X(p)), 0, 1)
        surv = np.ones(len(book)); e_n = np.zeros(len(book)); e_d = np.zeros(len(book)); mon_n = np.zeros(H); mon_d = np.zeros(H)
        lid = idx
        # months are ordered within loan: process step by step
        for k in range(H):
            sel = step == k
            if not sel.any(): break
            li = lid[sel]; h1, h2, s = hco[sel], hpf[sel], sev[sel]
            dn = surv[li] * h1; dd = dn * s * book.gross_approval.to_numpy()[li]
            e_n[li] += dn; e_d[li] += dd; mon_n[k] = dn.sum(); mon_d[k] = dd.sum()
            surv[li] *= (1 - h1 - h2).clip(0, 1)
        out[name] = (e_n, e_d, mon_n, mon_d)
    f = models["factor"]                          # deviation D4: M1 scaled by trailing 24-month actual / predicted
    out["M1r"] = tuple(x * f for x in out["M1"])
    return out

res, psi_rows, prev, monthly = [], [], None, []
for T in ORIGINS:
    mco, mpf = train_rows("co", T), train_rows("pif", T)
    models = {"co": fit_logit(mco, "co"), "pif": fit_logit(mpf, "pif"), "co0": lookup(mco), "pif0": lookup(mpf),
              "co_v": fit_logit(mco, "co", eras=True)}
    tr = mco[mco.m > T - 24]
    models["factor"] = float(np.clip(tr.y.sum() / (tr.w * pred_logit(models["co"], tr, "co")).sum(), 0.25, 4))
    co_fit = loans[(loans.status == "co") & (loans.co_m <= T) & (loans.co_m <= loans.censor_m) & ~loans.state.isin(HOLDOUT_STATES)].copy()
    co_fit["age"] = co_fit.co_m - co_fit.disb_m; co_fit["m"] = co_fit.co_m
    co_fit = add_macro(co_fit); s_y = (co_fit.co_amount / co_fit.gross_approval).clip(0, 1)
    models["sev"] = sm.GLM(s_y, sev_X(co_fit), family=sm.families.Binomial()).fit()
    book = loans[(np.maximum(loans.disb_m + 1, 72) <= T) & (loans.end_m > T) & (loans.censor_m > T)].copy()
    endm = np.minimum(book.end_m, book.censor_m)
    act_co = (book.status == "co") & (book.co_m <= T + H) & (book.co_m <= book.censor_m)
    actual_n, actual_d = act_co.sum(), (book.co_amount.where(act_co, 0)).sum()
    actual_d_lender = (book.co_amount.where(act_co, 0) * (1 - book.gty_share)).sum()
    for frozen in (False, True):
        o = project(book, T, models, frozen)
        for name, (e_n, e_d, mon_n, mon_d) in o.items():
            if name == "M0" and frozen:
                continue
            res.append({"origin": month_label(T), "model": name, "macro": "held at origin" if frozen else "realised",
                        "book_loans": len(book), "book_amount": book.gross_approval.sum(),
                        "pred_n": e_n.sum(), "actual_n": int(actual_n), "pred_dollars": e_d.sum(), "actual_dollars": actual_d,
                        "pred_dollars_lender": (e_d * (1 - book.gty_share.to_numpy())).sum(), "actual_dollars_lender": actual_d_lender,
                        "n_ratio": e_n.sum() / actual_n, "dollar_ratio": e_d.sum() / actual_d,
                        "recal_factor": models["factor"] if name == "M1r" else np.nan})
            if name == "M1" and not frozen:
                pd24 = e_n
            act_m = book[act_co].groupby("co_m").agg(n=("loan_id", "size"), d=("co_amount", "sum")).reindex(range(T + 1, T + H + 1), fill_value=0)
            for k in range(H):
                monthly.append({"origin": month_label(T), "model": name, "macro": "held at origin" if frozen else "realised",
                                "month": month_label(T + 1 + k), "pred_n": mon_n[k], "pred_dollars": mon_d[k],
                                "actual_n": int(act_m.n.iloc[k]), "actual_dollars": float(act_m.d.iloc[k])})
    # PSI of the 24-month charge-off probability between consecutive origins (deciles of the earlier origin)
    if prev is not None:
        edges = np.quantile(prev, np.linspace(0, 1, 11)); edges[0], edges[-1] = -np.inf, np.inf
        a = np.histogram(prev, edges)[0] / len(prev); b = np.histogram(pd24, edges)[0] / len(pd24)
        a, b = np.clip(a, 1e-6, None), np.clip(b, 1e-6, None)
        psi_rows.append({"from": month_label(ORIGINS[ORIGINS.index(T) - 1]), "to": month_label(T), "psi": float(((b - a) * np.log(b / a)).sum()),
                         "mean_pd24_from": float(prev.mean()), "mean_pd24_to": float(pd24.mean())})
    prev = pd24
    log(month_label(T), pd.DataFrame(res[-7:])[["model", "macro", "pred_n", "actual_n", "dollar_ratio"]].round(3).to_string(index=False))
bt = pd.DataFrame(res); bt.to_csv(R / "backtest_rolling.csv", index=False)
mo = pd.DataFrame(monthly); mo.to_csv(R / "backtest_rolling_monthly.csv", index=False)
c = mo[(mo.origin == "2019-12") & (mo.model == "M1") & (mo.macro == "realised") & (mo.month >= "2020-04") & (mo.month <= "2021-12")]
pandemic = {"window": "2020-04..2021-12", "origin": "2019-12", "pred_dollars": c.pred_dollars.sum(), "actual_dollars": c.actual_dollars.sum(),
            "ratio": c.pred_dollars.sum() / c.actual_dollars.sum(), "pred_n": c.pred_n.sum(), "actual_n": int(c.actual_n.sum()),
            "finding_rule": "record a finding if ratio > 1.25"}
(R / "pandemic_check.json").write_text(json.dumps(pandemic, indent=2, default=float)); log(pandemic)
pd.DataFrame(psi_rows).to_csv(R / "backtest_psi.csv", index=False)
summary = {"pass_rule": "realised-macro 24-month dollar forecast within +/-25% of actual at >= 6 of 8 origins"}
for mdl in ("M1", "M1v", "M1r", "M0"):
    a = bt[(bt.model == mdl) & (bt.macro == "realised")]
    k = int(((a.dollar_ratio - 1).abs() <= 0.25).sum())
    summary[mdl] = {"origins": len(a), "within_25pct": k, "passed": k >= 6, "median_dollar_ratio": float(a.dollar_ratio.median()),
                    "mean_abs_log_ratio": float(np.abs(np.log(a.dollar_ratio)).mean())}
(R / "backtest_summary.json").write_text(json.dumps(summary, indent=2)); log(summary)
