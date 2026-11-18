"""Credit-loss study: lifetime-to-date backtest (deviation D6), written before it was run.

At each origin T from 2009-12 to 2019-12, models fitted on data to T (relief overlay O1 in the macro inputs) project
the book at risk at T month by month to 2026-09 with the realised macro path; predictions are compared with the
book's actual charge-off dollars from T+1 to 2026-09. Writes reports/p3/r3_lifetime_backtest.csv and r3_summary.json.
"""
import sys, json, time; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from _paths import WORK, REPORTS
from mval.p3 import (W1, LOG_RATE, SAMPLE_RATE, HOLDOUT_STATES, mi, CreditDesign, macro_quarterly, macro_monthly,
                     downturn_exposure, month_label, ns_basis, AGE_KNOTS, SECTORS)

R = REPORTS / "p3"; OUT = WORK / "p3"; FED = Path("data_external/fed2026")
L = json.loads((R / "champion.json").read_text())["lag_months"]; WIN = 36
ORIG = [mi(y, 12) for y in (2009, 2011, 2013, 2015, 2017, 2019)]
T0 = time.time(); log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
mm = macro_monthly(macro_quarterly(FED, "baseline", relief_overlay=True), W1 + 12)
cc = pd.read_parquet(OUT / "cc.parquet"); cc["franchise"] = cc.franchise.astype(bool)
loans = pd.read_parquet(OUT / "loans.parquet"); loans = loans[loans.in_window].copy(); loans["franchise"] = loans.franchise.astype(bool)

def add_macro(p):
    p["dur4"] = mm.dur4.reindex(p.m - L).to_numpy(); p["bbb_spread"] = mm.bbb_spread.reindex(p.m - L).to_numpy()
    p["prime"] = mm.prime.reindex(p.m).to_numpy(); p["dprime4"] = mm.dprime4.reindex(p.m).to_numpy(); p["dur4_pif"] = mm.dur4.reindex(p.m).to_numpy()
    p["dx"] = downturn_exposure(mm.dur4, p.m.to_numpy(), p.disb_m.to_numpy(), lag=L, window=WIN)
    return p

def rows(cause, T):
    y = f"y_{cause}"
    p = cc[(cc.m <= T) & ~cc.holdout & ((cc[y] == 1) | cc.keep)].copy()
    p["y"] = p[y].astype(int); p["w"] = np.where(p.y == 1, 1.0, 1 / SAMPLE_RATE)
    return add_macro(p)

def fit(p, cause, **kw):
    d = CreditDesign(cause, L if cause == "co" else 0, **kw).fit(p)
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

def lookup(p):
    k = pd.DataFrame({"a": np.minimum(p.age // 12, 20), "g": p.proc, "y": p.y, "w": p.w})
    return (k.groupby(["a", "g"]).apply(lambda d: (d.y * d.w).sum() / d.w.sum(), include_groups=False),
            k.groupby("a").apply(lambda d: (d.y * d.w).sum() / d.w.sum(), include_groups=False).reindex(range(21)).ffill().bfill())

def pred_lookup(m, p):
    a = np.minimum(p.age // 12, 20)
    v = m[0].reindex(pd.MultiIndex.from_arrays([a, p.proc])).to_numpy()
    return np.where(np.isnan(v), m[1].reindex(a).to_numpy(), v)

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

res = []
for T in ORIG:
    fco, fpf = rows("co", T), rows("pif", T)
    mods = {"pif": fit(fpf, "pif"), "M1": fit(fco, "co"), "M1v": fit(fco, "co", eras=True), "R2b": fit(fco, "co", eras=True, downturn=True),
            "co0": lookup(fco), "pif0": lookup(fpf)}
    tr = fco[fco.m > T - 24]
    f_v = float(np.clip(tr.y.sum() / (tr.w * pred(mods["M1v"], tr, "co")).sum(), 0.25, 4))
    cf = loans[(loans.status == "co") & (loans.co_m <= T) & ~loans.state.isin(HOLDOUT_STATES)].copy()
    cf["age"] = cf.co_m - cf.disb_m; cf["m"] = cf.co_m; cf = add_macro(cf)
    sev = sm.GLM((cf.co_amount / cf.gross_approval).clip(0, 1), sev_X(cf), family=sm.families.Binomial()).fit()
    book = loans[(np.maximum(loans.disb_m + 1, 72) <= T) & (loans.end_m > T)].copy().reset_index(drop=True)
    act = (book.status == "co") & (book.co_m <= W1)
    actual = float(book.co_amount.where(act, 0).sum())
    tot = {k: 0.0 for k in ("M0", "M1", "M1v", "R2b")}
    n = W1 - T
    for s in range(0, len(book), 12_000):
        b = book.iloc[s:s + 12_000]
        idx = np.repeat(np.arange(len(b)), n); step = np.tile(np.arange(n), len(b))
        p = b.iloc[idx].reset_index(drop=True)
        p["m"] = T + 1 + step; p["age"] = p.m - p.disb_m; p["moy"] = (p.m % 12) + 1
        p = add_macro(p)
        sv = np.clip(sev.predict(sev_X(p)), 0, 1) * p.gross_approval.to_numpy()
        hp = pred(mods["pif"], p, "pif"); hp0 = pred_lookup(mods["pif0"], p)
        for key in tot:
            hc = pred_lookup(mods["co0"], p) if key == "M0" else pred(mods[key], p, "co")
            q = np.clip(1 - hc - (hp0 if key == "M0" else hp), 0, 1).reshape(len(b), n)
            surv_prev = np.concatenate([np.ones((len(b), 1)), np.cumprod(q, axis=1)[:, :-1]], axis=1).ravel()
            tot[key] += float((surv_prev * hc * sv).sum())
    for key, v in tot.items():
        res.append({"origin": month_label(T), "model": key, "months": n, "book_loans": len(book), "pred_dollars": v, "actual_dollars": actual,
                    "dollar_ratio": v / actual})
    res.append({"origin": month_label(T), "model": "R2a", "months": n, "book_loans": len(book), "pred_dollars": tot["M1v"] * f_v,
                "actual_dollars": actual, "dollar_ratio": tot["M1v"] * f_v / actual, "factor": f_v})
    log(month_label(T), {r["model"]: round(r["dollar_ratio"], 3) for r in res[-5:]})
out = pd.DataFrame(res); out.to_csv(R / "r3_lifetime_backtest.csv", index=False)
oot = pd.read_csv(R / "r2_oot_eval.csv")
summ = {"rule": "lifetime-to-date dollars within +/-25% at >= 5 of 6 origins (2009-2019); adoption also needs rule (i)"}
for key in ("M0", "M1", "M1v", "R2a", "R2b"):
    a = out[out.model == key]; k = int(((a.dollar_ratio - 1).abs() <= 0.25).sum())
    o = oot[(oot.overlay) & (oot.model == key) & (oot.window == "excluding 2020-04..2021-12")].pred_over_actual
    ri = bool(0.80 <= o.iloc[0] <= 1.25) if len(o) else None
    summ[key] = {"within_25pct": k, "mean_abs_log_ratio": float(np.abs(np.log(a.dollar_ratio)).mean()), "lifetime_rule": k >= 5,
                 "oot_excl_relief": float(o.iloc[0]) if len(o) else None, "rule_i": ri}
q = [(v["within_25pct"], -v["mean_abs_log_ratio"], k) for k, v in summ.items() if isinstance(v, dict) and v["lifetime_rule"] and v["rule_i"]]
summ["adopted_for_lifetime_allowance"] = max(q)[2] if q else None
(R / "r3_summary.json").write_text(json.dumps(summ, indent=2)); log(json.dumps(summ, indent=1))
