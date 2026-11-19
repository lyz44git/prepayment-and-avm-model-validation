"""Credit-loss study: rolling-origin backtest of 24-month charge-off forecasts, v2 (lag fix in the payoff hazard).

Same as scripts/43_p3_backtest.py (origins, refits on data to T, book at T, M0/M1/M1v/M1r, realised vs held-at-origin
macro, severity, horizon) except one thing: in 43 the payoff logit was FITTED with dur4 at the charge-off lag L
(add_macro sets p["dur4"] at lag L and fit_logit used p as is) but SCORED with dur4_pif (lag 0). Here the payoff logit
is fitted on p.assign(dur4=p.dur4_pif), as in scripts 45-47, so fit and score both use lag 0. M0 (lookups) and the
charge-off logits are unchanged by construction; M1, M1v and M1r change through the payoff hazard (survival).
The PSI table of 43 is not rewritten here (see scripts/49_p3_monitoring.py for an ex-ante PSI).
Writes reports/p3/r4_backtest_rolling_v2.csv, r4_backtest_rolling_monthly_v2.csv, r4_backtest_summary_v2.json and
r4_pandemic_check_v2.json. The summary also carries the v1 values read from backtest_rolling.csv for comparison.
"""
import sys, json, time; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
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
    X = d.transform(p if cause == "co" else p.assign(dur4=p.dur4_pif)); X.insert(0, "const", 1.0)   # v2: payoff fitted at lag 0
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
    hpf1 = pred_logit(models["pif"], p, "pif")
    for name in ("M1", "M0", "M1v"):
        if name == "M1":
            hco, hpf = pred_logit(models["co"], p, "co"), hpf1
        elif name == "M1v":                       # deviation D4: approval-era effects in the charge-off hazard
            hco, hpf = pred_logit(models["co_v"], p, "co"), hpf1
        else:
            hco, hpf = pred_lookup(models["co0"], p), pred_lookup(models["pif0"], p)
        sev = np.clip(models["sev"].predict(sev_X(p)), 0, 1)
        surv = np.ones(len(book)); e_n = np.zeros(len(book)); e_d = np.zeros(len(book)); mon_n = np.zeros(H); mon_d = np.zeros(H)
        lid = idx
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

res, monthly = [], []
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
    act_co = (book.status == "co") & (book.co_m <= T + H) & (book.co_m <= book.censor_m)
    actual_n, actual_d = act_co.sum(), (book.co_amount.where(act_co, 0)).sum()
    actual_d_lender = (book.co_amount.where(act_co, 0) * (1 - book.gty_share)).sum()
    act_m = book[act_co].groupby("co_m").agg(n=("loan_id", "size"), d=("co_amount", "sum")).reindex(range(T + 1, T + H + 1), fill_value=0)
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
            for k in range(H):
                monthly.append({"origin": month_label(T), "model": name, "macro": "held at origin" if frozen else "realised",
                                "month": month_label(T + 1 + k), "pred_n": mon_n[k], "pred_dollars": mon_d[k],
                                "actual_n": int(act_m.n.iloc[k]), "actual_dollars": float(act_m.d.iloc[k])})
    log(month_label(T), pd.DataFrame(res[-7:])[["model", "macro", "pred_n", "actual_n", "dollar_ratio"]].round(3).to_string(index=False))
bt = pd.DataFrame(res); bt.to_csv(R / "r4_backtest_rolling_v2.csv", index=False)
mo = pd.DataFrame(monthly); mo.to_csv(R / "r4_backtest_rolling_monthly_v2.csv", index=False)

# ---------------------------------------------------------------- summaries (v2) beside v1 (read from 43's outputs)
old = pd.read_csv(R / "backtest_rolling.csv"); old_mo = pd.read_csv(R / "backtest_rolling_monthly.csv")

def pandemic(m_):
    out = {}
    for mdl in ("M1", "M1v", "M1r", "M0"):
        for mac in ("realised", "held at origin"):
            c = m_[(m_.origin == "2019-12") & (m_.model == mdl) & (m_.macro == mac) & (m_.month >= "2020-04") & (m_.month <= "2021-12")]
            if len(c):
                out[f"{mdl} | {mac}"] = {"pred_dollars": float(c.pred_dollars.sum()), "actual_dollars": float(c.actual_dollars.sum()),
                                         "ratio": float(c.pred_dollars.sum() / c.actual_dollars.sum()),
                                         "pred_n": float(c.pred_n.sum()), "actual_n": int(c.actual_n.sum())}
    return out

pan = {"window": "2020-04..2021-12", "origin": "2019-12", "v2": pandemic(mo), "v1_from_backtest_rolling_monthly_csv": pandemic(old_mo),
       "finding_rule": "record a finding if ratio > 1.25"}
(R / "r4_pandemic_check_v2.json").write_text(json.dumps(pan, indent=2)); log(json.dumps(pan, indent=1))

def summarise(b):
    s = {}
    for mac in ("realised", "held at origin"):
        for mdl in ("M1", "M1v", "M1r", "M0"):
            a = b[(b.model == mdl) & (b.macro == mac)]
            if not len(a):
                continue
            ax = a[a.origin != "2019-12"]
            s[f"{mdl} | {mac}"] = {"origins": len(a), "within_25pct": int(((a.dollar_ratio - 1).abs() <= 0.25).sum()),
                                   "within_25pct_excl_2019_12": int(((ax.dollar_ratio - 1).abs() <= 0.25).sum()),
                                   "median_dollar_ratio": float(a.dollar_ratio.median()),
                                   "mean_abs_log_ratio_all": float(np.abs(np.log(a.dollar_ratio)).mean()),
                                   "mean_abs_log_ratio_excl_2019_12": float(np.abs(np.log(ax.dollar_ratio)).mean())}
    return s

def realised_vs_held(b, mdl="M1"):
    r = b[(b.model == mdl) & (b.macro == "realised")].set_index("origin").pred_dollars
    h = b[(b.model == mdl) & (b.macro == "held at origin")].set_index("origin").pred_dollars
    d = (r / h - 1)
    return {"per_origin": {k: float(v) for k, v in d.items()}, "origins_abs_lt_3pct": int((d.abs() < 0.03).sum())}

per_origin = []
for _, r_ in bt.iterrows():
    o_ = old[(old.origin == r_.origin) & (old.model == r_.model) & (old.macro == r_.macro)]
    per_origin.append({"origin": r_.origin, "model": r_.model, "macro": r_.macro, "dollar_ratio_v2": float(r_.dollar_ratio),
                       "dollar_ratio_v1": float(o_.dollar_ratio.iloc[0]) if len(o_) else None})
summary = {"pass_rule": "realised-macro 24-month dollar forecast within +/-25% of actual (|ratio - 1| <= 0.25) at >= 6 of 8 origins",
           "change_vs_43": "payoff logit fitted with dur4 at lag 0 (as scored); 43 fitted it with dur4 at lag L = %d" % L,
           "v2": summarise(bt), "v1_from_backtest_rolling_csv": summarise(old),
           "m1_realised_vs_held_relative_difference": {"v2": realised_vs_held(bt), "v1": realised_vs_held(old)},
           "per_origin_dollar_ratios": per_origin}
(R / "r4_backtest_summary_v2.json").write_text(json.dumps(summary, indent=2)); log(json.dumps({k: v for k, v in summary.items() if k != "per_origin_dollar_ratios"}, indent=1))
log("done")
