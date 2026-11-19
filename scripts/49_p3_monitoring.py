"""Credit-loss study: monitoring back-tests that use only information available at each origin.

Origins, fits, book, severity and helpers as scripts/48_p3_backtest_v2.py (= 43 with the payoff hazard fitted and
scored at lag 0). Champion = M1 logit (charge-off) with the M1 payoff logit.
(a) Outcome trigger. At each December origin T (2009, 2011, ..., 2023) the champion fitted on data to T is compared
    with actual charge-offs over the trailing 24 months T-23..T, on the fitting sample (hold-out states excluded,
    non-events weighted 20). Two versions, both reported:
      - counts: predicted / actual charge-off counts. This is exactly the inverse of 43's M1r recalibration factor
        (before 43's clip to 0.25-4), i.e. 43's logic reused.
      - dollars: predicted dollars (hazard x S1 severity fitted to T x gross approval, weighted) / actual charge-off
        dollars of the same rows. (The request said "dollars" and also "equals the M1r factor's inverse"; the M1r
        factor is count-based, so both are given.)
    Trigger fires if the ratio is outside 0.80-1.25. Miss = that origin's next-24-month M1 dollar forecast (v2,
    realised macro, from r4_backtest_rolling_v2.csv) outside 0.75-1.25 (|ratio - 1| > 0.25); held-macro miss also shown.
(b) Ex-ante PSI. For consecutive origins T0 -> T1: the book at risk at T1 scored with the models FITTED AT T0 and
    macro HELD at T1 (no macro after T1), against the T0 book scored with the T0 models and macro held at T0. Score =
    24-month cumulative charge-off probability (competing payoff), deciles of the T0 distribution, as 43. Flag if
    PSI > 0.25; miss = T1's v2 M1 realised-macro forecast outside 0.75-1.25. 43's PSI (realised macro, model refitted
    at each origin) is shown beside it.
Writes reports/p3/r4_monitoring_trigger.csv and r4_psi_exante.csv (per-origin / per-transition rows plus 2x2 counts).
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
LO_T, HI_T = 0.80, 1.25                       # trigger band
T0 = time.time(); log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
mm = macro_monthly(macro_quarterly(Path("data_external/fed2026"), "baseline"), W1 + 400)
cc = pd.read_parquet(OUT / "cc.parquet"); cc["franchise"] = cc.franchise.astype(bool)
loans = pd.read_parquet(OUT / "loans.parquet"); loans = loans[loans.in_window].copy(); loans["franchise"] = loans.franchise.astype(bool)
loans["censor_m"] = W1

def add_macro(p, frozen_at=None):
    mcol = lambda c, lag: mm[c].reindex((p.m - lag) if frozen_at is None else np.minimum(p.m - lag, frozen_at)).to_numpy()
    p["dur4"], p["bbb_spread"] = mcol("dur4", L), mcol("bbb_spread", L)
    p["prime"], p["dprime4"] = mcol("prime", 0), mcol("dprime4", 0)
    p["dur4_pif"] = mcol("dur4", 0)
    return p

def fit_logit(p, cause, eras=False):
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

def train_rows(cause, T):
    y = f"y_{cause}"
    p = cc[(cc.m <= T) & ~cc.holdout & ((cc[y] == 1) | cc.keep)].copy()
    p["y"] = p[y].astype(int); p["w"] = np.where(p.y == 1, 1.0, 1 / SAMPLE_RATE)
    return add_macro(p)

def pd24(book, T, models):
    """24-month cumulative charge-off probability per loan, macro held at T (no information after T)."""
    n = np.clip(np.minimum(T + H, book.censor_m.to_numpy()) - T, 0, H)
    idx = np.repeat(np.arange(len(book)), n); step = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    p = book.iloc[idx].reset_index(drop=True)
    p["m"] = T + 1 + step; p["age"] = p.m - p.disb_m; p["moy"] = (p.m % 12) + 1
    p = add_macro(p, T)
    hco, hpf = pred_logit(models["co"], p, "co"), pred_logit(models["pif"], p, "pif")
    surv = np.ones(len(book)); e_n = np.zeros(len(book))
    for k in range(H):
        sel = step == k
        if not sel.any(): break
        li = idx[sel]; e_n[li] += surv[li] * hco[sel]; surv[li] *= np.clip(1 - hco[sel] - hpf[sel], 0, 1)
    return e_n

def psi(a_scores, b_scores):
    edges = np.quantile(a_scores, np.linspace(0, 1, 11)); edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(a_scores, edges)[0] / len(a_scores); b = np.histogram(b_scores, edges)[0] / len(b_scores)
    a, b = np.clip(a, 1e-6, None), np.clip(b, 1e-6, None)
    return float(((b - a) * np.log(b / a)).sum())

bt = pd.read_csv(R / "r4_backtest_rolling_v2.csv")
ratio = lambda T, mac: float(bt[(bt.origin == month_label(T)) & (bt.model == "M1") & (bt.macro == mac)].dollar_ratio.iloc[0])
old_psi = pd.read_csv(R / "backtest_psi.csv").set_index("to")

trig, psis, prev = [], [], None
for T in ORIGINS:
    mco, mpf = train_rows("co", T), train_rows("pif", T)
    models = {"co": fit_logit(mco, "co"), "pif": fit_logit(mpf, "pif")}
    co_fit = loans[(loans.status == "co") & (loans.co_m <= T) & ~loans.state.isin(HOLDOUT_STATES)].copy()
    co_fit["age"] = co_fit.co_m - co_fit.disb_m; co_fit["m"] = co_fit.co_m; co_fit = add_macro(co_fit)
    sev = sm.GLM((co_fit.co_amount / co_fit.gross_approval).clip(0, 1), sev_X(co_fit), family=sm.families.Binomial()).fit()
    # (a) trailing 24 months T-23..T on the fitting sample
    tr = mco[mco.m > T - 24]
    h = pred_logit(models["co"], tr, "co")
    s_hat = np.clip(sev.predict(sev_X(tr)), 0, 1)
    pn, an = float((tr.w * h).sum()), int(tr.y.sum())
    pdl, adl = float((tr.w * h * s_hat * tr.gross_approval).sum()), float(tr.co_amount.where(tr.y == 1, 0).sum())
    r_real, r_held = ratio(T, "realised"), ratio(T, "held at origin")
    trig.append({"table": "per origin", "origin": month_label(T), "trailing_window": f"{month_label(T - 23)}..{month_label(T)}",
                 "trailing_pred_n": pn, "trailing_actual_n": an, "trailing_ratio_counts": pn / an,
                 "m1r_factor_equivalent": an / pn,
                 "trailing_pred_dollars": pdl, "trailing_actual_dollars": adl, "trailing_ratio_dollars": pdl / adl,
                 "trigger_counts": not (LO_T <= pn / an <= HI_T), "trigger_dollars": not (LO_T <= pdl / adl <= HI_T),
                 "next24_ratio_realised_v2": r_real, "next24_ratio_held_v2": r_held,
                 "miss_realised": abs(r_real - 1) > 0.25, "miss_held": abs(r_held - 1) > 0.25})
    # (b) ex-ante PSI
    book = loans[(np.maximum(loans.disb_m + 1, 72) <= T) & (loans.end_m > T) & (loans.censor_m > T)].copy()
    own = pd24(book, T, models)
    if prev is not None:
        Tp, prev_scores, prev_models = prev
        cross = pd24(book, T, prev_models)
        v = psi(prev_scores, cross)
        psis.append({"table": "per transition", "from": month_label(Tp), "to": month_label(T),
                     "psi_exante": v, "flag_psi_gt_0_25": v > 0.25,
                     "mean_pd24_from": float(prev_scores.mean()), "mean_pd24_to": float(cross.mean()),
                     "book_loans_from": len(prev_scores), "book_loans_to": len(cross),
                     "next24_ratio_realised_v2_at_to": r_real, "miss_realised_at_to": abs(r_real - 1) > 0.25,
                     "miss_held_at_to": abs(r_held - 1) > 0.25,
                     "psi_43_realised_macro_refit": float(old_psi.loc[month_label(T), "psi"])})
    prev = (T, own, models)
    log(month_label(T), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in trig[-1].items()})
    if psis:
        log(psis[-1])

def two_by_two(df, flag, miss, label):
    out = []
    for f in (True, False):
        for m_ in (True, False):
            out.append({"table": f"2x2 {label}", "flag": f, "missed": m_, "count": int(((df[flag] == f) & (df[miss] == m_)).sum())})
    return out

tg = pd.DataFrame(trig)
tab = (two_by_two(tg, "trigger_counts", "miss_realised", "count trigger x realised-macro miss") +
       two_by_two(tg, "trigger_dollars", "miss_realised", "dollar trigger x realised-macro miss") +
       two_by_two(tg, "trigger_counts", "miss_held", "count trigger x held-macro miss") +
       two_by_two(tg, "trigger_dollars", "miss_held", "dollar trigger x held-macro miss"))
pd.concat([tg, pd.DataFrame(tab)], ignore_index=True).to_csv(R / "r4_monitoring_trigger.csv", index=False)
ps = pd.DataFrame(psis)
tab2 = two_by_two(ps, "flag_psi_gt_0_25", "miss_realised_at_to", "PSI>0.25 x realised-macro miss at T1") + \
       two_by_two(ps, "flag_psi_gt_0_25", "miss_held_at_to", "PSI>0.25 x held-macro miss at T1")
pd.concat([ps, pd.DataFrame(tab2)], ignore_index=True).to_csv(R / "r4_psi_exante.csv", index=False)
log(tg.round(3).to_string()); log(pd.DataFrame(tab).to_string()); log(ps.round(3).to_string()); log(pd.DataFrame(tab2).to_string())
log("done")
