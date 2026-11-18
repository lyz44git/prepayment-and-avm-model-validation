"""Credit-loss study: second remediation round (deviation D5), out-of-time scores and rolling backtest.

Candidates: M1v (approval eras, from D4), R2a = M1v scaled by trailing 24-month actual / predicted, R2b = M1v plus
downturn exposure, R2c = R2b scaled. Each is run with and without the relief overlay O1. Rules and adoption as
written in docs/credit_loss_deviations.md before this script was run.
Writes reports/p3/r2_oot_eval.csv, r2_backtest.csv, r2_backtest_monthly.csv and r2_summary.json.
"""
import sys, json, time; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from _paths import WORK, REPORTS
from mval.p3 import (ORIGINS, VAL_END, W1, LOG_RATE, SAMPLE_RATE, HOLDOUT_STATES, mi, CreditDesign, macro_quarterly,
                     macro_monthly, downturn_exposure, month_label, ns_basis, AGE_KNOTS, SECTORS)

R = REPORTS / "p3"; OUT = WORK / "p3"; FED = Path("data_external/fed2026")
L = json.loads((R / "champion.json").read_text())["lag_months"]
H, WIN = 24, 36
T0 = time.time(); log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
MM = {ov: macro_monthly(macro_quarterly(FED, "baseline", relief_overlay=ov), W1 + 400) for ov in (False, True)}
cc = pd.read_parquet(OUT / "cc.parquet"); cc["franchise"] = cc.franchise.astype(bool)
loans = pd.read_parquet(OUT / "loans.parquet"); loans = loans[loans.in_window].copy(); loans["franchise"] = loans.franchise.astype(bool)

def add_macro(p, mm, frozen_at=None):
    col = lambda c, lag: mm[c].reindex((p.m - lag) if frozen_at is None else np.minimum(p.m - lag, frozen_at)).to_numpy()
    p["dur4"], p["bbb_spread"] = col("dur4", L), col("bbb_spread", L)
    p["prime"], p["dprime4"], p["dur4_pif"] = col("prime", 0), col("dprime4", 0), col("dur4", 0)
    p["dx"] = downturn_exposure(mm.dur4, p.m.to_numpy(), p.disb_m.to_numpy(), lag=L, window=WIN, frozen_at=frozen_at)
    return p

def rows(cause, lo, hi, mm, holdout=False):
    y = f"y_{cause}"
    p = cc[(cc.m >= lo) & (cc.m <= hi) & (cc.holdout == holdout) & ((cc[y] == 1) | cc.keep)].copy()
    p["y"] = p[y].astype(int); p["w"] = np.where(p.y == 1, 1.0, 1 / SAMPLE_RATE)
    return add_macro(p, mm)

def fit(p, cause, eras=False, downturn=False):
    d = CreditDesign(cause, L if cause == "co" else 0, eras=eras, downturn=downturn).fit(p)
    X = d.transform(p if cause == "co" else p.assign(dur4=p.dur4_pif)); X.insert(0, "const", 1.0)
    r = sm.Logit(p.y.to_numpy(), X.to_numpy(np.float64)).fit(method="newton", maxiter=80, disp=0)
    b = pd.Series(r.params, index=X.columns); b["const"] += LOG_RATE
    return {"beta": b, "se": pd.Series(r.bse, index=X.columns), "design": d, "shares": p.state.value_counts(normalize=True)}

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

# ---------------------------------------------------------------- A. out of time (fit 2006-01..2019-12)
oot_rows = []
for ov in (False, True):
    mm = MM[ov]
    fco = rows("co", 0, VAL_END, mm); oco = rows("co", VAL_END + 1, W1, mm)
    mv, mvd = fit(fco, "co", eras=True), fit(fco, "co", eras=True, downturn=True)
    fv, fvd = factor(mv, fco, VAL_END), factor(mvd, fco, VAL_END)
    relief = ((oco.m >= mi(2020, 4)) & (oco.m <= mi(2021, 12))).to_numpy()
    hv, hvd = pred(mv, oco, "co"), pred(mvd, oco, "co")
    for name, h in [("M1v", hv), ("R2a", fv * hv), ("R2b", hvd), ("R2c", fvd * hvd)]:
        for part, sel in [("2020-01..2026-09", np.ones(len(oco), bool)), ("excluding 2020-04..2021-12", ~relief), ("2020-04..2021-12 only", relief)]:
            q = oco[sel]
            oot_rows.append({"overlay": ov, "model": name, "window": part, "chargeoffs": int(q.y.sum()),
                             "pred_over_actual": float((q.w * h[sel]).sum() / q.y.sum()), "auc": roc_auc_score(q.y, h[sel], sample_weight=q.w),
                             "factor": {"R2a": fv, "R2c": fvd}.get(name, np.nan)})
    if ov:
        pd.DataFrame({"beta": mvd["beta"], "se": mvd["se"]}).to_csv(R / "r2b_co_coefficients.csv")
    log("oot overlay", ov, "done")
oot = pd.DataFrame(oot_rows); oot.to_csv(R / "r2_oot_eval.csv", index=False)
log(oot[oot.window == "excluding 2020-04..2021-12"].round(3).to_string(index=False))

# ---------------------------------------------------------------- B. rolling backtest
def project(book, T, mods, mm, frozen):
    n = np.clip(np.minimum(T + H, W1) - T, 0, H) * np.ones(len(book), int)
    idx = np.repeat(np.arange(len(book)), n); step = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    p = book.iloc[idx].reset_index(drop=True)
    p["m"] = T + 1 + step; p["age"] = p.m - p.disb_m; p["moy"] = (p.m % 12) + 1
    p = add_macro(p, mm, T if frozen else None)
    hp = pred(mods["pif"], p, "pif"); sv = np.clip(mods["sev"].predict(sev_X(p)), 0, 1); amt = book.gross_approval.to_numpy()
    out = {}
    for key in ("v", "vd"):
        hc = pred(mods[key], p, "co")
        surv = np.ones(len(book)); e_n = np.zeros(len(book)); e_d = np.zeros(len(book)); mon = np.zeros((H, 2))
        for k in range(H):
            sel = step == k
            if not sel.any(): break
            li = idx[sel]; dn = surv[li] * hc[sel]; dd = dn * sv[sel] * amt[li]
            e_n[li] += dn; e_d[li] += dd; mon[k] = (dn.sum(), dd.sum())
            surv[li] *= np.clip(1 - hc[sel] - hp[sel], 0, 1)
        out[key] = (e_n.sum(), e_d.sum(), mon)
    return {"M1v": out["v"], "R2a": tuple(x * mods["f_v"] for x in out["v"]), "R2b": out["vd"], "R2c": tuple(x * mods["f_vd"] for x in out["vd"])}

bt, mon_rows, cache = [], [], {}
for ov in (False, True):
    mm = MM[ov]
    for T in ORIGINS:
        if ov and T + H < mi(2020, 4):            # overlay changes nothing before the relief window: reuse
            bt += [{**r, "overlay": True} for r in cache[T]]; continue
        fco, fpf = rows("co", 0, T, mm), rows("pif", 0, T, mm)
        mods = {"pif": fit(fpf, "pif"), "v": fit(fco, "co", eras=True), "vd": fit(fco, "co", eras=True, downturn=True)}
        mods["f_v"], mods["f_vd"] = factor(mods["v"], fco, T), factor(mods["vd"], fco, T)
        cf = loans[(loans.status == "co") & (loans.co_m <= T) & ~loans.state.isin(HOLDOUT_STATES)].copy()
        cf["age"] = cf.co_m - cf.disb_m; cf["m"] = cf.co_m; cf = add_macro(cf, mm)
        mods["sev"] = sm.GLM((cf.co_amount / cf.gross_approval).clip(0, 1), sev_X(cf), family=sm.families.Binomial()).fit()
        book = loans[(np.maximum(loans.disb_m + 1, 72) <= T) & (loans.end_m > T)].copy()
        act = (book.status == "co") & (book.co_m <= T + H)
        an, ad = int(act.sum()), float(book.co_amount.where(act, 0).sum())
        act_m = book[act].groupby("co_m").co_amount.agg(["size", "sum"]).reindex(range(T + 1, T + H + 1), fill_value=0)
        res_T = []
        for frozen in (False, True):
            for name, (pn, pdl, mon) in project(book, T, mods, mm, frozen).items():
                r = {"overlay": ov, "origin": month_label(T), "model": name, "macro": "held at origin" if frozen else "realised",
                     "pred_n": pn, "actual_n": an, "pred_dollars": pdl, "actual_dollars": ad, "n_ratio": pn / an, "dollar_ratio": pdl / ad,
                     "factor": {"R2a": mods["f_v"], "R2c": mods["f_vd"]}.get(name, np.nan)}
                res_T.append(r)
                if not frozen and T == mi(2019, 12):
                    mon_rows += [{"overlay": ov, "model": name, "month": month_label(T + 1 + k), "pred_dollars": mon[k, 1],
                                  "actual_dollars": float(act_m["sum"].iloc[k])} for k in range(H)]
        cache[T] = res_T; bt += res_T
        log("overlay", ov, month_label(T), pd.DataFrame(res_T).query("macro == 'realised'")[["model", "dollar_ratio"]].round(2).to_dict("records"))
bt = pd.DataFrame(bt); bt.to_csv(R / "r2_backtest.csv", index=False)
pd.DataFrame(mon_rows).to_csv(R / "r2_backtest_monthly.csv", index=False)

# ---------------------------------------------------------------- C. rules and adoption (as written in D5)
summ = {"rules": {"i": "out-of-time pred/actual excluding 2020-04..2021-12 within 0.80-1.25",
                  "ii": "realised-macro 24-month dollars within +/-25% at >= 6 of 8 origins (with overlay O1)"}}
for ov in (False, True):
    for name in ("M1v", "R2a", "R2b", "R2c"):
        a = bt[(bt.overlay == ov) & (bt.model == name) & (bt.macro == "realised")]
        k = int(((a.dollar_ratio - 1).abs() <= 0.25).sum())
        o = oot[(oot.overlay == ov) & (oot.model == name) & (oot.window == "excluding 2020-04..2021-12")].pred_over_actual.iloc[0]
        summ[f"{name}{' +O1' if ov else ''}"] = {"within_25pct": k, "mean_abs_log_ratio": float(np.abs(np.log(a.dollar_ratio)).mean()),
                                               "oot_excl_relief": float(o), "rule_i": bool(0.80 <= o <= 1.25), "rule_ii": k >= 6}
cand = [(v["within_25pct"], -v["mean_abs_log_ratio"], n) for n, v in summ.items() if n.endswith("+O1") and v["rule_i"] and v["rule_ii"]]
summ["adopted"] = max(cand)[2] if cand else None
best = max((v["within_25pct"], -v["mean_abs_log_ratio"], n) for n, v in summ.items() if n.endswith("+O1"))
summ["best_with_overlay"] = best[2]
(R / "r2_summary.json").write_text(json.dumps(summ, indent=2)); log(json.dumps(summ, indent=1))
