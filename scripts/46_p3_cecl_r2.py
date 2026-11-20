"""Credit-loss study: lifetime expected loss of the active book at 2026-09-30 for the second-round candidates
(deviation D5), with the relief overlay O1 in the estimation history. Same book, horizon, scenarios and reversion as
scripts/44_p3_cecl.py. Writes reports/p3/r2_cecl_allowance.csv.
"""
import sys, json, time; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from _paths import WORK, REPORTS
from mval.p3 import (W1, LOG_RATE, SAMPLE_RATE, HOLDOUT_STATES, CreditDesign, macro_quarterly, macro_monthly,
                     downturn_exposure, ns_basis, AGE_KNOTS, SECTORS, balance_share, start_survival)

R = REPORTS / "p3"; OUT = WORK / "p3"; FED = Path("data_external/fed2026")
L = json.loads((R / "champion.json").read_text())["lag_months"]; WIN = 36
T0 = time.time(); log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
mm_hist = macro_monthly(macro_quarterly(FED, "baseline", relief_overlay=True), W1)
cc = pd.read_parquet(OUT / "cc.parquet"); cc["franchise"] = cc.franchise.astype(bool)

def add_macro(p, mm):
    p["dur4"] = mm.dur4.reindex(p.m - L).to_numpy(); p["bbb_spread"] = mm.bbb_spread.reindex(p.m - L).to_numpy()
    p["prime"] = mm.prime.reindex(p.m).to_numpy(); p["dprime4"] = mm.dprime4.reindex(p.m).to_numpy(); p["dur4_pif"] = mm.dur4.reindex(p.m).to_numpy()
    p["dx"] = downturn_exposure(mm.dur4, p.m.to_numpy(), p.disb_m.to_numpy(), lag=L, window=WIN)
    return p

def train(cause):
    y = f"y_{cause}"
    p = cc[~cc.holdout & ((cc[y] == 1) | cc.keep)].copy()
    p["y"] = p[y].astype(int); p["w"] = np.where(p.y == 1, 1.0, 1 / SAMPLE_RATE)
    return add_macro(p, mm_hist)

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

pco, ppf = train("co"), train("pif")
mods = {"pif": fit(ppf, "pif"), "M1": fit(pco, "co"), "M1v": fit(pco, "co", eras=True), "R2b": fit(pco, "co", eras=True, downturn=True)}
tr = pco[pco.m > W1 - 24]
fac = {k: float(np.clip(tr.y.sum() / (tr.w * pred(mods[k], tr, "co")).sum(), 0.25, 4)) for k in ("M1v", "R2b")}
log("fits done; trailing factors", fac)

loans = pd.read_parquet(OUT / "loans.parquet"); loans["franchise"] = loans.franchise.astype(bool)
co = loans[loans.in_window & (loans.status == "co") & ~loans.state.isin(HOLDOUT_STATES)].copy()
co["age"] = co.co_m - co.disb_m; co["m"] = co.co_m; co = add_macro(co, mm_hist)
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
sev = sm.GLM((co.co_amount / co.gross_approval).clip(0, 1), sev_X(co), family=sm.families.Binomial()).fit()

book = loans[loans.in_window & (loans.status == "active")].copy()
book["age0"] = W1 - book.disb_m; book["rem"] = np.maximum(book.term - book.age0, 12)
book["balance"] = book.gross_approval * balance_share(book.rate.fillna(book.rate.median()), book.term, book.age0)

MASS = []
def run(scen, key, chunk=25_000):
    mm = macro_monthly(macro_quarterly(FED, scen, relief_overlay=True), W1 + 480)
    tot = np.zeros(len(book))
    for s in range(0, len(book), chunk):
        b = book.iloc[s:s + chunk]; n = b.rem.to_numpy().astype(int)
        idx = np.repeat(np.arange(len(b)), n); step = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        p = b.iloc[idx].reset_index(drop=True)
        p["m"] = W1 + 1 + step; p["age"] = p.age0 + 1 + step; p["moy"] = (p.m % 12) + 1
        p = add_macro(p, mm)
        hc, hp = pred(mods[key], p, "co"), pred(mods["pif"], p, "pif")
        sv = np.clip(sev.predict(sev_X(p)), 0, 1)
        sp, n_over, mx = start_survival(idx, hc, hp)
        MASS.append({"scenario": scen, "model": key, "rows": len(idx), "rows_sum_above_1": n_over, "max_sum": mx})
        tot[s:s + len(b)] = np.bincount(idx, weights=sp * hc * sv * p.gross_approval.to_numpy(), minlength=len(b))
    return tot

out = []
for scen in ("baseline", "severe"):
    for key in ("M1", "M1v", "R2b"):
        el = run(scen, key)
        variants = [(key, 1.0)] + ([("R2a" if key == "M1v" else "R2c", fac[key])] if key in fac else [])
        for name, f in variants:
            e = el * f
            out.append({"scenario": scen, "model": name, "overlay": True, "lifetime_loss": e.sum(),
                        "lifetime_loss_lender": (e * (1 - book.gty_share.to_numpy())).sum(), "est_balance": book.balance.sum(),
                        "loss_over_balance": e.sum() / book.balance.sum(), "factor": f})
            log(scen, name, f"{e.sum() / book.balance.sum():.3%}")
pd.DataFrame(out).to_csv(R / "r2_cecl_allowance.csv", index=False)
pd.DataFrame(MASS).groupby(["scenario", "model"], as_index=False).agg(rows=("rows", "sum"), rows_sum_above_1=("rows_sum_above_1", "sum"),
    max_sum=("max_sum", "max")).to_csv(R / "r2_hazard_mass_checks.csv", index=False)
log("done")
