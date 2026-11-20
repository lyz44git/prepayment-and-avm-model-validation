"""Credit-loss study checks (2026-11-19) on the lifetime-loss estimate of scripts/46 (relief overlay O1 on).

A. Published history (finding L12). The Fed files end their history in 2025Q4, so 2026Q1-Q3 carry baseline-scenario
   values in the fits and each scenario starts in 2026Q1, three quarters before the data date. Here every scenario
   shares the published history through 2026Q3 (unemployment, prime rate and 10-year Treasury from FRED; the BBB
   spread kept at its baseline-scenario value, since no matching BBB yield is published) and its own path then starts
   in 2026Q4, shifted three quarters later. The final models are refitted on that history and the book re-projected.
B. Channels. The severe-minus-baseline increment is split exactly into the charge-off hazard, the payoff hazard and
   severity by running all eight combinations of baseline and severe inputs for the three parts and taking Shapley
   values (each part's average contribution over the six orders of switching); the two one-at-a-time orders
   (charge-off first, severity first) are reported beside.
Writes reports/p3/r6_allowance_published_history.csv and reports/p3/r6_scenario_channels.csv.
"""
import sys, json, time, itertools, math; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import multiprocessing as mp
import numpy as np, pandas as pd
import statsmodels.api as sm
from _paths import WORK, REPORTS
from mval.p3 import (W1, LOG_RATE, SAMPLE_RATE, HOLDOUT_STATES, CreditDesign, macro_quarterly, macro_monthly, mi,
                     downturn_exposure, ns_basis, AGE_KNOTS, SECTORS, balance_share, start_survival, RELIEF_QUARTERS, _read_fed)

R = REPORTS / "p3"; OUT = WORK / "p3"; FED = Path("data_external/fed2026")
L = json.loads((R / "champion.json").read_text())["lag_months"]; WIN = 36; NPROC = 8
T0 = time.time(); log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def q_published_history(scenario):
    """History to 2025Q4, published 2026Q1-Q3 shared by every scenario, the scenario's path shifted to start in 2026Q4."""
    hist = _read_fed(FED / "2026_Final_Historic_Domestic.csv")
    name = {"baseline": "Baseline", "severe": "Severely_Adverse"}[scenario]
    scen = _read_fed(FED / f"2026_Final_Supervisory_{name}_Domestic.csv"); scen = scen[scen.year >= 2026].reset_index(drop=True)
    base = _read_fed(FED / "2026_Final_Supervisory_Baseline_Domestic.csv")
    pub = pd.read_csv(R / "macro_2026_scenario_vs_published.csv"); pub = pub[pub.source == "published_fred"].reset_index(drop=True)
    p26 = base[(base.year == 2026) & (base.q <= 3)].reset_index(drop=True).copy()
    spread = p26.bbb - p26.t10
    p26["ur"], p26["prime"], p26["t10"] = pub.unemployment.to_numpy(), pub.prime.to_numpy(), pub.treasury_10y.to_numpy()
    p26["bbb"] = p26.t10 + spread                                  # BBB spread held at its baseline-scenario value
    qn = np.arange(len(scen)) + 3                                  # scenario quarter k -> 2026Q4 + k
    scen["year"], scen["q"] = 2026 + qn // 4, qn % 4 + 1
    q = pd.concat([hist[hist.year <= 2025], p26, scen]).sort_values(["year", "q"]).reset_index(drop=True)
    sel = pd.Series(list(zip(q.year, q.q))).isin(RELIEF_QUARTERS).to_numpy(); b20 = q[(q.year == 2020) & (q.q == 1)].iloc[0]
    for c in ("ur", "bbb", "t10"):
        q.loc[sel, c] = b20[c]                                     # relief overlay O1, as in macro_quarterly
    q["dur4"] = q.ur - q.ur.shift(4); q["bbb_spread"] = q.bbb - q.t10; q["dprime4"] = q.prime - q.prime.shift(4)
    return q


MACRO = {"as published": {s: macro_quarterly(FED, s, relief_overlay=True) for s in ("baseline", "severe")},
         "published history to 2026Q3": {s: q_published_history(s) for s in ("baseline", "severe")}}
cc = pd.read_parquet(OUT / "cc.parquet"); cc["franchise"] = cc.franchise.astype(bool)
loans = pd.read_parquet(OUT / "loans.parquet"); loans["franchise"] = loans.franchise.astype(bool)


def add_macro(p, mm_co, mm_pif=None, mm_sev=None):
    mm_pif = mm_co if mm_pif is None else mm_pif; mm_sev = mm_co if mm_sev is None else mm_sev
    p["dur4"] = mm_co.dur4.reindex(p.m - L).to_numpy(); p["bbb_spread"] = mm_co.bbb_spread.reindex(p.m - L).to_numpy()
    p["prime"] = mm_pif.prime.reindex(p.m).to_numpy(); p["dprime4"] = mm_pif.dprime4.reindex(p.m).to_numpy()
    p["dur4_pif"] = mm_pif.dur4.reindex(p.m).to_numpy()
    p["dx"] = downturn_exposure(mm_co.dur4, p.m.to_numpy(), p.disb_m.to_numpy(), lag=L, window=WIN)
    p["sev_dur4"] = mm_sev.dur4.reindex(p.m - L).to_numpy(); p["sev_bbb"] = mm_sev.bbb_spread.reindex(p.m - L).to_numpy()
    return p


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


def sev_X(p, own=False):
    X = pd.DataFrame(index=p.index)
    B = ns_basis(np.clip(p.age.to_numpy(float), 1, 400), AGE_KNOTS)
    for j in range(B.shape[1]): X[f"age_ns{j}"] = B[:, j]
    X["log_amount"] = np.log(p.gross_approval.clip(1000, 1e8)); X["gty_share"] = p.gty_share
    for g in ["express", "preferred", "lowdoc", "other"]: X[f"proc_{g}"] = (p.proc == g).astype(float)
    for g in ["startup", "new", "unknown"]: X[f"bage_{g}"] = (p.bage == g).astype(float)
    for s in SECTORS + ["na"]: X[f"naics_{s}"] = (p.naics2 == s).astype(float)
    X["dur4"], X["bbb_spread"] = (p.dur4, p.bbb_spread) if own else (p.sev_dur4, p.sev_bbb)
    X.insert(0, "const", 1.0); return X


def build_models(variant):
    mm_hist = macro_monthly(MACRO[variant]["baseline"], W1)
    def train(cause):
        y = f"y_{cause}"
        p = cc[~cc.holdout & ((cc[y] == 1) | cc.keep)].copy()
        p["y"] = p[y].astype(int); p["w"] = np.where(p.y == 1, 1.0, 1 / SAMPLE_RATE)
        return add_macro(p, mm_hist)
    pco, ppf = train("co"), train("pif")
    mods = {"pif": fit(ppf, "pif"), "M1v": fit(pco, "co", eras=True), "R2b": fit(pco, "co", eras=True, downturn=True)}
    tr = pco[pco.m > W1 - 24]
    fac = float(np.clip(tr.y.sum() / (tr.w * pred(mods["M1v"], tr, "co")).sum(), 0.25, 4))
    co = loans[loans.in_window & (loans.status == "co") & ~loans.state.isin(HOLDOUT_STATES)].copy()
    co["age"] = co.co_m - co.disb_m; co["m"] = co.co_m; co = add_macro(co, mm_hist)
    sev = sm.GLM((co.co_amount / co.gross_approval).clip(0, 1), sev_X(co, own=True), family=sm.families.Binomial()).fit()
    return mods, fac, sev


book = loans[loans.in_window & (loans.status == "active")].copy()
book["age0"] = W1 - book.disb_m; book["rem"] = np.maximum(book.term - book.age0, 12)
book["balance"] = book.gross_approval * balance_share(book.rate.fillna(book.rate.median()), book.term, book.age0)
MODELS = {v: build_models(v) for v in MACRO}
log("fits done", {v: round(MODELS[v][1], 4) for v in MODELS})
MM = {v: {s: macro_monthly(MACRO[v][s], W1 + 480) for s in ("baseline", "severe")} for v in MACRO}


def run(task, chunk=25_000):
    variant, key, s_co, s_pif, s_sev = task
    mods, fac, sev = MODELS[variant]; mm = MM[variant]
    tot = 0.0
    for s in range(0, len(book), chunk):
        b = book.iloc[s:s + chunk]; n = b.rem.to_numpy().astype(int)
        idx = np.repeat(np.arange(len(b)), n); step = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        p = b.iloc[idx].reset_index(drop=True)
        p["m"] = W1 + 1 + step; p["age"] = p.age0 + 1 + step; p["moy"] = (p.m % 12) + 1
        p = add_macro(p, mm[s_co], mm[s_pif], mm[s_sev])
        hc, hp = pred(mods[key], p, "co"), pred(mods["pif"], p, "pif")
        sv = np.clip(sev.predict(sev_X(p)), 0, 1)
        sp, _, _ = start_survival(idx, hc, hp)
        tot += float((sp * hc * sv * p.gross_approval.to_numpy()).sum())
    return task, tot / book.balance.sum()


tasks = [("published history to 2026Q3", k, s, s, s) for k in ("M1v", "R2b") for s in ("baseline", "severe")]
tasks += [("as published", k, a, b, c) for k in ("M1v", "R2b") for a, b, c in itertools.product(("baseline", "severe"), repeat=3)]
with mp.get_context("fork").Pool(NPROC) as pool:
    res = dict(pool.map(run, tasks))
log("projections done")

rows = []
for v in MACRO:
    fac = MODELS[v][1]
    for k in ("M1v", "R2b"):
        for s in ("baseline", "severe"):
            x = res[(v, k, s, s, s)]
            rows.append({"history": v, "model": k, "scenario": s, "loss_over_balance": x})
            if k == "M1v":
                rows.append({"history": v, "model": "R2a", "scenario": s, "loss_over_balance": x * fac, "factor": fac})
pd.DataFrame(rows).to_csv(R / "r6_allowance_published_history.csv", index=False)

ch = []
parts = ("charge-off hazard", "payoff hazard", "severity")
for k in ("M1v", "R2b"):
    v = lambda S: res[("as published", k, *["severe" if i in S else "baseline" for i in range(3)])]
    total = v({0, 1, 2}) - v(set())
    for i, part in enumerate(parts):
        others = [j for j in range(3) if j != i]
        shap = sum(math.factorial(len(S)) * math.factorial(2 - len(S)) / 6 * (v(set(S) | {i}) - v(set(S)))
                   for r_ in range(3) for S in itertools.combinations(others, r_))
        co_first = {0: v({0}) - v(set()), 1: v({0, 1}) - v({0}), 2: v({0, 1, 2}) - v({0, 1})}[i]
        sev_first = {2: v({2}) - v(set()), 1: v({2, 1}) - v({2}), 0: v({0, 1, 2}) - v({1, 2})}[i]
        ch.append({"model": k, "channel": part, "shapley_pp": 100 * shap, "order_chargeoff_first_pp": 100 * co_first,
                   "order_severity_first_pp": 100 * sev_first, "total_severe_minus_baseline_pp": 100 * total})
ch = pd.DataFrame(ch); ch.to_csv(R / "r6_scenario_channels.csv", index=False)
print(pd.DataFrame(rows).round(5).to_string(index=False)); print(ch.round(3).to_string(index=False)); log("done")
