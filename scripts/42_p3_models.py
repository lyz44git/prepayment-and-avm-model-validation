"""Credit-loss study, step 3: charge-off and payoff hazards (M0 lookup, M1 logit, M2 LightGBM) and severity models.

Development: fit 2006-01..2017-12, validate 2018-01..2019-12 (lag L for charge-off macro, M2 early stopping,
champion rule). Out of time: refit on 2006-01..2019-12, score 2020-01..2026-09. Hold-out states never enter a fit.
Checks the plan hash first. Writes reports/p3/*.csv|json and data_work/p3/models_dev.pkl.
"""
import sys, json, pickle, time; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from _paths import WORK, REPORTS
from mval.p3 import (FIT_END, VAL_END, W1, LOG_RATE, SAMPLE_RATE, CreditDesign, lgb_frame, macro_quarterly,
                     macro_monthly, plan_sha, ns_basis, AGE_KNOTS, SECTORS)

ROOT = Path(".").resolve(); R = REPORTS / "p3"; OUT = WORK / "p3"
rec = json.loads((R / "plan_hash.json").read_text())
assert plan_sha(ROOT) == rec["sha256"], "analysis plan changed after its hash was recorded"
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

# ---------------------------------------------------------------- data + macro
mm = macro_monthly(macro_quarterly(Path("data_external/fed2026"), "baseline"), W1)
cc = pd.read_parquet(OUT / "cc.parquet")
cc["franchise"] = cc.franchise.astype(bool)
for L in (0, 12, 24):
    cc[f"dur4_l{L}"] = mm.dur4.reindex(cc.m - L).to_numpy()
    cc[f"bbb_l{L}"] = mm.bbb_spread.reindex(cc.m - L).to_numpy()
for c in ("prime", "dprime4", "dur4"):
    cc[f"{c}_l0"] = mm[c].reindex(cc.m).to_numpy()
cc["prime"], cc["dprime4"] = cc.prime_l0, cc.dprime4_l0
cc["dur4"], cc["bbb_spread"] = cc.dur4_l0, cc.bbb_l0     # lag 0 defaults (payoff); set_lag() overrides for charge-off
log("sample rows", len(cc))

def set_lag(p, L):
    p = p.copy(); p["dur4"] = p[f"dur4_l{L}"]; p["bbb_spread"] = p[f"bbb_l{L}"]; return p

def rows(cause, lo, hi, holdout=False):
    y = f"y_{cause}"
    sel = (cc.m >= lo) & (cc.m <= hi) & (cc.holdout == holdout) & ((cc[y] == 1) | cc.keep)
    p = cc[sel].copy()
    p["y"] = p[y].astype(int)
    p["w"] = np.where(p.y == 1, 1.0, 1.0 / SAMPLE_RATE)    # population weights (non-events kept at 5%)
    return p

def state_share_fix(X, p, design, fit_rows):
    """Hold-out states: replace state dummies by the training-share-weighted average state effect."""
    st_cols = [c for c in design.columns if c.startswith("st_")]
    shares = fit_rows.state.value_counts(normalize=True)
    vec = np.array([shares.get(c[3:], 0.0) for c in st_cols], np.float32)
    ho = ~p.state.isin(design.states).to_numpy()
    X.loc[ho, st_cols] = vec
    return X

def fit_logit(p, design):
    X = design.transform(p); X.insert(0, "const", 1.0)
    res = sm.Logit(p.y.to_numpy(), X.to_numpy(np.float64)).fit(method="newton", maxiter=60, disp=0)
    beta = pd.Series(res.params, index=X.columns); beta["const"] += LOG_RATE
    se = pd.Series(res.bse, index=X.columns)
    return {"beta": beta, "se": se, "design": design, "converged": bool(res.mle_retvals.get("converged", True))}

def pred_logit(m, p, fit_rows=None):
    X = m["design"].transform(p)
    if fit_rows is not None:
        X = state_share_fix(X, p, m["design"], fit_rows)
    X.insert(0, "const", 1.0)
    return 1 / (1 + np.exp(-(X.to_numpy(np.float64) @ m["beta"].reindex(X.columns).to_numpy())))

def age_band(a): return np.minimum(a // 12, 20)

def fit_lookup(p):
    """M0: hazard by 12-month age band x processing group (term group dropped, deviation D1)."""
    k = pd.DataFrame({"a": age_band(p.age), "g": p.proc, "y": p.y, "w": p.w})
    g = k.groupby(["a", "g"]).apply(lambda d: (d.y * d.w).sum() / d.w.sum(), include_groups=False)
    g2 = k.groupby(["a"]).apply(lambda d: (d.y * d.w).sum() / d.w.sum(), include_groups=False)
    return {"cell": g, "coarse": g2}

def pred_lookup(m, p):
    v = m["cell"].reindex(pd.MultiIndex.from_arrays([age_band(p.age), p.proc])).to_numpy()
    v2 = m["coarse"].reindex(range(21)).ffill().bfill().reindex(age_band(p.age)).to_numpy()   # ages beyond the fit months: last band
    return np.where(np.isnan(v), v2, v)

def fit_lgb(p, pv, cause):
    macro = ["dur4", "bbb_spread"] if cause == "co" else ["prime", "dprime4", "dur4"]
    X, Xv = lgb_frame(p, macro), lgb_frame(pv, macro)
    for c in ["proc", "bage", "btype", "naics2", "state"]:
        cats = sorted(set(X[c].astype(str)) | set(Xv[c].astype(str)))
        X[c] = pd.Categorical(X[c].astype(str), categories=cats); Xv[c] = pd.Categorical(Xv[c].astype(str), categories=cats)
    mono = [1 if (cause == "co" and c in ("dur4", "bbb_spread")) else 0 for c in X.columns]
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, monotone_constraints=mono, verbose=-1, seed=7,
                  num_threads=24, max_cat_to_onehot=4, cat_smooth=50)
    dt = lgb.Dataset(X, p.y, weight=None, categorical_feature=["proc", "bage", "btype", "naics2", "state"])
    dv = lgb.Dataset(Xv, pv.y, reference=dt)
    b = lgb.train(params, dt, num_boost_round=2000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
    return {"booster": b, "macro": macro, "cats": {c: list(X[c].cat.categories) for c in ["proc", "bage", "btype", "naics2", "state"]},
            "n_trees": b.best_iteration}

def pred_lgb(m, p):
    X = lgb_frame(p, m["macro"])
    for c, cats in m["cats"].items():
        X[c] = pd.Categorical(X[c].astype(str), categories=cats)
    raw = m["booster"].predict(X, num_iteration=m["n_trees"], raw_score=True) + LOG_RATE
    return 1 / (1 + np.exp(-raw))

def metrics(p, ph):
    w = p.w.to_numpy(); y = p.y.to_numpy(); ph = np.clip(ph, 1e-9, 1 - 1e-9)
    ll = -np.sum(w * (y * np.log(ph) + (1 - y) * np.log(1 - ph))) / w.sum()
    return {"rows": len(p), "events": int(y.sum()), "auc": roc_auc_score(y, ph, sample_weight=w),
            "logloss_x1e3": 1e3 * ll, "pred_over_actual": float((w * ph).sum() / y.sum())}

# ---------------------------------------------------------------- 1. development: choose L on validation months
out, models = {}, {}
fit_co, val_co = rows("co", 0, FIT_END), rows("co", FIT_END + 1, VAL_END)
lag_rows = []
for L in (0, 12, 24):
    pf, pv = set_lag(fit_co, L), set_lag(val_co, L)
    m = fit_logit(pf, CreditDesign("co", L).fit(pf))
    r = metrics(pv, pred_logit(m, pv)); r.update(lag=L, b_dur4=m["beta"]["dur4"], se_dur4=m["se"]["dur4"],
                                              b_bbb=m["beta"]["bbb_spread"], se_bbb=m["se"]["bbb_spread"])
    lag_rows.append(r); models[f"m1_co_L{L}"] = m; log("lag", L, r)
lag_tab = pd.DataFrame(lag_rows); lag_tab.to_csv(R / "lag_selection_validation.csv", index=False)
L = int(lag_tab.sort_values("logloss_x1e3").lag.iloc[0])
log("chosen lag", L)
fit_co, val_co = set_lag(fit_co, L), set_lag(val_co, L)
fit_pif, val_pif = rows("pif", 0, FIT_END), rows("pif", FIT_END + 1, VAL_END)

dev = []
for cause, pf, pv in [("co", fit_co, val_co), ("pif", fit_pif, val_pif)]:
    m1 = models.get(f"m1_co_L{L}") if cause == "co" else fit_logit(pf, CreditDesign("pif").fit(pf))
    m0 = fit_lookup(pf); m2 = fit_lgb(pf, pv, cause)
    for name, ph in [("M0 lookup", pred_lookup(m0, pv)), ("M1 logit", pred_logit(m1, pv)), ("M2 LightGBM", pred_lgb(m2, pv))]:
        r = metrics(pv, ph); r.update(cause=cause, model=name, split="validation 2018-19"); dev.append(r)
    models[f"dev_{cause}"] = {"m0": m0, "m1": m1, "m2": m2}
    log(cause, "dev done; M2 trees", m2["n_trees"])
dev = pd.DataFrame(dev); dev.to_csv(R / "eval_validation.csv", index=False)
v = dev[dev.cause == "co"].set_index("model")
gain = 1 - v.loc["M2 LightGBM", "logloss_x1e3"] / v.loc["M1 logit", "logloss_x1e3"]
m2_ok = gain > 0.01 and 0.90 <= v.loc["M2 LightGBM", "pred_over_actual"] <= 1.10
champion = {"lag_months": L, "m2_logloss_gain_vs_m1": gain, "m2_validation_pred_over_actual": v.loc["M2 LightGBM", "pred_over_actual"],
            "rule": "M2 only if validation log-loss gain > 1% and predicted/actual within 0.90-1.10", "champion": "M2 LightGBM" if m2_ok else "M1 logit"}
log("champion", champion)

# ---------------------------------------------------------------- 2. refit on 2006-01..2019-12, score out of time and hold-out states
ev, mon = [], []
for cause in ("co", "pif"):
    pf = rows(cause, 0, VAL_END); pv = rows(cause, FIT_END + 1, VAL_END)    # pv only for M2 early stopping
    if cause == "co":
        pf, pv = set_lag(pf, L), set_lag(pv, L)
    d = CreditDesign(cause, L if cause == "co" else 0).fit(pf)
    m1 = fit_logit(pf, d); m0 = fit_lookup(pf); m2 = fit_lgb(rows(cause, 0, FIT_END) if cause == "pif" else set_lag(rows(cause, 0, FIT_END), L), pv, cause)
    models[f"oot_{cause}"] = {"m0": m0, "m1": m1, "m2": m2, "fit_states": pf.state}
    splits = {"in sample 2006-19": rows(cause, 0, VAL_END), "out of time 2020-26": rows(cause, VAL_END + 1, W1),
              "hold-out states 2006-19": rows(cause, 0, VAL_END, True), "hold-out states, out of time": rows(cause, VAL_END + 1, W1, True)}
    for s, p in splits.items():
        if cause == "co":
            p = set_lag(p, L)
        preds = {"M0 lookup": pred_lookup(m0, p), "M1 logit": pred_logit(m1, p, pf), "M2 LightGBM": pred_lgb(m2, p)}
        for name, ph in preds.items():
            r = metrics(p, ph); r.update(cause=cause, model=name, split=s); ev.append(r)
        if not s.startswith("hold"):
            for name, ph in preds.items():
                g = pd.DataFrame({"m": p.m, "pred": p.w * ph, "act": p.y}).groupby("m").sum()
                g["cause"], g["model"] = cause, name; mon.append(g.reset_index())
    pd.Series(m1["beta"]).to_frame("beta").join(pd.Series(m1["se"]).to_frame("se")).to_csv(R / f"m1_{cause}_coefficients.csv")
    log(cause, "oot done")
ev = pd.DataFrame(ev); ev.to_csv(R / "eval_hazard.csv", index=False)
pd.concat(mon).to_csv(R / "monthly_hazard_pred_vs_actual.csv", index=False)

# ---------------------------------------------------------------- 3. severity: charge-off amount / gross approval
loans = pd.read_parquet(OUT / "loans.parquet")
co = loans[loans.in_window & (loans.status == "co") & (loans.end_m <= W1)].copy()
co["age"] = co.co_m - co.disb_m; co["s"] = (co.co_amount / co.gross_approval).clip(0, 1)
co["dur4"] = mm.dur4.reindex(co.co_m - L).to_numpy(); co["bbb_spread"] = mm.bbb_spread.reindex(co.co_m - L).to_numpy()
co["holdout"] = co.state.isin(__import__("mval.p3", fromlist=["HOLDOUT_STATES"]).HOLDOUT_STATES)

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

sfit = co[(co.co_m <= VAL_END) & ~co.holdout]; stest = co[(co.co_m > VAL_END) & ~co.holdout]; sho = co[co.holdout]
glm = sm.GLM(sfit.s, sev_X(sfit), family=sm.families.Binomial()).fit(cov_type="HC1")
s0 = sfit.groupby([age_band(sfit.age), sfit.proc]).s.mean()
def s0_pred(p):
    v = s0.reindex(pd.MultiIndex.from_arrays([age_band(p.age), p.proc])).to_numpy()
    return np.where(np.isnan(v), sfit.s.mean(), v)
sev_feats = ["age", "log_amount", "gty_share", "proc", "bage", "naics2", "dur4", "bbb_spread"]
def sev_lgbX(p):
    X = p.assign(log_amount=np.log(p.gross_approval.clip(1000, 1e8)))[sev_feats].copy()
    for c in ["proc", "bage", "naics2"]: X[c] = pd.Categorical(X[c].astype(str), categories=sorted(co[c].astype(str).unique()))
    return X
sv = sfit[sfit.co_m > FIT_END]; st = sfit[sfit.co_m <= FIT_END]
sb = lgb.train(dict(objective="cross_entropy", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, verbose=-1, seed=7, num_threads=24),
               lgb.Dataset(sev_lgbX(st), st.s), 3000, valid_sets=[lgb.Dataset(sev_lgbX(sv), sv.s)], callbacks=[lgb.early_stopping(100, verbose=False)])
sev_rows = []
for name, d in [("in sample 2006-19", sfit), ("out of time 2020-26", stest), ("hold-out states", sho)]:
    for mname, ph in [("S0 lookup", s0_pred(d)), ("S1 fractional logit", glm.predict(sev_X(d))), ("S2 LightGBM", sb.predict(sev_lgbX(d), num_iteration=sb.best_iteration))]:
        ph = np.asarray(ph); dollars_pred = (ph * d.gross_approval).sum(); dollars_act = (d.s * d.gross_approval).sum()
        sev_rows.append({"split": name, "model": mname, "chargeoffs": len(d), "mean_actual": d.s.mean(), "mean_pred": ph.mean(),
                         "mae": np.abs(ph - d.s).mean(), "dollar_pred_over_actual": dollars_pred / dollars_act,
                         "spearman": pd.Series(ph).corr(pd.Series(d.s.to_numpy()), method="spearman")})
sev = pd.DataFrame(sev_rows); sev.to_csv(R / "eval_severity.csv", index=False)
pd.DataFrame({"beta": glm.params, "se": glm.bse}).to_csv(R / "s1_severity_coefficients.csv")
models["severity"] = {"glm_params": glm.params, "s0": s0, "lag": L}
log("severity", sev.to_string())

(R / "champion.json").write_text(json.dumps(champion, indent=2, default=float))
with open(OUT / "models_dev.pkl", "wb") as f:
    pickle.dump({"L": L, "models": {k: v for k, v in models.items() if k.startswith(("oot", "severity"))}}, f)
log("done")
