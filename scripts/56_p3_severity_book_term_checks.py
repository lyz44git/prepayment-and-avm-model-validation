"""Credit-loss study checks (2026-11-19): severity capping, the projected book, the term field, ranking within months.

1. Severity capping. Severity is charge-off amount / original amount capped to [0, 1]; the out-of-time check of 42
   compares predicted dollars with capped actual dollars. Here: how many charge-offs and dollars the cap touches, and
   the fractional logit's predicted dollars against the recorded (uncapped) charge-off dollars.
2. Book bridge. Which loans the 2026-09 projection and each backtest origin include and leave out: loans first
   disbursed in the origin month (they enter the risk set the next month), loans past their recorded term, loans with
   a zero estimated balance.
3. Term field ablation. LightGBM charge-off hazard fitted as in 42 (2006-2017, early stopping on 2018-19, scored
   2020-01..2026-09) with and without term-based features (recorded term, months to maturity), on the published risk
   set and on a risk set cut at the recorded term, all scored on the same rows of each risk set.
4. Ranking within months. Out-of-time charge-off AUC computed within each calendar month (event-weighted mean), so
   that differences in level between months do not count as ranking.
Writes reports/p3/r6_severity_capping.csv, r6_book_bridge.csv, r6_term_ablation.csv, r6_within_month_auc.csv.
"""
import sys, json, pickle, time; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from _paths import WORK, REPORTS
from mval.p3 import (FIT_END, VAL_END, W1, LOG_RATE, SAMPLE_RATE, HOLDOUT_STATES, ORIGINS, CreditDesign, lgb_frame,
                     macro_quarterly, macro_monthly, month_label, ns_basis, AGE_KNOTS, SECTORS, balance_share)

R = REPORTS / "p3"; OUT = WORK / "p3"
L = json.loads((R / "champion.json").read_text())["lag_months"]
T0 = time.time(); log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
mm = macro_monthly(macro_quarterly(Path("data_external/fed2026"), "baseline"), W1)
loans = pd.read_parquet(OUT / "loans.parquet")

# ---------------------------------------------------------------- 1. severity capping
co = loans[loans.in_window & (loans.status == "co") & (loans.end_m <= W1)].copy()
co["age"] = co.co_m - co.disb_m; co["raw"] = co.co_amount / co.gross_approval; co["s"] = co.raw.clip(0, 1)
co["dur4"] = mm.dur4.reindex(co.co_m - L).to_numpy(); co["bbb_spread"] = mm.bbb_spread.reindex(co.co_m - L).to_numpy()
co["holdout"] = co.state.isin(HOLDOUT_STATES)
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
sfit = co[(co.co_m <= VAL_END) & ~co.holdout]; stest = co[(co.co_m > VAL_END) & ~co.holdout]
glm = sm.GLM(sfit.s, sev_X(sfit), family=sm.families.Binomial()).fit()
rows = []
for name, d in [("in sample 2006-19", sfit), ("out of time 2020-26", stest)]:
    pred_d = float((np.asarray(glm.predict(sev_X(d))) * d.gross_approval).sum())
    rec_d = float(d.co_amount.sum()); cap_d = float((d.s * d.gross_approval).sum())
    rows.append({"split": name, "chargeoffs": len(d), "above_original_amount": int((d.raw > 1).sum()),
                 "negative_amount": int((d.raw < 0).sum()), "recorded_dollars": rec_d, "capped_dollars": cap_d,
                 "dollars_removed_by_cap_share": 1 - cap_d / rec_d, "pred_over_capped_actual": pred_d / cap_d,
                 "pred_over_recorded_actual": pred_d / rec_d})
pd.DataFrame(rows).to_csv(R / "r6_severity_capping.csv", index=False); log(pd.DataFrame(rows).round(4).to_string(index=False))

# ---------------------------------------------------------------- 2. book bridge
br = []
act = loans[loans.status == "active"]
iw = loans.in_window
br.append({"origin": "2026-09 (allowance)", "item": "active loans in the risk window (projected)", "loans": int((iw & (loans.status == "active")).sum())})
br.append({"origin": "2026-09 (allowance)", "item": "active, first disbursed in 2026-09 (enter the risk set next month; not projected)",
           "loans": int(((loans.status == "active") & (loans.disb_m == W1)).sum())})
book = loans[iw & (loans.status == "active")].copy(); book["age0"] = W1 - book.disb_m
bal = balance_share(book.rate.fillna(book.rate.median()), book.term, book.age0)
br.append({"origin": "2026-09 (allowance)", "item": "projected, past the recorded term (projected 12 months)", "loans": int((book.age0 >= book.term).sum())})
br.append({"origin": "2026-09 (allowance)", "item": "projected, zero estimated balance", "loans": int((bal <= 0).sum())})
br.append({"origin": "2026-09 (allowance)", "item": "projected, interest rate missing (median rate used)", "loans": int(book.rate.isna().sum())})
L_ = loans[iw].copy()
for T in ORIGINS:
    bk = L_[(np.maximum(L_.disb_m + 1, 72) <= T) & (L_.end_m > T)]
    br.append({"origin": month_label(T), "item": "in the backtest book", "loans": len(bk)})
    br.append({"origin": month_label(T), "item": "open at T but first disbursed in month T (not in the book)",
               "loans": int(((L_.disb_m == T) & (L_.end_m > T)).sum())})
pd.DataFrame(br).to_csv(R / "r6_book_bridge.csv", index=False); log(pd.DataFrame(br).to_string(index=False))

# ---------------------------------------------------------------- 3-4. term ablation and within-month AUC
cc = pd.read_parquet(OUT / "cc.parquet"); cc["franchise"] = cc.franchise.astype(bool)
cc["dur4"] = mm.dur4.reindex(cc.m - L).to_numpy(); cc["bbb_spread"] = mm.bbb_spread.reindex(cc.m - L).to_numpy()
def rows_(lo, hi, holdout=False, cut=False):
    p = cc[(cc.m >= lo) & (cc.m <= hi) & (cc.holdout == holdout) & ((cc.y_co == 1) | cc.keep)].copy()
    if cut:
        p = p[p.age <= p.term]                                       # risk set cut at the recorded term (first fit)
    p["y"] = p.y_co.astype(int); p["w"] = np.where(p.y == 1, 1.0, 1 / SAMPLE_RATE)
    return p
CATS = ["proc", "bage", "btype", "naics2", "state"]
def frame(p, term):
    X = lgb_frame(p, ["dur4", "bbb_spread"])
    if term:
        X["term"] = p.term.astype(float); X["months_to_maturity"] = (p.term - p.age).astype(float)
    return X
def fit_lgb(p, pv, term):
    X, Xv = frame(p, term), frame(pv, term)
    cats = {}
    for c in CATS:
        cats[c] = sorted(set(X[c].astype(str)) | set(Xv[c].astype(str)))
        X[c] = pd.Categorical(X[c].astype(str), categories=cats[c]); Xv[c] = pd.Categorical(Xv[c].astype(str), categories=cats[c])
    mono = [1 if c in ("dur4", "bbb_spread") else 0 for c in X.columns]
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, monotone_constraints=mono, verbose=-1, seed=7,
                  num_threads=24, max_cat_to_onehot=4, cat_smooth=50)
    b = lgb.train(params, lgb.Dataset(X, p.y, categorical_feature=CATS), 2000, valid_sets=[lgb.Dataset(Xv, pv.y, reference=None)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return b, cats
def score(b, cats, p, term):
    X = frame(p, term)
    for c in CATS: X[c] = pd.Categorical(X[c].astype(str), categories=sorted(set(cats[c]) | set(X[c].astype(str))))
    for c in CATS: X[c] = pd.Categorical(X[c].astype(str), categories=cats[c])
    return 1 / (1 + np.exp(-(b.predict(X, num_iteration=b.best_iteration, raw_score=True) + LOG_RATE)))
def within_month_auc(p, s):
    out, wts = [], []
    for m_, g in p.assign(s=s).groupby("m"):
        if 0 < g.y.sum() < len(g):
            out.append(roc_auc_score(g.y, g.s, sample_weight=g.w)); wts.append(g.y.sum())
    return float(np.average(out, weights=wts))
ab = []
for cut in (() if "--within-month-only" in sys.argv else (False, True)):
    fit_, val_, test_ = rows_(0, FIT_END, cut=cut), rows_(FIT_END + 1, VAL_END, cut=cut), rows_(VAL_END + 1, W1, cut=cut)
    for term in (False, True):
        b, cats = fit_lgb(fit_, val_, term); s = score(b, cats, test_, term)
        ab.append({"risk_set": "cut at the recorded term" if cut else "published (not cut)", "term_features": term,
                   "trees": int(b.best_iteration), "oot_rows": len(test_), "oot_chargeoffs": int(test_.y.sum()),
                   "oot_auc": roc_auc_score(test_.y, s, sample_weight=test_.w), "oot_within_month_auc": within_month_auc(test_, s)})
        log(ab[-1])
if ab: pd.DataFrame(ab).to_csv(R / "r6_term_ablation.csv", index=False)

dev = pickle.load(open(OUT / "models_dev.pkl", "rb"))["models"]["oot_co"]
for a_, v_ in (("eras", False), ("downturn", False)):                # objects pickled before these options existed
    if not hasattr(dev["m1"]["design"], a_): setattr(dev["m1"]["design"], a_, v_)
test = rows_(VAL_END + 1, W1)
d1 = dev["m1"]; X = d1["design"].transform(test)
st_cols = [c for c in d1["design"].columns if c.startswith("st_")]; shares = dev["fit_states"].value_counts(normalize=True)
ho = ~test.state.isin(d1["design"].states).to_numpy()
if ho.any(): X.loc[ho, st_cols] = np.array([shares.get(c[3:], 0.0) for c in st_cols], np.float32)
X.insert(0, "const", 1.0); s1 = 1 / (1 + np.exp(-(X.to_numpy(np.float64) @ d1["beta"].reindex(X.columns).to_numpy())))
m2 = dev["m2"]; X2 = lgb_frame(test, m2["macro"])
for c, cats in m2["cats"].items(): X2[c] = pd.Categorical(X2[c].astype(str), categories=cats)
s2 = 1 / (1 + np.exp(-(m2["booster"].predict(X2, num_iteration=m2["n_trees"], raw_score=True) + LOG_RATE)))
wm = [{"model": n, "oot_auc_pooled": roc_auc_score(test.y, s, sample_weight=test.w), "oot_auc_within_month": within_month_auc(test, s)}
      for n, s in (("M1 logit", s1), ("M2 LightGBM", s2))]
pd.DataFrame(wm).to_csv(R / "r6_within_month_auc.csv", index=False); log(pd.DataFrame(wm).round(4).to_string(index=False)); log("done")
