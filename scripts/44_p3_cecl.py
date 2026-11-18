"""Credit-loss study, step 5: lifetime expected loss of the active book at 2026-09-30 (CECL-style allowance).

Final models are fitted on 2006-01..2026-09 (hold-out states excluded from the fits; their loans are in the book).
Book: loans active at 2026-09. Horizon: each loan's remaining recorded term (at least 12 months), and a sensitivity
with 24 more months for charge-offs posted after maturity (deviation D1). Macro: Fed 2026 supervisory baseline and
severely adverse paths to 2028Q4, then straight-line reversion over four quarters to the 2006-2025 average.
Expected loss of loan i = sum_t S(t-1) h_co(t) s(t) x gross approval; lender's share x (1 - guarantee share).
Model uncertainty (deviation D2): the same calculation with M0, M1 and M2.
Writes reports/p3/cecl_allowance.csv, cecl_by_segment.csv, cecl_paths.csv.
"""
import sys, json, pickle, time; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
import lightgbm as lgb
from _paths import WORK, REPORTS
from mval.p3 import (W1, LOG_RATE, SAMPLE_RATE, HOLDOUT_STATES, CreditDesign, lgb_frame, macro_quarterly, macro_monthly,
                     plan_sha, ns_basis, AGE_KNOTS, SECTORS, balance_share, month_label)

ROOT = Path(".").resolve(); R = REPORTS / "p3"; OUT = WORK / "p3"
assert plan_sha(ROOT) == json.loads((R / "plan_hash.json").read_text())["sha256"]
champ = json.loads((R / "champion.json").read_text()); L = champ["lag_months"]
dev = pickle.load(open(OUT / "models_dev.pkl", "rb"))["models"]
T0 = time.time(); log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
fed = Path("data_external/fed2026")
mm_hist = macro_monthly(macro_quarterly(fed, "baseline"), W1)

cc = pd.read_parquet(OUT / "cc.parquet"); cc["franchise"] = cc.franchise.astype(bool)
def train(cause):
    y = f"y_{cause}"
    p = cc[~cc.holdout & ((cc[y] == 1) | cc.keep)].copy()
    p["y"] = p[y].astype(int); p["w"] = np.where(p.y == 1, 1.0, 1 / SAMPLE_RATE)
    lag = L if cause == "co" else 0
    p["dur4"] = mm_hist.dur4.reindex(p.m - lag).to_numpy(); p["bbb_spread"] = mm_hist.bbb_spread.reindex(p.m - lag).to_numpy()
    p["prime"] = mm_hist.prime.reindex(p.m).to_numpy(); p["dprime4"] = mm_hist.dprime4.reindex(p.m).to_numpy()
    return p

models = {}
for cause in ("co", "pif"):
    p = train(cause)
    d = CreditDesign(cause, L if cause == "co" else 0).fit(p)
    X = d.transform(p); X.insert(0, "const", 1.0)
    res = sm.Logit(p.y.to_numpy(), X.to_numpy(np.float64)).fit(method="newton", maxiter=60, disp=0)
    b = pd.Series(res.params, index=X.columns); b["const"] += LOG_RATE
    k = pd.DataFrame({"a": np.minimum(p.age // 12, 20), "g": p.proc, "y": p.y, "w": p.w})
    lk = (k.groupby(["a", "g"]).apply(lambda d: (d.y * d.w).sum() / d.w.sum(), include_groups=False),
          k.groupby("a").apply(lambda d: (d.y * d.w).sum() / d.w.sum(), include_groups=False))
    m2d = dev[f"oot_{cause}"]["m2"]
    Xl = lgb_frame(p, m2d["macro"])
    for c, cats in m2d["cats"].items():
        Xl[c] = pd.Categorical(Xl[c].astype(str), categories=sorted(set(cats) | set(Xl[c].astype(str))))
    mono = [1 if (cause == "co" and c in ("dur4", "bbb_spread")) else 0 for c in Xl.columns]
    booster = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8,
                             bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, monotone_constraints=mono, verbose=-1, seed=7,
                             num_threads=24, max_cat_to_onehot=4, cat_smooth=50), lgb.Dataset(Xl, p.y), num_boost_round=max(m2d["n_trees"], 50))
    models[cause] = {"design": d, "beta": b, "shares": p.state.value_counts(normalize=True), "lookup": lk,
                     "m2": booster, "m2_macro": m2d["macro"], "m2_cats": {c: list(Xl[c].cat.categories) for c in m2d["cats"]}}
    if cause == "co":                                  # deviation D4 candidates
        dv = CreditDesign("co", L, eras=True).fit(p)
        Xv = dv.transform(p); Xv.insert(0, "const", 1.0)
        rv = sm.Logit(p.y.to_numpy(), Xv.to_numpy(np.float64)).fit(method="newton", maxiter=60, disp=0)
        bv = pd.Series(rv.params, index=Xv.columns); bv["const"] += LOG_RATE
        models[cause].update(design_v=dv, beta_v=bv)
        tr = p[p.m > W1 - 24]; Xt = d.transform(tr); Xt.insert(0, "const", 1.0)
        ht = 1 / (1 + np.exp(-(Xt.to_numpy(np.float64) @ b.reindex(Xt.columns).to_numpy())))
        models["factor"] = float(np.clip(tr.y.sum() / (tr.w * ht).sum(), 0.25, 4))
        log("trailing 24-month recalibration factor", round(models["factor"], 3))
    log(cause, "final fit done")

loans = pd.read_parquet(OUT / "loans.parquet"); loans["franchise"] = loans.franchise.astype(bool)
co = loans[loans.in_window & (loans.status == "co")].copy()
co["age"] = co.co_m - co.disb_m
co["dur4"] = mm_hist.dur4.reindex(co.co_m - L).to_numpy(); co["bbb_spread"] = mm_hist.bbb_spread.reindex(co.co_m - L).to_numpy()
co = co[~co.state.isin(HOLDOUT_STATES)]
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
s0 = co.assign(s=(co.co_amount / co.gross_approval).clip(0, 1)).groupby([np.minimum(co.age // 12, 20), co.proc]).s.mean()

book = loans[loans.in_window & (loans.status == "active")].copy()
book["age0"] = W1 - book.disb_m
book["rem"] = np.maximum(book.term - book.age0, 12)
book["bal_share"] = balance_share(book.rate.fillna(book.rate.median()), book.term, book.age0)
book["balance"] = book.gross_approval * book.bal_share
log("book", len(book), "loans; gross", round(book.gross_approval.sum() / 1e9, 2), "bn; est. balance", round(book.balance.sum() / 1e9, 2), "bn")

def hazards(p, cause, model):
    m = models[cause]
    if model == "M0":
        a = np.minimum(p.age // 12, 20)
        v = m["lookup"][0].reindex(pd.MultiIndex.from_arrays([a, p.proc])).to_numpy()
        return np.where(np.isnan(v), m["lookup"][1].reindex(range(21)).ffill().bfill().reindex(a).to_numpy(), v)
    if model == "M2":
        X = lgb_frame(p if cause == "co" else p.assign(dur4=p.dur4_pif), m["m2_macro"])
        for c, cats in m["m2_cats"].items():
            X[c] = pd.Categorical(X[c].astype(str), categories=cats)
        return 1 / (1 + np.exp(-(m["m2"].predict(X, raw_score=True) + LOG_RATE)))
    q = p if cause == "co" else p.assign(dur4=p.dur4_pif)
    if model == "M1v" and cause == "co":
        m = {**m, "design": m["design_v"], "beta": m["beta_v"]}
    X = m["design"].transform(q)
    st_cols = [c for c in m["design"].columns if c.startswith("st_")]
    ho = ~q.state.isin(m["design"].states).to_numpy()
    if ho.any():
        X.loc[ho, st_cols] = np.array([m["shares"].get(c[3:], 0.0) for c in st_cols], np.float32)
    X.insert(0, "const", 1.0)
    return 1 / (1 + np.exp(-(X.to_numpy(np.float64) @ m["beta"].reindex(X.columns).to_numpy())))

def run(scen, model, extra=0, chunk=25_000):
    mm = macro_monthly(macro_quarterly(fed, scen), W1 + 480)
    tot = np.zeros(len(book)); paths = np.zeros(480)
    for s in range(0, len(book), chunk):
        b = book.iloc[s:s + chunk]
        n = (b.rem.to_numpy() + extra).astype(int)
        idx = np.repeat(np.arange(len(b)), n); step = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        p = b.iloc[idx].reset_index(drop=True)
        p["m"] = W1 + 1 + step; p["age"] = p.age0 + 1 + step; p["moy"] = (p.m % 12) + 1
        p["dur4"] = mm.dur4.reindex(p.m - L).to_numpy(); p["bbb_spread"] = mm.bbb_spread.reindex(p.m - L).to_numpy()
        p["prime"] = mm.prime.reindex(p.m).to_numpy(); p["dprime4"] = mm.dprime4.reindex(p.m).to_numpy(); p["dur4_pif"] = mm.dur4.reindex(p.m).to_numpy()
        hco, hpf = hazards(p, "co", model), hazards(p, "pif", model)
        if model == "M0":
            v = s0.reindex(pd.MultiIndex.from_arrays([np.minimum(p.age // 12, 20), p.proc])).to_numpy()
            sv = np.where(np.isnan(v), s0.mean(), v)
        else:
            sv = np.clip(sev.predict(sev_X(p)), 0, 1)
        # survival within loan: rows are ordered by step within each loan
        df = pd.DataFrame({"i": idx, "q": np.clip(1 - hco - hpf, 0, 1)})
        surv_prev = df.groupby("i").q.cumprod().to_numpy() / np.where(df.q.to_numpy() > 0, df.q.to_numpy(), 1)
        el = surv_prev * hco * sv * p.gross_approval.to_numpy()
        tot[s:s + len(b)] = np.bincount(idx, weights=el, minlength=len(b))
        paths += np.bincount(np.minimum(step, 479), weights=el, minlength=480)
    return tot, paths

rows, seg_rows, path_rows = [], [], []
for scen in ("baseline", "severe"):
    cache = {}
    for model in ("M1", "M1r", "M1v", "M2", "M0"):
        for extra in ((0, 24) if model in ("M1", "M1r", "M1v") else (0,)):
            if model == "M1r":                       # deviation D4: M1 scaled by the trailing 24-month actual / predicted
                el, paths = (x * models["factor"] for x in cache[("M1", extra)])
            else:
                el, paths = run(scen, model, extra); cache[(model, extra)] = (el, paths)
            lender = el * (1 - book.gty_share.to_numpy())
            rows.append({"scenario": scen, "model": model, "horizon": "recorded term" + (f" + {extra} months" if extra else ""),
                         "loans": len(book), "gross_approval": book.gross_approval.sum(), "est_balance": book.balance.sum(),
                         "lifetime_loss": el.sum(), "lifetime_loss_lender": lender.sum(),
                         "loss_over_gross": el.sum() / book.gross_approval.sum(), "loss_over_balance": el.sum() / book.balance.sum(),
                         "lender_loss_over_lender_balance": lender.sum() / (book.balance * (1 - book.gty_share)).sum()})
            path_rows += [{"scenario": scen, "model": model, "extra": extra, "month": month_label(W1 + 1 + k), "loss": paths[k]} for k in range(120)]
            if extra == 0 and model in ("M1", "M1r", "M1v", "M2"):
                b = book.assign(el=el, lender=lender)
                for segcol in ("proc", "bage"):
                    g = b.groupby(segcol).agg(loans=("el", "size"), balance=("balance", "sum"), loss=("el", "sum"))
                    g["loss_over_balance"] = g.loss / g.balance
                    for k_, r_ in g.iterrows():
                        seg_rows.append({"scenario": scen, "model": model, "segment": segcol, "value": k_, **r_.to_dict()})
            log(scen, model, extra, f"{el.sum() / 1e9:.2f}bn", f"{el.sum() / book.balance.sum():.3%} of balance")
pd.DataFrame(rows).assign(recal_factor=models["factor"]).to_csv(R / "cecl_allowance.csv", index=False)
pd.DataFrame(seg_rows).to_csv(R / "cecl_by_segment.csv", index=False)
pd.DataFrame(path_rows).to_csv(R / "cecl_paths.csv", index=False)
mmb, mms = macro_monthly(macro_quarterly(fed, "baseline"), W1 + 60), macro_monthly(macro_quarterly(fed, "severe"), W1 + 60)
pd.DataFrame({"month": [month_label(m) for m in range(W1 - 24, W1 + 61)],
              "ur_baseline": mmb.ur.reindex(range(W1 - 24, W1 + 61)).to_numpy(), "ur_severe": mms.ur.reindex(range(W1 - 24, W1 + 61)).to_numpy(),
              "dur4_baseline": mmb.dur4.reindex(range(W1 - 24, W1 + 61)).to_numpy(), "dur4_severe": mms.dur4.reindex(range(W1 - 24, W1 + 61)).to_numpy()}
             ).to_csv(R / "cecl_macro_paths.csv", index=False)
log("done")
