"""Prepayment study check (2026-11-19): refit with a house-price index known at the training cut-off.

The published panel (08) uses the 'full' repeat-sales index, estimated once on all states and on sales to 2025-08, so
later sales and the hold-out states shape the index values behind training features (log appreciation, estimated LTV,
purchase price indexed to 2022-04), and out-of-time features use the realised index. This script rebuilds the same
panel with the 'train_xh' index: pairs whose second sale is on or before 2024-06, hold-out states excluded (they take
their census-division index), frozen after 2024-06, the version the valuation study uses. Nothing after the training
cut-off, and nothing from the hold-out states, enters any feature. M1 (same design and case-control draw as 08, with
the unseen-state fix of 09) and M2 (same parameters and early stopping) are refitted and every split is scored, beside
the published fits. No other step of the study is rerun.
Writes reports/p1/r5_asof_index_eval.csv and reports/p1/r5_asof_index_summary.json.
"""
import sys, time, json, functools; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import WORK, REPORTS, duck
from mval import p1
from mval.metrics import auc, brier
from mval.amortization import cpr_from_smm
import numpy as np, pandas as pd, statsmodels.api as sm, lightgbm as lgb
print = functools.partial(print, flush=True)
t0 = time.time(); OUT = REPORTS / "p1"; SAMPLE = 0.04; CAUSES = ["sale", "new_lien"]
con = duck()
u = con.execute(f"SELECT * FROM read_parquet('{WORK}/p1_universe.parquet') WHERE u < {SAMPLE}").df()
cohort = con.execute(f"""SELECT t_purchase, median(mtgr) r FROM read_parquet('{WORK}/p1_universe.parquet')
   WHERE event='none' AND mtgr BETWEEN 0.5 AND 15 AND mtgd < 20220501 GROUP BY 1""").df().set_index("t_purchase").r
pm = p1.rate_vector()
st_idx, loghpi = p1.hpi_matrix(pd.read_parquet(WORK / "hpi_state_monthly.parquet"), pd.read_csv(WORK / "hpi_lookup_xh.csv"), "train_xh")
u = p1.add_static(u)
P = p1.build_panel(u, pm, st_idx, loghpi, cohort_rate=cohort)
P["state_code"] = P.state.astype("category").cat.codes
hold = P.state.isin(p1.HOLDOUT_STATES); late = P.t > p1.TRAIN_END
P["split"] = np.select([~hold & ~late, ~hold & late, hold & ~late, hold & late], ["train", "oot", "spatial", "spatial_oot"], default="na")
print(P.split.value_counts().to_dict(), f"({time.time() - t0:.0f}s)")
tr = P[P.split == "train"]
preds = P[["t", "split", "state", "y_sale", "y_new_lien"]].copy()

# ---------------- M1: same design and case-control draw as 08; unseen-state fix as 09
design = p1.LogitDesign().fit(tr)
X_all = design.transform(P)
st_share = tr.state.value_counts(normalize=True)
unseen = ~P.state.isin(design.states).to_numpy()
rng = np.random.default_rng(0)
coef_rows = []
for c in CAUSES:
    y = tr[f"y_{c}"].to_numpy(); q = 0.10
    keep = (y == 1) | (rng.random(len(y)) < q)
    Xc = sm.add_constant(X_all.loc[tr.index[keep]].astype(np.float64), has_constant="add")
    res = sm.GLM(y[keep], Xc, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(tr.fips5.to_numpy()[keep])[0]}, maxiter=100)
    params = res.params.copy(); params["const"] += np.log(q)
    for term in ["gap_pos", "gap_neg", "pmms", "pmms_chg3", "log_appr", "est_ltv", "log_value"]:
        coef_rows.append({"cause": c, "term": term, "coef": float(params[term]), "se_cluster": float(res.bse[term])})
    X = sm.add_constant(X_all, has_constant="add").reindex(columns=params.index, fill_value=0)
    for s_ in design.states[1:]:
        if f"st_{s_}" in X.columns:
            X.loc[unseen, f"st_{s_}"] = st_share.get(s_, 0.0)
    preds[f"m1_{c}"] = 1 / (1 + np.exp(-(X.to_numpy(np.float64) @ params.to_numpy())))
    print(c, "M1 done", f"({time.time() - t0:.0f}s)")

# ---------------- M2: same parameters and early stopping as 08
feat = p1.LGB_FEATURES
mono = {"sale": {"gap": -1}, "new_lien": {"gap": -1, "pmms": -1}}
trees = {}
for c in CAUSES:
    y = f"y_{c}"
    fit_rows = tr[tr.t <= p1.TRAIN_END - 6]; val_rows = tr[tr.t > p1.TRAIN_END - 6]
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=2000, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=30, seed=0,
                  monotone_constraints=[mono[c].get(f, 0) for f in feat], monotone_constraints_method="advanced")
    dtr = lgb.Dataset(fit_rows[feat], fit_rows[y], categorical_feature=["state_code", "moy"])
    dva = lgb.Dataset(val_rows[feat], val_rows[y], reference=dtr)
    b0 = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    trees[c] = int(b0.best_iteration)
    bst = lgb.train(params, lgb.Dataset(tr[feat], tr[y], categorical_feature=["state_code", "moy"]), trees[c])
    Xp = P[feat].copy(); Xp.loc[P.split.str.startswith("spatial"), "state_code"] = -1
    preds[f"m2_{c}"] = bst.predict(Xp)
    print(c, "M2 done", f"({time.time() - t0:.0f}s)")

# ---------------- evaluation beside the published fits
rows = []
for c in CAUSES:
    for sp in ["train", "oot", "spatial", "spatial_oot"]:
        g = preds[preds.split == sp]; y = g[f"y_{c}"].to_numpy()
        for m in ["m1", "m2"]:
            s = g[f"{m}_{c}"].to_numpy()
            rows.append({"cause": c, "split": sp, "model": m, "index": "as-of (train_xh, frozen after 2024-06)", "rows": len(g),
                         "events": int(y.sum()), "auc": auc(y, s), "brier_x1e4": brier(y, s) * 1e4,
                         "logloss": float(-np.mean(y * np.log(s) + (1 - y) * np.log(1 - s))),
                         "pred_over_actual": float(s.mean() / y.mean())})
pub = pd.read_csv(OUT / "eval_monthly_hazard_final.csv")
pub = pub[pub.model.isin(["m1", "m2"])].assign(index="published (full, all months and states)")
ev = pd.concat([pub, pd.DataFrame(rows)], ignore_index=True)[["cause", "split", "model", "index", "rows", "events", "auc",
                                                              "brier_x1e4", "logloss", "pred_over_actual"]]
ev.to_csv(OUT / "r5_asof_index_eval.csv", index=False)

# pool CPR backtest, same construction as 09, for both versions on the same months
old = pd.read_parquet(WORK / "p1" / "preds_v2.parquet", columns=["t", "split", "y_sale", "y_new_lien", "m1_sale", "m1_new_lien",
                                                                 "m2_sale", "m2_new_lien"])
def cpr_backtest(d):
    d = d.assign(month=d.t)
    a = d.groupby(["split", "month"]).agg(**{k: (k, "mean") for k in ["y_sale", "y_new_lien", "m1_sale", "m1_new_lien",
                                                                       "m2_sale", "m2_new_lien"]}).reset_index()
    out = {}
    for sp in ["oot", "spatial_oot"]:
        b = a[a.split == sp]; act = np.asarray(cpr_from_smm(b.y_sale + b.y_new_lien), float)
        for m in ["m1", "m2"]:
            pr = np.asarray(cpr_from_smm(b[f"{m}_sale"] + b[f"{m}_new_lien"]), float)
            out[f"{sp} | {m}"] = {"MAE_monthly_cpr_pp": float(np.abs(pr - act).mean() * 100), "corr": float(np.corrcoef(pr, act)[0, 1])}
    return out
coef = pd.DataFrame(coef_rows)
pubcoef = pd.concat([pd.read_csv(OUT / f"m1_coefficients_{c}.csv", index_col=0).assign(cause=c) for c in CAUSES])
coef["coef_published"] = [float(pubcoef[(pubcoef.cause == r.cause)].loc[r.term, "coef"]) for r in coef.itertuples()]
coef["se_published"] = [float(pubcoef[(pubcoef.cause == r.cause)].loc[r.term, "se_cluster"]) for r in coef.itertuples()]
summary = {"index": "train_xh: repeat-sales pairs with second sale <= 2024-06, hold-out states excluded (division index), frozen after 2024-06",
           "lgb_trees": trees,
           "gap_pos_odds_ratio": {c: {"as_of": float(np.exp(coef[(coef.cause == c) & (coef.term == "gap_pos")].coef.iloc[0])),
                                      "published": float(np.exp(coef[(coef.cause == c) & (coef.term == "gap_pos")].coef_published.iloc[0]))}
                                  for c in CAUSES},
           "m1_key_coefficients": coef.to_dict(orient="records"),
           "pool_cpr_backtest": {"as_of": cpr_backtest(preds), "published": cpr_backtest(old)}}
(OUT / "r5_asof_index_summary.json").write_text(json.dumps(summary, indent=2))
w = ev.pivot_table(index=["cause", "split", "model"], columns="index", values=["auc", "pred_over_actual"])
print(w.round(3).to_string()); print(json.dumps({k: summary[k] for k in ("lgb_trees", "gap_pos_odds_ratio", "pool_cpr_backtest")}, indent=1))
print(f"total {time.time() - t0:.0f}s")
