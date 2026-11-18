"""Prepayment study: build the property-month panel (4% national sample), fit M0/M1/M2 cause-specific hazards,
and evaluate out-of-time, out-of-state and both. Writes tables to reports/p1 and predictions to data_work/p1."""
import sys, time, json, pickle; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import WORK, REPORTS, duck
from mval import p1
from mval.metrics import auc, brier
import numpy as np, pandas as pd, statsmodels.api as sm, lightgbm as lgb, functools
print = functools.partial(print, flush=True)
t0 = time.time(); OUT = REPORTS / "p1"; DW = WORK / "p1"
SAMPLE = 0.04
con = duck()
u = con.execute(f"SELECT * FROM read_parquet('{WORK}/p1_universe.parquet') WHERE u < {SAMPLE}").df()
cohort = con.execute(f"""SELECT t_purchase, median(mtgr) r FROM read_parquet('{WORK}/p1_universe.parquet')
   WHERE event='none' AND mtgr BETWEEN 0.5 AND 15 AND mtgd < 20220501 GROUP BY 1""").df().set_index("t_purchase").r
print(f"sample properties {len(u):,}  ({time.time()-t0:.0f}s)")
pm = p1.rate_vector()
st_idx, loghpi = p1.hpi_matrix(pd.read_parquet(WORK / "hpi_state_monthly.parquet"), pd.read_csv(WORK / "hpi_lookup.csv"), "full")
u = p1.add_static(u)
P = p1.build_panel(u, pm, st_idx, loghpi, cohort_rate=cohort)
P["state_code"] = P.state.astype("category").cat.codes
hold = P.state.isin(p1.HOLDOUT_STATES); late = P.t > p1.TRAIN_END
P["split"] = np.select([~hold & ~late, ~hold & late, hold & ~late, hold & late], ["train", "oot", "spatial", "spatial_oot"], default="na")
print(P.split.value_counts().to_dict(), f"rows {len(P):,} ({time.time()-t0:.0f}s)")
print(P.groupby("split")[["y_sale", "y_new_lien", "y_other_transfer"]].agg(["sum", "mean"]).to_string())
P.to_parquet(DW / "panel.parquet", index=False)

CAUSES = ["sale", "new_lien"]
tr = P[P.split == "train"]
preds = P[["addr_id", "t", "split", "state", "y_sale", "y_new_lien", "y_other_transfer", "tenure_m", "gap", "pmms"]].copy()

# ---------------- M0: actuarial lookup (state x tenure bucket hazard, times month-of-year factor)
bins = [-1, 12, 24, 36, 60, 84, 120, 180, 240, 1000]
P["ten_b"] = pd.cut(P.tenure_m, bins, labels=False)
for c in CAUSES:
    y = f"y_{c}"; trb = P.loc[P.split == "train"]
    st_t = trb.groupby(["state", "ten_b"])[y].agg(["sum", "count"])
    nat_t = trb.groupby("ten_b")[y].mean()
    k = 200.0   # credibility weight toward the national tenure curve
    shr = (st_t["sum"] + k * nat_t.reindex(st_t.index.get_level_values(1)).to_numpy()) / (st_t["count"] + k)
    moy = trb.groupby("moy")[y].mean() / trb[y].mean()
    base = pd.Series(shr.reindex(pd.MultiIndex.from_arrays([P.state, P.ten_b])).to_numpy(), index=P.index)
    base = base.fillna(P.ten_b.map(nat_t))
    preds[f"m0_{c}"] = (base * P.moy.map(moy)).clip(1e-6, 0.5).to_numpy()

# ---------------- M1: logistic discrete-time hazard (weighted case-control) + full-data replication
design = p1.LogitDesign().fit(tr)
X_all = design.transform(P)
coefs, rep = {}, {}
rng = np.random.default_rng(0)
for c in CAUSES:
    y = tr[f"y_{c}"].to_numpy()
    q = 0.10                                     # case-control: all events + 10% of non-events
    keep = (y == 1) | (rng.random(len(y)) < q)
    Xc = sm.add_constant(X_all.loc[tr.index[keep]].astype(np.float64), has_constant="add")
    glm = sm.GLM(y[keep], Xc, family=sm.families.Binomial())
    res = glm.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(tr.fips5.to_numpy()[keep])[0]}, maxiter=100)
    params = res.params.copy(); params["const"] += np.log(q)      # prior correction (King & Zeng 2001)
    coefs[c] = pd.DataFrame({"coef": params, "se_cluster": res.bse, "z": res.tvalues, "p": res.pvalues})
    eta = sm.add_constant(X_all, has_constant="add").to_numpy(np.float64) @ params.to_numpy()
    preds[f"m1_{c}"] = 1 / (1 + np.exp(-eta))
    res = type("R", (), {"params": params})()
    with open(DW / f"m1_{c}.pkl", "wb") as f: pickle.dump(res.params, f)
    # replication: independent full-data fit (ALL training rows, no subsampling, no prior correction), IRLS
    full = sm.GLM(y, sm.add_constant(X_all.loc[tr.index].astype(np.float64), has_constant="add"),
                  family=sm.families.Binomial()).fit(maxiter=50)
    cmp = pd.DataFrame({"case_control": params, "full_irls": full.params})
    key = ["gap_pos", "gap_neg", "log_appr", "est_ltv", "pmms", "log_value"]
    p_rep = full.predict(sm.add_constant(X_all.astype(np.float64), has_constant="add")).to_numpy()
    rep[c] = {"max_abs_coef_diff_key": float((cmp.loc[key, "case_control"] - cmp.loc[key, "full_irls"]).abs().max()),
              "max_abs_diff_in_se_units_key": float(((cmp.loc[key, "case_control"] - cmp.loc[key, "full_irls"]).abs()
                                                     / coefs[c].loc[key, "se_cluster"]).max()),
              "coef_corr_all": float(cmp.drop("const").corr().iloc[0, 1]),
              "pred_corr_all_rows": float(np.corrcoef(p_rep, preds[f"m1_{c}"])[0, 1]),
              "mean_pred_ratio_cc_over_full": float(preds[f"m1_{c}"].mean() / p_rep.mean()),
              "key_coefs": cmp.loc[key].round(4).to_dict()}
    print(c, "M1 done", rep[c]["max_abs_coef_diff_key"], f"({time.time()-t0:.0f}s)", flush=True)
for c in CAUSES:
    coefs[c].to_csv(OUT / f"m1_coefficients_{c}.csv")
json.dump(rep, open(OUT / "m1_replication.json", "w"), indent=2)

# ---------------- M2: LightGBM challenger with economically motivated monotone constraints
feat = p1.LGB_FEATURES
mono = {"sale": {"gap": -1}, "new_lien": {"gap": -1, "pmms": -1}}
imp = {}
for c in CAUSES:
    y = f"y_{c}"
    fit_rows = tr[tr.t <= p1.TRAIN_END - 6]; val_rows = tr[tr.t > p1.TRAIN_END - 6]   # early stopping on the last 6 train months
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=2000, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=30, seed=0,
                  monotone_constraints=[mono[c].get(f, 0) for f in feat], monotone_constraints_method="advanced")
    dtr = lgb.Dataset(fit_rows[feat], fit_rows[y], categorical_feature=["state_code", "moy"])
    dva = lgb.Dataset(val_rows[feat], val_rows[y], reference=dtr)
    b0 = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    n_best = b0.best_iteration
    bst = lgb.train(params, lgb.Dataset(tr[feat], tr[y], categorical_feature=["state_code", "moy"]), n_best)
    Xp = P[feat].copy(); Xp.loc[P.split.str.startswith("spatial"), "state_code"] = -1   # held-out states: unseen category
    preds[f"m2_{c}"] = bst.predict(Xp)
    bst.save_model(str(DW / f"m2_{c}.txt"))
    imp[c] = pd.Series(bst.feature_importance("gain"), index=feat).sort_values(ascending=False)
    print(c, "M2 done, trees", n_best, f"({time.time()-t0:.0f}s)", flush=True)
pd.DataFrame(imp).to_csv(OUT / "m2_gain_importance.csv")
preds.to_parquet(DW / "preds.parquet", index=False)

# ---------------- evaluation
rows = []
for c in CAUSES:
    for sp in ["train", "oot", "spatial", "spatial_oot"]:
        g = preds[preds.split == sp]; y = g[f"y_{c}"].to_numpy()
        for m in ["m0", "m1", "m2"]:
            s = g[f"{m}_{c}"].to_numpy()
            rows.append({"cause": c, "split": sp, "model": m, "rows": len(g), "events": int(y.sum()),
                         "auc": auc(y, s), "brier_x1e4": brier(y, s) * 1e4,
                         "logloss": float(-np.mean(y * np.log(s) + (1 - y) * np.log(1 - s))),
                         "pred_over_actual": float(s.mean() / y.mean())})
ev = pd.DataFrame(rows); ev.to_csv(OUT / "eval_monthly_hazard.csv", index=False)
print(ev.round(4).to_string(index=False))
print(f"total {time.time()-t0:.0f}s")
