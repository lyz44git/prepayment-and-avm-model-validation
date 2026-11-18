"""Valuation study, part 2: challenger AVMs on identical rows.
B0 index AVM (prior sale x repeat-sales index), B1 hedonic OLS (state FE), C1 LightGBM point model,
C1q conformalised quantile LightGBM (CQR, 80% nominal). Training <= 2024-03, calibration 2024-04..06,
tests: OOT 2024-07..2025-08 (non-hold-out states) and spatial (9 hold-out states, all months)."""
import sys, time, json, functools; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import WORK, REPORTS
import numpy as np, pandas as pd, lightgbm as lgb, statsmodels.api as sm, shap
print = functools.partial(print, flush=True)
t0 = time.time(); OUT = REPORTS / "p2"; DW = WORK / "p2"
S = pd.read_parquet(DW / "features.parquet")
S["state_code"] = S.state.astype("category").cat.codes
S["moy"] = S.t_sale % 12 + 1
FEAT = ["prev_idx_logp", "months_since_prev", "log_mv0", "log_av0", "log_cv0", "log_tax0", "age", "log_sqft", "beds",
        "baths", "log_lot", "pool", "condo", "stories0", "units0", "comp_med5", "comp_med20", "comp_iqr20",
        "comp_lppsf_med20", "comp_dist1", "comp_dist20", "comp_lsqft_med20", "comp_age_med20", "comp_n_1km",
        "comp_implied", "mkt_idx", "lat", "lon", "log_density", "t_owned", "t_medage", "state_code", "moy"]
# race / ethnicity shares are deliberately NOT features (used only for the disparity audit)
tr, ca = S[S.split == "train"], S[S.split == "calib"]
P = S[["addr_id", "split", "state", "t_sale", "price", "y"]].copy()
# B0: index AVM
P["b0_index"] = S.prev_idx_logp
# B1: hedonic OLS with state fixed effects (NaNs -> train median + missing flag)
num = [f for f in FEAT if f not in ("state_code", "moy", "lat", "lon")]
med = tr[num].median()
def design(d):
    X = d[num].fillna(med)
    for c in ["prev_idx_logp", "log_mv0", "log_av0", "log_cv0", "log_tax0", "age", "log_sqft", "comp_med20"]:
        X[c + "_na"] = d[c].isna().astype(float)
    X["age2"] = X.age ** 2
    X = pd.concat([X, pd.get_dummies(d.state, prefix="st", dtype=float)], axis=1)
    return X
Xtr = design(tr); cols = Xtr.columns
ols = sm.OLS(tr.y, sm.add_constant(Xtr)).fit()
P["b1_hedonic"] = ols.predict(sm.add_constant(design(S).reindex(columns=cols, fill_value=0.0)))
print("OLS done", f"R2={ols.rsquared:.3f}", f"({time.time()-t0:.0f}s)")
# C1: LightGBM point model
base = dict(learning_rate=0.05, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=28, seed=0)
dtr = lgb.Dataset(tr[FEAT], tr.y, categorical_feature=["state_code", "moy"], free_raw_data=False)
dca = lgb.Dataset(ca[FEAT], ca.y, reference=dtr)
gbm = lgb.train({**base, "objective": "regression"}, dtr, 6000, valid_sets=[dca],
                callbacks=[lgb.early_stopping(200, verbose=False)])
Xs = S[FEAT].copy(); Xs.loc[S.split == "spatial", "state_code"] = -1          # unseen state for hold-out rows
P["c1_gbm"] = gbm.predict(Xs, num_iteration=gbm.best_iteration)
gbm.save_model(str(DW / "c1_gbm.txt"), num_iteration=gbm.best_iteration)
print("GBM done, trees", gbm.best_iteration, f"({time.time()-t0:.0f}s)")
# C1q: quantile models + split-conformal calibration on the calibration window (CQR, Romano et al. 2019)
q = {}
for a in (0.1, 0.9):
    m = lgb.train({**base, "objective": "quantile", "alpha": a}, dtr, gbm.best_iteration)
    q[a] = m.predict(Xs)
lo, hi = q[0.1], q[0.9]
cmask = (S.split == "calib").to_numpy()
E = np.maximum(lo[cmask] - S.y.to_numpy()[cmask], S.y.to_numpy()[cmask] - hi[cmask])
n = cmask.sum(); qhat = np.quantile(E, min(1, 0.8 * (1 + 1 / n)))
P["c1_lo"], P["c1_hi"] = lo - qhat, hi + qhat
P["c1_width_pct"] = np.exp(P.c1_hi - P.c1_lo) - 1
print("CQR qhat", round(qhat, 4), f"({time.time()-t0:.0f}s)")
P.to_parquet(DW / "preds_gbm.parquet", index=False)
imp = pd.Series(gbm.feature_importance("gain"), index=FEAT).sort_values(ascending=False)
imp.to_csv(OUT / "c1_gain_importance.csv")
# SHAP on 20k OOT rows, aggregated by feature family
oot_idx = S.index[S.split == "oot"]
smp = S.loc[np.random.default_rng(0).choice(oot_idx, 20000, replace=False), FEAT]
sv = shap.TreeExplainer(gbm).shap_values(smp)
fam = {"prior sale (indexed)": ["prev_idx_logp", "months_since_prev"], "assessor 2021": ["log_mv0", "log_av0", "log_cv0", "log_tax0"],
       "property attributes": ["age", "log_sqft", "beds", "baths", "log_lot", "pool", "condo", "stories0", "units0"],
       "neighbour comps (pre-2022)": [f for f in FEAT if f.startswith("comp_")], "market index": ["mkt_idx", "moy"],
       "location / tract": ["lat", "lon", "log_density", "t_owned", "t_medage", "state_code"]}
ma = pd.Series(np.abs(sv).mean(0), index=FEAT)
fs = pd.Series({k: ma[v].sum() for k, v in fam.items()}); fs = (fs / fs.sum()).sort_values(ascending=False)
fs.to_csv(OUT / "c1_shap_by_family.csv"); ma.sort_values(ascending=False).to_csv(OUT / "c1_shap_by_feature.csv")
print(fs.round(3).to_string()); print(f"done {time.time()-t0:.0f}s")
