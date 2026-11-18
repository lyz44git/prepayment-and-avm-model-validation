"""Prepayment study analysis: lock-in effects, OOT / spatial backtests, calibration, sensitivity refits, stability,
monitoring PSI, and the rate-scenario stress test with pool runoff. Reads the panel and fitted models from 08."""
import sys, time, json, pickle, functools; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import WORK, REPORTS, duck
from mval import p1
from mval.metrics import auc, psi
from mval.amortization import cpr_from_smm, pool_projection
import numpy as np, pandas as pd, statsmodels.api as sm, lightgbm as lgb
print = functools.partial(print, flush=True)
t0 = time.time(); OUT = REPORTS / "p1"; DW = WORK / "p1"
P = pd.read_parquet(DW / "panel.parquet"); preds = pd.read_parquet(DW / "preds.parquet")
tr = P[P.split == "train"]
design = p1.LogitDesign().fit(tr)
params = {c: pickle.load(open(DW / f"m1_{c}.pkl", "rb")) for c in ["sale", "new_lien"]}
def m1_predict(c, d, des=design, prm=None):
    prm = params[c] if prm is None else prm
    X = sm.add_constant(des.transform(d), has_constant="add").reindex(columns=prm.index, fill_value=0)
    return 1 / (1 + np.exp(-(X.to_numpy(np.float64) @ prm.to_numpy())))
m2 = {c: lgb.Booster(model_file=str(DW / f"m2_{c}.txt")) for c in ["sale", "new_lien"]}
res = {}
# ---------------- 0. unseen-state fix for M1: replace the reference-state default by the training-share average
st_share = tr.state.value_counts(normalize=True)
def m1_predict_marg(c, d):
    X = sm.add_constant(design.transform(d), has_constant="add").reindex(columns=params[c].index, fill_value=0)
    unseen = ~d.state.isin(design.states).to_numpy()
    for s_ in design.states[1:]:
        if f"st_{s_}" in X.columns:
            X.loc[unseen, f"st_{s_}"] = st_share.get(s_, 0.0)
    return 1 / (1 + np.exp(-(X.to_numpy(np.float64) @ params[c].to_numpy())))
sp_rows = preds.split.str.startswith("spatial").to_numpy()
fix_rows = []
for c in ["sale", "new_lien"]:
    preds[f"m1naive_{c}"] = preds[f"m1_{c}"]
    preds.loc[sp_rows, f"m1_{c}"] = m1_predict_marg(c, P.loc[sp_rows])
    for sp in ["spatial", "spatial_oot"]:
        g = preds[preds.split == sp]; y = g[f"y_{c}"].to_numpy()
        for m in ["m1naive", "m1"]:
            s_ = g[f"{m}_{c}"].to_numpy()
            fix_rows.append({"cause": c, "split": sp, "model": m, "auc": auc(y, s_), "pred_over_actual": s_.mean() / y.mean()})
fx = pd.DataFrame(fix_rows); fx.to_csv(OUT / "m1_unseen_state_fix.csv", index=False); print(fx.round(4).to_string(index=False))
ev = pd.read_csv(OUT / "eval_monthly_hazard.csv")        # from 08; replace naive M1 spatial rows with the fixed ones
for _, r in fx[fx.model == "m1"].iterrows():
    ev.loc[(ev.cause == r.cause) & (ev.split == r.split) & (ev.model == "m1"), ["auc", "pred_over_actual"]] = [r.auc, r.pred_over_actual]
ev.to_csv(OUT / "eval_monthly_hazard_final.csv", index=False)
preds.to_parquet(DW / "preds_v2.parquet", index=False)
# ---------------- 1. lock-in: odds ratios and average marginal effect of +1pp rate gap (training rows)
coef = {c: pd.read_csv(OUT / f"m1_coefficients_{c}.csv", index_col=0) for c in ["sale", "new_lien"]}
smp = tr.sample(200_000, random_state=1)
for c in ["sale", "new_lien"]:
    b, se = coef[c].loc["gap_pos", "coef"], coef[c].loc["gap_pos", "se_cluster"]
    h0 = m1_predict(c, smp); h1 = m1_predict(c, smp.assign(gap=smp.gap + 1.0))
    h0g = m2[c].predict(smp[p1.LGB_FEATURES]); h1g = m2[c].predict(smp.assign(gap=smp.gap + 1.0)[p1.LGB_FEATURES])
    res[f"lockin_{c}"] = {"gap_pos_coef": b, "se_cluster": se, "odds_ratio_per_pp": float(np.exp(b)),
                          "or_95ci": [float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se))],
                          "m1_ame_pct_change_hazard_per_pp": float(h1.mean() / h0.mean() - 1),
                          "m2_ame_pct_change_hazard_per_pp": float(h1g.mean() / h0g.mean() - 1),
                          "gap_neg_coef": float(coef[c].loc["gap_neg", "coef"]), "gap_neg_se": float(coef[c].loc["gap_neg", "se_cluster"])}
print(json.dumps({k: v for k, v in res.items()}, indent=1))
# partial dependence curves (sale & new lien) over the gap grid
grid = np.arange(-3.0, 5.01, 0.25); pd_rows = []
s2 = smp.sample(40_000, random_state=2)
for g in grid:
    d = s2.assign(gap=g)
    pd_rows.append({"gap": g, **{f"m1_{c}": m1_predict(c, d).mean() for c in ["sale", "new_lien"]},
                    **{f"m2_{c}": m2[c].predict(d[p1.LGB_FEATURES]).mean() for c in ["sale", "new_lien"]}})
pd.DataFrame(pd_rows).to_csv(OUT / "pdp_gap.csv", index=False)
gap_dist = tr.groupby(pd.cut(tr.gap, np.arange(-6, 7, 1.0)), observed=True).agg(rows=("gap", "size"), sale=("y_sale", "mean"), new_lien=("y_new_lien", "mean"))
gap_dist.to_csv(OUT / "empirical_hazard_by_gap.csv"); print(gap_dist.to_string())
# ---------------- 2. monthly aggregate CPR: actual vs predicted
preds["month"] = p1.MONTHS[preds.t.to_numpy()]
agg = preds.groupby(["split", "month"]).agg(n=("t", "size"), **{f"{k}": (k, "mean") for k in
       ["y_sale", "y_new_lien", "m0_sale", "m1_sale", "m2_sale", "m0_new_lien", "m1_new_lien", "m2_new_lien"]}).reset_index()
for k in ["y", "m0", "m1", "m2"]:
    agg[f"cpr_total_{k}"] = cpr_from_smm(agg[f"{k}_sale"] + agg[f"{k}_new_lien"])
agg.to_csv(OUT / "monthly_cpr_actual_vs_pred.csv", index=False)
mrows = []
for sp in ["train", "oot", "spatial", "spatial_oot"]:
    a = agg[agg.split == sp]
    for m in ["m0", "m1", "m2"]:
        mrows.append({"split": sp, "model": m, "months": len(a),
                      "actual_cpr_avg": a.cpr_total_y.mean(), "pred_cpr_avg": a[f"cpr_total_{m}"].mean(),
                      "MAE_monthly_cpr_pp": (a[f"cpr_total_{m}"] - a.cpr_total_y).abs().mean() * 100,
                      "corr_monthly": np.corrcoef(a[f"cpr_total_{m}"], a.cpr_total_y)[0, 1]})
mt = pd.DataFrame(mrows); mt.to_csv(OUT / "monthly_cpr_backtest_summary.csv", index=False); print(mt.round(3).to_string(index=False))
# ---------------- 3. 12-month cumulative forecast: see scripts/17_p1_cum12.py.
# (An earlier version summed hazards only over each property's observed months; that measures exposure length,
#  not risk, and inverted the ranking (AUC ~0.19). It was removed; 17 rolls every property forward 12 months.)
# ---------------- 4. sensitivity and stability refits (case-control logit, same design family)
def cc_fit(d, ycol, des, q=0.10, seed=0):
    y = d[ycol].to_numpy(); keep = (y == 1) | (np.random.default_rng(seed).random(len(y)) < q)
    X = sm.add_constant(des.transform(d.loc[d.index[keep]]).astype(np.float64), has_constant="add")
    r = sm.GLM(y[keep], X, family=sm.families.Binomial()).fit(cov_type="cluster",
              cov_kwds={"groups": pd.factorize(d.fips5.to_numpy()[keep])[0]}, maxiter=100)
    return r
sens = []
def add(label, c, r):
    sens.append({"spec": label, "cause": c, "gap_pos": r.params["gap_pos"], "se": r.bse["gap_pos"],
                 "odds_ratio": float(np.exp(r.params["gap_pos"])), "log_appr": r.params.get("log_appr", np.nan),
                 "n_cc_rows": int(r.nobs)})
for c in ["sale", "new_lien"]:
    add("baseline (purchase-month PMMS)", c, type("R", (), {"params": coef[c].coef, "bse": coef[c].se_cluster, "nobs": 0})())
    add("note rate = cohort median recorded rate (regression calibration)", c, cc_fit(tr.assign(gap=tr.gap_rc), f"y_{c}", p1.LogitDesign().fit(tr.assign(gap=tr.gap_rc))))
    add("tenure knots (12,60,120,240)", c, cc_fit(tr, f"y_{c}", p1.LogitDesign(tenure_knots=(12, 60, 120, 240)).fit(tr)))
    early, late = tr[tr.t <= p1.T0 + 13], tr[tr.t > p1.T0 + 13]
    add("stability: 2022-05..2023-06 only", c, cc_fit(early, f"y_{c}", p1.LogitDesign().fit(early)))
    add("stability: 2023-07..2024-06 only", c, cc_fit(late, f"y_{c}", p1.LogitDesign().fit(late)))
    add("case-control seed 1", c, cc_fit(tr, f"y_{c}", design, seed=1))
# owner-change-confirmed sales only (non-confirmed sales treated as censored at the sale month)
tr_oc = tr.assign(y_sale=(tr.y_sale * tr.owner_changed.fillna(False).astype(int)))
add("sale event requires owner surname change", "sale", cc_fit(tr_oc, "y_sale", design))
sens = pd.DataFrame(sens); sens.to_csv(OUT / "sensitivity_lockin.csv", index=False); print(sens.round(4).to_string(index=False))
# ---------------- 5. monitoring: PSI train -> OOT for scores and key drivers
mon = []
for col in ["m1_sale", "m2_sale", "m1_new_lien", "m2_new_lien"]:
    mon.append({"variable": col, "psi_train_vs_oot": psi(preds.loc[preds.split == "train", col], preds.loc[preds.split == "oot", col])})
for col in ["gap", "tenure_m", "pmms", "log_appr", "est_ltv", "log_value"]:
    mon.append({"variable": col, "psi_train_vs_oot": psi(P.loc[P.split == "train", col].dropna(), P.loc[P.split == "oot", col].dropna())})
mon = pd.DataFrame(mon); mon.to_csv(OUT / "monitoring_psi.csv", index=False); print(mon.round(4).to_string(index=False))
json.dump(res, open(OUT / "lockin_effects.json", "w"), indent=2)
print(f"analysis done {time.time()-t0:.0f}s")
