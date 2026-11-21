"""Prepayment study: sensitivity checks (2026-11-19).

1. Third exit in the 12-month backtest (17). Homes that leave through another transfer (quitclaim, trustee deed,
   gift, affidavit) count as "no event" in the observed 12-month rate, while the forecast only subtracts sales and new
   liens. Here those homes are removed (censored) and the 12-month rate, calibration and AUC are recomputed for the
   logit and LightGBM.
2. Recalibration by cause (22). The pool control rescales total predicted CPR by the trailing three-month actual /
   predicted ratio. Here sales and new liens are rescaled separately, each by its own ratio over months t-2..t-4
   (the two-month recording lag), and the pool CPR rebuilt.
3. Paired ranking by month. Sale and new-lien AUC of LightGBM and the logit computed within each month on the same
   rows; reported: the share of months in which LightGBM ranks better and the mean difference.
Writes reports/p1/r6_cum12_third_exit.csv, r6_recalibration_by_cause.csv and r6_auc_by_month.csv.
"""
import sys, pickle, functools, warnings; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
from _paths import WORK, REPORTS
from mval import p1
from mval.metrics import auc
from mval.amortization import cpr_from_smm
import numpy as np, pandas as pd, statsmodels.api as sm, lightgbm as lgb
print = functools.partial(print, flush=True)
OUT = REPORTS / "p1"; DW = WORK / "p1"; H = 12; T1 = p1.TRAIN_END + 1

# ---------------------------------------------------------------- 1. third exit
P = pd.read_parquet(DW / "panel.parquet"); tr = P[P.split == "train"]
design = p1.LogitDesign().fit(tr); st_share = tr.state.value_counts(normalize=True)
params = {c: pickle.load(open(DW / f"m1_{c}.pkl", "rb")) for c in ["sale", "new_lien"]}
m2 = {c: lgb.Booster(model_file=str(DW / f"m2_{c}.txt")) for c in ["sale", "new_lien"]}
def m1_pred(c, d):
    X = sm.add_constant(design.transform(d), has_constant="add").reindex(columns=params[c].index, fill_value=0)
    unseen = ~d.state.isin(design.states).to_numpy()
    for s_ in design.states[1:]:
        if f"st_{s_}" in X.columns:
            X.loc[unseen, f"st_{s_}"] = st_share.get(s_, 0.0)
    return 1 / (1 + np.exp(-(X.to_numpy(np.float64) @ params[c].to_numpy())))
pm = p1.rate_vector(); st_idx, loghpi = p1.hpi_matrix(pd.read_parquet(WORK / "hpi_state_monthly.parquet"), pd.read_csv(WORK / "hpi_lookup.csv"), "full")
rows = []
for sp in ["oot", "spatial_oot"]:
    start = P[(P.split == sp) & (P.t == T1)].copy(); later = P[P.split == sp]
    g = later.groupby("addr_id")
    act = (later.y_sale + later.y_new_lien).groupby(later.addr_id).max().reindex(start.addr_id).fillna(0).to_numpy()
    other = later.y_other_transfer.groupby(later.addr_id).max().reindex(start.addr_id).fillna(0).to_numpy()
    si = start.state.map(st_idx).to_numpy(); tp = start.t_purchase.to_numpy().astype(int)
    cum = {"m1": np.zeros(len(start)), "m2": np.zeros(len(start))}
    for k in range(H):
        t = T1 + k; d = start.copy()
        d["t"] = t; d["tenure_m"] = t - tp; d["pmms"] = pm[t]; d["gap"] = d.pmms - d.note_hat; d["pmms_chg3"] = pm[t] - pm[t - 3]
        d["log_appr"] = loghpi[si, t] - loghpi[si, tp]
        r = d.note_hat.to_numpy() / 1200; kk = np.minimum(d.tenure_m.to_numpy(), 360)
        d["est_ltv"] = 0.8 * ((1 + r) ** 360 - (1 + r) ** kk) / ((1 + r) ** 360 - 1) / np.exp(d.log_appr.to_numpy())
        d["moy"] = (t % 12) + 1
        Xg = d[p1.LGB_FEATURES].copy(); Xg.loc[~d.state.isin(design.states), "state_code"] = -1
        h = {"m1": m1_pred("sale", d) + m1_pred("new_lien", d), "m2": m2["sale"].predict(Xg) + m2["new_lien"].predict(Xg)}
        for m in cum: cum[m] += np.log1p(-np.clip(h[m], 0, 0.99))
    for m in cum:
        pr = 1 - np.exp(cum[m])
        for version, keep in (("published (other transfers count as no event)", np.ones(len(start), bool)),
                              ("other transfers removed (censored)", other == 0)):
            rows.append({"split": sp, "model": m, "version": version, "properties": int(keep.sum()),
                         "other_transfers_removed": int((~keep).sum()), "actual_rate_pct": act[keep].mean() * 100,
                         "pred_rate_pct": pr[keep].mean() * 100, "auc_12m": auc(act[keep], pr[keep])})
c12 = pd.DataFrame(rows); c12.to_csv(OUT / "r6_cum12_third_exit.csv", index=False); print(c12.round(4).to_string(index=False))

# ---------------------------------------------------------------- 2. recalibration by cause
a = pd.read_csv(OUT / "monthly_cpr_actual_vs_pred.csv")
rc = []
for sp in ["oot", "spatial_oot"]:
    full = a[a.split.isin(["train", "oot"] if sp == "oot" else ["spatial", "spatial_oot"])].sort_values("month").reset_index(drop=True)
    for m in ["m1", "m2"]:
        y = cpr_from_smm(full.y_sale + full.y_new_lien); raw = cpr_from_smm(full[f"{m}_sale"] + full[f"{m}_new_lien"])
        def scale(col_y, col_m):
            ratio = (full[col_y].shift(2) + full[col_y].shift(3) + full[col_y].shift(4)) / (full[col_m].shift(2) + full[col_m].shift(3) + full[col_m].shift(4))
            return full[col_m] * ratio
        pool_ratio = ((full.y_sale + full.y_new_lien).shift(2) + (full.y_sale + full.y_new_lien).shift(3) + (full.y_sale + full.y_new_lien).shift(4)) / \
                     ((full[f"{m}_sale"] + full[f"{m}_new_lien"]).shift(2) + (full[f"{m}_sale"] + full[f"{m}_new_lien"]).shift(3) + (full[f"{m}_sale"] + full[f"{m}_new_lien"]).shift(4))
        pool = cpr_from_smm((full[f"{m}_sale"] + full[f"{m}_new_lien"]) * pool_ratio)
        bycause = cpr_from_smm(scale("y_sale", f"{m}_sale") + scale("y_new_lien", f"{m}_new_lien"))
        o = (full.split == sp).to_numpy()
        own = {c: full[f"{m}_{c}"] * (full[f"y_{c}"].shift(2) + full[f"y_{c}"].shift(3) + full[f"y_{c}"].shift(4)) /
                  (full[f"{m}_{c}"].shift(2) + full[f"{m}_{c}"].shift(3) + full[f"{m}_{c}"].shift(4)) for c in ("sale", "new_lien")}
        by_cause = {"fixed": {c: full[f"{m}_{c}"] for c in own}, "pool recalibration (t-2..t-4)": {c: full[f"{m}_{c}"] * pool_ratio for c in own},
                    "recalibration by cause (t-2..t-4)": own}       # each exit's predicted / recorded count after the rescaling
        for name, pr in (("fixed", raw), ("pool recalibration (t-2..t-4)", pool), ("recalibration by cause (t-2..t-4)", bycause)):
            pr = np.asarray(pr, float)[o]; yy = np.asarray(y, float)[o]
            ratios = {f"pred_over_actual_{c}": float(np.asarray(by_cause[name][c], float)[o].sum() / full[f"y_{c}"].to_numpy()[o].sum()) for c in own}
            rc.append({"split": sp, "model": m, "forecast": name, "months": int(o.sum()),
                       "MAE_monthly_cpr_pp": float(np.abs(pr - yy).mean() * 100), "corr": float(np.corrcoef(pr, yy)[0, 1]), **ratios})
rc = pd.DataFrame(rc); rc.to_csv(OUT / "r6_recalibration_by_cause.csv", index=False); print(rc.round(4).to_string(index=False))

# ---------------------------------------------------------------- 3. paired ranking by month
pr_ = pd.read_parquet(DW / "preds_v2.parquet", columns=["t", "split", "y_sale", "y_new_lien", "m1_sale", "m1_new_lien", "m2_sale", "m2_new_lien"])
am = []
for sp in ["train", "oot", "spatial_oot"]:
    for c in ["sale", "new_lien"]:
        diffs = []
        for t, g in pr_[pr_.split == sp].groupby("t"):
            y = g[f"y_{c}"].to_numpy()
            if 0 < y.sum() < len(y):
                diffs.append(auc(y, g[f"m2_{c}"].to_numpy()) - auc(y, g[f"m1_{c}"].to_numpy()))
        diffs = np.array(diffs)
        am.append({"split": sp, "cause": c, "months": len(diffs), "months_lightgbm_ranks_better": int((diffs > 0).sum()),
                   "mean_auc_difference_lgb_minus_logit": float(diffs.mean()), "min_difference": float(diffs.min()), "max_difference": float(diffs.max())})
am = pd.DataFrame(am); am.to_csv(OUT / "r6_auc_by_month.csv", index=False); print(am.round(4).to_string(index=False))
print("done")
