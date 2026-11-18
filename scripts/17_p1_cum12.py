"""Prepayment study: proper 12-month-ahead termination forecast (replaces the exposure-biased cumulative in 09).

Every property at risk on 2024-07-01 (first OOT month) is rolled forward 12 months with the REALISED rate and
HPI paths (a conditional forecast, as in CCAR-style backtests): hazards for all 12 months are predicted from the
2024-07 state, regardless of when or whether the property actually terminates. Outcome = sale or new lien in
2024-07..2025-06. Reported: AUC, decile calibration, predicted vs actual 12-month rate, for M0/M1/M2."""
import sys, pickle, functools, warnings; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
from _paths import WORK, REPORTS
from mval import p1
from mval.metrics import auc, lift_table
import numpy as np, pandas as pd, statsmodels.api as sm, lightgbm as lgb
print = functools.partial(print, flush=True)
OUT = REPORTS / "p1"; DW = WORK / "p1"; H = 12; T1 = p1.TRAIN_END + 1
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
# M0 lookup tables (train only), as in 08
bins = [-1, 12, 24, 36, 60, 84, 120, 180, 240, 1000]
trb = tr.assign(ten_b=pd.cut(tr.tenure_m, bins, labels=False))
m0 = {}
for c in ["sale", "new_lien"]:
    y = f"y_{c}"; st_t = trb.groupby(["state", "ten_b"])[y].agg(["sum", "count"]); nat = trb.groupby("ten_b")[y].mean()
    shr = (st_t["sum"] + 200 * nat.reindex(st_t.index.get_level_values(1)).to_numpy()) / (st_t["count"] + 200)
    m0[c] = (shr, nat, trb.groupby("moy")[y].mean() / trb[y].mean())
def m0_pred(c, d):
    shr, nat, moy = m0[c]; tb = pd.cut(d.tenure_m, bins, labels=False)
    b = pd.Series(shr.reindex(pd.MultiIndex.from_arrays([d.state, tb])).to_numpy(), index=d.index).fillna(tb.map(nat))
    return (b * d.moy.map(moy)).clip(1e-6, 0.5).to_numpy()
pm = p1.rate_vector(); st_idx, loghpi = p1.hpi_matrix(pd.read_parquet(WORK / "hpi_state_monthly.parquet"), pd.read_csv(WORK / "hpi_lookup.csv"), "full")
rows = []
for sp in ["oot", "spatial_oot"]:
    start = P[(P.split == sp) & (P.t == T1)].copy()
    later = P[(P.split == sp)]
    act = (later.y_sale + later.y_new_lien).groupby(later.addr_id).max().reindex(start.addr_id).fillna(0).to_numpy()
    si = start.state.map(st_idx).to_numpy(); tp = start.t_purchase.to_numpy().astype(int)
    cum = {m: np.zeros(len(start)) for m in ["m0", "m1", "m2"]}
    for k in range(H):
        t = T1 + k; d = start.copy()
        d["t"] = t; d["tenure_m"] = t - tp; d["pmms"] = pm[t]; d["gap"] = d.pmms - d.note_hat; d["pmms_chg3"] = pm[t] - pm[t - 3]
        d["log_appr"] = loghpi[si, t] - loghpi[si, tp]
        r = d.note_hat.to_numpy() / 1200; kk = np.minimum(d.tenure_m.to_numpy(), 360)
        d["est_ltv"] = 0.8 * ((1 + r) ** 360 - (1 + r) ** kk) / ((1 + r) ** 360 - 1) / np.exp(d.log_appr.to_numpy())
        d["moy"] = (t % 12) + 1
        Xg = d[p1.LGB_FEATURES].copy(); Xg.loc[~d.state.isin(design.states), "state_code"] = -1
        h = {"m0": m0_pred("sale", d) + m0_pred("new_lien", d), "m1": m1_pred("sale", d) + m1_pred("new_lien", d),
             "m2": m2["sale"].predict(Xg) + m2["new_lien"].predict(Xg)}
        for m in cum:
            cum[m] += np.log1p(-np.clip(h[m], 0, 0.99))
    for m in cum:
        pr = 1 - np.exp(cum[m])
        lt = lift_table(act, pr, 10)
        dec = pd.qcut(pd.Series(pr).rank(method="first"), 10, labels=False)
        cal = pd.DataFrame({"pred": pr, "act": act, "dec": dec}).groupby("dec").mean()
        cal.to_csv(OUT / f"cum12_calibration_{sp}_{m}.csv")
        rows.append({"split": sp, "model": m, "properties": len(start), "events_12m": int(act.sum()),
                     "auc_12m": auc(act, pr), "pred_rate_pct": pr.mean() * 100, "actual_rate_pct": act.mean() * 100,
                     "top_decile_actual_pct": cal.act.iloc[-1] * 100, "bottom_decile_actual_pct": cal.act.iloc[0] * 100,
                     "top_decile_lift": cal.act.iloc[-1] / act.mean(), "max_decile_abs_gap_pp": (cal.pred - cal.act).abs().max() * 100})
R = pd.DataFrame(rows); R.to_csv(OUT / "cum12_backtest.csv", index=False); print(R.round(3).to_string(index=False))
