"""Prepayment study stress test: 24-month projection (2025-07..2027-06) of turnover + new-lien terminations for the
properties still at risk at 2025-06, under rate and house-price scenarios, with M1 (logit) and M2 (GBM).

Pool for the balance runoff is ILLUSTRATIVE: each at-risk property is assumed to carry its purchase-money
loan (80% LTV, 30-year level pay at the purchase-month PMMS), because the actual lien at risk is not observed
for every property. Count-based termination rates do not depend on that assumption.

The count-based CPR averages each month's hazard over the starting pool (every home weighted equally, a rate
standardised to the starting composition). Since 2026-11-19 the script also reports the survival-weighted CPR, which weights each
home by its probability of still being in the pool at the start of the month: the pool's actual run-off rate."""
import sys, json, pickle, functools, warnings; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
from _paths import WORK, REPORTS
from mval import p1
from mval.amortization import cpr_from_smm, pool_projection
import numpy as np, pandas as pd, statsmodels.api as sm, lightgbm as lgb
print = functools.partial(print, flush=True)
OUT = REPORTS / "p1"; DW = WORK / "p1"; H = 24
P = pd.read_parquet(DW / "panel.parquet")
tr = P[P.split == "train"]; design = p1.LogitDesign().fit(tr)
params = {c: pickle.load(open(DW / f"m1_{c}.pkl", "rb")) for c in ["sale", "new_lien"]}
m2 = {c: lgb.Booster(model_file=str(DW / f"m2_{c}.txt")) for c in ["sale", "new_lien"]}
st_share = tr.state.value_counts(normalize=True)
def m1_pred(c, d):
    X = sm.add_constant(design.transform(d), has_constant="add").reindex(columns=params[c].index, fill_value=0)
    unseen = ~d.state.isin(design.states).to_numpy()
    for s_ in design.states[1:]:
        if f"st_{s_}" in X.columns:
            X.loc[unseen, f"st_{s_}"] = st_share.get(s_, 0.0)
    return 1 / (1 + np.exp(-(X.to_numpy(np.float64) @ params[c].to_numpy())))
# starting pool: last observed month = 2025-06 and no event
last = P[(P.t == p1.T_END) & (P.y_sale == 0) & (P.y_new_lien == 0) & (P.y_other_transfer == 0)].copy()
print("at-risk properties at 2025-06:", len(last))
pm = p1.rate_vector(); st_idx, loghpi = p1.hpi_matrix(pd.read_parquet(WORK / "hpi_state_monthly.parquet"), pd.read_csv(WORK / "hpi_lookup.csv"), "full")
r0 = pm[p1.T_END]
months = np.arange(p1.T_END + 1, p1.T_END + H + 1)
ramp = np.minimum(np.arange(1, H + 1) / 6, 1.0)
realized = pm[months].copy()       # FRED file runs to 2026-10, then forward-filled
SCEN = {"base (flat at 2025-06)": (np.full(H, r0), 0.0), "realized PMMS path": (realized, 0.0),
        "-100bp": (r0 - 1.0 * ramp, 0.0), "-200bp": (r0 - 2.0 * ramp, 0.0), "+100bp": (r0 + 1.0 * ramp, 0.0),
        "+200bp": (r0 + 2.0 * ramp, 0.0), "HPI -10% (rates flat)": (np.full(H, r0), -0.10)}
si = last.state.map(st_idx).to_numpy(); tp = last.t_purchase.to_numpy().astype(int)
hpi0 = loghpi[si, p1.T_END]
out_paths, summ, haz_store = [], [], {}
for name, (path, hpi_shock) in SCEN.items():
    full_path = pm.copy(); full_path[months] = path
    for model in ["m1", "m2"]:
        hs, hl = np.zeros((len(last), H)), np.zeros((len(last), H))
        for k in range(H):
            t = months[k]; d = last.copy()
            d["t"] = t; d["tenure_m"] = t - tp; d["pmms"] = full_path[t]; d["gap"] = d.pmms - d.note_hat
            d["pmms_chg3"] = full_path[t] - full_path[t - 3]
            d["log_appr"] = hpi0 + np.log1p(hpi_shock * min((k + 1) / 12, 1.0)) - loghpi[si, tp]
            r = d.note_hat.to_numpy() / 1200; kk = np.minimum(d.tenure_m.to_numpy(), 360)
            d["est_ltv"] = 0.8 * ((1 + r) ** 360 - (1 + r) ** kk) / ((1 + r) ** 360 - 1) / np.exp(d.log_appr.to_numpy())
            d["moy"] = (t % 12) + 1
            if model == "m1":
                hs[:, k], hl[:, k] = m1_pred("sale", d), m1_pred("new_lien", d)
            else:
                Xg = d[p1.LGB_FEATURES].copy(); Xg.loc[~d.state.isin(design.states), "state_code"] = -1
                hs[:, k], hl[:, k] = m2["sale"].predict(Xg), m2["new_lien"].predict(Xg)
        haz_store[(name, model)] = (hs, hl)
        smm = (hs + hl).mean(0)
        surv = np.exp(np.log1p(-(hs + hl)).cumsum(1))
        s_prev = np.hstack([np.ones((len(last), 1)), surv[:, :-1]])             # in the pool at the start of month k
        smm_sw = ((hs + hl) * s_prev).sum(0) / s_prev.sum(0)
        for k in range(H):
            out_paths.append({"scenario": name, "model": model, "month": str(p1.MONTHS[months[k]].date())[:7], "pmms": path[k],
                              "smm_count": smm[k], "cpr_count": float(cpr_from_smm(smm[k])),
                              "cpr_count_survival_weighted": float(cpr_from_smm(smm_sw[k])),
                              "cpr_sale": float(cpr_from_smm(hs[:, k].mean())), "cpr_new_lien": float(cpr_from_smm(hl[:, k].mean()))})
        # illustrative balance runoff
        bal_frac = ((1 + last.note_hat / 1200) ** 360 - (1 + last.note_hat / 1200) ** np.minimum(last.tenure_m, 360)) / ((1 + last.note_hat / 1200) ** 360 - 1)
        pool = pd.DataFrame({"balance": 0.8 * last.price0.to_numpy() * bal_frac.to_numpy(), "note_rate": last.note_hat.to_numpy(),
                             "months_remaining": np.maximum(360 - last.tenure_m.to_numpy(), 0).astype(float)})
        live = (pool.months_remaining > 0).to_numpy()
        hsl, hll = hs[live], hl[live]
        proj = pool_projection(pool[live].reset_index(drop=True),
                               lambda a, k: pd.DataFrame({"h_refi": hll[:, k - 1], "h_sale": hsl[:, k - 1], "h_default": 0.0}),
                               pd.DataFrame({"month": [str(p1.MONTHS[m].date())[:7] for m in months]}), H)
        summ.append({"scenario": name, "model": model, "properties": len(last),
                     "cum_terminations_12m_pct": float((1 - surv[:, 11]).mean() * 100), "cum_terminations_24m_pct": float((1 - surv[:, 23]).mean() * 100),
                     "avg_cpr_count_24m_pct": float(cpr_from_smm(smm).mean() * 100),
                     "avg_cpr_count_survival_weighted_24m_pct": float(cpr_from_smm(smm_sw).mean() * 100),
                     "avg_cpr_balance_24m_pct": float(proj.cpr.mean() * 100),
                     "prepaid_share_of_start_balance_24m_pct": float(proj.prepaid_principal.sum() / proj.balance.iloc[0] * 100),
                     "wal_years_horizon_truncated": float(proj.wal_remaining.iloc[0])})
    print(name, "done")
S_ = pd.DataFrame(summ)
base = S_[S_.scenario == "base (flat at 2025-06)"].set_index("model")
S_["delta_cpr_count_vs_base_pp"] = S_.apply(lambda r: r.avg_cpr_count_24m_pct - base.loc[r.model, "avg_cpr_count_24m_pct"], axis=1)
S_["delta_cpr_survival_weighted_vs_base_pp"] = S_.apply(lambda r: r.avg_cpr_count_survival_weighted_24m_pct
                                                         - base.loc[r.model, "avg_cpr_count_survival_weighted_24m_pct"], axis=1)
S_.to_csv(OUT / "stress_summary.csv", index=False); pd.DataFrame(out_paths).to_csv(OUT / "stress_paths.csv", index=False)
print(S_.round(3).to_string(index=False))
rng_tr = (float(P.loc[P.split == "train", "pmms"].min()), float(P.loc[P.split == "train", "pmms"].max()))
json.dump({"pmms_2025_06": float(r0), "train_pmms_range": rng_tr, "horizon_months": H}, open(OUT / "stress_meta.json", "w"), indent=2)
print("train PMMS range", rng_tr, "start", r0)
