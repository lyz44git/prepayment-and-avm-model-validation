"""Remediation of findings F9 (house-price stress) and F2 (new-lien under-prediction).

Follows docs/prepayment_model_remediation_plan.md step by step. Reads the panel written by 08; writes reports/p1/remediation/.
"""
import sys, json, time, hashlib, functools, datetime; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import WORK, REPORTS
from mval import p1
from mval.metrics import auc
from mval.amortization import cpr_from_smm
import numpy as np, pandas as pd, statsmodels.api as sm
print = functools.partial(print, flush=True)
t0 = time.time(); OUT = REPORTS / "p1" / "remediation"; OUT.mkdir(parents=True, exist_ok=True); DW = WORK / "p1"
PLAN = REPORTS.parent / "docs" / "prepayment_model_remediation_plan.md"
meta = {"plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(), "run_started": datetime.datetime.now().isoformat(timespec="seconds")}
CAUSES = ["sale", "new_lien"]; Q = 0.10
ym = lambda y, m: (y - 1990) * 12 + m - 1
SEL_FIT_END, OOT0 = ym(2023, 12), p1.TRAIN_END + 1

P = pd.read_parquet(DW / "panel.parquet")
pm = p1.rate_vector(); t = P.t.to_numpy()
peak = np.array([np.nan if i < 12 else pm[i - 12:i].max() for i in range(len(pm))])
def rate_terms(path, tt):
    pk = np.array([path[i - 12:i].max() for i in tt])
    return {"dpeak12": np.minimum(path[tt] - pk, 0), "pmms_chg6": path[tt] - path[tt - 6], "pmms_chg12": path[tt] - path[tt - 12]}
P["dpeak12"] = np.minimum(pm[t] - peak[t], 0); P["pmms_chg6"] = pm[t] - pm[t - 6]; P["pmms_chg12"] = pm[t] - pm[t - 12]
hold = P.state.isin(p1.HOLDOUT_STATES).to_numpy(); split = P.split.to_numpy()
is_tr = split == "train"
Y = {c: P[f"y_{c}"].to_numpy(np.float64) for c in CAUSES}
print(f"panel {len(P):,} rows ({time.time()-t0:.0f}s)")

design = p1.LogitDesign().fit(P[is_tr])
Xb = design.transform(P)                                  # published design, float32
base_cols = list(Xb.columns)
EXTRA = ["dpeak12", "pmms_chg6", "pmms_chg12"]
X = np.column_stack([np.ones(len(P), np.float32), Xb.to_numpy(np.float32), P[EXTRA].to_numpy(np.float32)])
names = ["const"] + base_cols + EXTRA; ix = {n: i for i, n in enumerate(names)}
del Xb
SPECS = {"S0": [n for n in names if n not in EXTRA],
         "S1": [n for n in names if n not in EXTRA + ["log_appr"]],
         "S2": [n for n in names if n not in ["log_appr", "pmms_chg6", "pmms_chg12"]],
         "S3": [n for n in names if n not in ["log_appr", "dpeak12"]],
         "S4": [n for n in names if n != "log_appr"]}
st_cols = [n for n in names if n.startswith("st_")]
share = P.loc[is_tr, "state"].value_counts(normalize=True)
st_share = np.array([share.get(n[3:], 0.0) for n in st_cols])
print(f"design {X.shape} ({time.time()-t0:.0f}s)")

# case-control draws: the published fit drew sale first, then new lien, from one generator over the training rows
rng = np.random.default_rng(0); U = {}
for c in CAUSES:
    u = np.full(len(P), np.nan); u[is_tr] = rng.random(int(is_tr.sum())); U[c] = u
rng2 = np.random.default_rng(1)                            # draws for rows outside the training split (walk-forward)
for c in CAUSES:
    m = np.isnan(U[c]); U[c][m] = rng2.random(int(m.sum()))

def fit(spec, c, rows, cluster=None):
    cols = [ix[n] for n in SPECS[spec]]
    keep = rows & ((Y[c] == 1) | (U[c] < Q))
    Xk = X[keep][:, cols].astype(np.float64)
    live = Xk.std(0) > 0; live[0] = True                   # a state dummy can be empty in a shorter window
    kw = {"cov_type": "cluster", "cov_kwds": {"groups": cluster[keep]}} if cluster is not None else {}
    res = sm.GLM(Y[c][keep], Xk[:, live], family=sm.families.Binomial()).fit(maxiter=100, **kw)
    b = np.zeros(len(cols)); b[live] = res.params; b[0] += np.log(Q)
    se = np.full(len(cols), np.nan); se[live] = res.bse
    return pd.Series(b, index=SPECS[spec]), pd.Series(se, index=SPECS[spec])

def predict(b, rows):
    cols = [ix[n] for n in b.index]; bb = b.to_numpy(); idx = np.flatnonzero(rows); eta = np.empty(len(idx))
    for a in range(0, len(idx), 2_000_000):                # in blocks, to keep the float64 copy small
        j = idx[a:a + 2_000_000]; eta[a:a + len(j)] = X[j][:, cols].astype(np.float64) @ bb
    h = hold[idx]
    if h.any():                                            # unseen states: training-share-weighted state effect
        eta[h] += float(st_share @ b.reindex(st_cols).fillna(0).to_numpy())
    return 1 / (1 + np.exp(-eta))

def logloss(y, s): return float(-np.mean(y * np.log(s) + (1 - y) * np.log(1 - s)))

def cpr_table(pred, label):
    """Monthly pool CPR, actual vs predicted, by split group. pred: dict cause -> full-length array (nan where absent)."""
    d = pd.DataFrame({"t": t, "split": split, "ys": Y["sale"], "yl": Y["new_lien"], "ps": pred["sale"], "pl": pred["new_lien"]}).dropna()
    g = d.groupby(["split", "t"]).mean()
    g["cpr_y"] = cpr_from_smm(g.ys + g.yl); g["cpr_p"] = cpr_from_smm(g.ps + g.pl); g["model"] = label
    return g.reset_index()

def cpr_summary(tab):
    rows = []
    for (m, sp), a in tab.groupby(["model", "split"]):
        rows.append({"model": m, "split": sp, "months": len(a), "actual_cpr_avg_pct": a.cpr_y.mean() * 100, "pred_cpr_avg_pct": a.cpr_p.mean() * 100,
                     "MAE_monthly_cpr_pp": (a.cpr_p - a.cpr_y).abs().mean() * 100, "corr_monthly": float(np.corrcoef(a.cpr_p, a.cpr_y)[0, 1]),
                     "new_lien_pred_over_actual": a.pl.sum() / a.yl.sum(), "sale_pred_over_actual": a.ps.sum() / a.ys.sum()})
    return pd.DataFrame(rows)

# ---------------- Step 0: reproduce the published model
allrows = np.ones(len(P), bool); full = {}; B = {}
pub = {c: pd.read_csv(REPORTS / "p1" / f"m1_coefficients_{c}.csv", index_col=0).coef for c in CAUSES}
rep = {}
for c in CAUSES:
    B[("S0", c)], _ = fit("S0", c, is_tr)
    full[("S0", c)] = predict(B[("S0", c)], allrows)
    rep[c] = float((B[("S0", c)] - pub[c].reindex(B[("S0", c)].index)).abs().max())
tab0 = cpr_table({c: full[("S0", c)] for c in CAUSES}, "S0 fixed"); s0 = cpr_summary(tab0).set_index("split")
rep.update({"oot_cpr_mae_pp": float(s0.loc["oot", "MAE_monthly_cpr_pp"]), "oot_cpr_corr": float(s0.loc["oot", "corr_monthly"]),
            "published_oot_cpr_mae_pp": 0.4272786167549882, "published_oot_cpr_corr": 0.9003208596309573})
rep["pass"] = bool(max(rep["sale"], rep["new_lien"]) < 1e-6 and abs(rep["oot_cpr_mae_pp"] - 0.4273) < 5e-4 and abs(rep["oot_cpr_corr"] - 0.9003) < 5e-4)
json.dump(rep, open(OUT / "step0_reproduction.json", "w"), indent=2); print("step 0", rep, f"({time.time()-t0:.0f}s)")
if not rep["pass"]:
    raise SystemExit("published model not reproduced; stopping as the plan requires")

# ---------------- Step 2: selection on training months only
fit_rows = is_tr & (t <= SEL_FIT_END); val_rows = is_tr & (t > SEL_FIT_END)
sel = []
for s in ["S0", "S1", "S2", "S3", "S4"]:
    for c in CAUSES:
        b, _ = fit(s, c, fit_rows); pv = predict(b, val_rows); yv = Y[c][val_rows]
        sel.append({"spec": s, "cause": c, "val_logloss": logloss(yv, pv), "val_auc": auc(yv, pv), "val_pred_over_actual": pv.mean() / yv.mean()})
    print("selection", s, f"({time.time()-t0:.0f}s)")
sel = pd.DataFrame(sel); sel.to_csv(OUT / "step2_selection_validation.csv", index=False)
nl = sel[sel.cause == "new_lien"].set_index("spec").val_logloss
better = [s for s in ["S2", "S3", "S4"] if nl[s] <= nl["S1"] * (1 - 0.001)]
M1R = min(better, key=lambda s: nl[s]) if better else "S1"
meta["selected_spec"] = M1R; meta["new_lien_val_logloss"] = nl.to_dict(); print("selected", M1R, nl.round(6).to_dict())

# ---------------- Step 3: fixed-origin evaluation
fips = pd.factorize(P.fips5.to_numpy())[0]
for s in ["S1", "S2", "S3", "S4"]:
    for c in CAUSES:
        b, se = fit(s, c, is_tr, cluster=fips if s == M1R else None)
        B[(s, c)] = b; full[(s, c)] = predict(b, allrows)
        if s == M1R:
            pd.DataFrame({"coef": b, "se_cluster": se, "z": b / se}).to_csv(OUT / f"m1r_coefficients_{c}.csv")
    print("refit", s, f"({time.time()-t0:.0f}s)")
ev = []
for s in ["S0", "S1", "S2", "S3", "S4"]:
    for c in CAUSES:
        for sp in ["train", "oot", "spatial", "spatial_oot"]:
            r = split == sp; y = Y[c][r]; pz = full[(s, c)][r]
            ev.append({"spec": s, "selected": s == M1R, "used_for_selection": False, "cause": c, "split": sp, "auc": auc(y, pz),
                       "pred_over_actual": pz.mean() / y.mean(), "logloss": logloss(y, pz)})
ev = pd.DataFrame(ev)
head = ["S0", "S1"] + ([M1R] if M1R != "S1" else [])
ev[ev.spec.isin(head)].to_csv(OUT / "step3_eval_fixed_origin.csv", index=False)
ev[~ev.spec.isin(head)].to_csv(OUT / "step3_other_candidates_not_used_for_selection.csv", index=False)
tabs = [tab0] + [cpr_table({c: full[(s, c)] for c in CAUSES}, f"{s} fixed") for s in ["S1", "S2", "S3", "S4"]]

# ---------------- Step 4: walk-forward re-estimation
oot_months = list(range(OOT0, p1.T_END + 1)); wf_by_month = []; wf_ev = []
for s in dict.fromkeys(["S0", M1R]):
    wf = {c: np.full(len(P), np.nan) for c in CAUSES}
    for tm in oot_months:
        rows_fit = (~hold) & (t <= tm - 2); rows_pred = t == tm
        for c in CAUSES:
            b, _ = fit(s, c, rows_fit); wf[c][rows_pred] = predict(b, rows_pred)
            for grp, r in [("oot", rows_pred & ~hold), ("spatial_oot", rows_pred & hold)]:
                wf_by_month.append({"spec": s, "month": str(p1.MONTHS[tm].date())[:7], "split": grp, "cause": c, "actual_hazard": Y[c][r].mean(),
                                    "walk_forward": wf[c][r].mean(), "fixed_origin": full[(s, c)][r].mean()})
        print("walk-forward", s, str(p1.MONTHS[tm].date())[:7], f"({time.time()-t0:.0f}s)")
    tabs.append(cpr_table(wf, f"{s} walk-forward"))
    for c in CAUSES:                                       # row-weighted, the same definition as the published table
        for sp in ["oot", "spatial_oot"]:
            r = split == sp; y = Y[c][r]; pz = wf[c][r]
            wf_ev.append({"spec": s, "mode": "walk-forward", "cause": c, "split": sp, "auc": auc(y, pz), "pred_over_actual": pz.mean() / y.mean(), "logloss": logloss(y, pz)})
pd.DataFrame(wf_ev).to_csv(OUT / "step4_eval_walkforward.csv", index=False)
pd.DataFrame(wf_by_month).to_csv(OUT / "step4_walkforward_by_month.csv", index=False)
tab = pd.concat(tabs); tab["month"] = [str(p1.MONTHS[i].date())[:7] for i in tab.t]
tab = tab[tab.split.isin(["oot", "spatial_oot"])]
cs = cpr_summary(tab); cs.to_csv(OUT / "step3_4_cpr_backtest.csv", index=False)
tab[["model", "split", "month", "cpr_y", "cpr_p", "ys", "yl", "ps", "pl"]].to_csv(OUT / "step3_4_monthly_cpr_paths.csv", index=False)
print(cs.round(4).to_string(index=False))

# ---------------- Step 5: stress test with M1r (and S0 for reference), count-based
H = 24; st_idx, loghpi = p1.hpi_matrix(pd.read_parquet(WORK / "hpi_state_monthly.parquet"), pd.read_csv(WORK / "hpi_lookup.csv"), "full")
last_rows = (t == p1.T_END) & (P.y_sale.to_numpy() == 0) & (P.y_new_lien.to_numpy() == 0) & (P.y_other_transfer.to_numpy() == 0)
last = P[last_rows].copy(); Xl = X[last_rows].copy(); hl_ = hold[last_rows]
months = np.arange(p1.T_END + 1, p1.T_END + H + 1); r0 = pm[p1.T_END]; ramp = np.minimum(np.arange(1, H + 1) / 6, 1.0)
SCEN = {"base (flat at 2025-06)": (np.full(H, r0), 0.0), "realized PMMS path": (pm[months].copy(), 0.0),
        "-100bp": (r0 - 1.0 * ramp, 0.0), "-200bp": (r0 - 2.0 * ramp, 0.0), "+100bp": (r0 + 1.0 * ramp, 0.0), "+200bp": (r0 + 2.0 * ramp, 0.0),
        "HPI -10% (rates flat)": (np.full(H, r0), -0.10), "HPI -20% (rates flat)": (np.full(H, r0), -0.20), "HPI +10% (rates flat)": (np.full(H, r0), 0.10)}
si = last.state.map(st_idx).to_numpy(); tp = last.t_purchase.to_numpy().astype(int); note = last.note_hat.to_numpy()
hpi0 = loghpi[si, p1.T_END]; knots = design.knots / 12
summ, paths = [], []
for name, (path, shock) in SCEN.items():
    fp = pm.copy(); fp[months] = path
    for s in dict.fromkeys(["S0", "S1", M1R]):
        hs, hl = np.zeros((len(last), H)), np.zeros((len(last), H))
        for k in range(H):
            tm = months[k]; Xk = Xl.copy(); ten = tm - tp
            Bs = p1.ns_basis(np.clip(ten.astype(float), 1, 420) / 12, knots)
            for j in range(Bs.shape[1]):
                if f"tenure_ns{j}" in ix: Xk[:, ix[f"tenure_ns{j}"]] = Bs[:, j]
            gap = fp[tm] - note
            Xk[:, ix["gap_pos"]] = np.maximum(gap, 0); Xk[:, ix["gap_neg"]] = np.minimum(gap, 0)
            Xk[:, ix["pmms"]] = fp[tm]; Xk[:, ix["pmms_chg3"]] = fp[tm] - fp[tm - 3]
            la = hpi0 + np.log1p(shock * min((k + 1) / 12, 1.0)) - loghpi[si, tp]
            r = note / 1200; kk = np.minimum(ten, 360)
            Xk[:, ix["log_appr"]] = np.clip(la, -1.5, 2.5)
            Xk[:, ix["est_ltv"]] = np.clip(0.8 * ((1 + r) ** 360 - (1 + r) ** kk) / ((1 + r) ** 360 - 1) / np.exp(la), 0, 2)
            for m in range(2, 13):
                if f"moy{m}" in ix: Xk[:, ix[f"moy{m}"]] = float((tm % 12) + 1 == m)
            rt = rate_terms(fp, np.array([tm]))
            for n_ in EXTRA: Xk[:, ix[n_]] = rt[n_][0]
            for c, arr in [("sale", hs), ("new_lien", hl)]:
                b = B[(s, c)]; eta = Xk[:, [ix[n] for n in b.index]].astype(np.float64) @ b.to_numpy()
                eta[hl_] += float(st_share @ b.reindex(st_cols).fillna(0).to_numpy())
                arr[:, k] = 1 / (1 + np.exp(-eta))
        smm = (hs + hl).mean(0); surv = np.exp(np.log1p(-(hs + hl)).cumsum(1))
        summ.append({"scenario": name, "spec": s, "properties": len(last), "cum_terminations_12m_pct": float((1 - surv[:, 11]).mean() * 100),
                     "cum_terminations_24m_pct": float((1 - surv[:, 23]).mean() * 100), "avg_cpr_count_24m_pct": float(cpr_from_smm(smm).mean() * 100),
                     "avg_cpr_sale_24m_pct": float(cpr_from_smm(hs.mean(0)).mean() * 100), "avg_cpr_new_lien_24m_pct": float(cpr_from_smm(hl.mean(0)).mean() * 100)})
        for k in range(H):
            paths.append({"scenario": name, "spec": s, "month": str(p1.MONTHS[months[k]].date())[:7], "cpr_count": float(cpr_from_smm(smm[k]))})
    print("stress", name, f"({time.time()-t0:.0f}s)")
S_ = pd.DataFrame(summ); base = S_[S_.scenario == "base (flat at 2025-06)"].set_index("spec").avg_cpr_count_24m_pct
S_["delta_cpr_count_vs_base_pp"] = S_.avg_cpr_count_24m_pct - S_.spec.map(base)
S_.to_csv(OUT / "step5_stress_summary.csv", index=False); pd.DataFrame(paths).to_csv(OUT / "step5_stress_paths.csv", index=False)
print(S_.round(3).to_string(index=False))

# ---------------- closure rules
e = ev.set_index(["spec", "cause", "split"]); c_ = cs.set_index(["model", "split"]); w_ = pd.DataFrame(wf_ev).set_index(["spec", "cause", "split"]); d = S_[S_.spec == M1R].set_index("scenario").delta_cpr_count_vs_base_pp
f9 = {"hpi_m10_delta_pp": float(d["HPI -10% (rates flat)"]), "hpi_m20_delta_pp": float(d["HPI -20% (rates flat)"]), "hpi_p10_delta_pp": float(d["HPI +10% (rates flat)"]),
      "s1_minus_s0_oot_auc_sale": float(e.loc[("S1", "sale", "oot"), "auc"] - e.loc[("S0", "sale", "oot"), "auc"]),
      "s1_minus_s0_oot_auc_new_lien": float(e.loc[("S1", "new_lien", "oot"), "auc"] - e.loc[("S0", "new_lien", "oot"), "auc"]),
      "s1_minus_s0_oot_cpr_mae_pp": float(c_.loc[("S1 fixed", "oot"), "MAE_monthly_cpr_pp"] - c_.loc[("S0 fixed", "oot"), "MAE_monthly_cpr_pp"])}
f9["closed"] = bool(f9["hpi_m10_delta_pp"] <= -0.05 and f9["hpi_m20_delta_pp"] < f9["hpi_m10_delta_pp"] and f9["hpi_p10_delta_pp"] > 0
                    and f9["s1_minus_s0_oot_auc_sale"] >= -0.005 and f9["s1_minus_s0_oot_auc_new_lien"] >= -0.005 and f9["s1_minus_s0_oot_cpr_mae_pp"] <= 0.02)
f2 = {"published_oot_new_lien_pred_over_actual": float(e.loc[("S0", "new_lien", "oot"), "pred_over_actual"]),
      "m1r_fixed_oot_new_lien_pred_over_actual": float(e.loc[(M1R, "new_lien", "oot"), "pred_over_actual"]),
      "m1r_walkforward_oot_new_lien_pred_over_actual": float(w_.loc[(M1R, "new_lien", "oot"), "pred_over_actual"]),
      "m1r_walkforward_oot_cpr_mae_pp": float(c_.loc[(f"{M1R} walk-forward", "oot"), "MAE_monthly_cpr_pp"]),
      "s0_walkforward_oot_new_lien_pred_over_actual": float(w_.loc[("S0", "new_lien", "oot"), "pred_over_actual"]),
      "s0_walkforward_oot_cpr_mae_pp": float(c_.loc[("S0 walk-forward", "oot"), "MAE_monthly_cpr_pp"])}
f2["reduced_at_fixed_origin"] = bool(f2["m1r_fixed_oot_new_lien_pred_over_actual"] >= 0.85)
f2["mitigated_by_maintenance"] = bool(0.90 <= f2["m1r_walkforward_oot_new_lien_pred_over_actual"] <= 1.10 and f2["m1r_walkforward_oot_cpr_mae_pp"] <= 0.30)
json.dump({"F9": f9, "F2": f2}, open(OUT / "closure.json", "w"), indent=2)
meta["run_finished"] = datetime.datetime.now().isoformat(timespec="seconds"); meta["seconds"] = round(time.time() - t0)
json.dump(meta, open(OUT / "meta.json", "w"), indent=2)
print(json.dumps({"F9": f9, "F2": f2, "meta": meta}, indent=1))
