"""Decompose finding F2 (new-lien hazard under-predicted out of time) by calendar month (round-4 robustness).

Three pieces, all aggregated to month x geography (train states / hold-out states):
 1. Published M1 (scripts/08, with the unseen-state fix from 09; data_work/p1/preds_v2.parquet): predicted and
    actual monthly hazard for new lien and sale, 2022-05..2025-06. Months up to 2024-06 are training months for
    this model (flagged in_sample_published); its out-of-time window starts 2024-07.
 2. The same specification refitted on 2022-05..2023-12 only (the remediation step-2 selection fit in
    scripts/31: published design, published case-control draws restricted to those months), scored on
    2024-01..2025-06. This is the earliest validation window available, and it is out of sample for every
    month from 2024-01.
 3. Recording-coverage check on the full at-risk universe (data_work/p1_universe.parquet, 16.06M properties):
    recorded new first liens and sales by event month, new liens per sale, and the share of sales with a
    recorded first lien dated within +/-1 month of the sale (the buyer's purchase-money lien) or later.

Writes reports/p1/r4_f2_by_month.csv and reports/p1/r4_f2_summary.json (aggregate only).
"""
import sys, json, time, functools; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import WORK, REPORTS
from mval import p1
import numpy as np, pandas as pd, statsmodels.api as sm
print = functools.partial(print, flush=True)
t0 = time.time(); OUT = REPORTS / "p1"; DW = WORK / "p1"
CAUSES = ["sale", "new_lien"]; Q = 0.10
ym = lambda y, m: (y - 1990) * 12 + m - 1
SEL_FIT_END, V0 = ym(2023, 12), ym(2024, 1)
DIP = ym(2024, 9)
mlab = lambda t: str(p1.MONTHS[t].date())[:7]
pm = p1.rate_vector()

# ---------------------------------------------------------------- 1. published M1 by month
pv = pd.read_parquet(DW / "preds_v2.parquet", columns=["t", "split", "y_sale", "y_new_lien", "m1_sale", "m1_new_lien"])
pv["geo"] = np.where(pv.split.str.startswith("spatial"), "holdout_states", "train_states")
pub = pv.groupby(["geo", "t"]).agg(property_months=("t", "size"), actual_sale=("y_sale", "sum"), actual_new_lien=("y_new_lien", "sum"),
                                   pred_sale_published=("m1_sale", "sum"), pred_new_lien_published=("m1_new_lien", "sum")).reset_index()
chk = {}
o = pv[pv.split == "oot"]
chk["published_oot_pred_over_actual_new_lien"] = float(o.m1_new_lien.mean() / o.y_new_lien.mean())
chk["published_oot_pred_over_actual_sale"] = float(o.m1_sale.mean() / o.y_sale.mean())
del pv, o
print("published M1 aggregated", chk, f"({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------- 2. refit on 2022-05..2023-12, score 2024-01..2025-06
cols = ["state", "t", "split", "y_sale", "y_new_lien", "tenure_m", "gap", "pmms", "pmms_chg3", "log_appr", "est_ltv", "moy"] + p1.STATIC
P = pd.read_parquet(DW / "panel.parquet", columns=cols)
is_tr = (P.split == "train").to_numpy(); t = P.t.to_numpy(); hold = P.state.isin(p1.HOLDOUT_STATES).to_numpy()
design = p1.LogitDesign().fit(P[is_tr])
rng = np.random.default_rng(0); U = {}
for c in CAUSES:                                       # published draws: sale first, then new lien, over the training rows
    u = np.full(len(P), np.nan); u[is_tr] = rng.random(int(is_tr.sum())); U[c] = u
share = P.loc[is_tr, "state"].value_counts(normalize=True)
st_cols = [n for n in design.columns if n.startswith("st_")]
st_share = np.array([share.get(n[3:], 0.0) for n in st_cols])
pred_rows = t >= V0
Xp = sm.add_constant(design.transform(P[pred_rows]).astype(np.float32), has_constant="add")
names = list(Xp.columns)
refit = {}
for c in CAUSES:
    y = P[f"y_{c}"].to_numpy(np.float64)
    keep = is_tr & (t <= SEL_FIT_END) & ((y == 1) | (U[c] < Q))
    Xk = sm.add_constant(design.transform(P[keep]).astype(np.float64), has_constant="add").to_numpy()
    live = Xk.std(0) > 0; live[0] = True
    res = sm.GLM(y[keep], Xk[:, live], family=sm.families.Binomial()).fit(maxiter=100)
    b = np.zeros(len(names)); b[live] = res.params; b[0] += np.log(Q)
    b = pd.Series(b, index=names)
    eta = np.empty(pred_rows.sum()); Xa = Xp.to_numpy()
    for a in range(0, len(eta), 2_000_000):
        eta[a:a + 2_000_000] = Xa[a:a + 2_000_000].astype(np.float64) @ b.to_numpy()
    eta[hold[pred_rows]] += float(st_share @ b.reindex(st_cols).fillna(0).to_numpy())
    refit[c] = 1 / (1 + np.exp(-eta))
    print(c, f"refit on {keep.sum():,} case-control rows, {live.sum()} live columns ({time.time()-t0:.0f}s)")
R = pd.DataFrame({"t": t[pred_rows], "geo": np.where(hold[pred_rows], "holdout_states", "train_states"),
                  "pred_sale_refit_to_2023_12": refit["sale"], "pred_new_lien_refit_to_2023_12": refit["new_lien"],
                  "y_sale": P.y_sale.to_numpy()[pred_rows], "y_new_lien": P.y_new_lien.to_numpy()[pred_rows]})
v = R[(R.geo == "train_states") & (R.t < ym(2024, 7))]
chk["refit_val_2024H1_pred_over_actual_new_lien"] = float(v.pred_new_lien_refit_to_2023_12.mean() / v.y_new_lien.mean())
chk["refit_val_2024H1_pred_over_actual_sale"] = float(v.pred_sale_refit_to_2023_12.mean() / v.y_sale.mean())
chk["remediation_step2_published_values"] = {"new_lien": 0.8040461374396346, "sale": 0.9602154555788881}
print("refit check", chk, f"({time.time()-t0:.0f}s)")
ref = R.groupby(["geo", "t"])[["pred_sale_refit_to_2023_12", "pred_new_lien_refit_to_2023_12"]].sum().reset_index()
del P, Xp, Xa, R

# ---------------------------------------------------------------- 3. recording-coverage check on the full universe
U0 = pd.read_parquet(WORK / "p1_universe.parquet", columns=["state", "event", "t_event", "mtgd"])
U0["geo"] = np.where(U0.state.isin(p1.HOLDOUT_STATES), "holdout_states", "train_states")
md = U0.mtgd.where((U0.mtgd % 100) > 0)
U0["t_lien"] = (md // 10000 - 1990) * 12 + (md // 100) % 100 - 1
ev = U0[U0.event != "none"]
cnt = ev.groupby(["geo", "t_event", "event"]).size().unstack(fill_value=0)
cnt.index = cnt.index.set_names(["geo", "t"])
sales = U0[U0.event == "sale"].copy()
sales["lien_pm1"] = (sales.t_lien - sales.t_event).abs() <= 1
sales["lien_from_sale_m1"] = (sales.t_lien - sales.t_event) >= -1
cov = sales.groupby(["geo", "t_event"]).agg(sales_share_lien_within_1m=("lien_pm1", "mean"),
                                            sales_share_lien_dated_from_sale_month_minus1=("lien_from_sale_m1", "mean"))
cov.index = cov.index.set_names(["geo", "t"])
uni = []
for g, n_g in U0.groupby("geo").size().items():
    c_g = cnt.loc[g].reindex(range(p1.T0, p1.T_END + 1), fill_value=0)
    tot = c_g.sum(1)
    at_risk = n_g - tot.cumsum().shift(1, fill_value=0)
    uni.append(pd.DataFrame({"geo": g, "t": c_g.index, "universe_at_risk_start": at_risk.to_numpy(),
                             "universe_sales": c_g["sale"].to_numpy(), "universe_new_liens": c_g["new_lien"].to_numpy(),
                             "universe_other_transfers": c_g["other_transfer"].to_numpy()}))
uni = pd.concat(uni).merge(cov.reset_index(), on=["geo", "t"], how="left")
uni["universe_hazard_sale"] = uni.universe_sales / uni.universe_at_risk_start
uni["universe_hazard_new_lien"] = uni.universe_new_liens / uni.universe_at_risk_start
uni["universe_new_liens_per_sale"] = uni.universe_new_liens / uni.universe_sales
del U0, ev, sales
print("universe coverage done", f"({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------- assemble
M = pub.merge(ref, on=["geo", "t"], how="left").merge(uni, on=["geo", "t"], how="left")
M = M[(M.t >= p1.T0) & (M.t <= p1.T_END)].sort_values(["geo", "t"])
M.insert(2, "month", [mlab(x) for x in M.t]); M.insert(3, "pmms", pm[M.t.to_numpy()])
M.insert(4, "in_sample_published", (M.t <= p1.TRAIN_END) & (M.geo == "train_states"))
M.insert(5, "window_published", np.where(M.t <= p1.TRAIN_END, "training months", "out of time"))
for c in CAUSES:
    M[f"actual_hazard_{c}"] = M[f"actual_{c}"] / M.property_months
    M[f"pred_hazard_{c}_published"] = M[f"pred_{c}_published"] / M.property_months
    M[f"pred_over_actual_{c}_published"] = M[f"pred_{c}_published"] / M[f"actual_{c}"]
    M[f"pred_hazard_{c}_refit_to_2023_12"] = M[f"pred_{c}_refit_to_2023_12"] / M.property_months
    M[f"pred_over_actual_{c}_refit_to_2023_12"] = M[f"pred_{c}_refit_to_2023_12"] / M[f"actual_{c}"]
keep_cols = ["geo", "month", "pmms", "in_sample_published", "window_published", "property_months", "actual_sale", "actual_new_lien",
             "actual_hazard_new_lien", "pred_hazard_new_lien_published", "pred_over_actual_new_lien_published",
             "pred_hazard_new_lien_refit_to_2023_12", "pred_over_actual_new_lien_refit_to_2023_12",
             "actual_hazard_sale", "pred_hazard_sale_published", "pred_over_actual_sale_published",
             "pred_hazard_sale_refit_to_2023_12", "pred_over_actual_sale_refit_to_2023_12",
             "universe_at_risk_start", "universe_sales", "universe_new_liens", "universe_other_transfers",
             "universe_hazard_sale", "universe_hazard_new_lien", "universe_new_liens_per_sale",
             "sales_share_lien_within_1m", "sales_share_lien_dated_from_sale_month_minus1"]
M[keep_cols].to_csv(OUT / "r4_f2_by_month.csv", index=False)

# pooled windows (row-weighted predicted / actual, the published definition)
WIN = {"2022-05..2023-12": (p1.T0, SEL_FIT_END), "2024-01..2024-06": (V0, p1.TRAIN_END), "2024-07 only": (ym(2024, 7), ym(2024, 7)),
       "2024-07..2024-08 (OOT, before the Sep-2024 dip)": (ym(2024, 7), ym(2024, 8)), "2024-09..2025-06 (OOT, from the dip)": (DIP, p1.T_END),
       "2024-07..2025-06 (full OOT)": (ym(2024, 7), p1.T_END), "2024-01..2025-06": (V0, p1.T_END)}
summ = {"checks": chk, "windows": {}}
for g in ["train_states", "holdout_states"]:
    for w, (a, b_) in WIN.items():
        m = M[(M.geo == g) & (M.t >= a) & (M.t <= b_)]
        d = {"months": int(len(m)), "pmms_avg": float(m.pmms.mean()), "property_months": int(m.property_months.sum())}
        for c in CAUSES:
            d[f"{c}_events"] = int(m[f"actual_{c}"].sum())
            d[f"{c}_pred_over_actual_published"] = float(m[f"pred_{c}_published"].sum() / m[f"actual_{c}"].sum())
            if a >= V0:
                d[f"{c}_pred_over_actual_refit_to_2023_12"] = float(m[f"pred_{c}_refit_to_2023_12"].sum() / m[f"actual_{c}"].sum())
            d[f"{c}_monthly_ratio_range_published"] = [float(m[f"pred_over_actual_{c}_published"].min()), float(m[f"pred_over_actual_{c}_published"].max())]
        d["universe_new_liens_per_sale"] = float(m.universe_new_liens.sum() / m.universe_sales.sum())
        d["universe_sales_share_lien_within_1m"] = float((m.sales_share_lien_within_1m * m.universe_sales).sum() / m.universe_sales.sum())
        d["universe_sales_share_lien_dated_from_sale_month_minus1"] = float((m.sales_share_lien_dated_from_sale_month_minus1 * m.universe_sales).sum() / m.universe_sales.sum())
        d["universe_hazard_new_lien"] = float(m.universe_new_liens.sum() / m.universe_at_risk_start.sum())
        d["sample_hazard_new_lien"] = float(m.actual_new_lien.sum() / m.property_months.sum())
        summ["windows"][f"{g} | {w}"] = d
first_short = M[(M.geo == "train_states") & (M.t >= V0) & (M.pred_over_actual_new_lien_refit_to_2023_12 < 0.90)].month.min()
summ["first_month_refit_new_lien_ratio_below_0.90_train_states"] = first_short
json.dump(summ, open(OUT / "r4_f2_summary.json", "w"), indent=2, default=float)
pd.set_option("display.width", 250)
print(M[M.geo == "train_states"][["month", "pmms", "in_sample_published", "pred_over_actual_new_lien_published", "pred_over_actual_new_lien_refit_to_2023_12",
                                  "pred_over_actual_sale_published", "pred_over_actual_sale_refit_to_2023_12", "universe_new_liens_per_sale",
                                  "sales_share_lien_within_1m", "sales_share_lien_dated_from_sale_month_minus1"]].round(3).to_string(index=False))
print(json.dumps(summ, indent=1, default=float))
print(f"total {time.time()-t0:.0f}s")
