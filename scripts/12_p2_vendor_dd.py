"""Valuation study, part 1: due diligence on third-party valuation fields (the 'champions')."""
import sys, json; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import WORK, REPORTS
from mval import p1, p2
from mval.valuation import table, accuracy
from mval.metrics import auc
import numpy as np, pandas as pd, statsmodels.formula.api as smf
OUT = REPORTS / "p2"
S = pd.read_parquet(WORK / "p2" / "features.parquet")
VEND = {"Vendor A AVM": "v_vendor_a_avm", "Vendor B est. home value": "v_vendor_b", "Vendor A market value": "v_vendor_a_mktval",
        "Assessor value (Vendor C 2025)": "v_assessor_2025"}
S["sale_q"] = pd.PeriodIndex(pd.to_datetime(dict(year=1990 + S.t_sale // 12, month=S.t_sale % 12 + 1, day=1)), freq="Q").astype(str)
S["sale_year"] = 1990 + S.t_sale // 12
oot = S[S.split == "oot"]
t_all = table(oot, VEND); t_all.to_csv(OUT / "vendor_accuracy_oot.csv", index=False)
print(t_all.round(3).to_string(index=False))
t_q = table(S, VEND, by="sale_q"); t_q.to_csv(OUT / "vendor_accuracy_by_quarter.csv", index=False)
print(t_q.pivot(index="group", columns="estimator", values="median_ratio").round(3).to_string())
# --- staleness test: e = log(V/p) on market move since 2022-04 (state FE). slope -1 = frozen value, 0 = tracks sale date
st_idx, H = p1.hpi_matrix(pd.read_parquet(WORK / "hpi_state_monthly.parquet"), pd.read_csv(WORK / "hpi_lookup.csv"), "full")
si = S.state.map(st_idx).to_numpy()
S["mkt_move"] = H[si, S.t_sale.to_numpy()] - H[si, p2.T_REF]
stale = []
for name, col in VEND.items():
    d = S[(S[col] > 1000)].assign(e=lambda x: np.log(x[col] / x.price))
    d = d[d.e.between(-3, 3)]
    fit = smf.ols("e ~ mkt_move + C(state)", data=d.sample(min(len(d), 800_000), random_state=0)).fit(cov_type="cluster",
              cov_kwds={"groups": pd.factorize(d.sample(min(len(d), 800_000), random_state=0).fips5)[0]})
    stale.append({"estimator": name, "slope_on_market_move": fit.params["mkt_move"], "se": fit.bse["mkt_move"],
                  "n": int(fit.nobs)})
stale = pd.DataFrame(stale); stale.to_csv(OUT / "vendor_staleness_test.csv", index=False); print(stale.round(3).to_string(index=False))
# --- implied value date: month m with logHPI(m) = median(log V - log p + logHPI(t_sale)) (national index)
nat = pd.read_parquet(WORK / "hpi_state_monthly.parquet").query("version=='full' and level=='national'").set_index("month").log_hpi
natv = nat.reindex(p1.MONTHS).ffill().to_numpy()
imp = []
for name, col in VEND.items():
    d = S[S[col] > 1000]
    target = np.median(np.log(d[col] / d.price) + natv[d.t_sale.to_numpy()])
    cand = np.arange((2015 - 1990) * 12, (2025 - 1990) * 12 + 9)
    m = cand[np.argmin(np.abs(natv[cand] - target))]
    imp.append({"estimator": name, "implied_value_month_if_unbiased": str(p1.MONTHS[m].date())[:7]})
imp = pd.DataFrame(imp); imp.to_csv(OUT / "vendor_implied_value_date.csv", index=False); print(imp.to_string(index=False))
# --- vendor-stated confidence: does Vendor A's AVM error field rank realised error?
d = oot[(oot.v_vendor_a_avm > 1000) & (oot.v_vendor_a_avm_err > 0)].copy()
d["stated_rel_err"] = d.v_vendor_a_avm_err / d.v_vendor_a_avm
d["abs_err"] = np.abs(d.v_vendor_a_avm / d.price - 1)
conf = {"n": len(d), "spearman_stated_vs_realised": float(d.stated_rel_err.rank().corr(d.abs_err.rank())),
        "auc_stated_for_err_gt20": auc((d.abs_err > 0.2).astype(int), d.stated_rel_err),
        "stated_rel_err_median": float(d.stated_rel_err.median()), "distinct_values": int(d.v_vendor_a_avm_err.nunique())}
json.dump(conf, open(OUT / "vendor_confidence_check.json", "w"), indent=2); print(conf)
# --- disparity by tract minority share and by price tier (OOT)
oot = oot.assign(minority=lambda x: x.t_black.fillna(0) + x.t_hisp.fillna(0))
oot["minority_q"] = pd.qcut(oot.minority.rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
oot["tier"] = oot.groupby("state").price.transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
dm = table(oot, VEND, by="minority_q"); dm.to_csv(OUT / "vendor_disparity_minority.csv", index=False)
dt = table(oot, VEND, by="tier"); dt.to_csv(OUT / "vendor_by_price_tier.csv", index=False)
print(dm.pivot(index="group", columns="estimator", values="median_ratio").round(3).to_string())
print(dt.pivot(index="group", columns="estimator", values="MdAPE").round(3).to_string())
# --- upmarket screen: identify $1M+ sales
hv = (oot.price >= 1e6).astype(int); rows = []
for name, col in VEND.items():
    pr = (oot[col] >= 1e6).astype(int); m = oot[col] > 1000
    tp = ((pr == 1) & (hv == 1) & m).sum()
    rows.append({"estimator": name, "n_true_1m": int(hv[m].sum()), "precision": tp / max(((pr == 1) & m).sum(), 1),
                 "recall": tp / max(hv[m].sum(), 1), "auc_rank": auc(hv[m], oot.loc[m, col])})
up = pd.DataFrame(rows); up.to_csv(OUT / "vendor_upmarket_1m.csv", index=False); print(up.round(3).to_string(index=False))
