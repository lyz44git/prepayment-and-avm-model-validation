"""Valuation study evaluation: champions (vendor values) vs challengers on identical rows, OOT and spatial."""
import sys, json, functools; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import WORK, REPORTS
from mval import p2
from mval.valuation import accuracy, table
from mval.metrics import auc
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)
OUT = REPORTS / "p2"; DW = WORK / "p2"
S = pd.read_parquet(DW / "features.parquet")
G = pd.read_parquet(DW / "preds_gbm.parquet")
A = pd.read_parquet(DW / "preds_attn.parquet") if (DW / "preds_attn.parquet").exists() else None
D = S[["addr_id", "split", "state", "fips5", "t_sale", "price", "price_prev", "y", "t_black", "t_hisp",
       "v_vendor_a_avm", "v_vendor_b", "v_vendor_a_mktval", "v_assessor_2025", "v_vendor_a_avm_err"]].copy()
for c in ["b0_index", "b1_hedonic", "c1_gbm", "c1_lo", "c1_hi", "c1_width_pct"]:
    D[c] = G[c].to_numpy()
if A is not None:
    D["c2_attn"] = A.c2_attn.to_numpy(); D["baseline_comp_index"] = A.baseline_comp_index.to_numpy()
    D["c3_blend"] = 0.5 * D.c1_gbm + 0.5 * D.c2_attn
for c in ["b0_index", "b1_hedonic", "c1_gbm", "c2_attn", "c3_blend", "baseline_comp_index"]:
    if c in D:
        D["$" + c] = np.exp(D[c])
EST = {"Vendor A AVM (champion)": "v_vendor_a_avm", "Vendor B value": "v_vendor_b", "Vendor A market value": "v_vendor_a_mktval",
       "Assessor 2025": "v_assessor_2025", "B0 index AVM (prior sale x HPI)": "$b0_index",
       "B0' comp-median x HPI": "$baseline_comp_index", "B1 hedonic OLS": "$b1_hedonic", "C1 LightGBM": "$c1_gbm"}
if A is not None:
    EST.update({"C2 neighbour-attention NN": "$c2_attn", "C3 blend (C1+C2)": "$c3_blend"})
EST = {k: v for k, v in EST.items() if v in D}
res = {}
for sp in ["oot", "spatial"]:
    d = D[D.split == sp]
    t = table(d, EST); t.insert(0, "split", sp); t.to_csv(OUT / f"accuracy_{sp}_own_coverage.csv", index=False)
    common = d[(d.v_vendor_a_avm > 1000) & (d.v_vendor_b > 1000) & (d.v_vendor_a_mktval > 1000) & (d.v_assessor_2025 > 1000)]
    tc = table(common, EST); tc.insert(0, "split", sp + "_common_support"); tc.to_csv(OUT / f"accuracy_{sp}_common.csv", index=False)
    print(sp, "own coverage"); print(t[["estimator", "n", "coverage", "median_ratio", "MdAPE", "PPE10", "PPE20", "COD", "PRD"]].round(3).to_string(index=False))
    print(sp, f"common support n={len(common):,}"); print(tc[["estimator", "median_ratio", "MdAPE", "PPE10", "PPE20"]].round(3).to_string(index=False))
# cluster (county) bootstrap of MdAPE and PPE10: best challenger vs best vendor, OOT
def boot(d, cols, B=200, seed=0):
    rng = np.random.default_rng(seed); cnty = d.fips5.to_numpy(); u = np.unique(cnty)
    grp = {c: np.where(cnty == c)[0] for c in u}; out = []
    for _ in range(B):
        idx = np.concatenate([grp[c] for c in rng.choice(u, len(u), replace=True)])
        dd = d.iloc[idx]; row = {}
        for k, c in cols.items():
            ape = np.abs(dd[c] / dd.price - 1); row[k + "_MdAPE"] = np.nanmedian(ape); row[k + "_PPE10"] = np.nanmean(ape <= 0.1)
        out.append(row)
    return pd.DataFrame(out)
best_ch = "C3 blend (C1+C2)" if "C3 blend (C1+C2)" in EST else "C1 LightGBM"
for sp in ["oot", "spatial"]:
    d = D[(D.split == sp) & (D.v_vendor_a_mktval > 1000) & (D.v_vendor_a_avm > 1000)]
    bt = boot(d, {"challenger": EST[best_ch], "mktval": "v_vendor_a_mktval", "avm": "v_vendor_a_avm", "gbm": "$c1_gbm"})
    res[f"bootstrap_{sp}"] = {k: [float(bt[k].quantile(0.025)), float(bt[k].median()), float(bt[k].quantile(0.975))] for k in bt}
    res[f"bootstrap_{sp}"]["delta_MdAPE_challenger_minus_mktval"] = [float(q) for q in (bt.challenger_MdAPE - bt.mktval_MdAPE).quantile([0.025, 0.5, 0.975])]
    res[f"bootstrap_{sp}"]["n"] = len(d); res[f"bootstrap_{sp}"]["best_challenger"] = best_ch
# conformal intervals: coverage, width, and whether width ranks realised error (vs vendor's stated error)
for sp in ["calib", "oot", "spatial"]:
    d = D[D.split == sp]
    inside = (d.y >= d.c1_lo) & (d.y <= d.c1_hi)
    err = np.abs(np.exp(d.c1_gbm - d.y) - 1)
    res[f"conformal_{sp}"] = {"nominal": 0.80, "coverage": float(inside.mean()), "median_width_pct": float(d.c1_width_pct.median()),
                              "auc_width_for_err_gt20": auc((err > 0.2).astype(int), d.c1_width_pct),
                              "spearman_width_vs_abs_err": float(d.c1_width_pct.rank().corr(err.rank()))}
d = D[(D.split == "oot") & (D.v_vendor_a_avm > 1000) & (D.v_vendor_a_avm_err > 0)]
res["vendor_stated_error_auc_err_gt20"] = auc((np.abs(d.v_vendor_a_avm / d.price - 1) > 0.2).astype(int), d.v_vendor_a_avm_err / d.v_vendor_a_avm)
# drift: monthly MdAPE in the OOT window (models and index frozen at 2024-06)
d = D[D.split == "oot"].copy(); d["month"] = d.t_sale
cols = {"C1 LightGBM": "$c1_gbm", "Vendor A market value": "v_vendor_a_mktval", "B0 index AVM (prior sale x HPI)": "$b0_index"}
if "C3 blend (C1+C2)" in EST: cols["C3 blend (C1+C2)"] = "$c3_blend"
dr = d.groupby("month").apply(lambda g: pd.Series({k: np.nanmedian(np.abs(g[c] / g.price - 1)) for k, c in cols.items()}), include_groups=False)
dr.index = [f"{1990 + m // 12}-{m % 12 + 1:02d}" for m in dr.index]; dr.to_csv(OUT / "oot_monthly_MdAPE.csv"); print(dr.round(3).to_string())
# tiers, disparity (raw and within price tier), upmarket screen
o = D[D.split == "oot"].copy()
o["tier"] = o.groupby("state").price.transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
o["minority"] = o.t_black.fillna(0) + o.t_hisp.fillna(0)
o["minority_q"] = pd.qcut(o.minority.rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
tt = table(o, EST, by="tier"); tt.to_csv(OUT / "accuracy_by_price_tier_oot.csv", index=False)
print(tt.pivot(index="group", columns="estimator", values="MdAPE").round(3).to_string())
dm = table(o, EST, by="minority_q"); dm.to_csv(OUT / "disparity_minority_raw_oot.csv", index=False)
wt = []
for k, c in EST.items():
    ok = o[c] > 1000
    lr = np.log(o.loc[ok, c] / o.loc[ok, "price"])
    g = pd.DataFrame({"lr": lr, "tier": o.loc[ok, "tier"], "q": o.loc[ok, "minority_q"]})
    med = g.groupby(["tier", "q"], observed=True).lr.median().unstack()          # median log ratio per tier x quintile
    adj = med.sub(med.mean(1), axis=0).mean(0)                                   # within-tier deviation, averaged over tiers
    wt.append({"estimator": k, **{f"{q}_within_tier_logratio_dev": float(v) for q, v in adj.items()},
               "Q5_minus_Q1_within_tier": float(adj["Q5"] - adj["Q1"])})
wt = pd.DataFrame(wt); wt.to_csv(OUT / "disparity_minority_within_tier_oot.csv", index=False); print(wt.round(4).to_string(index=False))
up = []
hv = (o.price >= 1e6).astype(int)
for k, c in EST.items():
    m = o[c] > 1000; pr = (o[c] >= 1e6) & m
    tp = (pr & (hv == 1)).sum()
    up.append({"estimator": k, "coverage": float(m.mean()), "precision": tp / max(pr.sum(), 1), "recall": tp / max((hv[m] == 1).sum(), 1),
               "auc_rank": auc(hv[m], o.loc[m, c])})
up = pd.DataFrame(up); up.to_csv(OUT / "upmarket_1m_screen_oot.csv", index=False); print(up.round(3).to_string(index=False))
# sensitivity: exclude non-disclosure states; segment with / without a prior recorded sale
nd = table(o[~o.state.isin(p2.NON_DISCLOSURE)], EST); nd.to_csv(OUT / "accuracy_oot_excl_nondisclosure.csv", index=False)
seg = table(o.assign(has_prev=o.price_prev.notna().map({True: "has prior sale", False: "no prior sale"})), EST, by="has_prev")
seg.to_csv(OUT / "accuracy_oot_by_prior_sale.csv", index=False)
print(seg.pivot(index="estimator", columns="group", values="MdAPE").round(3).to_string())
json.dump(res, open(OUT / "eval_summary.json", "w"), indent=2, default=float); print(json.dumps(res, indent=1, default=float))
D.to_parquet(DW / "eval_frame.parquet", index=False)
