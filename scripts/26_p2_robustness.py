"""Valuation study round-4 robustness: (a) B = 1000 county-cluster bootstrap of the out-of-time MdAPE difference,
(b) the spatial x out-of-time cell, (c) a monitoring back-test table.

(a) Same rows, statistic and resampling as scripts/15 (OOT sales with both Vendor A fields > $1,000; counties drawn
    with replacement from np.random.default_rng(0)), but 1,000 draws instead of 200. The first 200 draws are the
    published ones, which is checked. Medians over a resample are computed exactly as integer-weighted medians.
(b) The nine hold-out states restricted to the out-of-time months 2024-07..2025-08 (each estimator on its own
    coverage, as in accuracy_spatial_own_coverage.csv), with the all-months and 2022-05..2024-06 cells for
    comparison, and a county-cluster bootstrap (B = 1000) of the blend minus Vendor A market value MdAPE on the
    rows where both Vendor A fields exist (the scripts/15 bootstrap sample definition).
(c) By out-of-time month: C1 prediction PSI and the largest PSI among inputs not expected to shift
    (psi_backfill_summary.csv), monthly MdAPE of C1 and C3 (oot_monthly_MdAPE.csv, recomputed and checked), the
    calibration-window (2024-04..06) MdAPE, and when the outcome trigger "monthly MdAPE more than 1.5 points
    above the calibration window" fires (single month, and two consecutive months as in the report's monitoring
    plan).

Reads data_work/p2/eval_frame.parquet and the two existing report CSVs. Writes reports/p2/r4_bootstrap_b1000.json,
reports/p2/r4_accuracy_spatial_oot.csv, reports/p2/r4_monitoring_backtest.csv (aggregate only).
"""
import sys, json, time, functools; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import WORK, REPORTS
from mval.valuation import table
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)
t0 = time.time(); OUT = REPORTS / "p2"; DW = WORK / "p2"
ym = lambda y, m: (y - 1990) * 12 + m - 1
lab = lambda t: f"{1990 + t // 12}-{t % 12 + 1:02d}"
D = pd.read_parquet(DW / "eval_frame.parquet")

def wmedian(x_sorted, grp_sorted, wc):
    """Exact median of the sample in which row i appears wc[grp_sorted[i]] times (x_sorted ascending)."""
    cum = np.cumsum(wc[grp_sorted]); W = cum[-1]
    i1 = np.searchsorted(cum, (W + 1) // 2, "left"); i2 = np.searchsorted(cum, W // 2 + 1, "left")
    return (x_sorted[i1] + x_sorted[i2]) / 2

def county_boot(d, cols, B, seed):
    """scripts/15 bootstrap (same draws for a given seed), returns per-draw MdAPE / PPE10 for each column."""
    rng = np.random.default_rng(seed); cnty = d.fips5.to_numpy(); u = np.unique(cnty); g = np.searchsorted(u, cnty)
    prep = {}
    for k, c in cols.items():
        ape = np.abs(d[c].to_numpy() / d.price.to_numpy() - 1); assert np.isfinite(ape).all()
        o = np.argsort(ape, kind="stable"); prep[k] = (ape[o], g[o], ape <= 0.1)
    out = []
    for _ in range(B):
        wc = np.bincount(np.searchsorted(u, rng.choice(u, len(u), replace=True)), minlength=len(u)).astype(np.int64)
        w = wc[g]; row = {}
        for k, (xs, gs, hit) in prep.items():
            row[k + "_MdAPE"] = wmedian(xs, gs, wc); row[k + "_PPE10"] = float((w * hit).sum() / w.sum())
        out.append(row)
    return pd.DataFrame(out)

qs = lambda s: [float(s.quantile(0.025)), float(s.quantile(0.5)), float(s.quantile(0.975))]

# ---------------------------------------------------------------- (a) B = 1000 bootstrap, OOT
COLS = {"challenger": "$c3_blend", "mktval": "v_vendor_a_mktval", "avm": "v_vendor_a_avm", "gbm": "$c1_gbm"}
d = D[(D.split == "oot") & (D.v_vendor_a_mktval > 1000) & (D.v_vendor_a_avm > 1000)]
bt = county_boot(d, COLS, 1000, 0)
bt["delta"] = bt.challenger_MdAPE - bt.mktval_MdAPE
pub = json.load(open(OUT / "eval_summary.json"))["bootstrap_oot"]
b200 = bt.iloc[:200]
res = {"sample": "OOT sales 2024-07..2025-08 with Vendor A market value and Vendor A AVM > $1,000 (as scripts/15)", "n": int(len(d)),
       "clusters_counties": int(d.fips5.nunique()), "B": 1000, "seed": 0, "quantiles": "[2.5%, 50%, 97.5%] of the bootstrap distribution",
       "challenger": "C3 blend (C1+C2)", "best_usable_vendor": "Vendor A market value",
       "point_estimate_full_sample": {k + "_MdAPE": float(np.median(np.abs(d[c] / d.price - 1))) for k, c in COLS.items()}}
res["point_estimate_full_sample"]["delta_MdAPE_challenger_minus_mktval"] = (res["point_estimate_full_sample"]["challenger_MdAPE"]
                                                                          - res["point_estimate_full_sample"]["mktval_MdAPE"])
res["bootstrap_B1000"] = {k: qs(bt[k]) for k in bt if k != "delta"}
res["bootstrap_B1000"]["delta_MdAPE_challenger_minus_mktval"] = qs(bt.delta)
res["bootstrap_B1000"]["delta_MdAPE_challenger_minus_mktval_se"] = float(bt.delta.std(ddof=1))
res["bootstrap_B1000"]["share_of_draws_delta_below_0"] = float((bt.delta < 0).mean())
res["reproduction_first_200_draws"] = {"delta_MdAPE_challenger_minus_mktval": qs(b200.delta),
                                       "published_B200": pub["delta_MdAPE_challenger_minus_mktval"],
                                       "max_abs_diff": float(np.max(np.abs(np.array(qs(b200.delta)) - np.array(pub["delta_MdAPE_challenger_minus_mktval"]))))}
for k in ["challenger_MdAPE", "mktval_MdAPE", "avm_MdAPE"]:
    res["reproduction_first_200_draws"][k + "_max_abs_diff"] = float(np.max(np.abs(np.array(qs(b200[k])) - np.array(pub[k]))))
json.dump(res, open(OUT / "r4_bootstrap_b1000.json", "w"), indent=2)
print(json.dumps(res, indent=1), f"({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------- (b) spatial x out-of-time cell
EST = {"C3 blend (C1+C2)": "$c3_blend", "Vendor A market value": "v_vendor_a_mktval", "Vendor A AVM (champion)": "v_vendor_a_avm",
       "C1 LightGBM": "$c1_gbm", "C2 neighbour-attention NN": "$c2_attn"}
sp = D[D.split == "spatial"]
WIN = {"all months 2022-05..2025-08": (ym(2022, 5), ym(2025, 8)), "2022-05..2024-06 (before OOT)": (ym(2022, 5), ym(2024, 6)),
       "2024-07..2025-08 (OOT months)": (ym(2024, 7), ym(2025, 8))}
rows = []
for w, (a, b) in WIN.items():
    s = sp[(sp.t_sale >= a) & (sp.t_sale <= b)]
    t = table(s, EST)[["estimator", "n", "coverage", "median_ratio", "MdAPE", "PPE10", "PPE20"]]
    t.insert(0, "window", w); t["states"] = s.state.nunique()
    common = s[(s.v_vendor_a_mktval > 1000) & (s.v_vendor_a_avm > 1000)]
    bb = county_boot(common, {"c3": "$c3_blend", "mktval": "v_vendor_a_mktval"}, 1000, 0)
    dd = bb.c3_MdAPE - bb.mktval_MdAPE
    pt = float(np.median(np.abs(common["$c3_blend"] / common.price - 1)) - np.median(np.abs(common.v_vendor_a_mktval / common.price - 1)))
    t["common_rows_both_vendor_a"] = len(common)
    t["c3_minus_mktval_MdAPE_common"] = pt
    t["c3_minus_mktval_boot_ci95_lo"] = float(dd.quantile(0.025)); t["c3_minus_mktval_boot_ci95_hi"] = float(dd.quantile(0.975))
    t["bootstrap_draws"] = 1000
    rows.append(t)
    print(w, f"n={len(s):,}", t[["estimator", "coverage", "MdAPE"]].round(4).values.tolist(), f"diff {pt:.4f} CI [{dd.quantile(0.025):.4f}, {dd.quantile(0.975):.4f}]")
SPO = pd.concat(rows, ignore_index=True)
old = pd.read_csv(OUT / "accuracy_spatial_own_coverage.csv").set_index("estimator").MdAPE
allm = SPO[SPO.window.str.startswith("all months")].set_index("estimator").MdAPE
print("reproduction of accuracy_spatial_own_coverage.csv MdAPE, max |diff|:", float((allm - old.reindex(allm.index)).abs().max()))
SPO.to_csv(OUT / "r4_accuracy_spatial_oot.csv", index=False)

# ---------------------------------------------------------------- (c) monitoring back-test
psi = pd.read_csv(OUT / "psi_backfill_summary.csv")
mdape = pd.read_csv(OUT / "oot_monthly_MdAPE.csv", index_col=0)
o = D[D.split == "oot"]
re = o.groupby("t_sale").apply(lambda g: pd.Series({"C1": np.nanmedian(np.abs(g["$c1_gbm"] / g.price - 1)),
                                                     "C3": np.nanmedian(np.abs(g["$c3_blend"] / g.price - 1))}), include_groups=False)
re.index = [lab(m) for m in re.index]
chk = float(max((re.C1 - mdape["C1 LightGBM"]).abs().max(), (re.C3 - mdape["C3 blend (C1+C2)"]).abs().max()))
print("monthly MdAPE recomputed vs oot_monthly_MdAPE.csv, max |diff|:", chk)
cal = D[D.split == "calib"]
cal_c1 = float(np.median(np.abs(cal["$c1_gbm"] / cal.price - 1))); cal_c3 = float(np.median(np.abs(cal["$c3_blend"] / cal.price - 1)))
TH = 0.015
m = psi[psi.split == "oot"].set_index("batch")
mon = pd.DataFrame({"window": "out of time", "month": m.index, "records": m.records.to_numpy(),
                    "prediction_psi_c1": m.prediction_psi.to_numpy(), "max_psi_inputs_not_expected_to_shift": m.max_unexpected_psi.to_numpy(),
                    "input_with_max_psi": m.top_unexpected.to_numpy(), "inputs_not_expected_to_shift_at_investigate": m.n_investigate_unexpected.to_numpy(),
                    "max_psi_inputs_expected_to_shift": m.expected_shift_max_psi.to_numpy()})
mon["MdAPE_C1_LightGBM"] = mon.month.map(mdape["C1 LightGBM"]); mon["MdAPE_C3_blend"] = mon.month.map(mdape["C3 blend (C1+C2)"])
mon["C1_minus_calibration_pp"] = (mon.MdAPE_C1_LightGBM - cal_c1) * 100; mon["C3_minus_calibration_pp"] = (mon.MdAPE_C3_blend - cal_c3) * 100
for k in ["C1", "C3"]:
    ex = mon[f"{k}_minus_calibration_pp"] > TH * 100
    mon[f"{k}_trigger_single_month"] = ex
    mon[f"{k}_trigger_two_consecutive"] = ex & ex.shift(1, fill_value=False)
cr = psi[psi.split == "calibration window"].iloc[0]
calrow = pd.DataFrame([{"window": "calibration window", "month": "2024-04..06", "records": int(cr.records), "prediction_psi_c1": cr.prediction_psi,
                        "max_psi_inputs_not_expected_to_shift": cr.max_unexpected_psi, "input_with_max_psi": cr.top_unexpected,
                        "inputs_not_expected_to_shift_at_investigate": int(cr.n_investigate_unexpected), "max_psi_inputs_expected_to_shift": cr.expected_shift_max_psi,
                        "MdAPE_C1_LightGBM": cal_c1, "MdAPE_C3_blend": cal_c3, "C1_minus_calibration_pp": 0.0, "C3_minus_calibration_pp": 0.0}])
mon = pd.concat([calrow, mon], ignore_index=True)
mon["trigger_threshold_pp"] = TH * 100; mon["psi_action_threshold"] = 0.25
mon.to_csv(OUT / "r4_monitoring_backtest.csv", index=False)
first = {f"{k}_{r}": (mon.loc[mon[f"{k}_trigger_{r}"] == True, "month"].min() if (mon[f"{k}_trigger_{r}"] == True).any() else None)
         for k in ["C1", "C3"] for r in ["single_month", "two_consecutive"]}
pd.set_option("display.width", 250)
print(mon.round(4).to_string(index=False))
print("calibration MdAPE C1", round(cal_c1, 5), "C3", round(cal_c3, 5), "| first trigger months", first)
print("max prediction PSI (OOT months)", float(mon.prediction_psi_c1[1:].max()), "max PSI of inputs not expected to shift", float(mon.max_psi_inputs_not_expected_to_shift[1:].max()))
print(f"total {time.time()-t0:.0f}s")
