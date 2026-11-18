"""The numbers quoted in the README and the two reports, checked against the tables in reports/.

If a table is regenerated and a headline moves, the matching line here fails, so the text cannot drift from the
results without a test noticing.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

R = Path(__file__).resolve().parents[1] / "reports"
P1, P2 = R / "p1", R / "p2"


def _hazard():
    return pd.read_csv(P1 / "eval_monthly_hazard_final.csv").set_index(["cause", "split", "model"])


def test_challenger_edge_shrinks_out_of_time():
    e = _hazard()
    assert e.loc[("sale", "train", "m2"), "auc"] == pytest.approx(0.704, abs=5e-4)
    assert e.loc[("sale", "train", "m1"), "auc"] == pytest.approx(0.650, abs=5e-4)
    assert e.loc[("sale", "oot", "m2"), "auc"] == pytest.approx(0.642, abs=5e-4)
    assert e.loc[("sale", "oot", "m1"), "auc"] == pytest.approx(0.639, abs=5e-4)
    in_sample = e.loc[("sale", "train", "m2"), "auc"] - e.loc[("sale", "train", "m1"), "auc"]
    out_of_time = e.loc[("sale", "oot", "m2"), "auc"] - e.loc[("sale", "oot", "m1"), "auc"]
    assert in_sample > 0.05 and out_of_time < 0.005


def test_new_lien_hazard_is_underpredicted_out_of_time():
    e = _hazard()
    assert e.loc[("new_lien", "oot", "m1"), "pred_over_actual"] == pytest.approx(0.70, abs=0.005)
    assert e.loc[("new_lien", "oot", "m2"), "pred_over_actual"] == pytest.approx(0.74, abs=0.005)
    assert e.loc[("sale", "oot", "m1"), "pred_over_actual"] == pytest.approx(1.04, abs=0.005)


def test_pool_cpr_backtest():
    b = pd.read_csv(P1 / "monthly_cpr_backtest_summary.csv").set_index(["split", "model"])
    assert b.loc[("oot", "m1"), "MAE_monthly_cpr_pp"] == pytest.approx(0.43, abs=0.005)
    assert b.loc[("oot", "m1"), "corr_monthly"] == pytest.approx(0.90, abs=0.005)
    assert b.loc[("oot", "m2"), "MAE_monthly_cpr_pp"] == pytest.approx(0.48, abs=0.005)


def test_time_series_benchmarks():
    t = pd.read_csv(P1 / "ts_benchmark.csv").set_index(["model", "mode"])
    assert t.loc[("hazard: logit + rolling 3-month recalibration", "rolling 1-month-ahead"), "MAE_pp"] == pytest.approx(0.30, abs=0.005)
    assert t.loc[("ARX(1) + PMMS", "rolling 1-month-ahead"), "MAE_pp"] == pytest.approx(0.31, abs=0.005)


def test_replication_and_rate_sensitivity():
    rep = json.loads((P1 / "m1_replication.json").read_text())
    assert rep["sale"]["max_abs_diff_in_se_units_key"] == pytest.approx(0.20, abs=0.005)
    assert rep["new_lien"]["max_abs_diff_in_se_units_key"] == pytest.approx(0.17, abs=0.005)
    lock = json.loads((P1 / "lockin_effects.json").read_text())["lockin_new_lien"]
    assert lock["odds_ratio_per_pp"] == pytest.approx(0.89, abs=0.005)
    assert lock["or_95ci"][0] == pytest.approx(0.86, abs=0.005) and lock["or_95ci"][1] == pytest.approx(0.93, abs=0.005)


def test_published_stress_test():
    s = pd.read_csv(P1 / "stress_summary.csv").set_index(["scenario", "model"])
    assert s.loc[("base (flat at 2025-06)", "m1"), "avg_cpr_count_24m_pct"] == pytest.approx(3.22, abs=0.005)
    assert s.loc[("-200bp", "m1"), "delta_cpr_count_vs_base_pp"] == pytest.approx(1.25, abs=0.005)
    assert s.loc[("+200bp", "m1"), "delta_cpr_count_vs_base_pp"] == pytest.approx(-0.85, abs=0.005)
    assert s.loc[("+200bp", "m2"), "delta_cpr_count_vs_base_pp"] == pytest.approx(-0.18, abs=0.005)
    assert abs(s.loc[("HPI -10% (rates flat)", "m1"), "delta_cpr_count_vs_base_pp"]) < 0.02      # finding F9


def test_twelve_month_forecast():
    c = pd.read_csv(P1 / "cum12_backtest.csv").set_index(["split", "model"])
    assert c.loc[("oot", "m1"), "properties"] == 482037
    assert c.loc[("oot", "m1"), "top_decile_lift"] == pytest.approx(1.45, abs=0.01)
    assert c.loc[("oot", "m2"), "auc_12m"] == pytest.approx(0.581, abs=5e-4)


def test_cash_flow_valuation():
    v = pd.read_csv(P1 / "valuation_risk.csv").set_index(["extension", "model", "prepay"])
    assert v.loc[("last", "m1", "dynamic"), "base_price"] == pytest.approx(85.7, abs=0.05)
    assert v.loc[("last", "m1", "dynamic"), "eff_duration_100"] == pytest.approx(6.80, abs=0.005)
    assert v.loc[("last", "m1", "static"), "eff_duration_100"] == pytest.approx(6.21, abs=0.005)


def test_valuation_challenger_out_of_time_and_out_of_geography():
    a = pd.read_csv(P2 / "accuracy_oot_own_coverage.csv").set_index("estimator")
    assert a.loc["C3 blend (C1+C2)", "MdAPE"] == pytest.approx(0.114, abs=5e-4)
    assert a.loc["Vendor A market value", "MdAPE"] == pytest.approx(0.163, abs=5e-4)
    assert a.loc["Vendor A AVM (champion)", "MdAPE"] == pytest.approx(0.344, abs=5e-4)
    assert a.loc["C3 blend (C1+C2)", "n"] == 591687
    s = pd.read_csv(P2 / "accuracy_spatial_own_coverage.csv").set_index("estimator")
    assert s.loc["C3 blend (C1+C2)", "MdAPE"] == pytest.approx(0.122, abs=5e-4)       # no better out of geography
    assert s.loc["Vendor A market value", "MdAPE"] == pytest.approx(0.120, abs=5e-4)


def test_vendor_diagnostics():
    st = pd.read_csv(P2 / "vendor_staleness_test.csv").set_index("estimator")
    assert st.loc["Vendor A AVM", "slope_on_market_move"] == pytest.approx(-1.02, abs=0.005)
    conf = json.loads((P2 / "vendor_confidence_check.json").read_text())
    assert conf["auc_stated_for_err_gt20"] == pytest.approx(0.50, abs=0.005)
    a = pd.read_csv(P2 / "accuracy_oot_own_coverage.csv").set_index("estimator")
    assert a.loc["Vendor B value", "eq_sale_0.5pct"] == pytest.approx(0.295, abs=5e-4)   # anchored to the sale price


def test_tables_are_small_and_have_no_id_columns():
    """Published tables are aggregates: no identifier or location columns, nothing near record level in size."""
    banned = {"addr_id", "prop_id", "apn", "owner", "name", "address", "lat", "lon", "tract", "fips5"}
    for f in R.rglob("*.csv"):
        df = pd.read_csv(f, nrows=5)
        assert not (banned & {c.lower() for c in df.columns}), f
        assert sum(1 for _ in open(f)) < 2000, f
    for f in R.rglob("*.json"):
        keys = json.dumps(json.loads(f.read_text())).lower()
        assert not any(f'"{b}"' in keys for b in banned), f


def test_more_headlines():
    t = pd.read_csv(P1 / "ts_benchmark.csv").set_index(["model", "mode"])
    assert t.loc[("naive", "static"), "MAE_pp"] == pytest.approx(0.42, abs=0.005)
    c = pd.read_csv(P1 / "cum12_backtest.csv").set_index(["split", "model"])
    assert c.loc[("oot", "m2"), "top_decile_lift"] == pytest.approx(1.50, abs=0.005)
    s = pd.read_csv(P1 / "stress_summary.csv").set_index(["scenario", "model"])
    assert s.loc[("-200bp", "m2"), "delta_cpr_count_vs_base_pp"] == pytest.approx(0.94, abs=0.005)
    ev = json.loads((P2 / "eval_summary.json").read_text())
    lo, mid, hi = ev["bootstrap_oot"]["delta_MdAPE_challenger_minus_mktval"]
    assert (round(lo * 100, 1), round(mid * 100, 1), round(hi * 100, 1)) == (-6.7, -5.4, -4.4)
    assert ev["bootstrap_oot"]["n"] == 454526
    assert ev["bootstrap_oot"]["challenger_MdAPE"][1] == pytest.approx(0.113, abs=5e-4)
    assert ev["bootstrap_oot"]["mktval_MdAPE"][1] == pytest.approx(0.167, abs=5e-4)
    lo, mid, hi = ev["bootstrap_spatial"]["delta_MdAPE_challenger_minus_mktval"]
    assert lo < 0 < hi and round(mid * 100, 1) == 0.1
    assert ev["conformal_oot"]["coverage"] == pytest.approx(0.775, abs=5e-4)
    assert ev["conformal_oot"]["auc_width_for_err_gt20"] == pytest.approx(0.77, abs=0.005)
    ps = pd.read_csv(P2 / "psi_backfill_summary.csv")
    oot = ps[ps.split == "oot"]
    assert len(oot) == 14 and oot.prediction_psi.max() <= 0.012 and oot.max_unexpected_psi.max() <= 0.012
    m = pd.read_csv(P2 / "oot_monthly_MdAPE.csv")
    assert m["C1 LightGBM"].iloc[0] == pytest.approx(0.111, abs=5e-4) and m["C1 LightGBM"].iloc[-1] == pytest.approx(0.129, abs=5e-4)
    d = pd.read_csv(P2 / "disparity_minority_within_tier_oot.csv").set_index("estimator").Q5_minus_Q1_within_tier
    assert d["Vendor A AVM (champion)"] == pytest.approx(-0.091, abs=5e-4) and d["C3 blend (C1+C2)"] == pytest.approx(-0.020, abs=5e-4)
    st = pd.read_csv(P2 / "vendor_staleness_test.csv").set_index("estimator").slope_on_market_move
    others = st.drop("Vendor A AVM")
    assert others.min() == pytest.approx(-1.39, abs=0.005) and others.max() == pytest.approx(-1.35, abs=0.005)
    q = pd.read_csv(P2 / "vendor_accuracy_by_quarter.csv")
    vb = q[q.estimator.str.startswith("Vendor B")].set_index("group")
    assert vb.loc[:"2024Q3", "MdAPE"].max() < 0.003 and vb.loc["2024Q4", "MdAPE"] == pytest.approx(0.064, abs=5e-4)
    a = pd.read_csv(P2 / "accuracy_oot_own_coverage.csv").set_index("estimator")
    assert a.loc["Vendor A AVM (champion)", "median_ratio"] == pytest.approx(0.68, abs=0.005)


REM = P1 / "remediation"


def test_remediation_started_from_the_published_model():
    rep = json.loads((REM / "step0_reproduction.json").read_text())
    assert rep["pass"] and max(rep["sale"], rep["new_lien"]) < 1e-12
    assert rep["oot_cpr_mae_pp"] == pytest.approx(rep["published_oot_cpr_mae_pp"], abs=1e-9)


def test_remediation_plan_was_hashed_and_selection_used_training_months():
    import hashlib
    meta = json.loads((REM / "meta.json").read_text())
    plan = R.parent / "docs" / "prepayment_model_remediation_plan.md"
    assert hashlib.sha256(plan.read_bytes()).hexdigest() == meta["plan_sha256"]
    assert meta["selected_spec"] == "S1"
    sel = pd.read_csv(REM / "step2_selection_validation.csv")
    nl = sel[sel.cause == "new_lien"].set_index("spec").val_logloss
    assert all(nl[s] > nl["S1"] * (1 - 0.001) for s in ["S2", "S3", "S4"])     # no rate term earned its place


def test_remediation_outcomes_as_reported():
    c = json.loads((REM / "closure.json").read_text())
    assert c["F9"]["closed"] is False and c["F2"]["reduced_at_fixed_origin"] is False and c["F2"]["mitigated_by_maintenance"] is False
    assert c["F9"]["hpi_m10_delta_pp"] == pytest.approx(-0.015, abs=5e-4)
    assert c["F9"]["hpi_m20_delta_pp"] == pytest.approx(-0.030, abs=5e-4)
    assert c["F9"]["hpi_p10_delta_pp"] == pytest.approx(0.015, abs=5e-4)
    assert abs(c["F9"]["s1_minus_s0_oot_auc_sale"]) < 0.001 and abs(c["F9"]["s1_minus_s0_oot_auc_new_lien"]) < 0.001
    assert c["F2"]["m1r_walkforward_oot_new_lien_pred_over_actual"] == pytest.approx(0.71, abs=0.005)
    b = pd.read_csv(REM / "step3_4_cpr_backtest.csv").set_index(["model", "split"])
    assert b.loc[("S1 walk-forward", "oot"), "MAE_monthly_cpr_pp"] == pytest.approx(0.43, abs=0.005)
    assert b.loc[("S3 fixed", "oot"), "new_lien_pred_over_actual"] == pytest.approx(0.83, abs=0.005)
    pool = json.loads((REM / "diag_stress_pool_ltv.json").read_text())
    assert pool["median_ltv"] == pytest.approx(0.30, abs=0.005) and pool["share_ltv_lt_0.5"] == pytest.approx(0.91, abs=0.005)
