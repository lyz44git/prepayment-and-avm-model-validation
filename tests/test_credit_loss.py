"""Credit-loss study: the numbers quoted in the README and docs/credit_loss_cecl_validation.md, checked against reports/p3/."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
P3 = ROOT / "reports" / "p3"


def _csv(name):
    return pd.read_csv(P3 / name)


def test_plan_was_hashed_before_fitting_and_is_unchanged():
    rec = json.loads((P3 / "plan_hash.json").read_text())
    assert hashlib.sha256((ROOT / "docs" / "credit_loss_plan.md").read_bytes()).hexdigest() == rec["sha256"]


def test_population_and_risk_set():
    prof = json.loads((P3 / "data_profile.json").read_text())
    assert prof["population"] == 994_992 and prof["in_risk_window"] == 919_479
    assert prof["by_status_in_window"] == {"co": 115_117, "active": 232_292, "pif": 572_070}
    assert prof["payoff_dates_2005_05"] == 45_770
    assert json.loads((P3 / "panel_info.json").read_text())["loan_months"] == 52_141_481


def test_term_field_is_revised_after_default():
    t = _csv("term_field_check.csv").set_index("final_status").term_not_multiple_of_12
    assert t["charged off"] == pytest.approx(0.90, abs=0.005) and t["cancelled (never disbursed)"] == pytest.approx(0.10, abs=0.005)
    assert t["active"] == pytest.approx(0.16, abs=0.005) and t["paid in full"] == pytest.approx(0.18, abs=0.005)
    old = _csv("first_fit/eval_hazard.csv").set_index(["cause", "split", "model"]).auc
    new = _csv("eval_hazard.csv").set_index(["cause", "split", "model"]).auc
    assert old[("co", "out of time 2020-26", "M2 LightGBM")] == pytest.approx(0.935, abs=5e-4)
    assert new[("co", "out of time 2020-26", "M2 LightGBM")] == pytest.approx(0.737, abs=5e-4)


def test_champion_and_out_of_time_level():
    c = json.loads((P3 / "champion.json").read_text())
    assert c["champion"] == "M1 logit" and c["lag_months"] == 12
    assert c["m2_logloss_gain_vs_m1"] == pytest.approx(0.050, abs=5e-4) and c["m2_validation_pred_over_actual"] == pytest.approx(1.66, abs=0.005)
    e = _csv("eval_hazard.csv").set_index(["cause", "split", "model"])
    oot = e.xs(("co", "out of time 2020-26"), level=("cause", "split"))
    assert oot.loc["M1 logit", "auc"] == pytest.approx(0.681, abs=5e-4) and oot.loc["M1 logit", "pred_over_actual"] == pytest.approx(2.79, abs=0.005)
    assert oot.pred_over_actual.min() > 2.2


def test_vintage_diagnostic_and_era_effects():
    v = _csv("vintage_pred_vs_actual.csv")
    v = v[v.period == "in sample 2006-19"].set_index("approval_fy").m1_pred_over_actual
    assert v[2007] == pytest.approx(0.69, abs=0.005) and v[2006] == pytest.approx(0.77, abs=0.005)
    assert v.loc[2009:2018].min() == pytest.approx(1.28, abs=0.005) and v.loc[2009:2018].max() == pytest.approx(2.34, abs=0.005)
    k = _csv("m1v_co_coefficients.csv").set_index("Unnamed: 0")
    assert k.loc["dur4", "beta_m1"] == pytest.approx(0.215, abs=5e-4) and k.loc["dur4", "beta_m1v"] == pytest.approx(0.103, abs=5e-4)


def test_rolling_backtest():
    s = json.loads((P3 / "backtest_summary.json").read_text())
    assert [s[m]["within_25pct"] for m in ("M0", "M1", "M1r", "M1v")] == [2, 2, 3, 4]
    assert not any(s[m]["passed"] for m in ("M0", "M1", "M1r", "M1v"))
    b = _csv("backtest_rolling.csv")
    m1 = b[b.model == "M1"].pivot(index="origin", columns="macro", values="dollar_ratio")
    rel = (m1["realised"] / m1["held at origin"] - 1).abs()
    assert (rel < 0.03).sum() == 6 and m1.loc["2009-12", "realised"] == pytest.approx(0.98, abs=0.005)


def test_payment_relief_finding():
    p = json.loads((P3 / "pandemic_check.json").read_text())
    assert p["ratio"] == pytest.approx(7.7, abs=0.05) and p["actual_dollars"] == pytest.approx(0.824e9, rel=1e-3)


def test_lifetime_loss_model_versus_scenario():
    c = _csv("cecl_allowance.csv"); c = c[c.horizon == "recorded term"].set_index(["scenario", "model"]).loss_over_balance
    assert c[("baseline", "M1v")] == pytest.approx(0.0342, abs=5e-5) and c[("baseline", "M2")] == pytest.approx(0.0694, abs=5e-5)
    assert c[("baseline", "M1")] == pytest.approx(0.0582, abs=5e-5) and c[("severe", "M1")] == pytest.approx(0.0836, abs=5e-5)
    assert c[("severe", "M1v")] - c[("baseline", "M1v")] == pytest.approx(0.0060, abs=1e-4)
    macro_models = ["M1", "M1r", "M1v", "M2"]
    model_spread = max(c[("baseline", m)] for m in macro_models) - min(c[("baseline", m)] for m in macro_models)
    scenario_spread = max(c[("severe", m)] - c[("baseline", m)] for m in macro_models)
    assert model_spread > scenario_spread


def test_monitoring_and_severity():
    ps = _csv("backtest_psi.csv").set_index("to").psi
    assert ps["2017-12"] == pytest.approx(0.023, abs=5e-4)
    sev = _csv("eval_severity.csv").set_index(["split", "model"]).dollar_pred_over_actual
    assert sev[("out of time 2020-26", "S1 fractional logit")] == pytest.approx(1.02, abs=0.005)
    r = _csv("remediation_eval.csv").set_index(["model", "window"]).pred_over_actual
    assert r[("M1v logit + approval eras", "excluding 2020-04..2021-12")] == pytest.approx(0.74, abs=0.005)
    assert r[("M1r logit, trailing recalibration", "excluding 2020-04..2021-12")] == pytest.approx(0.93, abs=0.005)


def test_remediation_rounds_as_reported():
    """D5: relief overlay and approval eras; D6: lifetime-to-date backtest. No pre-set rule is met."""
    s2 = json.loads((P3 / "r2_summary.json").read_text())
    assert s2["adopted"] is None
    assert s2["M1v +O1"]["within_25pct"] == 4 and s2["M1v +O1"]["oot_excl_relief"] == pytest.approx(0.81, abs=0.005)
    assert s2["R2a +O1"]["within_25pct"] == 5 and s2["R2a +O1"]["oot_excl_relief"] == pytest.approx(0.73, abs=0.005)
    assert s2["M1v +O1"]["mean_abs_log_ratio"] == pytest.approx(0.357, abs=5e-4)
    # no single candidate meets both rules
    assert not any(v["rule_i"] and v["rule_ii"] for k, v in s2.items() if isinstance(v, dict) and "rule_i" in v)
    with_o1 = [v["oot_excl_relief"] for k, v in s2.items() if k.endswith("+O1")]
    assert min(with_o1) == pytest.approx(0.72, abs=0.005) and max(with_o1) == pytest.approx(0.83, abs=0.005)
    assert s2["M1v"]["oot_excl_relief"] == pytest.approx(0.74, abs=0.005)  # same fit without O1: the gap is the overlay
    m1 = json.loads((P3 / "backtest_summary.json").read_text())["M1"]["mean_abs_log_ratio"]
    assert m1 == pytest.approx(0.68, abs=0.005) and s2["R2a +O1"]["mean_abs_log_ratio"] / m1 < 0.5
    oot = _csv("r2_oot_eval.csv")
    whole = oot[oot.overlay & (oot.model == "M1v") & (oot.window == "2020-01..2026-09")].pred_over_actual.iloc[0]
    assert whole == pytest.approx(0.91, abs=0.005)
    s3 = json.loads((P3 / "r3_summary.json").read_text())
    assert s3["adopted_for_lifetime_allowance"] is None
    assert [s3[m]["within_25pct"] for m in ("M0", "M1", "M1v", "R2a", "R2b")] == [2, 0, 3, 3, 4]
    lt = _csv("r3_lifetime_backtest.csv")
    late = lt[lt.origin.isin(["2015-12", "2017-12", "2019-12"]) & lt.model.isin(["M1v", "R2a", "R2b"])].dollar_ratio
    assert late.min() == pytest.approx(0.94, abs=0.005) and late.max() == pytest.approx(1.22, abs=0.005)
    early = lt[lt.origin.isin(["2009-12", "2011-12", "2013-12"]) & lt.model.isin(["M1", "M1v", "R2a", "R2b"])].dollar_ratio
    assert (early < 0.7).sum() == 11


def test_lifetime_backtest_without_overlay():
    s = json.loads((P3 / "r3_summary_no_overlay.json").read_text())
    assert [s[m]["within_25pct"] for m in ("M0", "M1", "M1v", "R2a", "R2b")] == [2, 0, 3, 3, 4]
    assert s["adopted_for_lifetime_allowance"] is None
    lt = _csv("r3_lifetime_backtest_no_overlay.csv")
    late = lt[lt.origin.isin(["2015-12", "2017-12", "2019-12"]) & lt.model.isin(["M1v", "R2a", "R2b"])].dollar_ratio
    assert late.min() == pytest.approx(0.87, abs=0.005) and late.max() == pytest.approx(1.21, abs=0.005)


def test_input_checks_added_at_review():
    """L11: business-age coding by approval year; L12: 2026Q1-Q3 scenario inputs against published series."""
    b = _csv("business_age_by_approval_fy.csv").set_index("approval_fy")
    assert b.loc[2000:2004, "startup"].max() < 0.01 and b.loc[2012:, "startup"].min() == pytest.approx(0.176, abs=5e-4)
    assert b.loc[2012:, "startup"].max() == pytest.approx(0.215, abs=5e-4)
    assert b.loc[2000:2008, "new"].min() == pytest.approx(0.236, abs=5e-4) and b.loc[2000:2008, "new"].max() == pytest.approx(0.311, abs=5e-4)
    assert b.loc[2019, "new"] == 0 and b.loc[2021:, "new"].min() == pytest.approx(0.130, abs=5e-4) and b.loc[2021:, "new"].max() == pytest.approx(0.180, abs=5e-4)
    assert b.loc[2018, "unknown"] == pytest.approx(0.118, abs=5e-4) and b.loc[2019, "unknown"] == pytest.approx(0.136, abs=5e-4)
    assert b.loc[2021:, "unknown"].max() < 0.01
    m = _csv("macro_2026_scenario_vs_published.csv").set_index(["source", "quarter"])
    assert (m.loc["fed_baseline", "unemployment"] == 4.6).all()
    pub = m.loc["published_fred"]
    assert pub.unemployment.min() == pytest.approx(4.13, abs=0.005) and pub.unemployment.max() == pytest.approx(4.33, abs=0.005)
    assert m.loc["fed_baseline", "prime"].min() == 6.2 and m.loc["fed_baseline", "prime"].max() == 6.6
    assert pub.prime.round(1).unique().tolist() == [6.8]
    sa = m.loc["fed_severely_adverse", "unemployment"]
    assert sa.min() == 5.9 and sa.max() == 8.2


def test_allowance_after_remediation():
    c = _csv("r2_cecl_allowance.csv").set_index(["scenario", "model"]).loss_over_balance
    base = {m: c[("baseline", m)] for m in ("M1v", "R2a", "R2b")}
    assert min(base.values()) == pytest.approx(0.0346, abs=5e-5) and max(base.values()) == pytest.approx(0.0448, abs=5e-5)
    adds = [c[("severe", m)] - c[("baseline", m)] for m in ("M1v", "R2a", "R2b")]
    assert min(adds) == pytest.approx(-0.0005, abs=1e-4) and max(adds) == pytest.approx(0.0053, abs=1e-4)
    assert c[("baseline", "M1")] == pytest.approx(0.062, abs=5e-4) and c[("severe", "M1")] == pytest.approx(0.082, abs=5e-4)
    assert max(base.values()) - min(base.values()) > max(adds)        # model choice moves the estimate more than the scenario


def test_disclosures_added_at_second_review():
    """Era-model cohort calibration, final-fit factor, payoff macro terms, hold-out states, PSI flags."""
    v = _csv("remediation_vintage.csv").set_index("approval_fy")
    assert v.loc[2014:2019, "m1v"].min() == pytest.approx(0.70, abs=0.005) and v.loc[2014:2019, "m1v"].max() == pytest.approx(0.82, abs=0.005)
    assert v.loc[2022:2023, "m1v"].min() == pytest.approx(0.64, abs=0.005) and v.loc[2022:2023, "m1v"].max() == pytest.approx(0.71, abs=0.005)
    assert v.loc[2009:2013, "m1v"].min() == pytest.approx(1.27, abs=0.005) and v.loc[2009:2013, "m1v"].max() == pytest.approx(1.78, abs=0.005)
    assert v.loc[2000:2008, "m1v"].min() == pytest.approx(2.8, abs=0.05) and v.loc[2000:2008, "m1v"].max() == pytest.approx(8.8, abs=0.05)
    share = v.loc[2014:2019, "chargeoffs"].sum() / v.chargeoffs.sum()
    assert share == pytest.approx(0.62, abs=0.005)
    c = _csv("r2_cecl_allowance.csv").set_index(["scenario", "model"]).factor
    assert c[("baseline", "R2a")] == pytest.approx(1.29, abs=0.005)
    k = _csv("m1_pif_coefficients.csv").set_index("Unnamed: 0").beta
    assert k["dur4"] == pytest.approx(-0.107, abs=5e-4) and k["prime"] == pytest.approx(0.121, abs=5e-4)
    e = _csv("eval_hazard.csv").query("cause == 'co'").set_index(["split", "model"])
    ho, hoo = "hold-out states 2006-19", "hold-out states, out of time"
    assert e.loc[(ho, "M1 logit"), "auc"] == pytest.approx(0.734, abs=5e-4) and e.loc[(ho, "M1 logit"), "pred_over_actual"] == pytest.approx(1.09, abs=0.005)
    assert e.loc[(ho, "M2 LightGBM"), "pred_over_actual"] == pytest.approx(0.71, abs=0.005)
    assert e.loc[(hoo, "M1 logit"), "pred_over_actual"] == pytest.approx(3.06, abs=0.005) and e.loc[(hoo, "M2 LightGBM"), "pred_over_actual"] == pytest.approx(1.77, abs=0.005)
    ps = _csv("backtest_psi.csv").set_index("to").psi
    assert ps["2011-12"] == pytest.approx(0.34, abs=0.005) and ps["2013-12"] == pytest.approx(0.40, abs=0.005) and ps["2023-12"] < 0.01
    b = _csv("backtest_rolling.csv"); r = b[(b.model == "M1") & (b.macro == "realised")].set_index("origin").dollar_ratio
    assert r["2023-12"] == pytest.approx(2.00, abs=0.005)
    m = _csv("monthly_actuals.csv"); last = m[m.m == m.m.max()]
    assert last.n_pif.sum() == 0


def test_robustness_checks_added_at_review():
    s = json.loads((P3 / "r4_backtest_summary_v2.json").read_text())
    v2 = s["v2"]
    assert [v2[f"{m} | realised"]["within_25pct"] for m in ("M0", "M1", "M1r", "M1v")] == [2, 2, 3, 4]
    assert v2["M1 | realised"]["mean_abs_log_ratio_all"] == pytest.approx(0.684, abs=5e-4)
    assert v2["M1 | realised"]["mean_abs_log_ratio_excl_2019_12"] == pytest.approx(0.499, abs=5e-4)
    rd = s["m1_realised_vs_held_relative_difference"]
    assert rd["v1"]["origins_abs_lt_3pct"] == 6 and rd["v2"]["origins_abs_lt_3pct"] == 5
    assert rd["v2"]["per_origin"]["2021-12"] == pytest.approx(0.041, abs=5e-4)
    pc = json.loads((P3 / "r4_pandemic_check_v2.json").read_text())
    assert pc["v2"]["M1 | realised"]["ratio"] == pytest.approx(7.66, abs=0.005)
    assert pc["v1_from_backtest_rolling_monthly_csv"]["M1 | realised"]["ratio"] == pytest.approx(7.69, abs=0.005)
    t = _csv("r4_monitoring_trigger.csv")
    def cells(tab):
        q = t[t.table == tab]
        return {(str(f), str(m)): int(c) for f, m, c in zip(q.flag, q.missed, q["count"])}
    c = cells("2x2 count trigger x realised-macro miss")
    assert c == {("True", "True"): 5, ("True", "False"): 0, ("False", "True"): 1, ("False", "False"): 2}
    d = cells("2x2 dollar trigger x realised-macro miss")
    assert d == {("True", "True"): 4, ("True", "False"): 1, ("False", "True"): 2, ("False", "False"): 1}
    po = t[t.table == "per origin"].set_index("origin")
    assert po.loc["2013-12", "trailing_ratio_counts"] == pytest.approx(1.49, abs=0.005)
    assert po.loc["2013-12", "next24_ratio_realised_v2"] == pytest.approx(0.56, abs=0.005)
    p = _csv("r4_psi_exante.csv"); q = p[p.table == "2x2 PSI>0.25 x realised-macro miss at T1"]
    assert {(str(f), str(m)): int(n) for f, m, n in zip(q.flag, q.missed, q["count"])} == {
        ("True", "True"): 2, ("True", "False"): 0, ("False", "True"): 4, ("False", "False"): 1}
    a = _csv("r4_allowance_sensitivities.csv"); a["pct"] = 100 * a.loss_over_balance
    v = a.set_index(["variant", "model", "scenario"]).pct
    for key, want in {("base", "M1v", "baseline"): 3.46, ("base", "R2a", "baseline"): 4.48, ("base", "R2b", "severe"): 3.42,
                      ("ii", "M1v", "baseline"): 3.64, ("ii", "R2a", "severe"): 5.22, ("iii", "M1v", "baseline"): 3.44,
                      ("iv", "R2a", "baseline"): 3.98, ("iv", "R2a", "severe"): 4.57, ("v", "M1v", "baseline"): 4.01,
                      ("v", "M1v", "severe"): 4.48}.items():
        assert v[key] == pytest.approx(want, abs=0.005), key
    base = a[a.variant == "base"].set_index(["model", "scenario"]).pct
    slow = a[a.variant == "iii"].set_index(["model", "scenario"]).pct
    assert (slow - base.reindex(slow.index)).abs().max() <= 0.035   # reversion speed: 0.03 points or less
    assert v[("i", "M1v", "increment: charge-off channel")] == pytest.approx(0.31, abs=0.005)
    assert v[("i", "M1v", "increment: payoff channel")] == pytest.approx(0.10, abs=0.005)
    assert v[("i", "R2b", "increment: charge-off channel")] == pytest.approx(-0.13, abs=0.005)
    assert v[("i", "R2b", "increment: payoff channel")] == pytest.approx(0.08, abs=0.005)
    lag = json.loads((P3 / "r4_recording_lag.json").read_text())
    assert lag["months_with_zero_recorded_payoffs_2006-01..2026-09"] == ["2026-09"]
    assert lag["payoff_oot_pred_over_actual_M1"]["2020-01..2026-08 (excluding 2026-09)"] == pytest.approx(1.105, abs=5e-4)
    assert lag["typical_month_2025-09..2026-08"]["mean_payoffs_as_share_of_2026-09_active_book"] == pytest.approx(0.0099, abs=5e-5)
    i = _csv("r4_inference_month_clustered.csv").set_index(["model", "term"])
    assert i.loc[("M1", "dur4"), "z_month_cluster"] == pytest.approx(8.8, abs=0.05)
    assert i.loc[("M1v", "dur4"), "z_month_cluster"] == pytest.approx(4.5, abs=0.05)
    assert i.loc[("R2b", "dx"), "z_month_cluster"] == pytest.approx(-3.7, abs=0.05)
    assert i.xs("bbb_spread", level="term").z_month_cluster.abs().max() < 1
    eras = i.loc[[k for k in i.index if k[0] == "M1v" and k[1].startswith("era")]].z_month_cluster.abs()
    assert eras.min() == pytest.approx(4.2, abs=0.05) and eras.max() == pytest.approx(13.2, abs=0.05)
    r = i.cluster_over_model_se
    assert r.min() >= 3.9 and r.max() <= 7.5
    b = json.loads((P3 / "r4_dur4_difference_bootstrap.json").read_text())
    assert b["point_difference"] == pytest.approx(0.112, abs=5e-4)
    assert b["bootstrap_difference_p2_5"] == pytest.approx(0.095, abs=5e-4) and b["bootstrap_difference_p97_5"] == pytest.approx(0.166, abs=5e-4)
    k = _csv("r4_final_fit_coefficients.csv").set_index(["model", "term"]).beta
    assert k[("M1v", "era4")] == pytest.approx(-1.14, abs=0.005) and k[("M1v", "era3")] == pytest.approx(-0.96, abs=0.005)


def test_scenario_path_survival_and_input_dates():
    """L16 scenario path, survival boundary, L15 input dates: the latest tables and the bridge to the earlier ones."""
    path = _csv("r5_scenario_path_reconciliation.csv")
    after = path[path.after_scenario]
    assert (after.dur4_from_levels - after.ur_12m_change).abs().max() < 1e-9          # changes follow the levels
    assert (after.dprime4_from_levels - after.prime_12m_change).abs().max() < 1e-9
    assert (after.dur4_own_reversion - after.ur_12m_change).abs().max() > 2          # the earlier runs did not
    before = path[~path.after_scenario]
    assert (before.dur4_own_reversion == before.dur4_from_levels).all()              # history and scenario untouched
    br = _csv("r5_allowance_update_bridge.csv")
    cand = br[br.source.str.startswith("46") & br.model.isin(["M1v", "R2a", "R2b"])]
    pub = cand.pivot(index="model", columns="scenario", values="loss_over_balance_earlier")
    cor = cand.pivot(index="model", columns="scenario", values="loss_over_balance_latest")
    assert round(100 * pub.baseline.min(), 1) == 3.4 and round(100 * pub.baseline.max(), 1) == 4.4
    assert round(100 * cor.baseline.min(), 1) == 3.5 and round(100 * cor.baseline.max(), 1) == 4.5
    assert round(100 * (pub.severe - pub.baseline).min(), 1) == 0.1 and round(100 * (pub.severe - pub.baseline).max(), 1) == 0.8
    assert round(100 * (cor.severe - cor.baseline).min(), 2) == -0.05 and round(100 * (cor.severe - cor.baseline).max(), 2) == 0.53
    assert round(100 * (pub.severe - pub.baseline).min(), 2) == 0.12 and round(100 * (pub.severe - pub.baseline).max(), 2) == 0.82
    champ = br[br.source.str.startswith("44") & (br.model == "M1") & (br.horizon == "recorded term") & (br.scenario == "baseline")].iloc[0]
    assert round(100 * champ.loss_over_balance_earlier, 1) == 5.7 and round(100 * champ.loss_over_balance_latest, 1) == 5.8
    old = pd.read_csv(P3 / "before_2026-11-19" / "r2_cecl_allowance.csv").set_index(["scenario", "model"]).loss_over_balance
    assert old[("baseline", "M1v")] == pytest.approx(0.0337, abs=5e-5)                 # the earlier table is kept as published
    for f in ("cecl_hazard_mass_checks.csv", "r2_hazard_mass_checks.csv"):
        m = _csv(f)
        assert (m.rows_sum_above_1 == 0).all() and m.max_sum.max() < 1                 # survival never hits the boundary
    d = _csv("r5_input_date_checks.csv").set_index("check").loans
    assert d["  of which in the modelled population (first build dated them 2026-09)"] == 16
    assert d["first disbursement more than 12 months before approval (population)"] == 9
    assert d["population, first build"] == 994_992 and d["population, first build"] - d["population, with the L15 checks"] == 25
    assert d["charge-offs in the risk window, first build"] - d["charge-offs in the risk window, with the L15 checks"] == 16
    u = _csv("r5_unemployment_2020.csv").set_index(["year", "q"]).ur
    assert u[(2020, 2)] - u[(2020, 1)] == pytest.approx(9.2, abs=0.05)


def test_backtest_with_the_lag_chosen_at_each_origin():
    """Report §14: choosing the macro lag at each origin from data available then does worse."""
    s = json.loads((P3 / "r5_backtest_lag_by_origin_summary.json").read_text())
    assert s["chosen_lags"] == {"2009-12": 24, "2011-12": 12, "2013-12": 12, "2015-12": 0, "2017-12": 12, "2019-12": 12,
                                "2021-12": 24, "2023-12": 12}
    got = {m: (s[m]["within_25pct_fixed_lag_12"], s[m]["within_25pct_lag_by_origin"]) for m in ("M1", "M1v", "M1r")}
    assert got == {"M1": (2, 1), "M1v": (4, 3), "M1r": (3, 1)}
    assert round(s["M1v"]["per_origin"]["2009-12"]["fixed_lag_12"], 2) == 0.92 and round(s["M1v"]["per_origin"]["2009-12"]["lag_by_origin"], 2) == 1.72
    for m in ("M1", "M1v", "M1r"):
        assert s[m]["mean_abs_log_ratio_lag_by_origin"] > s[m]["mean_abs_log_ratio_fixed_lag_12"]


def test_inference_history_channels_and_book():
    """Report §14: Newey-West inference, published 2026 history, exact scenario channels, severity cap, book, term field."""
    t = _csv("r6_inference_time_blocks.csv").set_index(["model", "term"])
    assert round(t.loc[("M1", "dur4"), "z_hac_12m"], 1) == 3.6 and round(t.loc[("M1", "dur4"), "z_hac_24m"], 1) == 3.2
    assert round(t.loc[("M1v", "dur4"), "z_hac_12m"], 1) == 2.2 and round(t.loc[("M1v", "dur4"), "z_hac_24m"], 1) == 2.0
    eras = t.loc[[k for k in t.index if k[0] == "M1v" and k[1].startswith("era")]]
    assert round(eras.z_hac_12m.abs().min(), 1) == 2.1 and round(eras.z_hac_24m.abs().min(), 1) == 2.2
    d = json.loads((P3 / "r6_dur4_difference_time_blocks.json").read_text())
    assert round(d["se_hac_12m"], 3) == 0.035 and round(d["se_hac_24m"], 3) == 0.042
    assert round(d["block_bootstrap"]["p2_5"], 3) == 0.036 and round(d["block_bootstrap"]["p97_5"], 3) == 0.173
    h = _csv("r6_allowance_published_history.csv"); h["pct"] = (100 * h.loss_over_balance).round(2)
    h = h.set_index(["history", "model", "scenario"]).pct
    assert [h[("published history to 2026Q3", m, "baseline")] for m in ("M1v", "R2a", "R2b")] == [3.45, 4.47, 3.48]
    assert [h[("published history to 2026Q3", m, "severe")] for m in ("M1v", "R2a", "R2b")] == [3.95, 5.11, 3.53]
    assert [h[("as published", m, "baseline")] for m in ("M1v", "R2a", "R2b")] == [3.46, 4.48, 3.47]
    c = _csv("r6_scenario_channels.csv").set_index(["model", "channel"])
    assert [round(c.loc[("M1v", k), "shapley_pp"], 2) for k in ("charge-off hazard", "payoff hazard")] == [0.32, 0.08]
    assert [round(c.loc[("R2b", k), "shapley_pp"], 2) for k in ("charge-off hazard", "payoff hazard")] == [-0.13, 0.07]
    assert round(c.loc[("M1v", "severity"), "shapley_pp"], 3) == 0.006 and round(c.loc[("R2b", "severity"), "shapley_pp"], 3) == 0.005
    assert (c.order_chargeoff_first_pp - c.shapley_pp).abs().max() <= 0.015 and (c.order_severity_first_pp - c.shapley_pp).abs().max() <= 0.015
    s = _csv("r6_severity_capping.csv").set_index("split").loc["out of time 2020-26"]
    assert s.above_original_amount == 411 and s.chargeoffs == 14_808 and round(s.dollars_removed_by_cap_share, 3) == 0.008
    assert round(s.pred_over_recorded_actual, 2) == 1.01 and round(s.pred_over_capped_actual, 2) == 1.02
    b = _csv("r6_book_bridge.csv"); a = b[b.origin == "2026-09 (allowance)"].set_index("item").loans
    assert a.iloc[0] == 232_292 and sorted(a.iloc[1:].tolist()) == [354, 2_790, 5_047, 5_047]
    left = b[b.item.str.startswith("open at T")].loans
    assert left.min() >= 2_400 and left.max() <= 3_600
    ab = _csv("r6_term_ablation.csv").set_index(["risk_set", "term_features"]).oot_auc.round(3)
    assert ab[("published (not cut)", False)] == 0.737 and ab[("published (not cut)", True)] == 0.935
    assert ab[("cut at the recorded term", False)] == 0.723 and ab[("cut at the recorded term", True)] == 0.926
    w = _csv("r6_within_month_auc.csv").set_index("model").oot_auc_within_month.round(3)
    assert w["M1 logit"] == 0.710 and w["M2 LightGBM"] == 0.741


def test_lag_by_origin_and_rebuild():
    """Report §14: the second-round candidates with the lag chosen at each origin; the rebuild with the L15 checks."""
    s = json.loads((P3 / "r6_backtest_lag_by_origin_r2_summary.json").read_text())
    got = {m: (s[m]["within_25pct_fixed_lag_12"], s[m]["within_25pct_lag_by_origin"]) for m in ("M1v", "R2a", "R2b", "R2c")}
    assert got == {"M1v": (4, 3), "R2a": (5, 4), "R2b": (4, 3), "R2c": (4, 4)}
    r = _csv("r6_rebuild_l15_comparison.csv").set_index("quantity")
    counts = r[r.index.str.contains("origins within 25%") & ~r.index.str.contains("chosen at each origin")]
    assert (counts.published == counts.rebuilt_with_l15_checks).all()           # no pass count changes
    assert r.loc["loans in the population", "rebuilt_with_l15_checks"] == 994_967
    assert r.loc["charge-offs in the risk window", "rebuilt_with_l15_checks"] == 115_101
    assert round(r.loc["M1 logit: out of time excluding 2020-04..2021-12", "rebuilt_with_l15_checks"], 2) == 1.77
    assert round(r.loc["M2 LightGBM: out-of-time AUC", "rebuilt_with_l15_checks"], 3) == 0.736
    assert r.loc["lag chosen at the 2009-12 origin (months)", "published"] == 24 and r.loc["lag chosen at the 2009-12 origin (months)", "rebuilt_with_l15_checks"] == 12
