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
    assert c[("baseline", "M1v")] == pytest.approx(0.0337, abs=5e-5) and c[("baseline", "M2")] == pytest.approx(0.0664, abs=5e-5)
    assert c[("baseline", "M1")] == pytest.approx(0.0574, abs=5e-5) and c[("severe", "M1")] == pytest.approx(0.0847, abs=5e-5)
    assert c[("severe", "M1v")] - c[("baseline", "M1v")] == pytest.approx(0.0071, abs=1e-4)
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
    assert min(base.values()) == pytest.approx(0.0337, abs=5e-5) and max(base.values()) == pytest.approx(0.0436, abs=5e-5)
    adds = [c[("severe", m)] - c[("baseline", m)] for m in ("M1v", "R2a", "R2b")]
    assert min(adds) == pytest.approx(0.0012, abs=1e-4) and max(adds) == pytest.approx(0.0082, abs=1e-4)
    assert c[("baseline", "M1")] == pytest.approx(0.060, abs=5e-4)
