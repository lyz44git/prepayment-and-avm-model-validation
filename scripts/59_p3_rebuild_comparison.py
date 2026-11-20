"""Credit-loss study check (2026-11-19): headline results of the published tables beside the rebuild with the L15 checks.

Usage: python3 scripts/59_p3_rebuild_comparison.py <rebuilt reports/p3 folder>   (see scripts/59_p3_rebuild_l15.sh)
Writes reports/p3/r6_rebuild_l15_comparison.csv: one row per quantity, published value, rebuilt value.
"""
import sys, json; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import pandas as pd
from _paths import REPORTS

A, B = REPORTS / "p3", Path(sys.argv[1])
J = lambda d, f: json.loads((d / f).read_text())
rows = []
def add(item, fn):
    rows.append({"quantity": item, "published": fn(A), "rebuilt_with_l15_checks": fn(B)})

add("loans in the population", lambda d: J(d, "data_profile.json")["population"])
add("charge-offs in the risk window", lambda d: J(d, "data_profile.json")["by_status_in_window"]["co"])
ev = lambda d: pd.read_csv(d / "eval_hazard.csv").query("cause == 'co'").set_index(["split", "model"])
for m in ("M1 logit", "M2 LightGBM"):
    add(f"{m}: out-of-time AUC", lambda d, m=m: ev(d).loc[("out of time 2020-26", m), "auc"])
    add(f"{m}: out-of-time predicted / actual", lambda d, m=m: ev(d).loc[("out of time 2020-26", m), "pred_over_actual"])
re_ = lambda d: pd.read_csv(d / "remediation_eval.csv").set_index(["model", "window"]).pred_over_actual
add("M1 logit: out of time excluding 2020-04..2021-12", lambda d: re_(d)[("M1 logit", "excluding 2020-04..2021-12")])
r2 = lambda d: J(d, "r2_summary.json")
for k in ("M1v +O1", "R2a +O1", "R2b +O1", "R2c +O1"):
    add(f"{k}: out of time excluding relief months", lambda d, k=k: r2(d)[k]["oot_excl_relief"])
    add(f"{k}: 24-month backtest, origins within 25% (of 8)", lambda d, k=k: r2(d)[k]["within_25pct"])
for k in ("M1", "M1v", "M1r"):
    add(f"{k}: first 24-month backtest, origins within 25% (of 8)", lambda d, k=k: J(d, "backtest_summary.json")[k]["within_25pct"])
for k in ("M1", "M1v", "R2a", "R2b"):
    add(f"{k}: lifetime-to-date backtest, origins within 25% (of 6)", lambda d, k=k: J(d, "r3_summary.json")[k]["within_25pct"])
add("champion: 2020-21 relief-window ratio from 2019-12", lambda d: J(d, "pandemic_check.json")["ratio"])
al = lambda d: pd.read_csv(d / "r2_cecl_allowance.csv").set_index(["scenario", "model"]).loss_over_balance
for k in ("M1v", "R2a", "R2b", "M1"):
    for s in ("baseline", "severe"):
        add(f"{k} (overlay O1): lifetime loss / balance, {s}", lambda d, k=k, s=s: al(d)[(s, k)])
c0 = lambda d: pd.read_csv(d / "cecl_allowance.csv").query("horizon == 'recorded term'").set_index(["scenario", "model"]).loss_over_balance
add("champion (no overlay): lifetime loss / balance, baseline", lambda d: c0(d)[("baseline", "M1")])
mc = lambda d: pd.read_csv(d / "m1v_co_coefficients.csv", index_col=0)
add("unemployment coefficient, M1", lambda d: mc(d).loc["dur4", "beta_m1"])
add("unemployment coefficient, M1v", lambda d: mc(d).loc["dur4", "beta_m1v"])
add("drop in the coefficient, month-cluster bootstrap 2.5%", lambda d: J(d, "r4_dur4_difference_bootstrap.json")["bootstrap_difference_p2_5"])
add("drop in the coefficient, month-cluster bootstrap 97.5%", lambda d: J(d, "r4_dur4_difference_bootstrap.json")["bootstrap_difference_p97_5"])
lag = lambda d: J(d, "r5_backtest_lag_by_origin_summary.json")
add("lag chosen at the 2009-12 origin (months)", lambda d: lag(d)["chosen_lags"]["2009-12"])
for k in ("M1", "M1v", "M1r"):
    add(f"{k}: backtest with the lag chosen at each origin, within 25% (of 8)", lambda d, k=k: lag(d)[k]["within_25pct_lag_by_origin"])
lag2 = lambda d: J(d, "r6_backtest_lag_by_origin_r2_summary.json")
for k in ("M1v", "R2a", "R2b", "R2c"):
    add(f"{k} +O1: backtest with the lag chosen at each origin, within 25% (of 8)", lambda d, k=k: lag2(d)[k]["within_25pct_lag_by_origin"])
out = pd.DataFrame(rows); out.to_csv(REPORTS / "p3" / "r6_rebuild_l15_comparison.csv", index=False)
print(out.to_string(index=False))
