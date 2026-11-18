"""Credit-loss study: vintage diagnostic (deviation D2). Predicted / actual charge-offs by approval fiscal year, for
the models fitted on 2006-01..2019-12, in sample (months to 2019-12) and out of time (2020-01..2026-09), hold-out
states excluded. Not used to choose a model. Writes reports/p3/vintage_pred_vs_actual.csv.
"""
import sys, json, pickle; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd
from _paths import WORK, REPORTS
from mval.p3 import VAL_END, W1, LOG_RATE, lgb_frame, macro_quarterly, macro_monthly

R = REPORTS / "p3"; OUT = WORK / "p3"
M = pickle.load(open(OUT / "models_dev.pkl", "rb")); L = M["L"]; mods = M["models"]["oot_co"]
mm = macro_monthly(macro_quarterly(Path("data_external/fed2026"), "baseline"), W1)
cc = pd.read_parquet(OUT / "cc.parquet"); cc["franchise"] = cc.franchise.astype(bool)
p = cc[~cc.holdout & ((cc.y_co == 1) | cc.keep)].copy()
p["dur4"] = mm.dur4.reindex(p.m - L).to_numpy(); p["bbb_spread"] = mm.bbb_spread.reindex(p.m - L).to_numpy()
p["y"] = p.y_co.astype(int); p["w"] = np.where(p.y == 1, 1.0, 20.0)

m1 = mods["m1"]; X = m1["design"].transform(p); X.insert(0, "const", 1.0)
p["h1"] = 1 / (1 + np.exp(-(X.to_numpy(np.float64) @ m1["beta"].reindex(X.columns).to_numpy())))
m2 = mods["m2"]; X2 = lgb_frame(p, m2["macro"])
for c, cats in m2["cats"].items():
    X2[c] = pd.Categorical(X2[c].astype(str), categories=cats)
p["h2"] = 1 / (1 + np.exp(-(m2["booster"].predict(X2, num_iteration=m2["n_trees"], raw_score=True) + LOG_RATE)))
p["period"] = np.where(p.m <= VAL_END, "in sample 2006-19", "out of time 2020-26")
g = p.groupby(["period", "approval_fy"]).apply(lambda d: pd.Series({
    "chargeoffs": d.y.sum(), "loan_months": d.w.sum(), "pred_m1": (d.w * d.h1).sum(), "pred_m2": (d.w * d.h2).sum()}), include_groups=False).reset_index()
g["m1_pred_over_actual"] = g.pred_m1 / g.chargeoffs; g["m2_pred_over_actual"] = g.pred_m2 / g.chargeoffs
g["actual_rate_per_1000_loan_months"] = 1000 * g.chargeoffs / g.loan_months
g.to_csv(R / "vintage_pred_vs_actual.csv", index=False)
print(g.round(3).to_string(index=False))
