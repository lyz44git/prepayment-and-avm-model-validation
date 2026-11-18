"""Credit-loss study: remediation candidates for the vintage finding (deviation D4), out-of-time part.

M1v = champion logit + approval-era effects; M1r = champion logit scaled by actual / predicted charge-offs over the
24 months before 2020-01. Both fitted on 2006-01..2019-12 (hold-out states excluded) and scored on 2020-01..2026-09.
Writes reports/p3/remediation_eval.csv, remediation_vintage.csv and m1v_co_coefficients.csv.
"""
import sys, json; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from _paths import WORK, REPORTS
from mval.p3 import VAL_END, W1, LOG_RATE, SAMPLE_RATE, mi, CreditDesign, macro_quarterly, macro_monthly

R = REPORTS / "p3"; OUT = WORK / "p3"
L = json.loads((R / "champion.json").read_text())["lag_months"]
mm = macro_monthly(macro_quarterly(Path("data_external/fed2026"), "baseline"), W1)
cc = pd.read_parquet(OUT / "cc.parquet"); cc["franchise"] = cc.franchise.astype(bool)
p = cc[~cc.holdout & ((cc.y_co == 1) | cc.keep)].copy()
p["dur4"] = mm.dur4.reindex(p.m - L).to_numpy(); p["bbb_spread"] = mm.bbb_spread.reindex(p.m - L).to_numpy()
p["y"] = p.y_co.astype(int); p["w"] = np.where(p.y == 1, 1.0, 1 / SAMPLE_RATE)
fit, oot = p[p.m <= VAL_END], p[p.m > VAL_END]

def fit_logit(d):
    X = d.transform(fit); X.insert(0, "const", 1.0)
    r = sm.Logit(fit.y.to_numpy(), X.to_numpy(np.float64)).fit(method="newton", maxiter=60, disp=0)
    b = pd.Series(r.params, index=X.columns); b["const"] += LOG_RATE
    return b, pd.Series(r.bse, index=X.columns), d
def pred(m, q):
    b, _, d = m; X = d.transform(q); X.insert(0, "const", 1.0)
    return 1 / (1 + np.exp(-(X.to_numpy(np.float64) @ b.reindex(X.columns).to_numpy())))

m1 = fit_logit(CreditDesign("co", L).fit(fit)); m1v = fit_logit(CreditDesign("co", L, eras=True).fit(fit))
relief = (oot.m >= mi(2020, 4)) & (oot.m <= mi(2021, 12))
trail = fit[fit.m > VAL_END - 24]
factor = float(np.clip(trail.y.sum() / (trail.w * pred(m1, trail)).sum(), 0.25, 4))
rows = []
for name, h in [("M1 logit", pred(m1, oot)), ("M1v logit + approval eras", pred(m1v, oot)), ("M1r logit, trailing recalibration", factor * pred(m1, oot))]:
    for part, sel in [("2020-01..2026-09", np.ones(len(oot), bool)), ("excluding 2020-04..2021-12", ~relief.to_numpy()), ("2020-04..2021-12 only", relief.to_numpy())]:
        q, hh = oot[sel], h[sel]
        rows.append({"model": name, "window": part, "chargeoffs": int(q.y.sum()), "pred_over_actual": (q.w * hh).sum() / q.y.sum(),
                     "auc": roc_auc_score(q.y, hh, sample_weight=q.w)})
ev = pd.DataFrame(rows); ev["recalibration_factor"] = np.where(ev.model.str.startswith("M1r"), factor, np.nan)
ev.to_csv(R / "remediation_eval.csv", index=False)
coef = pd.DataFrame({"beta_m1": m1[0], "se_m1": m1[1], "beta_m1v": m1v[0], "se_m1v": m1v[1]})
coef.to_csv(R / "m1v_co_coefficients.csv")
oot = oot.assign(h1=pred(m1, oot), hv=pred(m1v, oot))
vint = oot.groupby("approval_fy").apply(lambda d: pd.Series({"chargeoffs": d.y.sum(), "m1": (d.w * d.h1).sum() / max(d.y.sum(), 1),
                                                              "m1v": (d.w * d.hv).sum() / max(d.y.sum(), 1)}), include_groups=False)
vint.to_csv(R / "remediation_vintage.csv")
print(ev.round(3).to_string(index=False))
print(coef.reindex(["dur4", "bbb_spread", "era1", "era2", "era3", "era4"]).round(3).to_string())
print(vint.round(2).T.to_string())
