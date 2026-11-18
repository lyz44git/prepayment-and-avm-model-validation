"""Prepayment study extension: time-series benchmarks for the monthly pool CPR, against the hazard models.

Series: monthly pool CPR (sale + new lien, count-based) for the non-hold-out states, from
reports/p1/monthly_cpr_actual_vs_pred.csv. Training 2022-05..2024-06 (26 months), out-of-time 2024-07..2025-06
(12 months), the same split as the hazard models.

Benchmarks
- naive (last training value), seasonal naive (same month one year earlier), training mean
- simple exponential smoothing, damped-trend Holt
- AR(1) with mean, and ARX(1) with the monthly PMMS rate as a regressor (uses realised PMMS, like the hazard models)

Two forecast modes
- static: one 12-month forecast from 2024-06 (the hazard models' setting: fitted once, never updated)
- rolling: one-month-ahead, refitted each month on all actuals up to the previous month (this gives the
  time-series models information the hazard models never see)

Hazard models (m0 lookup, m1 logit, m2 LightGBM) are scored on the same 12 months.
"""
import sys, json, warnings, functools; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
from _paths import REPORTS
from mval import figstyle
from mval.macro import monthly_rates
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
print = functools.partial(print, flush=True)
OUT = REPORTS / "p1"

D = pd.read_csv(OUT / "monthly_cpr_actual_vs_pred.csv", parse_dates=["month"])
D = D[D.split.isin(["train", "oot"])].sort_values("month").reset_index(drop=True)
y = D.set_index("month").cpr_total_y * 100                      # CPR in percent
pm = monthly_rates("2022-01-01").pmms30.reindex(y.index)
n_tr = int((D.split == "train").sum()); idx_oot = y.index[n_tr:]
assert n_tr == 26 and len(idx_oot) == 12

def fc(name, hist, steps, x_hist=None, x_fut=None):
    """Forecast `steps` months after `hist` (a pd.Series with monthly index)."""
    h = hist.to_numpy()
    if name == "naive": return np.repeat(h[-1], steps)
    if name == "seasonal naive":
        full = pd.concat([hist, pd.Series(np.nan, index=pd.date_range(hist.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq="MS"))])
        out = []
        for k in range(steps):
            i = len(hist) + k; v = full.iloc[i - 12]
            full.iloc[i] = v; out.append(v)
        return np.array(out)
    if name == "training mean": return np.repeat(h.mean(), steps)
    if name == "exponential smoothing": return SimpleExpSmoothing(h, initialization_method="estimated").fit().forecast(steps)
    if name == "damped Holt": return ExponentialSmoothing(h, trend="add", damped_trend=True, initialization_method="estimated").fit().forecast(steps)
    if name == "AR(1)": return SARIMAX(h, order=(1, 0, 0), trend="c").fit(disp=False).forecast(steps)
    if name == "ARX(1) + PMMS":
        r = SARIMAX(h, exog=x_hist, order=(1, 0, 0), trend="c").fit(disp=False)
        return r.forecast(steps, exog=x_fut)
    raise ValueError(name)

TS = ["naive", "seasonal naive", "training mean", "exponential smoothing", "damped Holt", "AR(1)", "ARX(1) + PMMS"]
pred = {}
for name in TS:
    hist = y.iloc[:n_tr]
    pred[(name, "static")] = fc(name, hist, 12, pm.iloc[:n_tr].to_numpy()[:, None], pm.loc[idx_oot].to_numpy()[:, None])
    roll = []
    for k in range(12):
        hk = y.iloc[:n_tr + k]
        roll.append(fc(name, hk, 1, pm.iloc[:n_tr + k].to_numpy()[:, None], pm.iloc[[n_tr + k]].to_numpy()[:, None])[0])
    pred[(name, "rolling 1-month-ahead")] = np.array(roll)
for m, lab in [("m0", "hazard: actuarial lookup (M0)"), ("m1", "hazard: logit champion (M1)"), ("m2", "hazard: LightGBM (M2)")]:
    pred[(lab, "static")] = (D.loc[n_tr:, f"cpr_total_{m}"] * 100).to_numpy()

# hazard + monitoring recalibration (findings-log F2): scale each OOT month's logit prediction by the
# actual/predicted ratio over the previous 3 months, using only actuals already observed
m1_all = D.cpr_total_m1.to_numpy() * 100; y_all = y.to_numpy()
pred[("hazard: logit + rolling 3-month recalibration", "rolling 1-month-ahead")] = np.array(
    [m1_all[n_tr + k] * y_all[n_tr + k - 3:n_tr + k].sum() / m1_all[n_tr + k - 3:n_tr + k].sum() for k in range(12)])
act = y.loc[idx_oot].to_numpy()
rows = []
for (name, mode), p in pred.items():
    p = np.asarray(p, float)
    rows.append({"model": name, "mode": mode, "MAE_pp": float(np.abs(p - act).mean()), "bias_pp": float((p - act).mean()),
                 "corr": float(np.corrcoef(p, act)[0, 1]) if p.std() > 1e-9 else np.nan,
                 "family": "hazard" if name.startswith("hazard") else "time series"})
R = pd.DataFrame(rows).sort_values(["mode", "MAE_pp"])
R.to_csv(OUT / "ts_benchmark.csv", index=False)
P = pd.DataFrame({"month": idx_oot.strftime("%Y-%m"), "actual": act,
                  **{f"{n} | {m}": np.asarray(v, float) for (n, m), v in pred.items()}})
P.to_csv(OUT / "ts_benchmark_paths.csv", index=False)
pd.set_option("display.width", 200)
print(R.round(3).to_string(index=False))

# ---------------------------------------------------------------- figure
figstyle.apply()
fig, ax = plt.subplots(figsize=(8.2, 4.2))
best_static = R[(R["mode"] == "static") & (R.family == "time series")].iloc[0].model
best_roll = R[(R["mode"] == "rolling 1-month-ahead") & (R.family == "time series")].iloc[0].model
ax.plot(y.index, y.values, color=figstyle.INK, lw=1.6, label="Actual pool CPR")
ax.plot(idx_oot, pred[("hazard: logit champion (M1)", "static")], color=figstyle.SERIES[0], marker="o", ms=4,
        mec=figstyle.SURFACE, mew=1.2, label="Logit hazard (fitted once, realised rates)")
ax.plot(idx_oot, pred[(best_static, "static")], color=figstyle.SERIES[1], marker="o", ms=4, mec=figstyle.SURFACE, mew=1.2,
        label=f"Best fixed time-series forecast: {best_static}")
ax.plot(idx_oot, pred[(best_roll, "rolling 1-month-ahead")], color=figstyle.SERIES[2], marker="o", ms=4, mec=figstyle.SURFACE,
        mew=1.2, label=f"Best rolling 1-month-ahead: {best_roll}")
ax.plot(idx_oot, pred[("hazard: logit + rolling 3-month recalibration", "rolling 1-month-ahead")], color=figstyle.SERIES[3],
        marker="o", ms=4, mec=figstyle.SURFACE, mew=1.2, ls="--", label="Logit hazard + rolling 3-month recalibration")
ax.axvspan(idx_oot[0] - pd.Timedelta(days=15), idx_oot[-1] + pd.Timedelta(days=15), color=figstyle.GRID, alpha=0.35, lw=0)
ax.text(idx_oot[0], ax.get_ylim()[1], " out of time", va="top", fontsize=8, color=figstyle.INK2)
ax.set_ylabel("CPR (%)"); ax.set_title("Monthly pool CPR: hazard model vs. time-series benchmarks")
ax.legend(loc="lower left", fontsize=7.8)
g = R.set_index(["model", "mode"])
m1 = g.loc[("hazard: logit champion (M1)", "static")]; bs = g.loc[(best_static, "static")]; br = g.loc[(best_roll, "rolling 1-month-ahead")]; rc = g.loc[("hazard: logit + rolling 3-month recalibration", "rolling 1-month-ahead")]
figstyle.note(fig, f"Out-of-time MAE (12 months, 2024-07..2025-06): logit hazard {m1.MAE_pp:.2f} pp (r = {m1["corr"]:.2f}); best fixed "
                   f"time-series forecast ({best_static}) {bs.MAE_pp:.2f} pp; best rolling one-month-ahead ({best_roll}) {br.MAE_pp:.2f} pp, "
                   f"which sees each month's actual before forecasting the next; logit rescaled by its trailing 3-month actual/predicted "
                   f"ratio {rc.MAE_pp:.2f} pp (r = {rc["corr"]:.2f}). 26 training months only.")
fig.tight_layout(); fig.savefig(OUT / "fig8_ts_benchmark.png"); print("figure saved")
json.dump({"train_months": n_tr, "oot_months": 12, "best_static_ts": best_static, "best_rolling": best_roll},
          open(OUT / "ts_benchmark_meta.json", "w"), indent=2)
