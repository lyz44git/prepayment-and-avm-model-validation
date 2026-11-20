"""Credit-loss study check (2026-11-19): inference that allows correlation across months, not only within them.

51 clusters by calendar month, which allows any correlation within a month but none between months; loans run for
years and the macro terms move slowly, so neighbouring months are correlated. Here the same fits (charge-off M1 and
M1v, rows 2006-01..2019-12, hold-out states excluded, lag L) are given Newey-West (Bartlett) standard errors on the
monthly sums of the score, with bandwidths of 12 and 24 months, and the drop in the unemployment coefficient when
approval-era effects are added gets the same HAC standard error and a moving-block bootstrap of the monthly influence
sums (blocks of 12 consecutive months, 2,000 draws; linearised, so no refit per draw).
Writes reports/p3/r6_inference_time_blocks.csv and reports/p3/r6_dur4_difference_time_blocks.json.
"""
import sys, json, time; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from _paths import WORK, REPORTS
from mval.p3 import VAL_END, W1, LOG_RATE, SAMPLE_RATE, CreditDesign, macro_quarterly, macro_monthly

R = REPORTS / "p3"; OUT = WORK / "p3"; FED = Path("data_external/fed2026")
L = json.loads((R / "champion.json").read_text())["lag_months"]
T0 = time.time(); log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
mm = macro_monthly(macro_quarterly(FED, "baseline"), W1)
cc = pd.read_parquet(OUT / "cc.parquet"); cc["franchise"] = cc.franchise.astype(bool)
p = cc[~cc.holdout & ((cc.y_co == 1) | cc.keep)].copy()
p["dur4"] = mm.dur4.reindex(p.m - L).to_numpy(); p["bbb_spread"] = mm.bbb_spread.reindex(p.m - L).to_numpy()
p["y"] = p.y_co.astype(int)
fit = p[p.m <= VAL_END].reset_index(drop=True)
y = fit.y.to_numpy().astype(float); month = fit.m.to_numpy()
months = np.unique(month); mpos = np.searchsorted(months, month)


def fit_scores(d):
    X = d.transform(fit); X.insert(0, "const", 1.0); Xn = X.to_numpy(np.float64)
    r = sm.Logit(y, Xn).fit(method="newton", maxiter=80, disp=0)
    pr = 1 / (1 + np.exp(-(Xn @ r.params)))
    S = np.zeros((len(months), Xn.shape[1]))
    np.add.at(S, mpos, Xn * (y - pr)[:, None])          # monthly sums of the score
    Hinv = r.normalized_cov_params                        # inverse of the negative Hessian
    return list(X.columns), r.params, Hinv, S


def hac(S, bw):
    """Bartlett-kernel long-run covariance of the monthly sums (rows in calendar order)."""
    O = S.T @ S
    for l in range(1, bw + 1):
        G = S[l:].T @ S[:-l]; O += (1 - l / (bw + 1)) * (G + G.T)
    return O


rows, infl = [], {}
d1 = CreditDesign("co", L).fit(fit); d1v = CreditDesign("co", L, eras=True).fit(fit)
for name, d in (("M1", d1), ("M1v", d1v)):
    cols, b, Hinv, S = fit_scores(d)
    se_model = np.sqrt(np.diag(Hinv))
    se_month = np.sqrt(np.diag(Hinv @ (S.T @ S) @ Hinv))
    se_h12 = np.sqrt(np.diag(Hinv @ hac(S, 12) @ Hinv)); se_h24 = np.sqrt(np.diag(Hinv @ hac(S, 24) @ Hinv))
    for j, c in enumerate(cols):
        if c in ("dur4", "bbb_spread") or c.startswith("era"):
            rows.append({"model": name, "term": c, "beta": float(b[j] + (LOG_RATE if c == "const" else 0)),
                         "se_model": float(se_model[j]), "se_month_cluster": float(se_month[j]),
                         "se_hac_12m": float(se_h12[j]), "se_hac_24m": float(se_h24[j]),
                         "z_month_cluster": float(b[j] / se_month[j]), "z_hac_12m": float(b[j] / se_h12[j]),
                         "z_hac_24m": float(b[j] / se_h24[j])})
    k = cols.index("dur4")
    infl[name] = (b[k], (S @ Hinv)[:, k])                 # monthly influence of each month on the dur4 coefficient
    log(name, "done")
tab = pd.DataFrame(rows); tab.to_csv(R / "r6_inference_time_blocks.csv", index=False)

diff = float(infl["M1"][0] - infl["M1v"][0]); g = infl["M1"][1] - infl["M1v"][1]   # influence of each month on the difference
def hac1(v, bw):
    o = float(v @ v)
    for l in range(1, bw + 1): o += 2 * (1 - l / (bw + 1)) * float(v[l:] @ v[:-l])
    return o
rng = np.random.default_rng(0); n = len(g); blk = 12; nb = int(np.ceil(n / blk)); draws = []
for _ in range(2000):
    starts = rng.integers(0, n - blk + 1, nb)
    idx = np.concatenate([np.arange(s, s + blk) for s in starts])[:n]
    draws.append(g[idx].sum())
draws = diff + np.array(draws)                             # linearised: the estimate plus the resampled influence sum
out = {"difference_m1_minus_m1v_dur4": diff, "se_month_cluster": float(np.sqrt(g @ g)), "se_hac_12m": float(np.sqrt(hac1(g, 12))),
       "se_hac_24m": float(np.sqrt(hac1(g, 24))), "block_bootstrap": {"block_months": blk, "draws": 2000,
       "p2_5": float(np.quantile(draws, 0.025)), "p97_5": float(np.quantile(draws, 0.975)), "share_at_or_below_zero": float((draws <= 0).mean())},
       "months": int(n)}
(R / "r6_dur4_difference_time_blocks.json").write_text(json.dumps(out, indent=2))
print(tab.round(4).to_string(index=False)); print(json.dumps(out, indent=1)); log("done")
