"""Credit-loss study: inference that respects the time structure (calendar-month clustering).

Fit sample as scripts/42c_p3_remediation.py: charge-off case-control rows 2006-01..2019-12, hold-out states excluded,
macro without the relief overlay (identical before 2020 anyway), lag L. M1 and M1v (approval eras) charge-off logits
are refitted with statsmodels Logit, cov_type='cluster', groups = calendar month index (168 clusters; statsmodels'
default small-sample correction). Model-based s.e. = sqrt(diag(inverse negative Hessian)), as in the existing
coefficient files. R2b (M1v + downturn exposure dx) is fitted as in scripts/45_p3_remediation2.py on data to 2019-12
(overlay O1 on, which changes nothing before 2020) and reported the same way.
Month-cluster bootstrap for the difference between M1's and M1v's dur4 coefficients: calendar months 2006-01..2019-12
are drawn with replacement (168 per replicate); each replicate refits both logits by Newton-Raphson on the rows of
the drawn months with frequency weights (= number of times the month was drawn), which is the same likelihood as
stacking the duplicated months. The design (columns) is held at the full-sample design; starting values are the
full-sample estimates. B = 100 requested, stopped early if the bootstrap passes 90 minutes of wall time. A
month-clustered sandwich s.e. of the difference (scores of both models stacked by month) is reported beside it.
Writes reports/p3/r4_inference_month_clustered.csv and r4_dur4_difference_bootstrap.json.
"""
import sys, json, time; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import multiprocessing as mp
import numpy as np, pandas as pd
import scipy.sparse as sp
import statsmodels.api as sm
from _paths import WORK, REPORTS
from mval.p3 import VAL_END, W1, LOG_RATE, SAMPLE_RATE, CreditDesign, macro_quarterly, macro_monthly, downturn_exposure

R = REPORTS / "p3"; OUT = WORK / "p3"; FED = Path("data_external/fed2026")
L = json.loads((R / "champion.json").read_text())["lag_months"]; WIN = 36
B_REQ, BUDGET_S, NPROC = 100, 90 * 60, 8
T0 = time.time(); log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
mm = macro_monthly(macro_quarterly(FED, "baseline"), W1)
cc = pd.read_parquet(OUT / "cc.parquet"); cc["franchise"] = cc.franchise.astype(bool)
p = cc[~cc.holdout & ((cc.y_co == 1) | cc.keep)].copy()
p["dur4"] = mm.dur4.reindex(p.m - L).to_numpy(); p["bbb_spread"] = mm.bbb_spread.reindex(p.m - L).to_numpy()
p["y"] = p.y_co.astype(int); p["w"] = np.where(p.y == 1, 1.0, 1 / SAMPLE_RATE)
fit = p[p.m <= VAL_END]
y = fit.y.to_numpy().astype(float); groups = fit.m.to_numpy()
log("fit rows", len(fit), "charge-offs", int(y.sum()), "months", len(np.unique(groups)))

def logit_cluster(d, frame, yv, g):
    X = d.transform(frame); X.insert(0, "const", 1.0)
    Xn = X.to_numpy(np.float64)
    r = sm.Logit(yv, Xn).fit(method="newton", maxiter=80, disp=0, cov_type="cluster", cov_kwds={"groups": g})
    cols = list(X.columns)
    out = pd.DataFrame({"beta": r.params, "se_model": np.sqrt(np.diag(r.normalized_cov_params)), "se_month_cluster": r.bse}, index=cols)
    out.loc["const", "beta"] += LOG_RATE
    return out, r, Xn, cols

d1 = CreditDesign("co", L).fit(fit); d1v = CreditDesign("co", L, eras=True).fit(fit)
c1, r1, X1, cols1 = logit_cluster(d1, fit, y, groups); log("M1 done")
c1v, r1v, X1v, cols1v = logit_cluster(d1v, fit, y, groups); log("M1v done")

# R2b as in 45 (overlay O1 on; rows 2006-01..2019-12, hold-out excluded)
mmo = macro_monthly(macro_quarterly(FED, "baseline", relief_overlay=True), W1 + 400)
fb = cc[(cc.m <= VAL_END) & ~cc.holdout & ((cc.y_co == 1) | cc.keep)].copy()
fb["y"] = fb.y_co.astype(int); fb["w"] = np.where(fb.y == 1, 1.0, 1 / SAMPLE_RATE)
fb["dur4"], fb["bbb_spread"] = mmo.dur4.reindex(fb.m - L).to_numpy(), mmo.bbb_spread.reindex(fb.m - L).to_numpy()
fb["prime"], fb["dprime4"], fb["dur4_pif"] = mmo.prime.reindex(fb.m).to_numpy(), mmo.dprime4.reindex(fb.m).to_numpy(), mmo.dur4.reindex(fb.m).to_numpy()
fb["dx"] = downturn_exposure(mmo.dur4, fb.m.to_numpy(), fb.disb_m.to_numpy(), lag=L, window=WIN)
db = CreditDesign("co", L, eras=True, downturn=True).fit(fb)
cb, rb, _, _ = logit_cluster(db, fb, fb.y.to_numpy().astype(float), fb.m.to_numpy()); del _; log("R2b done")

KEEP = ["dur4", "bbb_spread", "dx", "era1", "era2", "era3", "era4"]
ref = {"M1": pd.read_csv(R / "m1v_co_coefficients.csv", index_col=0)[["beta_m1", "se_m1"]].set_axis(["beta", "se"], axis=1),
       "M1v": pd.read_csv(R / "m1v_co_coefficients.csv", index_col=0)[["beta_m1v", "se_m1v"]].set_axis(["beta", "se"], axis=1),
       "R2b": pd.read_csv(R / "r2b_co_coefficients.csv", index_col=0)}
tab = []
for name, c in [("M1", c1), ("M1v", c1v), ("R2b", cb)]:
    for k in KEEP:
        if k in c.index:
            r_ = c.loc[k]
            tab.append({"model": name, "fit": "2006-01..2019-12", "term": k, "beta": r_.beta, "se_model": r_.se_model,
                        "se_month_cluster": r_.se_month_cluster, "cluster_over_model_se": r_.se_month_cluster / r_.se_model,
                        "z_model": r_.beta / r_.se_model, "z_month_cluster": r_.beta / r_.se_month_cluster,
                        "beta_in_existing_file": float(ref[name].loc[k, "beta"]) if k in ref[name].index else np.nan,
                        "se_in_existing_file": float(ref[name].loc[k, "se"]) if k in ref[name].index else np.nan})
tab = pd.DataFrame(tab); tab.to_csv(R / "r4_inference_month_clustered.csv", index=False)
log(tab.round(4).to_string(index=False))

# ---------------------------------------------------------------- month-clustered sandwich s.e. of the dur4 difference
months = np.unique(groups); code = np.searchsorted(months, groups); G = len(months); N = len(y)
Mi = sp.csr_matrix((np.ones(N), (code, np.arange(N))), shape=(G, N))
def score_sums(r, X):
    pr = 1 / (1 + np.exp(-(X @ r.params)))
    return Mi @ (X * (y - pr)[:, None])
S1, S1v = score_sums(r1, X1), score_sums(r1v, X1v)
corr = G / (G - 1)                                    # statsmodels' cluster correction G/(G-1) x (N-1)/(N-K), per model
k1, k1v = X1.shape[1], X1v.shape[1]
A1 = r1.normalized_cov_params @ S1.T; A1v = r1v.normalized_cov_params @ S1v.T   # K x G influence by month
j1, j1v = cols1.index("dur4"), cols1v.index("dur4")
v11 = corr * (N - 1) / (N - k1) * (A1[j1] @ A1[j1]); v22 = corr * (N - 1) / (N - k1v) * (A1v[j1v] @ A1v[j1v])
v12 = corr * np.sqrt((N - 1) / (N - k1) * (N - 1) / (N - k1v)) * (A1[j1] @ A1v[j1v])
se_diff_sandwich = float(np.sqrt(v11 + v22 - 2 * v12))
log("sandwich check: M1 dur4 se", np.sqrt(v11), "vs statsmodels", c1.loc["dur4", "se_month_cluster"],
    "| M1v", np.sqrt(v22), "vs", c1v.loc["dur4", "se_month_cluster"], "| corr", v12 / np.sqrt(v11 * v22))

# ---------------------------------------------------------------- month-cluster bootstrap
B0_1, B0_1v = np.asarray(r1.params), np.asarray(r1v.params)
def newton_w(X, yv, f, b0, tol=1e-9, maxit=30):
    b = b0.copy()
    for it in range(maxit):
        pr = 1 / (1 + np.exp(-(X @ b)))
        g = X.T @ (f * (yv - pr))
        Hm = (X * (f * pr * (1 - pr))[:, None]).T @ X
        step = np.linalg.solve(Hm, g); b += step
        if np.max(np.abs(step)) < tol:
            return b, it + 1, True
    return b, maxit, False

def replicate(b):
    rng = np.random.default_rng(20261118 + b)
    draw = rng.integers(0, G, G)
    cnt = np.bincount(draw, minlength=G).astype(float)
    f = cnt[code]; sel = f > 0
    b1, it1, ok1 = newton_w(X1[sel], y[sel], f[sel], B0_1)
    b1v, it1v, ok1v = newton_w(X1v[sel], y[sel], f[sel], B0_1v)
    return {"b": b, "dur4_m1": float(b1[j1]), "dur4_m1v": float(b1v[j1v]), "diff": float(b1[j1] - b1v[j1v]),
            "iters": (it1, it1v), "converged": bool(ok1 and ok1v), "months_distinct": int((cnt > 0).sum())}

def _init():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)

del cc, p
tb = time.time(); reps = []
with mp.get_context("fork").Pool(NPROC, initializer=_init) as pool:
    for r_ in pool.imap_unordered(replicate, range(B_REQ)):
        reps.append(r_)
        if len(reps) % 10 == 0:
            log("bootstrap", len(reps), "replicates", f"{time.time() - tb:.0f}s")
        if time.time() - tb > BUDGET_S:
            log("time budget reached"); pool.terminate(); break
boot_s = time.time() - tb
rp = pd.DataFrame(reps).sort_values("b")
ok = rp[rp.converged]
point = float(c1.loc["dur4", "beta"] - c1v.loc["dur4", "beta"])
res = {"B_requested": B_REQ, "B_completed": int(len(rp)), "B_converged_used": int(len(ok)), "bootstrap_wall_seconds": round(boot_s, 1),
       "processes": NPROC, "clusters": "calendar months 2006-01..2019-12", "G": int(G), "fit_rows": int(N), "charge_offs": int(y.sum()),
       "point_dur4_M1": float(c1.loc["dur4", "beta"]), "point_dur4_M1v": float(c1v.loc["dur4", "beta"]), "point_difference": point,
       "bootstrap_difference_p2_5": float(np.percentile(ok["diff"], 2.5)), "bootstrap_difference_p97_5": float(np.percentile(ok["diff"], 97.5)),
       "bootstrap_difference_mean": float(ok["diff"].mean()), "bootstrap_difference_sd": float(ok["diff"].std(ddof=1)),
       "share_of_replicates_with_difference_le_0": float((ok["diff"] <= 0).mean()),
       "bootstrap_dur4_M1_p2_5_p97_5": [float(np.percentile(ok.dur4_m1, 2.5)), float(np.percentile(ok.dur4_m1, 97.5))],
       "bootstrap_dur4_M1v_p2_5_p97_5": [float(np.percentile(ok.dur4_m1v, 2.5)), float(np.percentile(ok.dur4_m1v, 97.5))],
       "bootstrap_sd_dur4_M1": float(ok.dur4_m1.std(ddof=1)), "bootstrap_sd_dur4_M1v": float(ok.dur4_m1v.std(ddof=1)),
       "sandwich_month_cluster_se_difference": se_diff_sandwich,
       "sandwich_month_cluster_corr_dur4_M1_M1v": float(v12 / np.sqrt(v11 * v22)),
       "model_based_se_dur4_M1": float(c1.loc["dur4", "se_model"]), "model_based_se_dur4_M1v": float(c1v.loc["dur4", "se_model"]),
       "month_cluster_se_dur4_M1": float(c1.loc["dur4", "se_month_cluster"]), "month_cluster_se_dur4_M1v": float(c1v.loc["dur4", "se_month_cluster"]),
       "newton_iterations_max": int(max(max(i) for i in rp.iters)), "seeds": "numpy default_rng(20261118 + b), b = 0..B-1",
       "design": "columns fixed at the full-sample design; frequency-weighted Newton from full-sample estimates"}
(R / "r4_dur4_difference_bootstrap.json").write_text(json.dumps(res, indent=2)); log(json.dumps(res, indent=1))
log("done")
