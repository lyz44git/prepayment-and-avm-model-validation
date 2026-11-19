"""Time-robust inference for the published M1 logit (round-4 robustness).

The published standard errors (scripts/08) cluster by county. PMMS, its 3-month change and the month-of-year
dummies vary only by calendar month (26 training months, 2022-05..2024-06), and the rate gap varies mostly
over time, so county clustering can understate their uncertainty. This script refits M1 for both causes with the
same design, sample and case-control draws as 08 (all events + 10% of non-events, King-Zeng intercept
correction), then recomputes the sandwich covariance with
  - county clusters (reproduces the published SEs),
  - calendar-month clusters,
  - county x month intersection clusters,
  - two-way county and month clusters (Cameron-Gelbach-Miller: V_county + V_month - V_intersection).
Each one-way piece uses the statsmodels small-sample factor G/(G-1) * (N-1)/(N-K).
With only 26 month clusters and 13 regressors that vary only by month (11 seasonal dummies, PMMS, its 3-month
change), the CR1 month-clustered SE is known to be biased down. The script therefore also reports the
leave-one-month-out jackknife (CV3: (G-1)/G * sum_g (b_(-g) - b)(b_(-g) - b)', MacKinnon-Nielsen-Webb), refitting
the case-control logit 26 times per cause by Newton-Raphson from the full-sample estimate.

Reads data_work/p1/panel.parquet (written by 08). Writes reports/p1/r4_inference_twoway.csv (aggregate only).
"""
import sys, time, functools; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import WORK, REPORTS
from mval import p1
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
print = functools.partial(print, flush=True)
t0 = time.time(); OUT = REPORTS / "p1"; DW = WORK / "p1"
CAUSES = ["sale", "new_lien"]; Q = 0.10
KEY = ["gap_pos", "gap_neg", "pmms", "pmms_chg3", "log_appr", "est_ltv", "log_value"] + [f"moy{m}" for m in range(2, 13)]
LEVEL = {"gap_pos": "cohort x month", "gap_neg": "cohort x month", "pmms": "month only", "pmms_chg3": "month only",
         "log_appr": "state x cohort x month", "est_ltv": "state x cohort x month", "log_value": "property (static)"}
LEVEL.update({f"moy{m}": "month only (seasonality)" for m in range(2, 13)})
LABEL = {"gap_pos": "rate gap above 0 (owner rate below market), per +1 pp", "gap_neg": "rate gap below 0 (owner in the money), per +1 pp",
         "pmms": "PMMS level, per +1 pp", "pmms_chg3": "PMMS 3-month change, per +1 pp", "log_appr": "log appreciation since purchase",
         "est_ltv": "estimated current LTV", "log_value": "log purchase price indexed to 2022-04"}

cols = ["state", "fips5", "t", "split", "y_sale", "y_new_lien", "tenure_m", "gap", "pmms", "pmms_chg3", "log_appr", "est_ltv", "moy"] + p1.STATIC
P = pd.read_parquet(DW / "panel.parquet", columns=cols)
tr = P[P.split == "train"]; del P
print(f"training rows {len(tr):,} ({time.time()-t0:.0f}s)")
design = p1.LogitDesign().fit(tr)
pub = {c: pd.read_csv(OUT / f"m1_coefficients_{c}.csv", index_col=0) for c in CAUSES}

def cluster_cov(score, Hinv, groups):
    """One-way cluster-robust covariance with the statsmodels correction. groups: integer codes."""
    G = int(groups.max()) + 1; n, k = score.shape
    Sg = np.zeros((G, k))
    order = np.argsort(groups, kind="stable"); g_sorted = groups[order]
    starts = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
    Sg[g_sorted[starts]] = np.add.reduceat(score[order], starts, axis=0)
    meat = Sg.T @ Sg
    n_g = len(starts)
    return Hinv @ meat @ Hinv * (n_g / (n_g - 1.0)) * ((n - 1.0) / (n - k)), n_g

def newton_logit(X, y, b0, tol=1e-10, maxit=25):
    b = b0.copy()
    for _ in range(maxit):
        mu = 1 / (1 + np.exp(-(X @ b)))
        step = np.linalg.solve((X * (mu * (1 - mu))[:, None]).T @ X, X.T @ (y - mu))
        b += step
        if np.abs(step).max() < tol:
            break
    return b

rng = np.random.default_rng(0)                         # same generator and draw order as 08 (sale first, then new lien)
draws = {c: rng.random(len(tr)) for c in CAUSES}
rows = []
for c in CAUSES:
    y = tr[f"y_{c}"].to_numpy()
    keep = (y == 1) | (draws[c] < Q)
    sub = tr.loc[tr.index[keep]]
    Xc = sm.add_constant(design.transform(sub).astype(np.float64), has_constant="add")
    county = pd.factorize(sub.fips5.to_numpy())[0]
    month = pd.factorize(sub.t.to_numpy())[0]
    inter = pd.factorize(county.astype(np.int64) * 1000 + month)[0]
    res = sm.GLM(y[keep], Xc, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": county}, maxiter=100)
    print(c, f"fit on {keep.sum():,} case-control rows ({time.time()-t0:.0f}s)")
    X = Xc.to_numpy(); mu = res.fittedvalues if isinstance(res.fittedvalues, np.ndarray) else res.fittedvalues.to_numpy()
    score = X * (y[keep] - mu)[:, None]
    Hinv = np.linalg.inv((X * (mu * (1 - mu))[:, None]).T @ X)
    V = {}; ng = {}
    V["county"], ng["county"] = cluster_cov(score, Hinv, county)
    V["month"], ng["month"] = cluster_cov(score, Hinv, month)
    V["county_x_month"], ng["county_x_month"] = cluster_cov(score, Hinv, inter)
    V["twoway"] = V["county"] + V["month"] - V["county_x_month"]
    names = list(Xc.columns); ix = {n: i for i, n in enumerate(names)}
    # leave-one-month-out jackknife (CV3), centred on the full-sample estimate
    b_full = res.params.to_numpy(); yk = y[keep].astype(np.float64)
    chk_newton = float(np.abs(newton_logit(X, yk, b_full) - b_full).max())
    jk = np.stack([newton_logit(X[month != g], yk[month != g], b_full) for g in range(ng["month"])])
    dev = jk - b_full
    V["month_jackknife_cv3"] = (ng["month"] - 1) / ng["month"] * dev.T @ dev
    print(c, f"jackknife done; Newton vs IRLS max diff {chk_newton:.1e} ({time.time()-t0:.0f}s)")
    se = {k: np.sqrt(np.where(np.diag(v) > 0, np.diag(v), np.nan)) for k, v in V.items()}
    neg2w = [n for n in names if np.diag(V["twoway"])[ix[n]] <= 0]
    # checks against the published fit
    b = res.params.copy(); b["const"] += np.log(Q)
    chk_b = float((b - pub[c].coef.reindex(b.index)).abs().max())
    chk_se_sm = float(np.max(np.abs(se["county"] - res.bse.to_numpy()) / res.bse.to_numpy()))
    chk_se_pub = float(np.max(np.abs(se["county"] - pub[c].se_cluster.reindex(names).to_numpy()) / pub[c].se_cluster.reindex(names).to_numpy()))
    print(c, f"max |coef - published| {chk_b:.2e}; county SE rel. diff vs statsmodels {chk_se_sm:.2e}, vs published {chk_se_pub:.2e}; "
          f"clusters county {ng['county']}, month {ng['month']}, county x month {ng['county_x_month']}; two-way non-positive variances: {neg2w}")
    df_t = min(ng["county"], ng["month"]) - 1                         # conservative t reference for the time-clustered SEs
    for n in KEY:
        if n not in ix:
            continue
        i = ix[n]; bb = float(b[n])
        r = {"cause": c, "term": n, "description": LABEL.get(n, "month-of-year dummy (vs January)"), "varies_at": LEVEL[n],
             "beta": bb, "se_county_published": float(pub[c].se_cluster.get(n, np.nan)), "se_county": se["county"][i],
             "se_month": se["month"][i], "se_county_x_month": se["county_x_month"][i], "se_twoway_county_month": se["twoway"][i],
             "se_month_jackknife_cv3": se["month_jackknife_cv3"][i],
             "ratio_twoway_to_county": se["twoway"][i] / se["county"][i], "ratio_month_to_county": se["month"][i] / se["county"][i],
             "ratio_jackknife_to_county": se["month_jackknife_cv3"][i] / se["county"][i],
             "z_county": bb / se["county"][i], "z_twoway": bb / se["twoway"][i], "z_month": bb / se["month"][i],
             "p_county_normal": float(2 * stats.norm.sf(abs(bb / se["county"][i]))),
             "p_twoway_normal": float(2 * stats.norm.sf(abs(bb / se["twoway"][i]))),
             "p_twoway_t": float(2 * stats.t.sf(abs(bb / se["twoway"][i]), df_t)),
             "p_month_normal": float(2 * stats.norm.sf(abs(bb / se["month"][i]))),
             "p_month_t": float(2 * stats.t.sf(abs(bb / se["month"][i]), df_t)),
             "p_month_jackknife_t": float(2 * stats.t.sf(abs(bb / se["month_jackknife_cv3"][i]), df_t))}
        if n in ["gap_pos", "gap_neg", "pmms", "pmms_chg3"]:          # odds ratio per +1 pp and 95% CIs under each SE
            z = stats.norm.ppf(0.975); tq = stats.t.ppf(0.975, df_t)
            r["odds_ratio"] = float(np.exp(bb))
            for lab, s_, q_ in [("county", se["county"][i], z), ("twoway", se["twoway"][i], z), ("twoway_t", se["twoway"][i], tq),
                                ("month", se["month"][i], z), ("month_t", se["month"][i], tq),
                                ("month_jackknife_t", se["month_jackknife_cv3"][i], tq)]:
                r[f"or_ci95_lo_{lab}"] = float(np.exp(bb - q_ * s_)); r[f"or_ci95_hi_{lab}"] = float(np.exp(bb + q_ * s_))
        r.update({"n_case_control_rows": int(keep.sum()), "n_events": int(y.sum()), "clusters_county": ng["county"],
                  "clusters_month": ng["month"], "clusters_county_x_month": ng["county_x_month"], "t_df_for_time_clusters": df_t,
                  "check_max_abs_coef_diff_vs_published": chk_b, "check_max_rel_county_se_diff_vs_published": chk_se_pub})
        rows.append(r)
out = pd.DataFrame(rows)
out.to_csv(OUT / "r4_inference_twoway.csv", index=False)
pd.set_option("display.width", 250)
print(out[["cause", "term", "beta", "se_county", "se_twoway_county_month", "se_month", "se_month_jackknife_cv3", "ratio_twoway_to_county",
           "ratio_jackknife_to_county", "p_twoway_t", "p_month_jackknife_t"]].round(4).to_string(index=False))
g = out[(out.cause == "new_lien") & (out.term == "gap_pos")].iloc[0]
print("new-lien gap_pos OR", round(g.odds_ratio, 4), {k: (round(g[f"or_ci95_lo_{k}"], 4), round(g[f"or_ci95_hi_{k}"], 4))
                                                         for k in ["county", "twoway", "twoway_t", "month", "month_t", "month_jackknife_t"]})
print(f"total {time.time()-t0:.0f}s")
