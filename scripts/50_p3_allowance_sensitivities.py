"""Credit-loss study: sensitivities of the 2026-09 lifetime allowance (relief overlay O1 on), around scripts/46_p3_cecl_r2.py.

Same book (loans active at 2026-09), final fits (2006-01..2026-09, hold-out states excluded, O1 in the history),
models (M1v, R2a = M1v x trailing 24-month factor, R2b = M1v + downturn exposure, R2c = R2b x its factor), Fed 2026
baseline/severely adverse scenarios, horizon (remaining recorded term, at least 12 months) and severity model as 46.
What differs, by variant (each changes one thing relative to "base", which reproduces 46):
  i    severe run with the payoff hazard's macro inputs (prime, dprime4, dur4 at lag 0) on the BASELINE path; the
       charge-off hazard (dur4, bbb_spread at lag L, downturn exposure) and the severity model (dur4, bbb_spread at
       lag L) follow the severe path. Increment severe - baseline = charge-off channel (this run - baseline) + payoff
       channel (severe - this run).
  ii   payoff hazard divided by its out-of-time predicted / actual (M1, fit 2006-19, scored 2020-01..2026-09, from
       eval_hazard.csv = 1.1173), applied as a MULTIPLICATIVE HAZARD SCALE (h_pif / ratio), not an odds shift.
  iib  as ii but with the out-of-time ratio excluding 2026-09 (zero payoffs recorded that month; see r4_recording_lag.json).
  iii  macro reversion over 8 quarters instead of 4 after the last scenario quarter (macro_monthly revert_quarters=8).
  iv   R2a / R2c factor applied in full through the end of the scenario (2029-03), then reverting linearly to 1 over
       4 quarters (reaches 1 at 2030-03); computed from the base monthly loss paths (no new projection).
  v    M1v with the FY2020+ approval-era coefficient (era4) set equal to the FY2014-19 coefficient (era3). R2a (v)
       uses the factor recomputed for the modified M1v (labelled so).
Also writes (vi) r4_recording_lag.json (payoff out-of-time predicted / actual excluding 2026-09; typical monthly
payoffs as a share of the active book) and r4_final_fit_coefficients.csv (2026 final-fit M1v and R2b charge-off
coefficients with model-based s.e.; const includes the log(0.05) case-control correction, as elsewhere).
Projections run in parallel worker processes (fork); results are identical to running them one by one.
Writes reports/p3/r4_allowance_sensitivities.csv, r4_recording_lag.json, r4_final_fit_coefficients.csv.
"""
import sys, json, time, gc; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import multiprocessing as mp
import numpy as np, pandas as pd
import statsmodels.api as sm
from _paths import WORK, REPORTS
from mval.p3 import (W1, LOG_RATE, SAMPLE_RATE, HOLDOUT_STATES, mi, CreditDesign, macro_quarterly, macro_monthly,
                     downturn_exposure, ns_basis, AGE_KNOTS, SECTORS, balance_share, month_label, era_of)

R = REPORTS / "p3"; OUT = WORK / "p3"; FED = Path("data_external/fed2026")
L = json.loads((R / "champion.json").read_text())["lag_months"]; WIN = 36
SCEN_END = mi(2029, 3)                        # last month of the last scenario quarter (2029Q1)
NPROC = 10
T0 = time.time(); log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
mm_hist = macro_monthly(macro_quarterly(FED, "baseline", relief_overlay=True), W1)
cc = pd.read_parquet(OUT / "cc.parquet"); cc["franchise"] = cc.franchise.astype(bool)

def add_macro(p, mm):
    p["dur4"] = mm.dur4.reindex(p.m - L).to_numpy(); p["bbb_spread"] = mm.bbb_spread.reindex(p.m - L).to_numpy()
    p["prime"] = mm.prime.reindex(p.m).to_numpy(); p["dprime4"] = mm.dprime4.reindex(p.m).to_numpy(); p["dur4_pif"] = mm.dur4.reindex(p.m).to_numpy()
    p["dx"] = downturn_exposure(mm.dur4, p.m.to_numpy(), p.disb_m.to_numpy(), lag=L, window=WIN)
    return p

def train(cause):
    y = f"y_{cause}"
    p = cc[~cc.holdout & ((cc[y] == 1) | cc.keep)].copy()
    p["y"] = p[y].astype(int); p["w"] = np.where(p.y == 1, 1.0, 1 / SAMPLE_RATE)
    return add_macro(p, mm_hist)

def fit(p, cause, **kw):
    d = CreditDesign(cause, L if cause == "co" else 0, **kw).fit(p)
    X = d.transform(p if cause == "co" else p.assign(dur4=p.dur4_pif)); X.insert(0, "const", 1.0)
    r = sm.Logit(p.y.to_numpy(), X.to_numpy(np.float64)).fit(method="newton", maxiter=80, disp=0)
    b = pd.Series(r.params, index=X.columns); b["const"] += LOG_RATE
    return {"beta": b, "se": pd.Series(r.bse, index=X.columns), "design": d, "shares": p.state.value_counts(normalize=True)}

def pred(m, p, cause):
    q = p if cause == "co" else p.assign(dur4=p.dur4_pif)
    X = m["design"].transform(q)
    st = [c for c in m["design"].columns if c.startswith("st_")]
    ho = ~q.state.isin(m["design"].states).to_numpy()
    if ho.any():
        X.loc[ho, st] = np.array([m["shares"].get(c[3:], 0.0) for c in st], np.float32)
    X.insert(0, "const", 1.0)
    return 1 / (1 + np.exp(-(X.to_numpy(np.float64) @ m["beta"].reindex(X.columns).to_numpy())))

pco, ppf = train("co"), train("pif")
mods = {"pif": fit(ppf, "pif"), "M1v": fit(pco, "co", eras=True), "R2b": fit(pco, "co", eras=True, downturn=True)}
dv = mods["M1v"]["design"]
has_era4 = "era4" in dv.columns
mods["M1v_e4"] = {**mods["M1v"], "beta": mods["M1v"]["beta"].copy()}
if has_era4:
    mods["M1v_e4"]["beta"]["era4"] = mods["M1v"]["beta"]["era3"]
tr = pco[pco.m > W1 - 24]
fac = {k: float(np.clip(tr.y.sum() / (tr.w * pred(mods[k], tr, "co")).sum(), 0.25, 4)) for k in ("M1v", "R2b", "M1v_e4")}
era_events = pd.Series(era_of(pco.approval_fy[pco.y == 1])).value_counts().sort_index()
log("fits done; max_era", dv.max_era, "era4 in design:", has_era4, "; trailing factors", fac, "; charge-offs by era in fit", era_events.to_dict())
coef = []
for k in ("M1v", "R2b"):
    for name in mods[k]["beta"].index:
        coef.append({"model": k, "term": name, "beta": float(mods[k]["beta"][name]), "se_model": float(mods[k]["se"][name])})
pd.DataFrame(coef).to_csv(R / "r4_final_fit_coefficients.csv", index=False)
del cc, pco, ppf, tr; gc.collect()

loans = pd.read_parquet(OUT / "loans.parquet"); loans["franchise"] = loans.franchise.astype(bool)
co = loans[loans.in_window & (loans.status == "co") & ~loans.state.isin(HOLDOUT_STATES)].copy()
co["age"] = co.co_m - co.disb_m; co["m"] = co.co_m; co = add_macro(co, mm_hist)
def sev_X(p):
    X = pd.DataFrame(index=p.index)
    B = ns_basis(np.clip(p.age.to_numpy(float), 1, 400), AGE_KNOTS)
    for j in range(B.shape[1]): X[f"age_ns{j}"] = B[:, j]
    X["log_amount"] = np.log(p.gross_approval.clip(1000, 1e8)); X["gty_share"] = p.gty_share
    for g in ["express", "preferred", "lowdoc", "other"]: X[f"proc_{g}"] = (p.proc == g).astype(float)
    for g in ["startup", "new", "unknown"]: X[f"bage_{g}"] = (p.bage == g).astype(float)
    for s in SECTORS + ["na"]: X[f"naics_{s}"] = (p.naics2 == s).astype(float)
    X["dur4"], X["bbb_spread"] = p.dur4, p.bbb_spread
    X.insert(0, "const", 1.0); return X
sev = sm.GLM((co.co_amount / co.gross_approval).clip(0, 1), sev_X(co), family=sm.families.Binomial()).fit()

book = loans[loans.in_window & (loans.status == "active")].copy()
book["age0"] = W1 - book.disb_m; book["rem"] = np.maximum(book.term - book.age0, 12)
book["balance"] = book.gross_approval * balance_share(book.rate.fillna(book.rate.median()), book.term, book.age0)
del loans, co; gc.collect()
NSTEP = int(book.rem.max()) + 1
BAL = book.balance.sum(); GTY = book.gty_share.to_numpy()

# ---------------------------------------------------------------- (vi) recording lag (from existing outputs, no refit)
mh = pd.read_csv(R / "monthly_hazard_pred_vs_actual.csv")
pf = mh[(mh.cause == "pif") & (mh.model == "M1 logit") & (mh.m > mi(2019, 12))]
ratio_all = float(pf.pred.sum() / pf.act.sum()); pfx = pf[pf.m < W1]; ratio_x = float(pfx.pred.sum() / pfx.act.sum())
ev = pd.read_csv(R / "eval_hazard.csv")
ratio_eval = float(ev[(ev.cause == "pif") & (ev.model == "M1 logit") & (ev.split == "out of time 2020-26")].pred_over_actual.iloc[0])
ma = pd.read_csv(R / "monthly_actuals.csv")
g_all = ma.groupby("m")[["at_risk", "n_co", "n_pif"]].sum(); g_nho = ma[~ma.holdout].set_index("m")[["at_risk", "n_co", "n_pif"]]
last12 = list(range(W1 - 12, W1))
rec = {"payoff_oot_pred_over_actual_M1": {"2020-01..2026-09 (eval_hazard.csv)": ratio_eval,
                                          "2020-01..2026-09 (recomputed from monthly_hazard_pred_vs_actual.csv)": ratio_all,
                                          "2020-01..2026-08 (excluding 2026-09)": ratio_x},
       "2026-09": {"recorded_payoffs_all_states": int(g_all.loc[W1, "n_pif"]), "recorded_chargeoffs_all_states": int(g_all.loc[W1, "n_co"]),
                   "at_risk_all_states": int(g_all.loc[W1, "at_risk"]),
                   "M1_predicted_payoffs_non_holdout": float(pf[pf.m == W1].pred.sum()), "recorded_payoffs_non_holdout": int(g_nho.loc[W1, "n_pif"])},
       "typical_month_2025-09..2026-08": {"mean_payoffs_all_states": float(g_all.loc[last12, "n_pif"].mean()),
                                          "min_payoffs_all_states": int(g_all.loc[last12, "n_pif"].min()),
                                          "max_payoffs_all_states": int(g_all.loc[last12, "n_pif"].max()),
                                          "mean_at_risk_all_states": float(g_all.loc[last12, "at_risk"].mean()),
                                          "mean_monthly_payoff_rate_all_states": float((g_all.loc[last12, "n_pif"] / g_all.loc[last12, "at_risk"]).mean()),
                                          "mean_payoffs_as_share_of_2026-09_active_book": float(g_all.loc[last12, "n_pif"].mean() / len(book)),
                                          "active_book_loans_2026-09": int(len(book))},
       "months_with_zero_recorded_payoffs_2006-01..2026-09": [month_label(int(m)) for m in g_all.index[g_all.n_pif == 0]]}
(R / "r4_recording_lag.json").write_text(json.dumps(rec, indent=2)); log(json.dumps(rec, indent=1))

# ---------------------------------------------------------------- projections
MMS = {}
def mmf(scen, rq):
    if (scen, rq) not in MMS:
        MMS[(scen, rq)] = macro_monthly(macro_quarterly(FED, scen, relief_overlay=True), W1 + 480, revert_quarters=rq)
    return MMS[(scen, rq)]
for s_ in ("baseline", "severe"):
    for rq in (4, 8):
        mmf(s_, rq)

def run(task, chunk=25_000):
    key, scen, pif_scen, pif_scale, rq = task["key"], task["scen"], task["pif_scen"], task["pif_scale"], task["rq"]
    mm, mmp = mmf(scen, rq), mmf(pif_scen, rq)
    tot = np.zeros(len(book)); path = np.zeros(NSTEP); path_l = np.zeros(NSTEP)
    for s in range(0, len(book), chunk):
        b = book.iloc[s:s + chunk]; n = b.rem.to_numpy().astype(int)
        idx = np.repeat(np.arange(len(b)), n); step = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        p = b.iloc[idx].reset_index(drop=True)
        p["m"] = W1 + 1 + step; p["age"] = p.age0 + 1 + step; p["moy"] = (p.m % 12) + 1
        p = add_macro(p, mm)
        if pif_scen != scen:                       # variant i: payoff macro inputs from another path
            p["prime"] = mmp.prime.reindex(p.m).to_numpy(); p["dprime4"] = mmp.dprime4.reindex(p.m).to_numpy()
            p["dur4_pif"] = mmp.dur4.reindex(p.m).to_numpy()
        hc, hp = pred(mods[key], p, "co"), pred(mods["pif"], p, "pif") * pif_scale
        sv = np.clip(sev.predict(sev_X(p)), 0, 1)
        q = np.clip(1 - hc - hp, 0, 1)
        sp = pd.DataFrame({"i": idx, "q": q}).groupby("i").q.cumprod().to_numpy() / np.where(q > 0, q, 1)
        el = sp * hc * sv * p.gross_approval.to_numpy()
        tot[s:s + len(b)] = np.bincount(idx, weights=el, minlength=len(b))
        path += np.bincount(step, weights=el, minlength=NSTEP)
        path_l += np.bincount(step, weights=el * (1 - p.gty_share.to_numpy()), minlength=NSTEP)
    return task["label"], tot, path, path_l

S_X = 1.0 / ratio_eval; S_XB = 1.0 / ratio_x
tasks = []
for scen in ("baseline", "severe"):
    for key in ("M1v", "R2b"):
        tasks.append({"label": ("base", key, scen), "key": key, "scen": scen, "pif_scen": scen, "pif_scale": 1.0, "rq": 4})
        tasks.append({"label": ("ii", key, scen), "key": key, "scen": scen, "pif_scen": scen, "pif_scale": S_X, "rq": 4})
        tasks.append({"label": ("iib", key, scen), "key": key, "scen": scen, "pif_scen": scen, "pif_scale": S_XB, "rq": 4})
        tasks.append({"label": ("iii", key, scen), "key": key, "scen": scen, "pif_scen": scen, "pif_scale": 1.0, "rq": 8})
    if has_era4:
        tasks.append({"label": ("v", "M1v_e4", scen), "key": "M1v_e4", "scen": scen, "pif_scen": scen, "pif_scale": 1.0, "rq": 4})
for key in ("M1v", "R2b"):
    tasks.append({"label": ("i", key, "severe"), "key": key, "scen": "severe", "pif_scen": "baseline", "pif_scale": 1.0, "rq": 4})
log(len(tasks), "projections on", NPROC, "processes")
gc.collect()
def _init():
    from threadpoolctl import threadpool_limits
    threadpool_limits(3)
with mp.get_context("fork").Pool(NPROC, initializer=_init) as pool:
    results = {}
    for lab, tot, path, path_l in pool.imap_unordered(run, tasks):
        results[lab] = (tot, path, path_l); log("done", lab, f"{tot.sum() / BAL:.4%}")

# ---------------------------------------------------------------- assemble
rows = []
def add(variant, model, scen, loss, loss_l, factor=np.nan, note="", **extra):
    rows.append({"variant": variant, "model": model, "scenario": scen, "lifetime_loss": loss, "lifetime_loss_lender": loss_l,
                 "est_balance": BAL, "loss_over_balance": loss / BAL, "factor": factor, "note": note, **extra})
def tl(lab):
    tot, path, path_l = results[lab]; return tot.sum(), (tot * (1 - GTY)).sum(), path, path_l

notes = {"base": "reproduces scripts/46 (r2_cecl_allowance.csv)",
         "ii": f"payoff hazard x 1/{ratio_eval:.4f} (multiplicative hazard scale)",
         "iib": f"payoff hazard x 1/{ratio_x:.4f} (multiplicative hazard scale; OOT ratio excluding 2026-09)",
         "iii": "macro reversion over 8 quarters (4 in base)"}
for variant in ("base", "ii", "iib", "iii"):
    for scen in ("baseline", "severe"):
        lv, lvl, _, _ = tl((variant, "M1v", scen)); lb, lbl, _, _ = tl((variant, "R2b", scen))
        add(variant, "M1v", scen, lv, lvl, note=notes[variant])
        add(variant, "R2a", scen, lv * fac["M1v"], lvl * fac["M1v"], fac["M1v"], notes[variant])
        add(variant, "R2b", scen, lb, lbl, note=notes[variant])
        if variant == "base":
            add(variant, "R2c", scen, lb * fac["R2b"], lbl * fac["R2b"], fac["R2b"], notes[variant])

# (i) decomposition
for key, mname, f in [("M1v", "M1v", 1.0), ("M1v", "R2a", fac["M1v"]), ("R2b", "R2b", 1.0)]:
    b0 = tl(("base", key, "baseline")); s1 = tl(("base", key, "severe")); c1 = tl(("i", key, "severe"))
    fv = f if mname == "R2a" else np.nan
    add("i", mname, "severe, payoff macro at baseline (charge-off channel only)", c1[0] * f, c1[1] * f, fv,
        "charge-off hazard and severity on severe path; payoff hazard on baseline path")
    tot_inc = (s1[0] - b0[0]) * f; co_inc = (c1[0] - b0[0]) * f; pf_inc = (s1[0] - c1[0]) * f
    tot_l = (s1[1] - b0[1]) * f; co_l = (c1[1] - b0[1]) * f; pf_l = (s1[1] - c1[1]) * f
    add("i", mname, "increment: severe - baseline (total)", tot_inc, tot_l, fv, "loss_over_balance column = increment / balance")
    add("i", mname, "increment: charge-off channel", co_inc, co_l, fv, "(severe co, baseline payoff) - baseline")
    add("i", mname, "increment: payoff channel", pf_inc, pf_l, fv, "severe - (severe co, baseline payoff)")

# (iv) factor through the scenario end, then linear reversion to 1 over 4 quarters
tm = W1 + 1 + np.arange(NSTEP)
for key, mname, F in [("M1v", "R2a", fac["M1v"]), ("R2b", "R2c", fac["R2b"])]:
    w = np.clip((tm - SCEN_END) / 12.0, 0, 1)
    fpath = F * (1 - w) + 1.0 * w
    for scen in ("baseline", "severe"):
        _, _, path, path_l = tl(("base", key, scen))
        add("iv", mname, scen, float((path * fpath).sum()), float((path_l * fpath).sum()), F,
            "factor in full to 2029-03, then linear to 1 by 2030-03",
            share_of_loss_after_2029_03=float(path[tm > SCEN_END].sum() / path.sum()))

# (v) FY2020+ era effect = FY2014-19 era effect
if has_era4:
    for scen in ("baseline", "severe"):
        lv, lvl, _, _ = tl(("v", "M1v_e4", scen))
        add("v", "M1v", scen, lv, lvl, note=f"era4 beta set to era3 beta ({mods['M1v']['beta']['era3']:.4f}; fitted era4 {mods['M1v']['beta']['era4']:.4f})")
        add("v", "R2a (factor recomputed)", scen, lv * fac["M1v_e4"], lvl * fac["M1v_e4"], fac["M1v_e4"],
            "modified M1v x its own trailing 24-month actual / predicted")
else:
    for scen in ("baseline", "severe"):
        add("v", "M1v", scen, np.nan, np.nan, note="2026 final fit has no separate era4 effect: FY2020+ already takes era3")

out = pd.DataFrame(rows)
base = out[out.variant == "base"].set_index(["model", "scenario"]).loss_over_balance
out["base_loss_over_balance"] = [base.get((m_.split(" ")[0], s_), np.nan) for m_, s_ in zip(out.model, out.scenario)]
out["change_vs_base_pp"] = 100 * (out.loss_over_balance - out.base_loss_over_balance)
out.to_csv(R / "r4_allowance_sensitivities.csv", index=False)
log(out[["variant", "model", "scenario", "loss_over_balance", "change_vs_base_pp", "factor"]].round(5).to_string(index=False))
log("done")
