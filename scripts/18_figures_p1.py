"""Prepayment study figures (all drawn from CSVs written by scripts 05b, 07, 09, 16, 17)."""
import sys; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import REPORTS
from mval import figstyle as fs
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.dates as mdates
fs.apply(); O = REPORTS / "p1"; S = fs.SERIES
# F1 monthly CPR actual vs predicted, non-hold-out states (train + OOT), PMMS panel above (no dual axis)
m = pd.read_csv(O / "monthly_cpr_actual_vs_pred.csv", parse_dates=["month"])
d = m[m.split.isin(["train", "oot"])].sort_values("month")
from mval.macro import monthly_rates
r = monthly_rates("2022-01-01").pmms30.loc["2022-05":"2025-06"]
fig, (a0, a1) = plt.subplots(2, 1, figsize=(8.2, 5.4), sharex=True, gridspec_kw={"height_ratios": [1, 2.4]})
a0.plot(r.index, r.values, color=fs.INK2, lw=1.6); a0.set_ylabel("PMMS 30y (%)"); a0.set_title("Freddie Mac 30-year rate", fontsize=9.5)
for k, (col, lab) in enumerate([("cpr_total_y", "Actual"), ("cpr_total_m1", "M1 logit hazard"), ("cpr_total_m2", "M2 LightGBM"), ("cpr_total_m0", "M0 lookup")]):
    a1.plot(d.month, d[col] * 100, color=[fs.INK, S[0], S[1], S[2]][k], lw=2.2 if k == 0 else 1.8,
            ls="-" if k < 3 else ":", label=lab)
cut = pd.Timestamp("2024-07-01")
for a in (a0, a1):
    a.axvline(cut, color=fs.INK2, lw=1, ls="--")
a1.text(cut, a1.get_ylim()[0] + 0.05, "  out-of-time test →", va="bottom", fontsize=8, color=fs.INK2)
a1.set_ylabel("Annualised termination rate (CPR, %)"); a1.legend(loc="upper right", ncol=2)
a1.set_title("Turnover + new-lien CPR: actual vs predicted (train states, 4% national sample)")
a1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fs.note(fig, "Models fit on 2022-05..2024-06 only. CPR = 1-(1-SMM)^12 with SMM = monthly hazard of sale + new first lien. "
             "Covariates use realised macro paths (conditional forecast).")
fig.savefig(O / "fig1_cpr_backtest.png"); plt.close(fig)
# F2 rate proxy validation: median Vendor A recorded rate vs PMMS by purchase year (purchase liens of stayers)
b = pd.read_csv(O / "p1_rate_proxy_by_year.csv").set_index("yr").loc[1995:2022]
fig, a = plt.subplots(figsize=(7.2, 3.6))
a.plot(b.index, b.pmms, color=S[0], marker="o", ms=4, label="PMMS 30y in purchase month (proxy)")
a.plot(b.index, b.median_recorded, color=S[1], marker="s", ms=4, ls="--", label="Vendor A 'recorded' rate (median)")
a.set_ylabel("Rate (%)"); a.legend(); a.set_title("Vendor A's lien rate tracks the market rate of the lien month")
fs.note(fig, "Stayers whose Vendor A first lien is dated within 2 months of purchase (n = 3.94M). For 30-year fixed conventional liens "
             "MAE = 0.02 pp, corr = 0.996: the field is a modelled lookup, not a recorded note rate.")
fig.savefig(O / "fig2_rate_proxy_validation.png"); plt.close(fig)
# F3 partial dependence on the rate gap
p = pd.read_csv(O / "pdp_gap.csv"); e = pd.read_csv(O / "empirical_hazard_by_gap.csv")
fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.6), sharex=True)
for a, c, t in [(axs[0], "new_lien", "New first lien (refinance / cash-out)"), (axs[1], "sale", "Sale (turnover)")]:
    a.plot(p.gap, p[f"m1_{c}"] * 1200, color=S[0], label="M1 logit")
    a.plot(p.gap, p[f"m2_{c}"] * 1200, color=S[1], ls="--", label="M2 LightGBM (monotone)")
    a.set_title(t); a.set_xlabel("Rate gap: market PMMS − purchase-month PMMS (pp)"); a.set_ylabel("Annualised hazard (%)")
    a.axvspan(-3, 0, color=fs.GRID, alpha=0.35, lw=0)
axs[0].legend()
fs.note(fig, "Partial dependence on 40k training property-months, other covariates held at observed values. Shaded = owner's rate above market "
             "(in the money). Logit: new-lien odds ratio 0.89 per +1 pp (95% CI 0.86-0.93); sale odds ratio 1.02 (0.99-1.05), not significant.")
fig.savefig(O / "fig3_rate_gap_pdp.png"); plt.close(fig)
# F4 12-month calibration by decile (OOT)
fig, a = plt.subplots(figsize=(5.6, 4.2))
for k, mname in enumerate(["m1", "m2"]):
    c = pd.read_csv(O / f"cum12_calibration_oot_{mname}.csv")
    a.plot(c.pred * 100, c.act * 100, marker="o", ms=6, color=S[k], label={"m1": "M1 logit", "m2": "M2 LightGBM"}[mname])
lim = [0, max(a.get_xlim()[1], a.get_ylim()[1])]
a.plot(lim, lim, color=fs.INK2, lw=1, ls=":"); a.set_xlim(lim); a.set_ylim(lim)
a.set_xlabel("Predicted 12-month termination (%)"); a.set_ylabel("Actual (%)"); a.legend()
a.set_title("12-month forecast calibration, OOT (2024-07..2025-06)")
fs.note(fig, "482,037 properties at risk on 2024-07-01, rolled forward 12 months; deciles of predicted probability.")
fig.savefig(O / "fig4_cum12_calibration.png"); plt.close(fig)
# F5 HPI benchmark
q = pd.read_csv(O / "hpi_ca_vs_fhfa_quarterly.csv", index_col=0, parse_dates=True)
fig, a = plt.subplots(figsize=(7.2, 3.4))
a.plot(q.index, np.exp(q.fhfa) * 100, color=S[0], label="FHFA CA all-transactions HPI")
a.plot(q.index, np.exp(q["self"]) * 100, color=S[1], ls="--", label="Self-built repeat-sales index (this data)")
a.set_ylabel("Index (2000-01 = 100)"); a.legend(); a.set_title("California: self-built repeat-sales index vs FHFA")
fs.note(fig, "Bailey-Muth-Nourse OLS with a light smoothness penalty on 60,302 CA repeat-sale pairs. Annual-change correlation 0.88, 5-year 0.95.")
fig.savefig(O / "fig5_hpi_vs_fhfa.png"); plt.close(fig)
# F6 stress CPR paths (M1) - if available
try:
    st = pd.read_csv(O / "stress_paths.csv"); st = st[st.model == "m1"]
    fig, a = plt.subplots(figsize=(8.0, 4.0))
    order = ["-200bp", "-100bp", "HPI -10% (rates flat)", "realized PMMS path", "+100bp", "+200bp", "base (flat at 2025-06)"]
    cols = {"-200bp": S[0], "-100bp": S[1], "HPI -10% (rates flat)": S[6], "realized PMMS path": S[3], "+100bp": S[4],
            "+200bp": S[5], "base (flat at 2025-06)": fs.INK}
    for sc in order:
        g = st[st.scenario == sc]
        a.plot(pd.to_datetime(g.month), g.cpr_count * 100, color=cols[sc], lw=2.4 if "base" in sc else 1.7,
               ls="--" if "base" in sc else "-", label=sc)
    a.set_ylabel("Projected CPR (%)"); a.legend(ncol=2, fontsize=8); a.set_title("Stress test: projected turnover + new-lien CPR (M1)")
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fs.note(fig, "Properties at risk on 2025-06 (4% sample); shocks phased in over 6 months. Training PMMS range 5.2-7.6%: both +/-200bp paths "
                 "(8.8%, 4.8%) leave it, so those responses extrapolate the logit form. The HPI -10% path sits on top of base: its two "
                 "equity terms offset (finding F9).")
    fig.savefig(O / "fig6_stress_paths.png"); plt.close(fig)
except FileNotFoundError:
    print("stress paths not yet available")
print("P1 figures written")
