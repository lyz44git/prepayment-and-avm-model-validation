"""Figure 9: remediation of F2 and F9 (drawn from reports/p1 CSVs only)."""
import sys; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import REPORTS
from mval import figstyle as fs
import pandas as pd, numpy as np, matplotlib.pyplot as plt
fs.apply(); O = REPORTS / "p1"; R = O / "remediation"; S = fs.SERIES
m = pd.read_csv(R / "step3_4_monthly_cpr_paths.csv"); m = m[m.split == "oot"]
ts = pd.read_csv(O / "ts_benchmark_paths.csv")
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"width_ratios": [1.35, 1]})
a = ax[0]; x = pd.to_datetime(m[m.model == "S0 fixed"].month)
a.plot(x, m[m.model == "S0 fixed"].cpr_y * 100, color=fs.INK, marker="o", ms=4, label="Actual")
a.plot(x, m[m.model == "S0 fixed"].cpr_p * 100, color=S[1], ls="--", label="Logit, fixed at 2024-06 (published)")
a.plot(x, m[m.model == "S1 walk-forward"].cpr_p * 100, color=S[0], label="Logit S1, re-estimated each month")
a.plot(pd.to_datetime(ts.month), ts["hazard: logit + rolling 3-month recalibration | rolling 1-month-ahead"], color=S[2],
       label="Logit + trailing 3-month recalibration")
a.set_ylabel("Pool CPR (%)"); a.set_title("a  Re-estimation does not fix the refinance shortfall"); a.legend(loc="upper left", fontsize=7.8)
st = pd.read_csv(R / "step5_stress_summary.csv")
sc = ["HPI -20% (rates flat)", "HPI -10% (rates flat)", "HPI +10% (rates flat)"]; lab = ["House prices −20%", "−10%", "+10%"]
b = ax[1]; y = np.arange(len(sc)); w = 0.36
for k, (spec, col, name) in enumerate([("S0", S[1], "Published (two equity terms)"), ("S1", S[0], "S1 (estimated LTV only)")]):
    v = st[st.spec == spec].set_index("scenario").delta_cpr_count_vs_base_pp.reindex(sc).to_numpy()
    b.barh(y + (k - 0.5) * w, v, height=w, color=col, label=name)
    for yy, vv in zip(y + (k - 0.5) * w, v):
        b.text(vv + (0.001 if vv >= 0 else -0.001), yy, f"{vv:+.3f}", va="center", ha="left" if vv >= 0 else "right", fontsize=7.8, color=fs.INK2)
b.axvline(0, color=fs.INK2, lw=0.8); b.axvline(-0.05, color=fs.INK2, lw=0.8, ls=":")
b.text(-0.0495, -0.42, "F9 closure bar for −10%: −0.05", fontsize=7.5, color=fs.INK2, va="top", ha="left")
b.set_yticks(y, lab); b.invert_yaxis(); b.set_xlim(-0.065, 0.035); b.set_xlabel("Change in 24-month average CPR vs base (pp)")
b.set_title("b  House-price response: about doubled, still small"); b.legend(loc="lower left", fontsize=7.8)
fs.note(fig, "Out-of-time months 2024-07..2025-06, training states. Re-estimation refits on data to month t − 2 and predicts t. Stress pool: "
             "564,732 homes at risk in 2025-06, median estimated LTV 0.30; only 0.004% of training property-months are above 80% LTV.")
fig.tight_layout(); fig.savefig(O / "fig9_remediation.png"); print("figure saved")
