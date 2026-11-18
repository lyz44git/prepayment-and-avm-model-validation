"""Valuation study figures (drawn from CSVs written by scripts 12 and 15)."""
import sys; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import REPORTS
from mval import figstyle as fs
import numpy as np, pandas as pd, matplotlib.pyplot as plt
fs.apply(); O = REPORTS / "p2"; S = fs.SERIES
# F1 vendor value / sale price by sale quarter
q = pd.read_csv(O / "vendor_accuracy_by_quarter.csv").pivot(index="group", columns="estimator", values="median_ratio")
fig, a = plt.subplots(figsize=(8.0, 3.8))
names = ["Vendor A AVM", "Vendor A market value", "Assessor value (Vendor C 2025)", "Vendor B est. home value"]
for k, n in enumerate(names):
    a.plot(q.index, q[n], marker="o", ms=4, color=S[k], label=n)
a.axhline(1.0, color=fs.INK2, lw=1, ls=":"); a.set_ylabel("Median vendor value / sale price")
a.set_title("Third-party values vs recorded arm's-length prices, by sale quarter")
a.legend(ncol=2, loc="lower left", bbox_to_anchor=(0.50, 0.11), fontsize=8); plt.setp(a.get_xticklabels(), rotation=45, ha="right")
fs.note(fig, "2.4M arm's-length resales, 49 states + DC. Vendor B = 1.000 through 2024Q3: its value is anchored to the recorded sale price "
             "(look-ahead). Vendor A AVM: median value 0.68 of the sale price; its error moves one-for-one with appreciation (slope -1.02).")
fig.savefig(O / "fig1_vendor_ratio_by_quarter.png"); plt.close(fig)
# F2 head-to-head MdAPE and PPE10 (OOT, common support)
t = pd.read_csv(O / "accuracy_oot_common.csv").sort_values("MdAPE", ascending=False)
fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
col = [S[1] if "Vendor A" in e or "Assessor" in e or "Vendor B" in e else (S[0] if e.startswith("C") else fs.INK2) for e in t.estimator]
axs[0].barh(t.estimator, t.MdAPE * 100, color=col, height=0.62); axs[0].set_xlabel("Median absolute % error (lower is better)")
axs[1].barh(t.estimator, t.PPE10 * 100, color=col, height=0.62); axs[1].set_xlabel("% of sales within ±10% (higher is better)")
for a, c in zip(axs, ["MdAPE", "PPE10"]):
    for y, v in enumerate(t[c] * 100):
        a.text(v + 0.6, y, f"{v:.1f}", va="center", fontsize=8, color=fs.INK)
    a.grid(axis="y", visible=False)
axs[0].set_title("Champions (orange), baselines (gray), challengers (blue): out-of-time")
fs.note(fig, "Identical rows: 419,615 sales 2024-07..2025-08 where all four vendor values exist; models trained on sales <= 2024-03 with "
             "features dated <= 2022-04 and a market index frozen at 2024-06. Vendor B is anchored to the sale itself (not a valid benchmark).")
fig.savefig(O / "fig2_champion_challenger.png"); plt.close(fig)
# F3 MdAPE by price tier
tt = pd.read_csv(O / "accuracy_by_price_tier_oot.csv").pivot(index="group", columns="estimator", values="MdAPE")
fig, a = plt.subplots(figsize=(7.6, 3.8))
for k, n in enumerate(["Vendor A AVM (champion)", "Vendor A market value", "C1 LightGBM", "C2 neighbour-attention NN", "C3 blend (C1+C2)"]):
    a.plot(tt.index, tt[n] * 100, marker="o", ms=4, color=S[k], label=n)
a.set_xlabel("Within-state price decile (1 = lowest, 10 = highest)"); a.set_ylabel("Median absolute % error")
a.set_xticks(range(1, 11)); a.legend(ncol=2); a.set_title("Accuracy by price tier (OOT)")
fs.note(fig, "Top decile = the upmarket segment; bottom decile is dominated by non-standard / distressed resales for every estimator.")
fig.savefig(O / "fig3_accuracy_by_tier.png"); plt.close(fig)
# F4 within-tier disparity by tract minority share
w = pd.read_csv(O / "disparity_minority_within_tier_oot.csv").set_index("estimator")
qs = [f"Q{i}" for i in range(1, 6)]
fig, a = plt.subplots(figsize=(7.4, 3.8))
for k, n in enumerate(["Vendor A AVM (champion)", "Assessor 2025", "Vendor B value", "C3 blend (C1+C2)"]):
    a.plot(qs, [w.loc[n, f"{q}_within_tier_logratio_dev"] * 100 for q in qs], marker="o", ms=6, color=S[k], label=n)
a.axhline(0, color=fs.INK2, lw=1, ls=":")
a.set_xlabel("Tract Black + Hispanic population share quintile (Q1 lowest)"); a.set_ylabel("Value/price deviation within tier (%)")
a.legend(); a.set_title("Valuation gap by neighbourhood composition, same price tier (OOT)")
fs.note(fig, "Median log(value/price) per price decile x quintile, centred within decile, averaged over deciles. Race shares are never model "
             "features; used only for this audit. Vendor A AVM: Q5-Q1 = -9.1%; challenger blend: -2.0%.")
fig.savefig(O / "fig4_disparity_within_tier.png"); plt.close(fig)
# F5 drift
d = pd.read_csv(O / "oot_monthly_MdAPE.csv", index_col=0)
fig, a = plt.subplots(figsize=(7.6, 3.6))
for k, n in enumerate(["Vendor A market value", "B0 index AVM (prior sale x HPI)", "C1 LightGBM", "C3 blend (C1+C2)"]):
    a.plot(d.index, d[n] * 100, marker="o", ms=4, color=[S[1], fs.INK2, S[0], S[2]][k], label=n)
a.set_ylabel("Median absolute % error"); a.legend(ncol=2); a.set_title("Error drift after the model freeze (2024-06)")
plt.setp(a.get_xticklabels(), rotation=45, ha="right")
fs.note(fig, "Models and market index frozen at 2024-06; error grows ~0.14 pp per month for the blend -> retraining / re-indexing trigger.")
fig.savefig(O / "fig5_drift.png"); plt.close(fig)
print("P2 figures written")
