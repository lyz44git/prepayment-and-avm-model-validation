"""Prepayment study extension: link the prepayment scenarios to a pass-through cash-flow valuation.

Inputs
- reports/p1/stress_paths.csv: monthly pool SMM for months 1-24 under base / +-100 / +-200 bp, for the logit
  champion (m1) and the LightGBM challenger (m2).
- data_work/p1/panel.parquet: the 564,732 properties still at risk at 2025-06 (4% sample), used to build the
  same ILLUSTRATIVE pool as script 16: each property carries its purchase-money loan (80% LTV, 30-year level
  pay at the purchase-month PMMS). The actual lien at risk is not observed.

Method
- Loans are grouped by purchase month (note rate and remaining term are both set by it).
- Each month: scheduled principal at the gross note rate, then the pool SMM applied to the post-scheduled
  balance; the investor receives scheduled + prepaid principal and interest at net coupon = note - 0.50%.
- SMM beyond month 24 is held at the month-24 value (sensitivity: mean of months 13-24).
- Flat discount yield = 10-year Treasury at 2025-06 + 150 bp. Scenario shifts move the discount yield in
  parallel and instantly, and the prepayment path comes from the matching model scenario.
- Effective duration and convexity from +-100 bp and +-200 bp; "static" uses the base SMM path for every
  shift (no prepayment response), which isolates the prepayment option's contribution.

Limits: no term structure, no rate-path Monte Carlo / OAS, no payment delay, no market price data.
"""
import sys, json, functools, warnings; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
from _paths import WORK, REPORTS
from mval import p1, figstyle
from mval.amortization import scheduled_principal_from_balance, scheduled_balance
from mval.macro import monthly_rates
import numpy as np, pandas as pd, matplotlib.pyplot as plt
print = functools.partial(print, flush=True)
OUT = REPORTS / "p1"
SERVICING = 0.50          # servicing + guarantee fee, percentage points
SPREAD = 1.50             # nominal spread to the 10-year Treasury, percentage points
SHIFTS = {"base (flat at 2025-06)": 0, "-100bp": -100, "+100bp": 100, "-200bp": -200, "+200bp": 200}

# ---------------------------------------------------------------- pool (same construction as script 16)
cols = ["t", "y_sale", "y_new_lien", "y_other_transfer", "t_purchase", "note_hat", "tenure_m", "price0"]
P = pd.read_parquet(WORK / "p1" / "panel.parquet", columns=cols)
last = P[(P.t == p1.T_END) & (P.y_sale == 0) & (P.y_new_lien == 0) & (P.y_other_transfer == 0)].copy()
assert len(last) == 564_732, len(last)
last["bal"] = scheduled_balance(0.8 * last.price0, last.note_hat, 360, last.tenure_m)
last["rem"] = 360 - last.tenure_m
live = last[(last.rem > 0) & (last.bal > 0)]
coh = live.groupby("t_purchase").agg(bal=("bal", "sum"), note=("note_hat", "first"), rem=("rem", "first"),
                                     n=("bal", "size")).reset_index()
B0 = coh.bal.sum()
wac = float((coh.note * coh.bal).sum() / B0); wam = float((coh.rem * coh.bal).sum() / B0)
print(f"pool: {len(live):,} live loans in {len(coh)} purchase-month cohorts; balance ${B0/1e9:.1f}bn; "
      f"WAC {wac:.2f}%; WAM {wam:.0f} months; excluded (term ended) {len(last)-len(live):,}")
dgs10 = float(monthly_rates("2025-01-01").dgs10.loc["2025-06-01"]); y0 = dgs10 + SPREAD
print(f"discount yield: 10y UST {dgs10:.3f}% + {SPREAD:.2f} = {y0:.3f}%")

# ---------------------------------------------------------------- cash-flow engine
K = int(coh.rem.max())
def cashflows(smm_path, note=coh.note.to_numpy(), bal0=coh.bal.to_numpy(), rem0=coh.rem.to_numpy(), servicing=SERVICING):
    """Monthly investor cash flows for the pool under one SMM path (length K)."""
    bal, rem = bal0.astype(float).copy(), rem0.astype(float).copy()
    prin = np.zeros(K); intr = np.zeros(K)
    for k in range(K):
        sp = scheduled_principal_from_balance(bal, note, rem)
        intr[k] = (bal * (note - servicing) / 1200).sum()
        after = bal - sp; pp = after * smm_path[k]
        prin[k] = sp.sum() + pp.sum(); bal = after - pp; rem = np.maximum(rem - 1, 0)
    return prin, intr

def value(smm_path, y, **kw):
    prin, intr = cashflows(smm_path, **kw)
    t = np.arange(1, K + 1); df = (1 + y / 1200) ** (-t)
    b0 = kw.get("bal0", coh.bal.to_numpy()).sum()
    return {"price": 100 * ((prin + intr) * df).sum() / b0, "wal_years": float((t / 12 * prin).sum() / prin.sum())}

def extend(path24, how="last"):
    tail = path24[-1] if how == "last" else path24[12:].mean()
    return np.concatenate([path24, np.full(K - len(path24), tail)])

# sanity check: a loan discounted at its own note rate, with no servicing strip, prices at par for any SMM
for smm in (0.0, 0.01, 0.05):
    chk = value(np.full(K, smm), 6.0, note=np.array([6.0]), bal0=np.array([1.0]), rem0=np.array([300.0]), servicing=0.0)
    assert abs(chk["price"] - 100) < 1e-6, chk
print("engine check passed: own-rate discounting prices at par for SMM 0 / 1% / 5%")

# ---------------------------------------------------------------- scenarios
SP = pd.read_csv(OUT / "stress_paths.csv")
rows, risk = [], []
for how in ("last", "mean13_24"):
    for model in ("m1", "m2"):
        paths = {s: extend(SP[(SP.scenario == s) & (SP.model == model)].sort_values("month").smm_count.to_numpy(), how)
                 for s in SHIFTS}
        for prepay in ("dynamic", "static"):
            pr = {}
            for s, bp in SHIFTS.items():
                v = value(paths[s] if prepay == "dynamic" else paths["base (flat at 2025-06)"], y0 + bp / 100)
                pr[bp] = v["price"]
                rows.append({"extension": how, "model": model, "prepay": prepay, "scenario": s, "shift_bp": bp,
                             "discount_yield_pct": y0 + bp / 100, "price": v["price"], "wal_years": v["wal_years"],
                             "avg_cpr_24m_pct": float((1 - (1 - paths[s][:24]) ** 12).mean() * 100) if prepay == "dynamic"
                             else float((1 - (1 - paths["base (flat at 2025-06)"][:24]) ** 12).mean() * 100)})
            p0 = pr[0]
            risk.append({"extension": how, "model": model, "prepay": prepay, "base_price": p0,
                         "eff_duration_100": (pr[-100] - pr[100]) / (2 * p0 * 0.01),
                         "eff_convexity_100": (pr[-100] + pr[100] - 2 * p0) / (p0 * 0.01 ** 2),
                         "eff_duration_200": (pr[-200] - pr[200]) / (2 * p0 * 0.02),
                         "eff_convexity_200": (pr[-200] + pr[200] - 2 * p0) / (p0 * 0.02 ** 2)})
V = pd.DataFrame(rows); R = pd.DataFrame(risk)
V.to_csv(OUT / "valuation_scenarios.csv", index=False); R.to_csv(OUT / "valuation_risk.csv", index=False)
json.dump({"loans": int(len(live)), "cohorts": int(len(coh)), "excluded_term_ended": int(len(last) - len(live)),
           "balance_usd": float(B0), "wac_pct": wac, "wam_months": wam, "net_coupon_pct": wac - SERVICING,
           "servicing_pct": SERVICING, "dgs10_2025_06_pct": dgs10, "spread_pct": SPREAD, "discount_yield_pct": y0,
           "smm_extension_main": "month-24 SMM held flat", "smm_extension_sensitivity": "mean SMM of months 13-24"},
          open(OUT / "valuation_meta.json", "w"), indent=2)
pd.set_option("display.width", 200)
print(V[V.extension == "last"].round(3).to_string(index=False))
print(R.round(3).to_string(index=False))

# ---------------------------------------------------------------- figure: price vs parallel shift
figstyle.apply()
fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
main = V[V.extension == "last"]
series = [("m1", "dynamic", "Logit champion, prepayment responds", figstyle.SERIES[0]),
          ("m2", "dynamic", "LightGBM challenger, prepayment responds", figstyle.SERIES[1]),
          ("m1", "static", "Prepayment held at base path", figstyle.SERIES[2])]
for model, prepay, lab, col in series:
    d = main[(main.model == model) & (main.prepay == prepay)].sort_values("shift_bp")
    ax.plot(d.shift_bp, d.price, color=col, marker="o", ms=5, mec=figstyle.SURFACE, mew=1.5, label=lab)
ax.set_xlabel("Parallel rate shift (bp)"); ax.set_ylabel("Price per 100 of balance")
ax.set_xticks(list(sorted(SHIFTS.values()))); ax.legend(loc="upper right")
ax.set_title("Price under parallel rate shifts")
# right panel: price effect of the prepayment response = dynamic - static, by model
w = 30
for j, (model, col, lab) in enumerate([("m1", figstyle.SERIES[0], "Logit"), ("m2", figstyle.SERIES[1], "LightGBM")]):
    dyn = main[(main.model == model) & (main.prepay == "dynamic")].set_index("shift_bp").price
    sta = main[(main.model == model) & (main.prepay == "static")].set_index("shift_bp").price
    diff = (dyn - sta).reindex(sorted(SHIFTS.values()))
    xs = np.array(diff.index) + (j - 0.5) * (w + 12)
    bx.bar(xs, diff.values, width=w, color=col, label=lab)
    for x_, v_ in zip(xs, diff.values):
        if abs(v_) > 0.05:
            bx.text(x_, v_ + (0.04 if v_ > 0 else -0.04), f"{v_:+.2f}", ha="center", fontsize=7.2, va="bottom" if v_ > 0 else "top",
                    color=figstyle.INK2)
bx.axhline(0, color=figstyle.INK2, lw=0.8)
bx.set_xticks(list(sorted(SHIFTS.values()))); bx.set_xlabel("Parallel rate shift (bp)")
bx.set_ylabel("Price points (responding minus static)"); bx.legend(loc="upper right")
bx.set_title("Price effect of the prepayment response")
r1 = R[(R.extension == "last") & (R.model == "m1")].set_index("prepay")
figstyle.note(fig, f"Illustrative pool of {len(live):,} loans (WAC {wac:.2f}%, WAM {wam:.0f} months, net coupon {wac-SERVICING:.2f}%) "
                   f"discounted at a flat {y0:.2f}% (10-year Treasury at 2025-06 + {SPREAD*100:.0f} bp); SMM after month 24 held at the "
                   f"month-24 value. Logit effective duration {r1.loc['dynamic','eff_duration_100']:.2f} with responding prepayment vs "
                   f"{r1.loc['static','eff_duration_100']:.2f} with prepayment held fixed (±100 bp): for this discount pool, slower "
                   f"prepayment when rates rise (extension) adds price loss. Property-level prepayment proxy; no term structure or OAS.")
fig.tight_layout(); fig.savefig(OUT / "fig7_valuation_price_shift.png"); print("figure saved")
