"""Price tiers that do not condition on the outcome (round-4 robustness for the valuation study).

The published tier accuracy and within-tier minority-share disparity (scripts/15) sort out-of-time sales into
within-state deciles of the realised sale price. Sorting on the outcome mechanically pushes over-valued sales into
low tiers and under-valued sales into high tiers. This script repeats both analyses with tiers built from a value
known before the sale:
  - prior_sale_indexed: prior arm's-length sale x repeat-sales index frozen at 2024-06 (the B0 value; preferred,
    but only sales with a recorded prior sale have it);
  - assessor_mv_2021: assessor market value from the pre-2022-05 property vintage (~2021 tax roll), broader coverage.
Tiers are within-state deciles of the ex-ante value (rank, ties by row order, as in 15). Minority-share quintiles
(tract Black + Hispanic share) keep the published assignment computed on all OOT sales. Each ex-ante analysis is
paired with a realised-price-tier analysis on the same rows, so the comparison isolates the tier definition from
the change in sample. Disparity construction as in 15: median log(value/price) per tier x quintile, minus the tier
mean over quintiles, averaged over tiers; Q5 - Q1. County-cluster bootstrap (B = 500; counties resampled with
replacement, tier and quintile assignments held fixed, same draws for every row of the table).

Estimators: Vendor A AVM (champion), C3 blend (C1+C2), and Vendor A market value (best usable vendor field) for context.
Reads data_work/p2/eval_frame.parquet and features.parquet. Writes reports/p2/r4_accuracy_by_exante_tier.csv and
reports/p2/r4_disparity_exante_tier.csv (aggregate only).
"""
import sys, time, functools; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import WORK, REPORTS
from mval.valuation import table
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)
t0 = time.time(); OUT = REPORTS / "p2"; DW = WORK / "p2"
B = 500; SEED = 20261118
EST = {"Vendor A AVM (champion)": "v_vendor_a_avm", "C3 blend (C1+C2)": "$c3_blend", "Vendor A market value": "v_vendor_a_mktval"}
EXISTING = pd.read_csv(OUT / "disparity_minority_within_tier_oot.csv").set_index("estimator").Q5_minus_Q1_within_tier

D = pd.read_parquet(DW / "eval_frame.parquet")
F = pd.read_parquet(DW / "features.parquet", columns=["addr_id", "mv0"])
assert (F.addr_id.to_numpy() == D.addr_id.to_numpy()).all()
D["mv0"] = F.mv0.to_numpy(); del F
o = D[D.split == "oot"].copy(); del D
o["minority"] = o.t_black.fillna(0) + o.t_hisp.fillna(0)
o["minority_q"] = pd.qcut(o.minority.rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
o["q_idx"] = o.minority_q.cat.codes.astype(int)
o["exante_prior_sale"] = o["$b0_index"]
o["exante_assessor_mv"] = o.mv0.where(o.mv0 > 1000)
N_OOT = len(o)
print(f"OOT sales {N_OOT:,}; prior-sale coverage {o.exante_prior_sale.notna().mean():.4f}; "
      f"assessor-mv coverage {o.exante_assessor_mv.notna().mean():.4f} ({time.time()-t0:.0f}s)")

def deciles(df, col, min_rows=50):
    """Within-state deciles of df[col] (rank with ties broken by row order); states with < min_rows get NaN."""
    n = df.groupby("state")[col].transform("size")
    out = pd.Series(np.nan, index=df.index)
    ok = n >= min_rows
    out[ok] = df[ok].groupby("state")[col].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
    return out

SAMPLES = {"all OOT sales": o.index, "sales with a prior recorded sale": o.index[o.exante_prior_sale.notna()],
           "sales with a 2021 assessor market value": o.index[o.exante_assessor_mv.notna()]}
SPECS = [("realised_price", "all OOT sales", "price"),
         ("prior_sale_indexed", "sales with a prior recorded sale", "exante_prior_sale"),
         ("realised_price", "sales with a prior recorded sale", "price"),
         ("assessor_mv_2021", "sales with a 2021 assessor market value", "exante_assessor_mv"),
         ("realised_price", "sales with a 2021 assessor market value", "price")]
tiers = {}
for basis, samp, col in SPECS:
    s = o.loc[SAMPLES[samp]]
    tiers[(basis, samp)] = deciles(s, col)
    print(basis, "|", samp, f"rows {len(s):,}, tiered {tiers[(basis, samp)].notna().sum():,}")

# ---------------------------------------------------------------- (a) accuracy by tier
acc = []
for basis, samp, col in SPECS:
    s = o.loc[SAMPLES[samp]].assign(tier=tiers[(basis, samp)]).dropna(subset=["tier"])
    s["tier"] = s.tier.astype(int)
    t = table(s, EST, by="tier")
    t = t[["group", "estimator", "n", "coverage", "median_ratio", "MdAPE", "PPE10", "PPE20"]].rename(columns={"group": "decile"})
    t.insert(0, "sample", samp); t.insert(0, "tier_basis", basis)
    t["sample_rows"] = len(s); t["sample_share_of_oot"] = len(s) / N_OOT
    if col != "price":
        t["exante_over_price_median_in_decile"] = t.decile.map(s.groupby("tier").apply(lambda g: float(np.median(g[col] / g.price)), include_groups=False))
    t["realised_price_median_in_decile"] = t.decile.map(s.groupby("tier").price.median())
    acc.append(t)
acc = pd.concat(acc, ignore_index=True)
acc.to_csv(OUT / "r4_accuracy_by_exante_tier.csv", index=False)
old = pd.read_csv(OUT / "accuracy_by_price_tier_oot.csv")
chk = acc[(acc.tier_basis == "realised_price") & (acc["sample"] == "all OOT sales")].merge(
    old.rename(columns={"group": "decile"}), on=["decile", "estimator"], suffixes=("", "_old"))
print("reproduction of accuracy_by_price_tier_oot.csv: max |MdAPE diff|", float((chk.MdAPE - chk.MdAPE_old).abs().max()),
      "max |median ratio diff|", float((chk.median_ratio - chk.median_ratio_old).abs().max()), f"({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------- (b) within-tier disparity with county-cluster bootstrap
cty, cty_u = pd.factorize(o.fips5); o["cty"] = cty; G = len(cty_u)
rng = np.random.default_rng(SEED)
W = np.stack([np.bincount(rng.choice(G, G, replace=True), minlength=G) for _ in range(B)]).astype(np.int64)

class CellMedians:
    """Exact weighted (integer-weight) medians of x by cell; weight w per row equals the count of its county in a draw."""
    def __init__(self, x, cell, grp, n_cells=50):
        order = np.lexsort((x, cell)); self.x = x[order]; self.g = grp[order]; c = cell[order]
        self.starts = np.searchsorted(c, np.arange(n_cells), "left"); self.ends = np.searchsorted(c, np.arange(n_cells), "right")
    def __call__(self, wc):
        w = wc[self.g]; cum = np.cumsum(w)
        off = np.where(self.starts > 0, cum[np.maximum(self.starts - 1, 0)], 0)
        tot = np.where(self.ends > self.starts, cum[np.maximum(self.ends - 1, 0)], 0) - off
        tot = np.where(self.ends > self.starts, tot, 0)
        k1 = (tot + 1) // 2; k2 = tot // 2 + 1
        i1 = np.minimum(np.searchsorted(cum, off + k1, "left"), len(cum) - 1)
        i2 = np.minimum(np.searchsorted(cum, off + k2, "left"), len(cum) - 1)
        med = (self.x[i1] + self.x[i2]) / 2
        return np.where(tot > 0, med, np.nan).reshape(10, 5)

def q5_minus_q1(med):
    dev = med - np.nanmean(med, axis=1, keepdims=True)
    adj = np.nanmean(dev, axis=0)
    return adj, float(adj[4] - adj[0])

rows = []; draws = {}
for basis, samp, col in SPECS:
    s = o.loc[SAMPLES[samp]].assign(tier=tiers[(basis, samp)]).dropna(subset=["tier"])
    for k, c in EST.items():
        ok = (s[c] > 1000).to_numpy()
        lr = np.log(s.loc[ok, c].to_numpy() / s.loc[ok, "price"].to_numpy())
        cell = ((s.tier.to_numpy()[ok].astype(int) - 1) * 5 + s.q_idx.to_numpy()[ok]).astype(np.int64)
        # point estimate exactly as in scripts/15 (pandas medians)
        g = pd.DataFrame({"lr": lr, "tier": s.tier.to_numpy()[ok], "q": s.minority_q.to_numpy()[ok]})
        med = g.groupby(["tier", "q"], observed=True).lr.median().unstack()
        adj = med.sub(med.mean(1), axis=0).mean(0); point = float(adj["Q5"] - adj["Q1"])
        cm = CellMedians(lr, cell, s.cty.to_numpy()[ok])
        _, point_w = q5_minus_q1(cm(np.ones(G, np.int64)))
        assert abs(point_w - point) < 1e-12, (basis, samp, k, point, point_w)
        bt = np.array([q5_minus_q1(cm(W[b]))[1] for b in range(B)]); draws[(basis, samp, k)] = bt
        rows.append({"tier_basis": basis, "sample": samp, "estimator": k, "sample_rows": len(s), "sample_share_of_oot": len(s) / N_OOT,
                     "rows_with_value": int(ok.sum()), **{f"{q}_within_tier_logratio_dev": float(adj[q]) for q in adj.index},
                     "Q5_minus_Q1_within_tier": point, "boot_ci95_lo": float(np.percentile(bt, 2.5)), "boot_ci95_hi": float(np.percentile(bt, 97.5)),
                     "boot_se": float(bt.std(ddof=1)), "bootstrap_draws": B, "existing_realised_tier_all_oot": float(EXISTING.get(k, np.nan))})
    print(basis, "|", samp, "done", f"({time.time()-t0:.0f}s)")
R = pd.DataFrame(rows)
# paired difference: ex-ante tiers minus realised-price tiers on the same rows (same bootstrap draws)
for i, r in R.iterrows():
    if r.tier_basis == "realised_price":
        continue
    base = R[(R.tier_basis == "realised_price") & (R["sample"] == r["sample"]) & (R.estimator == r.estimator)].iloc[0]
    d = draws[(r.tier_basis, r["sample"], r.estimator)] - draws[("realised_price", r["sample"], r.estimator)]
    R.loc[i, "delta_vs_realised_tier_same_rows"] = r.Q5_minus_Q1_within_tier - base.Q5_minus_Q1_within_tier
    R.loc[i, "delta_boot_ci95_lo"] = float(np.percentile(d, 2.5)); R.loc[i, "delta_boot_ci95_hi"] = float(np.percentile(d, 97.5))
R.to_csv(OUT / "r4_disparity_exante_tier.csv", index=False)
pd.set_option("display.width", 250)
print(R[["tier_basis", "sample", "estimator", "rows_with_value", "Q5_minus_Q1_within_tier", "boot_ci95_lo", "boot_ci95_hi",
         "delta_vs_realised_tier_same_rows", "delta_boot_ci95_lo", "delta_boot_ci95_hi"]].round(4).to_string(index=False))
print(acc[acc.estimator != "Vendor A market value"].pivot_table(index="decile", columns=["tier_basis", "sample", "estimator"], values="MdAPE").round(3).T.to_string())
print(f"total {time.time()-t0:.0f}s")
