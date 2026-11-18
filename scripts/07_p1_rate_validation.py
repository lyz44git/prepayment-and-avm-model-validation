"""Measurement validation: how well does the symmetric note-rate proxy match Vendor A's lien rate on stayers?

Stayers (no event in the window) whose Vendor A first-lien date is within 2 months of the purchase = purchase
mortgage still outstanding -> recorded rate vs PMMS30 in the purchase month. Stayers with a pre-window
refinance (lien date > purchase + 3 months, before 2022-05) -> recorded rate vs (a) PMMS at purchase,
(b) min PMMS between purchase and 2022-04 (the 'best achievable rate' proxy).
"""
import sys; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import WORK, REPORTS, duck
from mval.macro import monthly_rates
import numpy as np, pandas as pd
r = monthly_rates("1990-01-01").pmms30
pm = r.reindex(pd.date_range("1990-01-01", "2026-10-01", freq="MS")).ffill().to_numpy()
cummin_to_2204 = np.array([pm[i:(2022 - 1990) * 12 + 4].min() if i < (2022 - 1990) * 12 + 4 else pm[i] for i in range(len(pm))])
con = duck()
d = con.execute(f"""SELECT t_purchase, ((mtgd//10000 - 1990)*12 + (mtgd//100)%100 - 1) t_mtg, mtgr, mtgterm, mtgloan, mtgrtype
  FROM read_parquet('{WORK}/p1_universe.parquet') WHERE event='none' AND mtgr BETWEEN 0.5 AND 15 AND (mtgd%100)>0
  AND mtgd < 20220501""").df()
d["pmms_purchase"] = pm[d.t_purchase]; d["pmms_lien"] = pm[d.t_mtg.clip(0, len(pm) - 1)]
d["best_rate"] = cummin_to_2204[d.t_purchase]
d["kind"] = np.where((d.t_mtg - d.t_purchase).abs() <= 2, "purchase lien", np.where(d.t_mtg > d.t_purchase + 3, "pre-window refi", "other"))
d["fixed30"] = (d.mtgterm == 30) & d.mtgrtype.isin(["FXA", "FIX"]) & (d.mtgloan == "CNV")
rows = []
for kind in ["purchase lien", "pre-window refi"]:
    g = d[d.kind == kind]
    for sub, gg in [("all", g), ("30y fixed conventional", g[g.fixed30])]:
        row = {"group": kind, "subset": sub, "n": len(gg)}
        for proxy in ["pmms_purchase", "best_rate", "pmms_lien"]:
            e = gg.mtgr - gg[proxy]
            row[f"{proxy}: median spread"] = e.median(); row[f"{proxy}: MAE"] = e.abs().mean()
            row[f"{proxy}: corr"] = np.corrcoef(gg.mtgr, gg[proxy])[0, 1]
        rows.append(row)
res = pd.DataFrame(rows).T; print(res.to_string()); res.to_csv(REPORTS / "p1" / "p1_rate_proxy_validation.csv")
# share of recorded rates that look like vendor-modelled values (exactly equal to PMMS in some month, 2 decimals)
pm_set = set(np.round(r.to_numpy(), 2))
print("share of recorded rates equal to a monthly PMMS30 value (2 dp):", round(d.mtgr.round(2).isin(pm_set).mean(), 3))
print("distinct recorded rate values:", d.mtgr.nunique(), " most common:", d.mtgr.value_counts().head(8).to_dict())
by_year = d[d.kind == "purchase lien"].assign(yr=lambda x: 1990 + x.t_purchase // 12).groupby("yr").apply(
    lambda g: pd.Series({"n": len(g), "median_recorded": g.mtgr.median(), "pmms": g.pmms_purchase.median()}))
by_year.to_csv(REPORTS / "p1" / "p1_rate_proxy_by_year.csv"); print(by_year.loc[2005:].round(2).T.to_string())
