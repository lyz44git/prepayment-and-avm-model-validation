"""Credit-loss study: two input checks added at review (findings L11 and L12); no model is refitted.

L11  business-age categories by approval fiscal year (the coding changes across eras, and the field is a predictor).
L12  macro inputs for 2026Q1-Q3: the Fed 2026 files end their history in 2025Q4, so these quarters carry scenario
     values inside the estimation window; compared here with the published series (FRED UNRATE, DPRIME, DGS10).
Writes reports/p3/business_age_by_approval_fy.csv and reports/p3/macro_2026_scenario_vs_published.csv.
"""
import sys; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import pandas as pd
from _paths import WORK, REPORTS

R = REPORTS / "p3"
loans = pd.read_parquet(WORK / "p3" / "loans.parquet", columns=["approval_fy", "bage"])
t = pd.crosstab(loans.approval_fy, loans.bage, normalize="index").round(3)
t.insert(0, "loans", loans.approval_fy.value_counts().sort_index())
t.to_csv(R / "business_age_by_approval_fy.csv")

FED, FRED = Path("data_external/fed2026"), Path("data_external/fred_credit")
rows = []
for scen, f in [("baseline", "2026_Final_Supervisory_Baseline_Domestic.csv"),
                ("severely_adverse", "2026_Final_Supervisory_Severely_Adverse_Domestic.csv")]:
    d = pd.read_csv(FED / f)
    d = d[d.Date.isin(["2026 Q1", "2026 Q2", "2026 Q3"])]
    for _, r in d.iterrows():
        rows.append({"quarter": r.Date.replace(" ", ""), "source": f"fed_{scen}", "unemployment": r["Unemployment rate"],
                     "prime": r["Prime rate"], "treasury_10y": r["10-year Treasury yield"]})
pub = {}
for sid, col in [("UNRATE", "unemployment"), ("DPRIME", "prime"), ("DGS10", "treasury_10y")]:
    s = pd.read_csv(FRED / f"{sid}.csv"); s.columns = ["date", "v"]
    s["v"] = pd.to_numeric(s.v, errors="coerce"); s["date"] = pd.to_datetime(s.date)
    s = s[(s.date >= "2026-01-01") & (s.date < "2026-10-01")]
    pub[col] = s.groupby(s.date.dt.to_period("Q")).v.mean().round(2)
p = pd.DataFrame(pub); p.index = p.index.astype(str)
for q, r in p.iterrows():
    rows.append({"quarter": q, "source": "published_fred", **r.to_dict()})
out = pd.DataFrame(rows).sort_values(["quarter", "source"])
out.to_csv(R / "macro_2026_scenario_vs_published.csv", index=False)
print(t.to_string()); print(out.to_string(index=False))
