"""Benchmark the self-built CA repeat-sales index against FHFA's published CA all-transactions HPI."""
import sys; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import WORK, REPORTS
from mval.macro import fhfa_ca_hpi
import numpy as np, pandas as pd
h = pd.read_parquet(WORK / "hpi_state_monthly.parquet")
ca = h[(h.version == "full") & (h.state == "CA")].set_index("month").log_hpi
f = np.log(fhfa_ca_hpi())
df = pd.concat([ca.rename("self"), f.rename("fhfa")], axis=1).dropna().loc["1995":"2025-06"]
df = df - df.loc["2000-01-01"]
q = df.resample("QS").mean()
rows = []
for lag, name in [(4, "annual (4q) log change"), (20, "5-year log change")]:
    d = q.diff(lag).dropna()
    rows.append({"horizon": name, "corr": d.corr().iloc[0, 1], "mae_pp": (d.self - d.fhfa).abs().mean() * 100,
                 "n": len(d)})
for a, b in [("2000-01-01", "2006-07-01"), ("2006-07-01", "2012-01-01"), ("2012-01-01", "2022-04-01"), ("2022-04-01", "2025-04-01")]:
    rows.append({"horizon": f"cum change {a[:7]}->{b[:7]}", "self_pct": (np.exp(q.self[b] - q.self[a]) - 1) * 100,
                 "fhfa_pct": (np.exp(q.fhfa[b] - q.fhfa[a]) - 1) * 100})
r = pd.DataFrame(rows); print(r.round(3).to_string(index=False)); r.to_csv(REPORTS / "hpi_ca_vs_fhfa.csv", index=False)
q.to_csv(WORK / "hpi_ca_vs_fhfa_quarterly.csv")
