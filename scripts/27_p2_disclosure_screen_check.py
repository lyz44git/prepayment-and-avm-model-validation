"""Valuation study: the price-disclosure screen with the denominator it was meant to have (finding V10, 2026-11-19).

10_p2_sales.py kept a state when >= 90% of its window resales "carry a sale amount", computed as
AVG((sa1 >= 30000)::int). AVG skips NULL, so resales with no recorded price dropped out of the denominator and every
state passed: the screen did not bind, and the published sample covers all 50 states with recorded prices. This script
recomputes each state's share over all window resales, lists the states that fail the intended screen, and scores the
existing out-of-time and spatial predictions on the states that pass. No model is refitted: the challengers were
trained with the failing states included, so this separates the evaluation sample, not the training sample.
Writes reports/p2/r5_disclosure_denominator_bridge.csv and reports/p2/r5_accuracy_by_screen_version.csv, and
reports/p2/r5_avm_data_counts.csv (sample, comparable-sale pool and repeat-sales pair counts quoted in the report).
"""
import sys; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from _paths import WORK, REPORTS, duck
from mval import p2
from mval.sql import RES_SQL, ARMS_SQL
from mval.valuation import table

OUT = REPORTS / "p2"; DW = WORK / "p2"
con = duck(threads=12)
P = f"read_parquet('{WORK}/props/*.parquet')"
cov = con.execute(f"""SELECT state, count(*) n_resale, count(sa1) n_price_recorded,
   avg((sa1 >= 30000)::int) share_as_coded, avg((COALESCE(sa1, 0) >= 30000)::int) share_of_all_resales
   FROM {P} WHERE {RES_SQL} AND in_vc_new AND sd1 >= 20220501 AND sd1 < 20250900 AND tt1='1' AND doc1 IN {ARMS_SQL}
   GROUP BY 1 ORDER BY 5""").df()
cov["passes_as_coded"] = cov.share_as_coded >= 0.90
cov["passes_intended_screen"] = cov.share_of_all_resales >= 0.90
cov["in_nondisclosure_list"] = cov.state.isin(p2.NON_DISCLOSURE)
cov.to_csv(OUT / "r5_disclosure_denominator_bridge.csv", index=False)
fail = sorted(cov.loc[~cov.passes_intended_screen, "state"])
print("fail the intended screen:", fail)

S = pd.read_parquet(DW / "features.parquet", columns=["split", "state", "price", "v_vendor_a_avm", "v_vendor_b", "v_vendor_a_mktval",
                                                      "v_assessor_2025"])
G = pd.read_parquet(DW / "preds_gbm.parquet", columns=["b0_index", "b1_hedonic", "c1_gbm"])
A = pd.read_parquet(DW / "preds_attn.parquet", columns=["c2_attn"])
for c in ["b0_index", "b1_hedonic", "c1_gbm"]:
    S["$" + c] = np.exp(G[c].to_numpy())
S["$c2_attn"] = np.exp(A.c2_attn.to_numpy())
S["$c3_blend"] = np.exp(0.5 * G.c1_gbm.to_numpy() + 0.5 * A.c2_attn.to_numpy())
EST = {"Vendor A AVM (champion)": "v_vendor_a_avm", "Vendor B value": "v_vendor_b", "Vendor A market value": "v_vendor_a_mktval",
       "Assessor 2025": "v_assessor_2025", "B0 index AVM (prior sale x HPI)": "$b0_index", "B1 hedonic OLS": "$b1_hedonic",
       "C1 LightGBM": "$c1_gbm", "C2 neighbour-attention NN": "$c2_attn", "C3 blend (C1+C2)": "$c3_blend"}
versions = {"as published (all 50 states)": np.ones(len(S), bool),
            "excluding the 12-state non-disclosure list": ~S.state.isin(p2.NON_DISCLOSURE).to_numpy(),
            "excluding the states that fail the intended screen": ~S.state.isin(fail).to_numpy()}
res = []
for sp in ("oot", "spatial"):
    for v, keep in versions.items():
        d = S[(S.split == sp).to_numpy() & keep]
        t = table(d, EST); t.insert(0, "version", v); t.insert(0, "split", sp)
        res.append(t)
res = pd.concat(res, ignore_index=True)
res.to_csv(OUT / "r5_accuracy_by_screen_version.csv", index=False)
print(res[res.estimator.isin(["C3 blend (C1+C2)", "Vendor A market value"])][["split", "version", "estimator", "n", "coverage", "MdAPE"]]
      .round(4).to_string(index=False))

# data counts quoted in the report that were printed but not exported
import pyarrow.parquet as pq
hd = pd.read_csv(REPORTS / "hpi_diagnostics.csv")
nat = hd[hd.level == "national"].set_index("version").n_pairs
counts = pd.DataFrame([
    {"item": "arm's-length resales in the sample (2022-05..2025-08)", "value": int(len(S))},
    {"item": "comparable-sale pool, arm's-length sales 2012-01..2022-04", "value": int(pq.ParquetFile(DW / "comp_pool.parquet").metadata.num_rows)},
    {"item": "repeat-sales pairs, full index (all states, all months; prepayment study)", "value": int(nat["full"])},
    {"item": "repeat-sales pairs, index used here (second sale <= 2024-06, hold-out states excluded)", "value": int(nat["train_xh"])},
])
counts.to_csv(OUT / "r5_avm_data_counts.csv", index=False)
print(counts.to_string(index=False))
