"""Valuation study extension: build the PSI reference for the scoring service and back-fill monitoring results.

Reference = the training window (sales <= 2024-03). Stored as bin edges and shares only (no records):
service/reference/psi_reference.json.

Back-fill: run the service's own scoring and PSI code on each out-of-time month (2024-07..2025-08) and on the
9 hold-out states, and save reports/p2/psi_backfill.csv, as a monitoring team would have seen it.
"""
import sys, json, functools, time; sys.path.insert(0, "scripts"); sys.path.insert(0, "src"); sys.path.insert(0, ".")
from _paths import WORK, REPORTS
import numpy as np, pandas as pd
from service import app as svc
from service.monitoring import numeric_reference, categorical_reference, psi_table
print = functools.partial(print, flush=True)
OUT = REPORTS / "p2"; t0 = time.time()

S = pd.read_parquet(WORK / "p2" / "features.parquet")
S["state_code"] = S.state.astype("category").cat.codes
S["moy"] = S.t_sale % 12 + 1
codes = {s: int(c) for c, s in enumerate(S.state.astype("category").cat.categories)}
training_states = sorted(S.loc[S.split == "train", "state"].unique())
X = S[svc.FEATURES].copy()
X.loc[S.split == "spatial", "state_code"] = -1                  # unseen-state treatment, as offline
pred = svc.model().predict(X)

# parity with the offline predictions saved by scripts/13
off = pd.read_parquet(WORK / "p2" / "preds_gbm.parquet", columns=["addr_id", "c1_gbm"])
assert (off.addr_id.values == S.addr_id.values).all()
max_diff = float(np.abs(pred - off.c1_gbm.to_numpy()).max())
print(f"parity with offline predictions on {len(S):,} rows: max |diff| = {max_diff:.2e}")
assert max_diff < 1e-9

CATEGORICAL = {"state_code", "moy", "pool", "condo"}
tr = S.split == "train"
ref = {"window": "training sales <= 2024-03", "rows": int(tr.sum()), "state_codes": codes,
       "training_states": training_states, "columns": {}}
for c in svc.FEATURES:
    ref["columns"][c] = categorical_reference(X.loc[tr, c]) if c in CATEGORICAL else numeric_reference(X.loc[tr, c])
ref["columns"]["prediction"] = numeric_reference(pd.Series(pred[tr.to_numpy()]))
ref["expected_shift"] = {
    "moy": "sale month: a single-month batch never matches the 12-month training mix",
    "mkt_idx": "market index level rises with calendar time by design",
    "months_since_prev": "time since the prior sale grows with the sale date",
    "state_code": "new geographies map to the unseen-state code by design",
    "lat": "new geographies shift location by design", "lon": "new geographies shift location by design"}
(svc.HERE / "reference").mkdir(exist_ok=True)
json.dump(ref, open(svc.HERE / "reference" / "psi_reference.json", "w"), indent=1)
print("reference saved:", len(ref["columns"]), "columns")
svc.reference.cache_clear()

# back-fill
F = X.copy(); F["prediction"] = pred
month = (S.t_sale // 12 + 1990).astype(str) + "-" + (S.t_sale % 12 + 1).astype(str).str.zfill(2)
rows = []
batches = [(m, "oot", (S.split == "oot") & (month == m)) for m in sorted(month[S.split == "oot"].unique())]
batches.append(("all months", "spatial hold-out states", S.split == "spatial"))
batches.append(("2024-04..06", "calibration window", S.split == "calib"))
for label, split, mask in batches:
    t = psi_table(F[mask.to_numpy()], ref)
    for r in t.itertuples():
        rows.append({"batch": label, "split": split, "records": int(mask.sum()), "column": r.column, "psi": r.psi, "status": r.status})
B = pd.DataFrame(rows)
B.to_csv(OUT / "psi_backfill.csv", index=False)
wide = B.pivot_table(index=["split", "batch"], columns="column", values="psi").round(3)
EXP = set(ref["expected_shift"])
def summarise(g):
    u = g[(g.column != "prediction") & ~g.column.isin(EXP)].sort_values("psi")
    return pd.Series({"records": g.records.iloc[0], "prediction_psi": g.loc[g.column == "prediction", "psi"].iloc[0],
                      "expected_shift_max_psi": g.loc[g.column.isin(EXP), "psi"].max(),
                      "max_unexpected_psi": u.psi.iloc[-1], "top_unexpected": u.column.iloc[-1],
                      "n_investigate_unexpected": int((u.status == "investigate").sum()),
                      "n_watch_unexpected": int((u.status == "watch").sum())})
summary = B.groupby(["split", "batch"])[["column", "psi", "status", "records"]].apply(summarise).reset_index()
summary.to_csv(OUT / "psi_backfill_summary.csv", index=False)
pd.set_option("display.width", 220)
print(summary.round(3).to_string(index=False))
print(f"done {time.time()-t0:.0f}s")
