"""Prepayment study: recompute every monthly-hazard metric from one set of final predictions (2026-11-19).

09_p1_analysis.py replaced the spatial-split M1 predictions (unseen-state fix) but rewrote only AUC and predicted ÷
actual in eval_monthly_hazard_final.csv, leaving Brier score and log loss from the naive predictions. This script
recomputes all metrics, for every cause, split and model, from data_work/p1/preds_v2.parquet (the predictions 09
saved after the fix), with the same formulas as 08, and records which cells changed. No model is refitted.
Writes reports/p1/eval_monthly_hazard_final.csv and reports/p1/r5_eval_reconciliation.csv.
"""
import sys; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from _paths import WORK, REPORTS
from mval.metrics import auc, brier

OUT = REPORTS / "p1"
preds = pd.read_parquet(WORK / "p1" / "preds_v2.parquet")
old = pd.read_csv(OUT / "eval_monthly_hazard_final.csv")
rows = []
for c in ["sale", "new_lien"]:
    for sp in ["train", "oot", "spatial", "spatial_oot"]:
        g = preds[preds.split == sp]; y = g[f"y_{c}"].to_numpy()
        for m in ["m0", "m1", "m2"]:
            s = g[f"{m}_{c}"].to_numpy()
            rows.append({"cause": c, "split": sp, "model": m, "rows": len(g), "events": int(y.sum()),
                         "auc": auc(y, s), "brier_x1e4": brier(y, s) * 1e4,
                         "logloss": float(-np.mean(y * np.log(s) + (1 - y) * np.log(1 - s))),
                         "pred_over_actual": float(s.mean() / y.mean())})
new = pd.DataFrame(rows)
key = ["cause", "split", "model"]
m = old.merge(new, on=key, suffixes=("_published", "_recomputed"))
diff = []
for col in ["rows", "events", "auc", "brier_x1e4", "logloss", "pred_over_actual"]:
    a, b = m[f"{col}_published"].astype(float), m[f"{col}_recomputed"].astype(float)
    ch = ~np.isclose(a, b, rtol=1e-9, atol=1e-12)
    for i in np.where(ch)[0]:
        diff.append({**m.loc[i, key].to_dict(), "metric": col, "published": a[i], "recomputed": b[i]})
diff = pd.DataFrame(diff, columns=key + ["metric", "published", "recomputed"])
diff.to_csv(OUT / "r5_eval_reconciliation.csv", index=False)
new.to_csv(OUT / "eval_monthly_hazard_final.csv", index=False)
print(diff.to_string(index=False) if len(diff) else "no cell changed")
