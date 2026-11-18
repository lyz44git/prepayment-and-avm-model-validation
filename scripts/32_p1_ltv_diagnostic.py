"""Why the house-price stress response stays small (finding F9): estimated-LTV distribution of the stress pool and the
empirical new-lien hazard by LTV band in the training months. Descriptive only; no model is chosen here."""
import sys; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import WORK, REPORTS
import numpy as np, pandas as pd, json
from mval import p1
P = pd.read_parquet(WORK / "p1" / "panel.parquet", columns=["t", "split", "est_ltv", "y_sale", "y_new_lien", "y_other_transfer"])
OUT = REPORTS / "p1" / "remediation"
bins = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 9]
tr = P[P.split == "train"].copy(); tr["band"] = pd.cut(tr.est_ltv, bins, right=False)
h = tr.groupby("band", observed=True).agg(rows=("y_new_lien", "size"), new_lien_hazard=("y_new_lien", "mean"), sale_hazard=("y_sale", "mean"))
h["share_of_rows"] = h.rows / h.rows.sum(); h.to_csv(OUT / "diag_hazard_by_ltv_band_train.csv")
last = P[(P.t == p1.T_END) & (P.y_sale == 0) & (P.y_new_lien == 0) & (P.y_other_transfer == 0)]
ltv = last.est_ltv.to_numpy()
pool = {"pool_properties": int(len(ltv)), "median_ltv": float(np.median(ltv)), "share_ltv_ge_0.8": float((ltv >= 0.8).mean()),
        "share_ltv_ge_0.8_after_minus10pct": float((ltv / 0.9 >= 0.8).mean()), "share_ltv_ge_0.8_after_minus20pct": float((ltv / 0.8 >= 0.8).mean()),
        "share_ltv_lt_0.5": float((ltv < 0.5).mean())}
json.dump(pool, open(OUT / "diag_stress_pool_ltv.json", "w"), indent=2)
print(h.round(5).to_string()); print(json.dumps(pool, indent=1))
