"""Valuation study features: prior-sale index value, assessor values, property attributes, k-NN comparables
(from pre-2022-05 sales only), and the comp tensor for the neighbour-attention network."""
import sys, time; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import WORK
from mval import p2, p1
import numpy as np, pandas as pd
t0 = time.time(); DW = WORK / "p2"
S = pd.read_parquet(DW / "sales.parquet"); pool = pd.read_parquet(DW / "comp_pool.parquet")
S = S[S.lat.notna()].reset_index(drop=True)
st_idx, H = p1.hpi_matrix(pd.read_parquet(WORK / "hpi_state_monthly.parquet"), pd.read_csv(WORK / "hpi_lookup_xh.csv"), "train_xh")
def lh(states, t): return H[states.map(st_idx).to_numpy(), np.asarray(t, int)]
S["y"] = np.log(S.price)
S["mkt_idx"] = lh(S.state, S.t_sale) - lh(S.state, np.full(len(S), p2.T_REF))          # frozen after 2024-06
has = S.t_prev.notna()
S["prev_idx_logp"] = np.nan
S.loc[has, "prev_idx_logp"] = np.log(S.price_prev[has]) + lh(S.state[has], S.t_sale[has]) - lh(S.state[has], S.t_prev[has].astype(int))
S["months_since_prev"] = S.t_sale - S.t_prev
pool["logp_ref"] = np.log(pool.price_prev) + lh(pool.state, np.full(len(pool), p2.T_REF)) - lh(pool.state, pool.t_prev)
pool["log_sqft"] = np.log(pool.sqft.where(pool.sqft.between(300, 20000)))
pool["age"] = (2022 - pool.yr_built).where(pool.yr_built.between(1800, 2022))
pool["lppsf"] = pool.logp_ref - pool.log_sqft
K = p2.K_DL
comp_feat = np.full((len(S), K, 10), np.nan, dtype=np.float32)
agg = {c: np.full(len(S), np.nan) for c in ["comp_med5", "comp_med20", "comp_iqr20", "comp_lppsf_med20", "comp_dist1",
                                            "comp_dist20", "comp_lsqft_med20", "comp_age_med20", "comp_n_1km"]}
for st, g in S.groupby("state"):
    pl = pool[pool.state == st].reset_index(drop=True)
    if len(pl) < K + 5:
        continue
    idx, dist = p2.knn_comps(g, pl, K)
    rows = g.index.to_numpy()
    lp = pl.logp_ref.to_numpy()[idx]; lpsf = pl.lppsf.to_numpy()[idx]
    agg["comp_med5"][rows] = np.nanmedian(lp[:, :5], 1); agg["comp_med20"][rows] = np.nanmedian(lp[:, :20], 1)
    agg["comp_iqr20"][rows] = np.nanpercentile(lp[:, :20], 75, 1) - np.nanpercentile(lp[:, :20], 25, 1)
    agg["comp_lppsf_med20"][rows] = np.nanmedian(lpsf[:, :20], 1)
    agg["comp_dist1"][rows] = dist[:, 0]; agg["comp_dist20"][rows] = dist[:, 19]
    agg["comp_lsqft_med20"][rows] = np.nanmedian(pl.log_sqft.to_numpy()[idx][:, :20], 1)
    agg["comp_age_med20"][rows] = np.nanmedian(pl.age.to_numpy()[idx][:, :20], 1)
    agg["comp_n_1km"][rows] = (dist < 1.0).sum(1)
    lat0 = np.radians(g.lat.to_numpy())[:, None]
    dx = (pl.lon.to_numpy()[idx] - g.lon.to_numpy()[:, None]) * np.cos(lat0) * 111.32
    dy = (pl.lat.to_numpy()[idx] - g.lat.to_numpy()[:, None]) * 110.57
    comp_feat[rows] = np.stack([lp, pl.log_sqft.to_numpy()[idx], pl.age.to_numpy()[idx], pl.beds.to_numpy()[idx],
                                pl.baths.to_numpy()[idx], pl.condo.to_numpy()[idx].astype(float), dx, dy, dist,
                                (p2.T_REF - pl.t_prev.to_numpy()[idx]) / 12.0], -1).astype(np.float32)
    print(st, len(g), f"{time.time()-t0:.0f}s", flush=True)
for c, v in agg.items():
    S[c] = v
S["comp_implied"] = S.comp_lppsf_med20 + np.log(S.sqft.where(S.sqft.between(300, 20000)))
S["age"] = (2022 - S.yr_built).where(S.yr_built.between(1800, 2022))
S["log_sqft"] = np.log(S.sqft.where(S.sqft.between(300, 20000)))
S["log_lot"] = np.log1p(S.lot_sqft.where(S.lot_sqft.between(0, 5e7)))
for c in ["mv0", "av0", "cv0", "tax0"]:
    S[f"log_{c}"] = np.log(S[c].where(S[c] > 1000))
S["condo"] = (S.proptype0 == "11").astype(int); S["pool"] = S.pool.astype(int)
S["log_density"] = np.log1p(S.t_density)
S["split"] = np.select([S.state.isin(p2.HOLDOUT_STATES), S.t_sale <= p2.TRAIN_END, S.t_sale <= p2.CALIB_END],
                       ["spatial", "train", "calib"], default="oot")
S.to_parquet(DW / "features.parquet", index=False)
np.save(DW / "comp_tensor.npy", comp_feat)
print(S.split.value_counts().to_dict(), f"done {time.time()-t0:.0f}s")
