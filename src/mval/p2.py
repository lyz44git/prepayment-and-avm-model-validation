"""Valuation study helpers: time indexing, splits and k-nearest prior-sale comparables."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

T_REF = (2022 - 1990) * 12 + 3       # 2022-04: last month of the comp pool / old vintage
TRAIN_END = (2024 - 1990) * 12 + 2   # 2024-03 last training month
CALIB_END = (2024 - 1990) * 12 + 5   # 2024-04..2024-06 calibration (early stopping, conformal)
# OOT test: 2024-07..2025-08. Spatial holdout: one state per census division among disclosure states.
HOLDOUT_STATES = ["MA", "NJ", "WI", "MN", "SC", "KY", "OK", "AZ", "WA"]
NON_DISCLOSURE = ["TX", "UT", "ID", "KS", "MO", "MS", "MT", "NM", "ND", "WY", "LA", "AK"]
K_DL = 32
R_EARTH = 6371.0

def unit_xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])

def knn_comps(tgt: pd.DataFrame, pool: pd.DataFrame, k: int = K_DL):
    """For each target row, the k nearest pool sales in the same state, excluding the target parcel itself.
    Returns (idx [n,k] into pool, dist_km [n,k])."""
    tree = cKDTree(unit_xyz(pool.lat.to_numpy(), pool.lon.to_numpy()))
    d, i = tree.query(unit_xyz(tgt.lat.to_numpy(), tgt.lon.to_numpy()), k=k + 3, workers=-1)
    pool_addr = pool.addr_id.to_numpy(); pool_pid = pool.prop_id.to_numpy()
    same = (pool_addr[i] == tgt.addr_id.to_numpy()[:, None]) | (pool_pid[i] == tgt.prop_id.to_numpy()[:, None])
    d = np.where(same, np.inf, d)
    order = np.argsort(d, axis=1, kind="stable")[:, :k]
    i = np.take_along_axis(i, order, 1); d = np.take_along_axis(d, order, 1)
    return i, 2 * R_EARTH * np.arcsin(np.clip(d / 2, 0, 1))      # chord -> great-circle km
