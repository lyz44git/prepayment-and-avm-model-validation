"""Repeat-sales (Bailey-Muth-Nourse) monthly house price index by state, built from the data itself.

Pairs: last arm's-length sale in the old Vendor C vintage (<= 2022-04) -> arm's-length resale in the new
vintage (2022-05..2025-08), same parcel, residential. Estimator: OLS on log price relatives with month
dummies (+1 second sale, -1 first sale) plus a light random-walk smoothness penalty, solved with sparse LSQR.
Two versions: 'full' (all pairs) and 'train' (second sale <= 2024-06) used by the valuation study to avoid look-ahead.
"""
import sys; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import WORK, REPORTS, duck
from mval.sql import arms_new, arms_old, RES_SQL, ym_to_idx
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import lsqr

T = (2025 - 1990) * 12 + 8   # months 1990-01 .. 2025-08
con = duck()
pairs = con.execute(f"""
  SELECT state, {ym_to_idx('sd0')} AS t1, {ym_to_idx('sd1')} AS t2, ln(sa1) - ln(sa0) AS dy
  FROM read_parquet('{WORK}/props/*.parquet')
  WHERE in_vc_old AND in_vc_new AND same_prop_id AND {RES_SQL} AND {arms_old()} AND {arms_new()}
    AND sd0 >= 19900101 AND sd0 < 20220501 AND sd1 >= 20220501 AND sd1 < 20250900 AND (sd1 % 100) > 0 AND (sd0 % 100) > 0
""").df()
pairs = pairs[(pairs.t2 - pairs.t1) >= 6]
yrs = (pairs.t2 - pairs.t1) / 12
ann = pairs.dy / yrs
keep = ann.between(-0.30, 0.40) & pairs.dy.abs().lt(np.log(6))
print(f"pairs {len(pairs):,} -> kept {keep.sum():,} after outlier screen")
pairs = pairs[keep]

def bmn(df, lam=3.0):
    n = len(df); rows = np.arange(n)
    A = sp.csr_matrix((np.r_[np.ones(n), -np.ones(n)], (np.r_[rows, rows], np.r_[df.t2.values, df.t1.values])), shape=(n, T))
    D = sp.diags([-np.ones(T - 1), np.ones(T - 1)], [0, 1], shape=(T - 1, T)) * lam   # smoothness rows
    M = sp.vstack([A, D]).tocsr()
    b = np.r_[df.dy.values, np.zeros(T - 1)]
    beta = lsqr(M, b, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
    resid = df.dy.values - A @ beta
    return beta, resid

DIV = {"New England": "CT ME MA NH RI VT", "Middle Atlantic": "NJ NY PA", "East North Central": "IL IN MI OH WI",
       "West North Central": "IA KS MN MO NE ND SD", "South Atlantic": "DE DC FL GA MD NC SC VA WV",
       "East South Central": "AL KY MS TN", "West South Central": "AR LA OK TX",
       "Mountain": "AZ CO ID MT NV NM UT WY", "Pacific": "AK CA HI OR WA"}
st2div = {st: d for d, sts in DIV.items() for st in sts.split()}
pairs["division"] = pairs.state.map(st2div)
groups = [("state", st, g) for st, g in pairs.groupby("state")] + \
         [("division", d, g) for d, g in pairs.groupby("division")] + [("national", "US", pairs)]
out, diag = [], []
HOLD = ["MA", "NJ", "WI", "MN", "SC", "KY", "OK", "AZ", "WA"]     # spatial hold-out states (Projects 1 and 2)
for version in ["full", "train", "train_xh"]:
    for level, st, g in groups:
        if version in ("train", "train_xh"):
            g = g[g.t2 <= (2024 - 1990) * 12 + 5]
        if version == "train_xh":                     # strict spatial test for the valuation study: hold-out states contribute nothing
            if level == "state" and st in HOLD:
                continue
            g = g[~g.state.isin(HOLD)]
        if len(g) < 3000:
            continue
        beta, resid = bmn(g)
        last = g.t2.max()
        beta = beta - beta[(2000 - 1990) * 12]          # normalise: 2000-01 = 0 (log)
        if version in ("train", "train_xh"):             # freeze after the last observed month (no look-ahead)
            beta[last + 1:] = beta[last]
        months = pd.date_range("1990-01-01", periods=T, freq="MS")
        out.append(pd.DataFrame({"version": version, "level": level, "state": st, "month": months, "log_hpi": beta}))
        diag.append({"version": version, "level": level, "state": st, "n_pairs": len(g), "resid_sd": resid.std(),
                     "first_sale_median": int(g.t1.median()), "last_month_observed": str(months[last].date())})
hpi = pd.concat(out); diag = pd.DataFrame(diag)
hpi.to_parquet(WORK / "hpi_state_monthly.parquet", index=False)
diag.to_csv(REPORTS / "hpi_diagnostics.csv", index=False)
print(diag[(diag.version == "full") & (diag.level != "state")].to_string(index=False))
print("indices:", diag.groupby(["version", "level"]).state.nunique().to_dict())
# lookup table state -> index used (own state if >= 3000 pairs else its census division)
own = set(diag[(diag.level == "state") & (diag.version == "train")].state)
lk = pd.DataFrame({"state": sorted(st2div)}); lk["division"] = lk.state.map(st2div)
lk["hpi_key"] = np.where(lk.state.isin(own), lk.state, lk.division)
lk.to_csv(WORK / "hpi_lookup.csv", index=False); print(lk[~lk.state.isin(own)].to_string(index=False))
own_xh = set(diag[(diag.level == "state") & (diag.version == "train_xh")].state)
lk["hpi_key"] = np.where(lk.state.isin(own_xh), lk.state, lk.division)
lk.to_csv(WORK / "hpi_lookup_xh.csv", index=False)
