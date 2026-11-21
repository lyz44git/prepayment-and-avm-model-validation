"""Prepayment study universe: one row per owner-occupied residential property at risk on 2022-05-01.

Only pre-window (old Vendor C vintage, ~2022-04) property attributes and the purchase record are kept as
covariates, so event and non-event properties are described by the same, symmetric information. The Vendor A
current-lien fields (rate, term, loan type) are kept ONLY for the measurement-validation exercise on stayers.
"""
import sys; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import WORK, REPORTS, duck
from mval.sql import RES_SQL, arms_old, ym_to_idx
con = duck()
W0, W1 = 20220501, 20250701           # risk window [2022-05, 2025-06]
# True reproduces the published panel: a home whose 2025-vintage sale date is missing fails the "transferred just
# before the window" test below (NULL), so it is dropped (3,497 of 16,142,418 otherwise eligible homes, 188 of them
# with a new first lien in the window). False keeps those homes, as the new-lien rule intends.
LEGACY_NULL_SALE_DATE = True
PRE_WINDOW = "sd1 >= 20220101 AND sd1 < {W0} AND sd1 <> sd0".format(W0=W0)
PRE_WINDOW_SQL = f"NOT ({PRE_WINDOW})" if LEGACY_NULL_SALE_DATE else f"NOT COALESCE({PRE_WINDOW}, FALSE)"
q = f"""
COPY (
WITH base AS (
  SELECT *, {ym_to_idx('sd0')} AS t_purchase FROM read_parquet('{WORK}/props/*.parquet')
  WHERE in_vc_old AND in_vc_new AND same_prop_id AND {RES_SQL} AND ownocc0 = 'O' AND corp0 IS NULL
    AND {arms_old()} AND sd0 >= 19900101 AND sd0 < {W0} AND (sd0 % 100) > 0 AND state <> 'VT'
    AND {PRE_WINDOW_SQL}     -- transferred just before the window: ambiguous
    AND NOT (sd1 IS NOT NULL AND sd1 < sd0)                     -- inconsistent vintages
),
ev AS (
  SELECT *,
    CASE WHEN sd1 >= {W0} AND sd1 < {W1} AND sd1 <> sd0 AND tt1 = '1' AND COALESCE(doc1,'') NOT IN ('QC','TD','AF','GF') THEN 'sale'
         WHEN sd1 >= {W0} AND sd1 < {W1} AND sd1 <> sd0 THEN 'other_transfer'
         WHEN (sd1 IS NULL OR sd1 < {W0} OR sd1 >= {W1}) AND mtgd >= {W0} AND mtgd < {W1} AND (mtgd % 100) > 0 THEN 'new_lien'
         ELSE 'none' END AS event
  FROM base)
SELECT addr_id, prop_id, state, fips5, cbsa, tract, lat, lon,
  t_purchase, sa0 AS price0, mv0, av0, cv0, tax0, taxyr0,
  COALESCE(yrblt0, yrblt_c) AS yr_built, COALESCE(sqft0, sqft_c) AS sqft, COALESCE(beds0, beds_c) AS beds,
  COALESCE(baths0, baths_c) AS baths, COALESCE(lot0, lot_c) AS lot_sqft, proptype0, units0, stories0,
  (COALESCE(pool0, pool_c) IS NOT NULL AND COALESCE(pool0, pool_c) NOT IN ('N','0')) AS pool,
  t_density, t_owned, t_medage, t_black, t_hisp,
  event,
  CASE WHEN event IN ('sale','other_transfer') THEN {ym_to_idx('sd1')}
       WHEN event = 'new_lien' THEN {ym_to_idx('mtgd')} END AS t_event,
  owner_changed, doc1,
  -- validation-only (asymmetric) Vendor A current-lien fields
  mtgd, mtgr, mtgterm, mtgloan, mtgrtype,
  (hash(addr_id) % 1000000) / 1000000.0 AS u
FROM ev) TO '{WORK}/p1_universe.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)"""
con.execute(q)
u = f"read_parquet('{WORK}/p1_universe.parquet')"
print(con.execute(f"SELECT count(*) n, count(DISTINCT prop_id) props FROM {u}").df())
s = con.execute(f"""SELECT event, count(*) n, round(100.0*count(*)/sum(count(*)) OVER (),2) pct,
   avg(owner_changed::int) owner_changed FROM {u} GROUP BY 1 ORDER BY 2 DESC""").df()
print(s.to_string(index=False)); s.to_csv(REPORTS / "p1" / "p1_event_counts.csv", index=False)
bys = con.execute(f"""SELECT state, count(*) n, avg((event='sale')::int) sale, avg((event='new_lien')::int) new_lien,
   avg((event='other_transfer')::int) other FROM {u} GROUP BY 1 ORDER BY 2 DESC""").df()
bys.to_csv(REPORTS / "p1" / "p1_event_rates_by_state.csv", index=False)
print(bys.head(12).round(4).to_string(index=False)); print('...'); print(bys.tail(6).round(4).to_string(index=False))
