"""Valuation study ground truth + vendor values: arm's-length resales 2022-05..2025-08 in price-disclosure states.

Disclosure screen: a state is kept when >= 90% of its window resales carry a sale amount (non-disclosure
states such as TX/UT/KS/MO record few prices and the ones recorded are a selected subsample).
Every feature is dated at or before 2022-04 (old Vendor C vintage) except the market index, which is the
train-only repeat-sales HPI (frozen after 2024-06). Vendor values are as of 2025 and are evaluated, never used as features.
"""
import sys; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import WORK, REPORTS, duck
from mval.sql import RES_SQL, arms_new, arms_old, ARMS_SQL, ym_to_idx
con = duck(threads=12)
P = f"read_parquet('{WORK}/props/*.parquet')"
cov = con.execute(f"""SELECT state, count(*) n_resale, avg((sa1 >= 30000)::int) amt_share FROM {P}
   WHERE {RES_SQL} AND in_vc_new AND sd1 >= 20220501 AND sd1 < 20250900 AND tt1='1' AND doc1 IN {ARMS_SQL}
   GROUP BY 1 ORDER BY 3""").df()
cov["disclosure"] = cov.amt_share >= 0.90
cov.to_csv(REPORTS / "p2" / "disclosure_screen.csv", index=False)
keep = cov[cov.disclosure].state.tolist()
print("excluded (non-disclosure):", cov[~cov.disclosure][["state", "amt_share"]].round(2).values.tolist())
st_list = "(" + ",".join(f"'{s}'" for s in keep) + ")"
con.execute(f"""COPY (
 SELECT addr_id, prop_id, state, fips5, cbsa, tract, lat, lon,
   {ym_to_idx('sd1')} AS t_sale, sa1 AS price,
   CASE WHEN {arms_old()} AND sd0 >= 19900101 AND sd0 < 20220501 AND (sd0 % 100) > 0 THEN {ym_to_idx('sd0')} END AS t_prev,
   CASE WHEN {arms_old()} AND sd0 >= 19900101 AND sd0 < 20220501 THEN sa0 END AS price_prev,
   mv0, av0, cv0, tax0, COALESCE(yrblt0, yrblt_c) yr_built, COALESCE(sqft0, sqft_c) sqft, COALESCE(beds0, beds_c) beds,
   COALESCE(baths0, baths_c) baths, COALESCE(lot0, lot_c) lot_sqft, proptype0, units0, stories0, qual0,
   (COALESCE(pool0, pool_c) IS NOT NULL AND COALESCE(pool0, pool_c) NOT IN ('N','0')) AS pool,
   t_density, t_owned, t_medage, t_black, t_hisp, t_white, t_asian, ownocc0,
   -- vendor / third-party values (as of 2025) - evaluated only
   avm AS v_vendor_a_avm, avm_err AS v_vendor_a_avm_err, vb_val AS v_vendor_b, prop_mv AS v_vendor_a_mktval,
   cv1 AS v_assessor_2025, mrkthomeval_band AS v_band, heq AS v_heq,
   (hash(addr_id) % 1000000) / 1000000.0 AS u
 FROM {P}
 WHERE {RES_SQL} AND in_vc_old AND in_vc_new AND same_prop_id AND state IN {st_list}
   AND sd1 >= 20220501 AND sd1 < 20250900 AND (sd1 % 100) > 0 AND {arms_new()} AND sd1 IS DISTINCT FROM sd0
) TO '{WORK}/p2/sales.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)""")
s = f"read_parquet('{WORK}/p2/sales.parquet')"
print(con.execute(f"""SELECT count(*) n, count(DISTINCT state) states, avg((price_prev IS NOT NULL)::int) has_prev,
   median(price) med_price, avg((lat IS NOT NULL)::int) has_ll, avg((v_vendor_a_avm>0)::int) avm_cov FROM {s}""").df().T)
# comp pool: every residential property's last arm's-length sale 2012-01..2022-04 (pre-window), same states
con.execute(f"""COPY (
 SELECT addr_id, prop_id, state, lat, lon, {ym_to_idx('sd0')} AS t_prev, sa0 AS price_prev,
   COALESCE(sqft0, sqft_c) sqft, COALESCE(yrblt0, yrblt_c) yr_built, COALESCE(beds0, beds_c) beds, COALESCE(baths0, baths_c) baths,
   (proptype0 = '11') AS condo
 FROM {P} WHERE {RES_SQL} AND in_vc_old AND state IN {st_list} AND {arms_old()} AND sd0 >= 20120101 AND sd0 < 20220501
   AND (sd0 % 100) > 0 AND lat IS NOT NULL
) TO '{WORK}/p2/comp_pool.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)""")
print(con.execute(f"SELECT count(*) pool FROM read_parquet('{WORK}/p2/comp_pool.parquet')").df())
