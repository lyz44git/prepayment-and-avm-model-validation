"""State-level profile of the national property extract: coverage of the fields both projects need."""
import sys; sys.path.insert(0, "scripts")
from _paths import WORK, REPORTS, duck
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100); pd.set_option("display.max_columns", 40)
con = duck()
con.execute(f"CREATE VIEW p AS SELECT * FROM read_parquet('{WORK}/props/*.parquet')")
tot = con.execute("SELECT count(*) n, count(DISTINCT prop_id) nprop, sum(n_veh) veh FROM p").df()
print(tot)
q = """SELECT state, count(*) n_addr,
 avg((in_vc_old AND in_vc_new)::int) both_v, avg(same_prop_id::int) same_prop_id,
 avg((ownocc0='O')::int) oo_old, avg((proptype0 IN ('10','11'))::int) res,
 avg((sd0 BETWEEN 19900101 AND 20220430)::int) has_sd0, avg((sa0>=10000)::int) has_sa0,
 avg((sd1>=20220501)::int) sale_in_window, avg((sd1>=20220501 AND sa1>=10000)::int) sale_w_amt,
 avg((mtgd IS NOT NULL)::int) has_mtg, avg((avm>0)::int) has_avm, avg((vb_val>0)::int) has_vb,
 avg((prop_mv>0)::int) has_pmv, avg((lat IS NOT NULL)::int) has_ll, avg((tract IS NOT NULL)::int) has_tract,
 avg((fc1 IS NOT NULL)::int) fc1
 FROM p GROUP BY 1 ORDER BY 1"""
df = con.execute(q).df()
print(df.round(3).to_string(index=False))
df.to_csv(REPORTS / "profile_national_by_state.csv", index=False)
print(con.execute("SELECT proptype0, count(*) FROM p GROUP BY 1 ORDER BY 2 DESC LIMIT 12").df().T)
print(con.execute("SELECT ownocc0, count(*) FROM p GROUP BY 1 ORDER BY 2 DESC").df().T)
print(con.execute("SELECT corp0, count(*) FROM p GROUP BY 1 ORDER BY 2 DESC").df().T)
print("recording lag (days) for window sales:")
print(con.execute("""SELECT quantile_cont(datediff('day', strptime(sd1::VARCHAR,'%Y%m%d'), strptime(srd1::VARCHAR,'%Y%m%d')), [0.5,0.9,0.95,0.99]) q
   FROM p WHERE sd1>=20220501 AND srd1>=sd1 AND sd1<99999999""").df())
print("monthly count of window sales (national):")
print(con.execute("SELECT sd1//100 ym, count(*) n FROM p WHERE sd1>=20220101 AND sd1<20251001 GROUP BY 1 ORDER BY 1").df().T.to_string())
print("Vendor A new first liens by month (no sale in window):")
print(con.execute("SELECT mtgd//100 ym, count(*) n FROM p WHERE mtgd>=20220101 AND mtgd<20251101 AND (sd1 IS NULL OR sd1<20220501) GROUP BY 1 ORDER BY 1").df().T.to_string())
