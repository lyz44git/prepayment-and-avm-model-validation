"""Credit-loss study: is the recorded loan term an origination attribute? (Finding behind deviation D1.)

Compares, by final status, the share of loans whose term is not a multiple of 12 and the share of round amounts and
standard guarantee shares. Cancelled loans were never disbursed and keep their approved terms, so they show what
the field looks like when nothing happened to the loan. Writes reports/p3/term_field_check.csv and
term_charge_off_share_by_term.csv.
"""
import sys; sys.path.insert(0, "scripts")
from _paths import REPORTS, duck

con = duck(threads=16, mem="60GB")
con.execute("""CREATE VIEW raw AS SELECT trim(LoanStatus) AS st, try_cast(TermInMonths AS DOUBLE) AS term,
  try_cast(GrossApproval AS DOUBLE) AS amt, try_cast(SBAGuaranteedApproval AS DOUBLE) AS gty, RevolverStatus
  FROM read_csv('data_external/sba/FOIA_7a_FY*_asof_260930.csv', all_varchar=true, header=true, union_by_name=true)""")
t = con.execute("""
SELECT CASE st WHEN 'CANCLD' THEN 'cancelled (never disbursed)' WHEN 'P I F' THEN 'paid in full' WHEN 'EXEMPT' THEN 'active'
               WHEN 'CHGOFF' THEN 'charged off' ELSE st END AS final_status,
  count(*) AS loans, avg((term % 12 <> 0)::INT) AS term_not_multiple_of_12, median(term) AS median_term,
  avg((amt % 1000 = 0)::INT) AS amount_multiple_of_1000,
  avg((abs(100 * gty / amt - round(100 * gty / amt)) < 0.01 AND round(100 * gty / amt) IN (50, 75, 85, 90))::INT) AS standard_guarantee_share
FROM raw WHERE RevolverStatus = 'N' AND st IN ('CANCLD','P I F','EXEMPT','CHGOFF') GROUP BY 1 ORDER BY 1""").df()
t.to_csv(REPORTS / "p3" / "term_field_check.csv", index=False)
u = con.execute(f"""SELECT term, count(*) AS loans, avg((status = 'co')::INT) AS charged_off_share
  FROM read_parquet('data_work/p3/loans.parquet') WHERE in_window GROUP BY 1 ORDER BY loans DESC LIMIT 15""").df()
u.to_csv(REPORTS / "p3" / "term_charge_off_share_by_term.csv", index=False)
print(t.round(3).to_string(index=False)); print(u.round(4).to_string(index=False))
