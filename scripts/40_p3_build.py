"""Credit-loss study, step 1: build a de-identified loan table from the SBA 7(a) FOIA files.

Reads data_external/sba/FOIA_7a_*_asof_260930.csv (public, U.S. Government Works). Writes data_work/p3/loans.parquet
(no borrower names, addresses or lender names) and reports/p3/data_profile.json (counts at each filter, field
coverage by approval era). Plan: docs/credit_loss_plan.md.
"""
import sys, json; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import WORK, REPORTS, duck
from mval.p3 import STATES51, W0, W1

OUT = WORK / "p3"; OUT.mkdir(parents=True, exist_ok=True)
(REPORTS / "p3").mkdir(parents=True, exist_ok=True)
con = duck(threads=24, mem="120GB")
src = "data_external/sba/FOIA_7a_FY*_asof_260930.csv"
con.execute(f"CREATE TABLE raw AS SELECT * EXCLUDE (filename) FROM read_csv('{src}', all_varchar=true, header=true, union_by_name=true, filename=true)")
prof = {"rows_all_files": con.execute("SELECT count(*) FROM raw").fetchone()[0]}
con.execute("CREATE TABLE dd AS SELECT DISTINCT * FROM raw")
prof["rows_after_exact_duplicates"] = con.execute("SELECT count(*) FROM dd").fetchone()[0]

states = ",".join(f"'{s}'" for s in STATES51)
con.execute(f"""
CREATE TABLE loans AS
WITH b AS (
  SELECT *, trim(LoanStatus) AS st,
    try_cast(FirstDisbursementDate AS DATE) AS disb, try_cast(PaidInFullDate AS DATE) AS pif,
    least(try_cast(ChargeOffDate AS DATE), DATE '2026-09-30') AS co,
    try_cast(TermInMonths AS DOUBLE)::INT AS term, try_cast(GrossApproval AS DOUBLE) AS gross_approval,
    try_cast(SBAGuaranteedApproval AS DOUBLE) AS sba_gty, try_cast(GrossChargeOffAmount AS DOUBLE) AS co_amount,
    try_cast(InitialInterestRate AS DOUBLE) AS rate, try_cast(ApprovalFY AS INT) AS approval_fy,
    try_cast(ApprovalDate AS DATE) AS approval
  FROM dd
)
SELECT row_number() OVER (ORDER BY approval, gross_approval, term, ProjectState, md5(concat(BorrName, BorrStreet))) AS loan_id,
  approval_fy, (year(approval) - 2000) * 12 + month(approval) - 1 AS approval_m,
  (year(disb) - 2000) * 12 + month(disb) - 1 AS disb_m, term, gross_approval, sba_gty,
  least(greatest(sba_gty / gross_approval, 0), 1) AS gty_share,
  CASE WHEN ProcessingMethod IN ('SBA Express Program','Community Express','Patriot Express Loans','Export Express') THEN 'express'
       WHEN ProcessingMethod IN ('Preferred Lenders Program','Preferred Lenders with EWCP') THEN 'preferred'
       WHEN ProcessingMethod IN ('7a General','Certified Lenders Program','7a with EWCP') THEN 'standard'
       WHEN ProcessingMethod = 'Low Documentation Program' THEN 'lowdoc' ELSE 'other' END AS proc,
  CASE WHEN BusinessAge = 'Startup, Loan Funds will Open Business' THEN 'startup'
       WHEN BusinessAge IN ('New, Less than 1 Year old','New Business or 2 years or less') THEN 'new'
       WHEN BusinessAge IN ('Existing or more than 2 years old','Existing, 5 or more years','Less than 3 years old but at least 2',
                            'Less than 4 years old but at least 3','Less than 5 years old but at least 4','Change of Ownership') THEN 'existing'
       ELSE 'unknown' END AS bage,
  CASE WHEN upper(BusinessType) LIKE 'INDIV%' THEN 'individual' WHEN upper(BusinessType) LIKE 'PARTN%' THEN 'partnership' ELSE 'corporation' END AS btype,
  CASE WHEN NaicsCode IS NULL OR length(trim(NaicsCode)) < 2 THEN 'na'
       WHEN substr(NaicsCode,1,2) IN ('31','32','33') THEN '31' WHEN substr(NaicsCode,1,2) IN ('44','45') THEN '44'
       WHEN substr(NaicsCode,1,2) IN ('48','49') THEN '48' ELSE substr(NaicsCode,1,2) END AS naics2,
  (FranchiseCode IS NOT NULL AND regexp_matches(FranchiseCode, '[1-9]')) AS franchise,
  ProjectState AS state,
  CASE st WHEN 'P I F' THEN 'pif' WHEN 'CHGOFF' THEN 'co' ELSE 'active' END AS status,
  CASE WHEN st = 'P I F' THEN (year(pif) - 2000) * 12 + month(pif) - 1 END AS pif_m,
  CASE WHEN st = 'CHGOFF' THEN (year(co) - 2000) * 12 + month(co) - 1 END AS co_m,
  CASE WHEN st = 'CHGOFF' THEN co_amount END AS co_amount,
  rate, FixedorVariableInterestInd AS fv, CollateralInd AS collateral, SoldSecMrktInd AS sold_sec
FROM b
WHERE RevolverStatus = 'N' AND st IN ('P I F','CHGOFF','EXEMPT') AND disb IS NOT NULL AND term >= 6
  AND gross_approval > 0 AND ProjectState IN ({states})
  AND NOT (st = 'P I F' AND pif IS NULL) AND NOT (st = 'CHGOFF' AND co IS NULL)
""")
steps = {
    "term_loans": "SELECT count(*) FROM dd WHERE RevolverStatus = 'N'",
    "final_status_pif_co_active": "SELECT count(*) FROM dd WHERE RevolverStatus = 'N' AND trim(LoanStatus) IN ('P I F','CHGOFF','EXEMPT')",
    "population": "SELECT count(*) FROM loans",
}
for k, s in steps.items():
    prof[k] = con.execute(s).fetchone()[0]
con.execute(f"""ALTER TABLE loans ADD COLUMN end_m INT;
UPDATE loans SET end_m = coalesce(pif_m, co_m, {W1});
ALTER TABLE loans ADD COLUMN in_window BOOLEAN;
UPDATE loans SET in_window = (end_m >= {W0}) AND (disb_m < {W1}) AND (end_m > disb_m);""")
prof["in_risk_window"] = con.execute("SELECT count(*) FROM loans WHERE in_window").fetchone()[0]
prof["event_before_or_at_disbursement_month"] = con.execute("SELECT count(*) FROM loans WHERE end_m <= disb_m AND status <> 'active'").fetchone()[0]
prof["by_status_in_window"] = dict(con.execute("SELECT status, count(*) FROM loans WHERE in_window GROUP BY 1").fetchall())
prof["coverage_by_era"] = con.execute("""
  SELECT CASE WHEN approval_fy < 2010 THEN 'FY2000-09' WHEN approval_fy < 2020 THEN 'FY2010-19' ELSE 'FY2020-26' END AS era,
    count(*) AS loans, round(avg((rate IS NULL)::INT), 3) AS rate_missing, round(avg((collateral = 'Y')::INT), 3) AS collateral_y,
    round(avg((bage = 'unknown')::INT), 3) AS business_age_unknown, round(avg((naics2 = 'na')::INT), 3) AS naics_missing,
    round(median(gross_approval)) AS median_amount, round(avg(gty_share), 3) AS mean_guarantee_share
  FROM loans GROUP BY 1 ORDER BY 1""").df().to_dict(orient="records")
prof["payoff_dates_2005_05"] = con.execute("SELECT count(*) FROM loans WHERE pif_m = 64").fetchone()[0]
prof["payoff_dates_per_month_2005_08_to_2006_12"] = round(con.execute("SELECT count(*) / 17.0 FROM loans WHERE pif_m BETWEEN 67 AND 83").fetchone()[0])
prof["chargeoff_amount_over_approval_quantiles"] = [round(x, 3) for x in con.execute(
    "SELECT quantile_cont(co_amount / gross_approval, [0.1, 0.25, 0.5, 0.75, 0.9]) FROM loans WHERE status = 'co'").fetchone()[0]]
prof["months_disbursement_to_chargeoff_quantiles"] = con.execute(
    "SELECT quantile_cont(co_m - disb_m, [0.1, 0.25, 0.5, 0.75, 0.9]) FROM loans WHERE status = 'co'").fetchone()[0]
con.execute(f"COPY (SELECT * FROM loans ORDER BY loan_id) TO '{OUT / 'loans.parquet'}' (FORMAT parquet)")
(REPORTS / "p3" / "data_profile.json").write_text(json.dumps(prof, indent=2, default=str))
print(json.dumps(prof, indent=2, default=str))
