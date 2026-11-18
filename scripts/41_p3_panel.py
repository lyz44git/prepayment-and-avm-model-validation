"""Credit-loss study, step 2: loan-month risk set 2006-01..2026-09, case-control sample and exact monthly totals.

For each loan in the window: months from max(disbursement month + 1, 2006-01) to min(event month, 2026-09)
(the plan's maturity + 60 censoring is dropped because the recorded term is revised after default; deviation D1). Keeps every charge-off and payoff month plus 5% of all loan-months (deterministic hash), so the
charge-off model uses {charge-offs} + hash-kept months and the payoff model {payoffs} + hash-kept months.
Writes data_work/p3/cc.parquet and reports/p3/monthly_actuals.csv (exact counts and dollars by month and split).
"""
import sys, json; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
from _paths import WORK, REPORTS, duck
from mval.p3 import W0, W1, HOLDOUT_STATES, SAMPLE_RATE, macro_quarterly, macro_monthly

OUT = WORK / "p3"
con = duck(threads=24, mem="160GB")
con.execute(f"CREATE TABLE loans AS SELECT * FROM read_parquet('{OUT / 'loans.parquet'}') WHERE in_window")
ho = ",".join(f"'{s}'" for s in HOLDOUT_STATES)
con.execute(f"""
CREATE TABLE spans AS SELECT loan_id, disb_m, term, status, end_m,
  greatest(disb_m + 1, {W0}) AS m0,
  least(end_m, {W1}) AS m1,                       -- censored only at the data date (deviation D1: no term-based censoring)
  (status = 'co'  AND end_m <= {W1}) AS ends_co,
  (status = 'pif' AND end_m <= {W1}) AS ends_pif,
  state IN ({ho}) AS holdout
FROM loans""")
n_months = con.execute("SELECT sum(m1 - m0 + 1) FROM spans WHERE m1 >= m0").fetchone()[0]
thr = int(SAMPLE_RATE * 1_000_000)
con.execute(f"""
CREATE TABLE lm AS
SELECT s.loan_id, m, m - s.disb_m AS age, s.holdout,
  (s.ends_co AND m = s.m1)::TINYINT AS y_co, (s.ends_pif AND m = s.m1)::TINYINT AS y_pif,
  (hash(s.loan_id, m) % 1000000 < {thr}) AS keep
FROM spans s, LATERAL (SELECT unnest(range(s.m0, s.m1 + 1)) AS m)
WHERE s.m1 >= s.m0""")
tot = con.execute("SELECT count(*), sum(y_co), sum(y_pif), avg(keep::INT) FROM lm").fetchone()
# exact monthly totals by split (all loan-months, no sampling)
con.execute("""CREATE TABLE monthly AS SELECT lm.m, lm.holdout, count(*) AS at_risk, sum(y_co) AS n_co, sum(y_pif) AS n_pif,
  sum(CASE WHEN y_co = 1 THEN l.co_amount ELSE 0 END) AS co_dollars,
  sum(CASE WHEN y_co = 1 THEN l.co_amount * (1 - l.gty_share) ELSE 0 END) AS co_dollars_lender
  FROM lm JOIN loans l USING (loan_id) GROUP BY 1, 2 ORDER BY 1, 2""")
con.execute(f"COPY monthly TO '{REPORTS / 'p3' / 'monthly_actuals.csv'}' (HEADER)")
# sample with loan attributes
con.execute(f"""COPY (
  SELECT lm.loan_id, lm.m, lm.age, lm.holdout, lm.y_co, lm.y_pif, lm.keep, (lm.m % 12) + 1 AS moy,
    l.term, l.gross_approval, l.gty_share, l.proc, l.bage, l.btype, l.naics2, l.franchise, l.state, l.approval_fy,
    l.disb_m, l.rate, l.co_amount
  FROM lm JOIN loans l USING (loan_id) WHERE lm.keep OR lm.y_co = 1 OR lm.y_pif = 1
) TO '{OUT / 'cc.parquet'}' (FORMAT parquet)""")
n_cc = con.execute(f"SELECT count(*) FROM read_parquet('{OUT / 'cc.parquet'}')").fetchone()[0]
info = {"loan_months": int(n_months), "lm_rows": int(tot[0]), "chargeoff_months": int(tot[1]), "payoff_months": int(tot[2]),
        "kept_share": round(float(tot[3]), 5), "sample_rows": int(n_cc), "loans_in_window": con.execute("SELECT count(*) FROM spans").fetchone()[0]}
(REPORTS / "p3" / "panel_info.json").write_text(json.dumps(info, indent=2)); print(info)
