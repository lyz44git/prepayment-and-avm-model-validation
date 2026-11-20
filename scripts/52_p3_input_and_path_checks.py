"""Credit-loss study: input-date and scenario-path checks (2026-11-19; aggregates only).

1. Input dates (finding L15). Counts, from the public SBA files, the charge-offs with no parseable charge-off date
   (the first build dated them at the 2026-09 cutoff, because DuckDB's least() ignores NULL), the charge-offs dated
   after the cutoff (capped at 2026-09 under the plan) and the loans whose first disbursement precedes approval by
   more than 12 months; and how many of each entered the modelled population.
   -> reports/p3/r5_input_date_checks.csv
2. Scenario path (finding L16). For each Fed scenario, with and without the relief overlay, the monthly unemployment
   and prime levels and their 4-quarter changes as first published (each change reverted to its own average after
   2029-03) and as now computed (changes taken from the reverted levels). -> reports/p3/r5_scenario_path_reconciliation.csv
3. Allowance bridge. Lifetime loss / balance from the earlier runs (reports/p3/before_2026-11-19/)
   against the latest runs of scripts 44 and 46. -> reports/p3/r5_allowance_update_bridge.csv
4. Two figures quoted in the report that had no exported table: the active book's split by approval era
   (-> reports/p3/r5_active_book_by_approval_era.csv) and quarterly unemployment 2019Q4-2021Q4 from the Fed 2026
   historic file (-> reports/p3/r5_unemployment_2020.csv).
"""
import sys; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd
from _paths import WORK, REPORTS, duck
from mval.p3 import STATES51, W0, W1, mi, month_label, macro_quarterly, macro_monthly, balance_share

R = REPORTS / "p3"; FED = Path("data_external/fed2026")
src = "data_external/sba/FOIA_7a_FY*_asof_260930.csv"
con = duck(threads=12, mem="60GB")
states = ",".join(f"'{s}'" for s in STATES51)

# ---------------------------------------------------------------- 1. input dates
con.execute(f"""
CREATE TABLE b AS
SELECT trim(LoanStatus) AS st, RevolverStatus, ProjectState,
  try_cast(FirstDisbursementDate AS DATE) AS disb, try_cast(PaidInFullDate AS DATE) AS pif,
  try_cast(ChargeOffDate AS DATE) AS co_raw, try_cast(ApprovalDate AS DATE) AS approval,
  try_cast(TermInMonths AS DOUBLE)::INT AS term, try_cast(GrossApproval AS DOUBLE) AS gross_approval
FROM (SELECT DISTINCT * EXCLUDE (filename) FROM read_csv('{src}', all_varchar=true, header=true, union_by_name=true, filename=true))
""")
base = f"""RevolverStatus = 'N' AND st IN ('P I F','CHGOFF','EXEMPT') AND disb IS NOT NULL AND term >= 6
  AND gross_approval > 0 AND ProjectState IN ({states}) AND NOT (st = 'P I F' AND pif IS NULL)"""
co_m = "(year(least(co_raw, DATE '2026-09-30')) - 2000) * 12 + month(least(co_raw, DATE '2026-09-30')) - 1"
end_m = f"""CASE WHEN st = 'P I F' THEN (year(pif) - 2000) * 12 + month(pif) - 1
  WHEN st = 'CHGOFF' THEN coalesce({co_m}, {W1}) ELSE {W1} END"""
disb_m = "(year(disb) - 2000) * 12 + month(disb) - 1"
in_window = f"(({end_m}) >= {W0} AND ({disb_m}) < {W1} AND ({end_m}) > ({disb_m}))"
early = "approval IS NOT NULL AND disb < approval - INTERVAL 12 MONTH"
checks = {
    "charge-off status, no parseable charge-off date (all records)": "st = 'CHGOFF' AND co_raw IS NULL",
    "  of which in the modelled population (first build dated them 2026-09)": f"{base} AND st = 'CHGOFF' AND co_raw IS NULL",
    "  of which in the risk window (undated charge-offs)": f"{base} AND st = 'CHGOFF' AND co_raw IS NULL AND {in_window}",
    "charge-off dated after 2026-09-30 (capped at 2026-09 under the plan)": "st = 'CHGOFF' AND co_raw > DATE '2026-09-30'",
    "first disbursement more than 12 months before approval (population)": f"{base} AND (st <> 'CHGOFF' OR co_raw IS NOT NULL) AND {early}",
    "  of which in the risk window (date-order errors)": f"{base} AND (st <> 'CHGOFF' OR co_raw IS NOT NULL) AND {early} AND {in_window}",
    "  of which charged off": f"{base} AND st = 'CHGOFF' AND co_raw IS NOT NULL AND {early}",
    "population, first build": f"{base} AND NOT (st = 'CHGOFF' AND least(co_raw, DATE '2026-09-30') IS NULL)",
    "population, with the L15 checks": f"{base} AND NOT (st = 'CHGOFF' AND co_raw IS NULL) AND NOT ({early})",
    "charge-offs in the risk window, first build": f"{base} AND st = 'CHGOFF' AND {in_window}",
    "charge-offs in the risk window, with the L15 checks": f"{base} AND st = 'CHGOFF' AND co_raw IS NOT NULL AND NOT ({early}) AND {in_window}",
    "charge-offs dated 2026-09, first build": f"{base} AND st = 'CHGOFF' AND coalesce({co_m}, {W1}) = {W1}",
    "charge-offs dated 2026-09, with the L15 checks": f"{base} AND st = 'CHGOFF' AND co_raw IS NOT NULL AND NOT ({early}) AND {co_m} = {W1}",
}
assert len(checks) == 13
rows = []
for k, cond in checks.items():
    n, amt = con.execute(f"SELECT count(*), coalesce(sum(gross_approval), 0) FROM b WHERE {cond}").fetchone()
    rows.append({"check": k, "loans": int(n), "gross_approval_usd": float(amt)})
chk = pd.DataFrame(rows)
chk.to_csv(R / "r5_input_date_checks.csv", index=False)
print(chk.to_string(index=False))

# ---------------------------------------------------------------- 2. scenario path
out = []
months = range(W1 - 12, W1 + 61)
for scen in ("baseline", "severe"):
    for ov in (False, True):
        q = macro_quarterly(FED, scen, relief_overlay=ov)
        old = macro_monthly(q, W1 + 480, changes_from_levels=False)
        new = macro_monthly(q, W1 + 480, changes_from_levels=True)
        assert old.loc[:mi(2029, 3)].equals(new.loc[:mi(2029, 3)])
        for m in months:
            out.append({"scenario": scen, "relief_overlay": ov, "month": month_label(m), "after_scenario": m > mi(2029, 3),
                        "ur": new.ur[m], "ur_12m_change": new.ur[m] - new.ur[m - 12],
                        "dur4_own_reversion": old.dur4[m], "dur4_from_levels": new.dur4[m],
                        "prime": new.prime[m], "prime_12m_change": new.prime[m] - new.prime[m - 12],
                        "dprime4_own_reversion": old.dprime4[m], "dprime4_from_levels": new.dprime4[m]})
path = pd.DataFrame(out)
path.to_csv(R / "r5_scenario_path_reconciliation.csv", index=False)
gap = (path.dur4_own_reversion - path.ur_12m_change).abs()
print("largest |dur4 (own reversion) - 12-month change in ur|:", round(gap.max(), 3), "| from levels:", round((path.dur4_from_levels - path.ur_12m_change).abs().max(), 12))

# ---------------------------------------------------------------- 3. allowance bridge
S = R / "before_2026-11-19"
br = []
for f, keys, src_label in (("cecl_allowance.csv", ["scenario", "model", "horizon"], "44 first round, no overlay"),
                           ("r2_cecl_allowance.csv", ["scenario", "model"], "46 second round, overlay O1")):
    a, b = pd.read_csv(S / f), pd.read_csv(R / f)
    m = a.merge(b, on=keys, suffixes=("_pub", "_cor"))
    for _, r in m.iterrows():
        br.append({"source": src_label, "scenario": r.scenario, "model": r.model, "horizon": r.get("horizon", "recorded term"),
                   "loss_over_balance_earlier": r.loss_over_balance_pub, "loss_over_balance_latest": r.loss_over_balance_cor,
                   "change_pp": 100 * (r.loss_over_balance_cor - r.loss_over_balance_pub)})
br = pd.DataFrame(br)
br.to_csv(R / "r5_allowance_update_bridge.csv", index=False)
cand = br[(br.source.str.startswith("46")) & br.model.isin(["M1v", "R2a", "R2b", "R2c"])]
for v in ("earlier", "latest"):
    p = cand.pivot(index="model", columns="scenario", values=f"loss_over_balance_{v}") * 100
    print(v, "baseline %.2f-%.2f" % (p.baseline.min(), p.baseline.max()), "severe - baseline:", (p.severe - p.baseline).round(2).to_dict())

# ---------------------------------------------------------------- 4. untraced figures
loans = pd.read_parquet(WORK / "p3" / "loans.parquet")
book = loans[loans.in_window & (loans.status == "active")].copy()
book["age0"] = W1 - book.disb_m
book["balance"] = book.gross_approval * balance_share(book.rate.fillna(book.rate.median()), book.term, book.age0)
book["era"] = np.where(book.approval_fy >= 2020, "FY2020 or later", np.where(book.approval_fy >= 2014, "FY2014-2019", "before FY2014"))
g = book.groupby("era").agg(loans=("balance", "size"), est_balance=("balance", "sum"))
g["share_of_loans"], g["share_of_balance"] = g.loans / g.loans.sum(), g.est_balance / g.est_balance.sum()
g.reset_index().to_csv(R / "r5_active_book_by_approval_era.csv", index=False)
print(g.round(3).to_string())
q = macro_quarterly(FED, "baseline")
u = q[(q.year * 10 + q.q >= 20194) & (q.year * 10 + q.q <= 20214)][["year", "q", "ur", "dur4"]]
u.to_csv(R / "r5_unemployment_2020.csv", index=False)
print(u.to_string(index=False))
