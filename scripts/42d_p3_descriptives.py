"""Credit-loss study: descriptive tables for the report (no model). Annual charge-off rate and dollar severity by
charge-off year, and the months with administrative write-off batches. Writes reports/p3/annual_chargeoffs.csv and
chargeoff_batch_months.csv.
"""
import sys; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import pandas as pd
from _paths import WORK, REPORTS
from mval.p3 import month_label

R = REPORTS / "p3"
m = pd.read_csv(R / "monthly_actuals.csv").groupby("m")[["at_risk", "n_co", "n_pif", "co_dollars", "co_dollars_lender"]].sum()
loans = pd.read_parquet(WORK / "p3" / "loans.parquet")
co = loans[loans.in_window & (loans.status == "co") & (loans.co_m >= 72)]
sev = co.groupby(co.co_m // 12 + 2000).apply(lambda d: pd.Series({"dollar_severity": d.co_amount.sum() / d.gross_approval.sum(),
                                                                  "mean_severity": (d.co_amount / d.gross_approval).clip(0, 1).mean()}), include_groups=False)
y = m.groupby(m.index // 12 + 2000).agg(loan_months=("at_risk", "sum"), chargeoffs=("n_co", "sum"), payoffs=("n_pif", "sum"),
                                       chargeoff_dollars=("co_dollars", "sum"), lender_dollars=("co_dollars_lender", "sum"))
y["annualised_chargeoff_rate"] = 12 * y.chargeoffs / y.loan_months
y = y.join(sev); y.index.name = "year"; y.to_csv(R / "annual_chargeoffs.csv")
med = m.n_co.rolling(13, center=True, min_periods=7).median()
b = m[m.n_co > 2 * med].assign(month=lambda d: [month_label(i) for i in d.index], rolling_median=med[m.n_co > 2 * med])
b[["month", "n_co", "rolling_median"]].to_csv(R / "chargeoff_batch_months.csv", index=False)
print(y.round(3).to_string()); print(b[["month", "n_co", "rolling_median"]].to_string(index=False))
