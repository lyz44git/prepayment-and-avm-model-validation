# Credit-loss study: deviations from the analysis plan

The plan (`docs/credit_loss_plan.md`, SHA-256 in `reports/p3/plan_hash.json`, recorded 2026-11-18 12:49) is
unchanged. Changes made after the first model fit are listed here, with the reason and the evidence that prompted
them. Results from the first fit are kept in `reports/p3/first_fit/` for comparison.

## D1. The loan term is dropped from every model and from censoring (2026-11-18, after the first fit)

**Evidence.** The first fit gave the LightGBM challenger an out-of-time AUC of 0.935 against 0.836 for the logit,
with the term and months-to-maturity features carrying 57% of its split gain. A check of the term field by final
status (`scripts/40b_p3_term_check.py`, `reports/p3/term_field_check.csv`):

- Loans that were cancelled before disbursement, which keep their approved terms, have a term that is not a multiple
  of 12 in 10% of cases; active loans in 16%; paid-off loans in 18%; charged-off loans in 90%.
- 120-month loans, the most common term, show a charge-off share of 0.4%, against 12.5% for all loans in the window.
- Amount and guarantee share show no such pattern.

The recorded term is evidently revised after a loan runs into trouble, so it carries information about the outcome.
Any feature built from it (term group, months to maturity, past maturity) leaks the target, and so does the plan's
censoring at maturity + 60 months, which also made "42% of charge-offs fall after maturity" an artefact.

**Change.** No hazard or severity model uses the term. Loans are censored only at 2026-09. The lookup M0 and the
severity lookup S0 use processing group in place of term group. The rolling backtest uses the same risk set. For the
lifetime projection of the active book, the recorded term is used only to set each loan's horizon (its remaining
contractual life as recorded), never as a predictor; a sensitivity extends the horizon by 24 months to allow for
charge-offs posted after maturity.

**Consequence for the plan's procedure.** The lag L, the M2 early stopping and the champion rule are re-run as
specified on the term-free design.

## D2. Diagnostics added to answer the study's questions (2026-11-18, after the first fit)

None of these is used to choose a model.

- **Vintage diagnostic.** Predicted ÷ actual charge-offs by approval cohort (fiscal year), in sample and out of time,
  to separate macro sensitivity from changes in the loans themselves. In the first fit the logit over-predicted
  charge-offs in 2017–2019 by about 50% and in 2020–2026 by about 90%.
- **Value of the macro path.** At each rolling origin, the realised-macro forecast is compared with the
  frozen-macro forecast and with the lookup M0, not only with the ±25% rule.
- **Model uncertainty of the allowance.** Lifetime expected loss under M0, M1 and M2, set beside the spread between
  the baseline and severely adverse scenarios and beside the rolling backtest error.
- **Outcome-based monitoring.** A trigger on predicted ÷ actual 24-month charge-offs at successive origins, next to
  the population stability index.

## D3. Scenario timing (2026-11-18, before the allowance was computed)

The plan applies the supervisory paths from 2026Q4. The Fed's scenarios start from a 2025Q4 jump-off and run
2026Q1–2029Q1, and the severely adverse path is already at 8.2% unemployment by 2026Q3 against 4.6% in the
baseline. Starting it at 2026Q4 would make unemployment jump from 4.6% to 9.0% in one quarter. Each scenario is
therefore applied from its own start (2026Q1), so the lagged unemployment change in the first projection year
follows the scenario; reversion to the 2006–2025 average begins after the last scenario quarter, 2029Q1. The
baseline values for 2026Q1–Q3 also stand in for realised macro in those months of the estimation sample, as the plan
states.

## D4. Remediation candidates for the vintage finding, written before they were run (2026-11-18)

**Evidence.** With the term-free design, the champion logit fitted on 2006–2019 is calibrated in sample overall
(predicted ÷ actual 1.00) but not by approval cohort: 0.69–0.77 for FY2006–FY2007 loans and 1.3–2.3 for every cohort
from FY2009 to FY2019 (`reports/p3/vintage_pred_vs_actual.csv`). Out of time it over-predicts 2020–2026 charge-offs
2.8-fold. The unemployment term was estimated on one crisis whose charge-offs came mostly from FY2005–FY2008 loans,
so it may be carrying cohort quality as well as macro sensitivity.

**Candidates.**
- **M1v: M1 plus approval-era effects.** Eras fixed now: FY2000–04 (base), FY2005–08, FY2009–13, FY2014–19,
  FY2020 onward. When a model is fitted at a date before an era has loans, those loans take the latest fitted era.
- **M1r: M1 with trailing recalibration.** At each origin, M1's forecasts are scaled by actual ÷ predicted
  charge-offs of the book in the 24 months before the origin (in-sample predictions), capped to [0.25, 4].

**Rule, fixed before running.** A candidate corrects the calibration if (i) its out-of-time predicted ÷ actual
for 2020-01..2026-09, excluding the payment-relief months 2020-04..2021-12, is within 0.80–1.25, and (ii) its rolling
24-month dollar forecasts with realised macro are within ±25% of actual at 6 or more of the 8 origins. Also reported,
with no rule attached: the change in the unemployment coefficient when era effects are added.

Implementation note added after the first M1v run: loans approved in October–December 2019 (FY2020) gave the
FY2020+ era 3,080 loan-months and no charge-offs in the 2006–2019 fit, and a separated coefficient (−10.3, s.e. 177).
An era now gets its own effect only with at least 50 charge-offs in the fit; later loans take the latest such era.
