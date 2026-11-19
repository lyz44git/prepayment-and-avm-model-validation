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

## D5. Second remediation round, written before it was run (2026-11-18, after D4 results)

**Evidence.** Neither D4 candidate meets its rule. M1v (approval eras) has the smallest typical backtest error but
misses at the 2011, 2013, 2015 and 2019 origins; M1r (recalibration) fixes the out-of-time level but not the
origins. Two mechanisms are not in any model so far:

- **Default pipeline.** Charge-off follows default by one to three years (liquidation). After a downturn, loans that
  defaulted keep being written off while unemployment is already falling, which no current-macro term captures. The
  2013 origin (24-month charge-offs in 2014–2015, including the largest dollar year of the window) is the clearest
  case: all models under-predict it (0.35–0.56) except the lookup, which carries crisis-era rates.
- **Payment relief.** From 2020-04 to 2021-09 SBA made the scheduled payments of 7(a) borrowers in regular servicing
  (CARES Act §1112 and its 2021 extension), so the unemployment shock of 2020 did not reach charge-offs.

**Candidates (all keep the approval-era effects of M1v).**
- **R2a.** M1v scaled by the trailing 24-month actual ÷ predicted (as M1r), capped to [0.25, 4].
- **R2b.** M1v plus a downturn-exposure term: the largest 4-quarter rise in unemployment dated between
  max(first disbursement, t − L − 36) and t − L, floored at 0 (L is the charge-off lag, 12 months). The 36-month
  window is fixed now and not tuned.
- **R2c.** R2b scaled as in R2a.

**Relief overlay O1, applied to every model and reported beside the unadjusted results.** In the quarterly macro
series, the unemployment rate and the BBB spread for 2020Q2–2021Q3 are replaced by their 2020Q1 values before the
4-quarter changes are computed: the relief window is treated as one in which the shock did not reach borrowers.
It changes the inputs of fits and forecasts from 2020-04 on, so it affects the 2019, 2021 and 2023 origins, the
out-of-time scores and the final fit. It is a documented qualitative adjustment, not an estimated effect.

**Rules (unchanged from D4).** (i) Out-of-time predicted ÷ actual for 2020-01..2026-09 excluding 2020-04..2021-12
within 0.80–1.25; (ii) rolling 24-month dollar forecasts with realised macro (with O1) within ±25% of actual at 6 or
more of the 8 origins. **Adoption.** The candidate meeting both rules with the most origins inside ±25% (ties: the
lower mean absolute log ratio) replaces M1 for the allowance; if none meets both, the vintage finding stays open and
the best candidate is reported as the leading challenger. Results without O1 are reported for every candidate.

**D5 results (recorded 2026-11-18 before D6 was written).** No candidate meets both rules (`r2_summary.json`). With
O1, M1v is at 0.81 out of time (rule i met) and 4 of 8 origins; R2a reaches 5 of 8 but 0.73 out of time; the
downturn-exposure term (R2b, R2c) does not help. The remaining misses alternate in sign at the 2011, 2013 and 2015
origins (about 1.5, 0.3 and 0.7 times actual): 24-month charge-offs of the book were $1.17bn after 2011-12 and
$3.01bn after 2013-12, during an expansion, when SBA wrote off the crisis-era loans it had been liquidating.

## D6. Lifetime-to-date backtest, written before it was run (2026-11-18)

**Why.** The allowance is a lifetime quantity, and the 24-month test is dominated by when SBA completes write-offs.
This test asks the question the allowance depends on: did each origin's book eventually lose what the model said?
It is added after the 24-month test failed and does not replace it; the 24-month failure stays in the findings log.

**Test.** At each origin T from 2009-12 to 2019-12 (six origins, each with at least 81 months of outcomes), models
fitted on data to T project the book at T month by month to 2026-09 with the realised macro path (with O1); the
prediction is compared with the actual charge-off dollars of that book from T+1 to 2026-09. Models: M0, M1, M1v, R2a
(M1v scaled at the origin), R2b.

**Rule.** Within ±25% of actual dollars at 5 or more of the 6 origins. **Adoption for the allowance.** A candidate
that meets this rule and rule (i) of D4/D5 is adopted for the lifetime allowance, with use limited to lifetime loss:
not for 24-month or quarterly charge-off forecasts, which fail rule (ii). If several qualify: most origins inside
±25%, then the lower mean absolute log ratio.

**D6 results (2026-11-18).** No model meets the lifetime rule (`r3_summary.json`), so none is adopted under it.
Lifetime-to-date predicted ÷ actual dollars by origin (`r3_lifetime_backtest.csv`):

| Origin | M0 lookup | M1 | M1v (+O1) | R2a (+O1) | R2b (+O1) |
|---|---|---|---|---|---|
| 2009-12 | 0.55 | 0.39 | 0.40 | 0.42 | 0.96 |
| 2011-12 | 0.93 | 0.69 | 0.57 | 0.58 | 0.52 |
| 2013-12 | 1.01 | 0.65 | 0.38 | 0.30 | 0.37 |
| 2015-12 | 2.54 | 1.81 | 0.97 | 1.19 | 1.16 |
| 2017-12 | 3.41 | 2.47 | 1.11 | 1.14 | 1.22 |
| 2019-12 | 3.14 | 2.24 | 1.03 | 0.94 | 1.09 |

From the 2015 origin on, when the fit covers the crisis and the recovery, all three remediated candidates are within
±25% at every origin; fitted on four to eight years (2009–2013 origins), the logits under-predict in 11 of 12
model-origin pairs (R2b at 2009 is the exception). **Validator's
recommendation (outside the pre-set adoption rule, labelled as such):** use M1v with O1 for the lifetime allowance
only when it is fitted on at least one full credit cycle, report the range across M1v, R2a and R2b, and do not use it
for 24-month or quarterly charge-off forecasts. The vintage finding stays open.

## Review notes (2026-11-18): checks after the three rounds; no model refitted, no result changed

- **Sampling design.** The 5% of non-event months is drawn from all loan-months by one hash, so each cause-specific
  fit samples its non-events, including months with the competing event, at the same rate; the log 0.05 intercept
  correction is the right one for that design (`scripts/41_p3_panel.py`).
- **Reach of overlay O1.** Holding the levels before differencing (as D5 specifies) also changes the 4-quarter
  unemployment change for 2021Q4–2022Q3 and, through the 12-month lag, charge-off inputs until 2023Q3. For M1v,
  out-of-time predicted ÷ actual outside the relief months is 0.74 without O1 and 0.81 with it (`r2_oot_eval.csv`), so
  its rule (i) result depends on the overlay. Results at the 2019-12 origin with O1 use knowledge of the relief
  programme that was not available then; they are reported as a policy-hindsight sensitivity.
- **L11.** Business-age categories are coded differently across approval eras (`scripts/42e_p3_input_checks.py`,
  `business_age_by_approval_fy.csv`).
- **L12.** The Fed files end their history in 2025Q4, so 2026Q1–Q3 carry scenario values in the estimation window
  and in each projection (`macro_2026_scenario_vs_published.csv`). Not re-run.
- **Disposition.** Research and benchmark use only; outputs are expected gross SBA write-offs under stated scenarios.
- **Corrections to the text above.** D4's "FY2009 to FY2019" should read FY2009–FY2018 (FY2019 is 2.73). D1's
  "57% of its split gain" was read from the first-fit model and is not in a published table. "Within ±25%" in rule
  (ii) and D6 is 0.75–1.25; rule (i) and the monitoring trigger use 0.80–1.25.
- **Hashes.** The plan hash can be checked against `docs/credit_loss_plan.md`. The D5 and D6 hashes are of this
  file in the private workspace before each run; the published copy has renamed paths, so they cannot be re-derived
  from this repository.
- **D6 without O1 (sensitivity, run at review).** `scripts/47_p3_lifetime_backtest.py --no-overlay`: the same counts
  (M1v 3, R2a 3, R2b 4 of 6; no model meets the rule), and the three candidates within ±25% at 2015–2019 (0.87–1.21)
  (`r3_lifetime_backtest_no_overlay.csv`, `r3_summary_no_overlay.json`).
- **Payoff hazard specification.** The payoff logit includes the unlagged 4-quarter change in unemployment
  (`src/mval/p3.py`, CreditDesign for cause "pif"); the plan listed only the prime rate and its 4-quarter change. The
  term was in every fit from the first; it is logged here because the plan did not name it.
- **24-month backtest script (43).** The payoff logit is fitted with the unemployment change at the charge-off lag but
  scored with the unlagged value; scripts 45–47 are consistent. The corrected run is `scripts/48_p3_backtest_v2.py`
  (report §14); the original tables are kept and the published figures use them.
- **Adaptive design.** Each round's rule was written before it ran, but each round's candidates were designed after the
  previous round's results, and overlay O1 with knowledge of 2020–21. Results across rounds are therefore not
  independent tests.
