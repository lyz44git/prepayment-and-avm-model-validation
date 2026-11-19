# Small-business credit loss: charge-off hazard, severity and lifetime expected loss, with validation

Public SBA 7(a) loan data, as of 2026-09-30. Analysis plan written and hashed before any model was fitted
(`docs/credit_loss_plan.md`; hash in `reports/p3/plan_hash.json`); every change made afterwards is listed with
its reason in `docs/credit_loss_deviations.md` (D1–D6, including three remediation rounds whose rules were written before they ran). The developer is also the validator, so this is a self-administered
review in the structure of a model validation report, not an independent validation.

## 1. Purpose and scope

An allowance for credit losses under CECL needs, for every loan on the books, a lifetime expected loss: when the
loan might default, how much would be lost, and how long it stays on the books. This study builds those pieces on
public data and asks of each one what a model risk reviewer asks: is the number usable, under which conditions does
it fail, and what would detect the failure?

- **Portfolio.** SBA 7(a) term loans: small-business loans made by banks and other lenders with a partial SBA
  guarantee. They are not a bank's own portfolio and carry no bureau or delinquency data.
- **Event.** SBA's charge-off of the loan, which follows default and liquidation. The hazard therefore times
  write-off, not default.
- **Outputs.** A monthly charge-off hazard and a competing payoff hazard (champion and challengers), a severity model
  (charge-off amount over the original loan amount), a rolling 24-month backtest, and the lifetime expected loss of
  the loans active at 2026-09 under the Federal Reserve's 2026 supervisory scenarios.

## 2. What the data record

The files hold one row per loan: amount, SBA guarantee, approval and first-disbursement dates, processing method,
business age and type, sector, state, term, final status (paid in full, charged off, active), payoff date,
charge-off date and charge-off amount (`reports/p3/data_profile.json`).

| Step | Loans |
|---|---|
| Rows in the three files (FY2000 onward) | 1,639,728 |
| After exact duplicates | 1,637,918 |
| Term loans, final status paid / charged off / active, disbursed, 50 states and DC | 994,992 |
| At risk in the window 2006-01..2026-09 | 919,479 (115,117 charged off, 572,070 paid off, 232,292 active) |

Five properties of the data decide what can be modelled. Each is in the findings log (§12).

- **The recorded term is revised after a loan runs into trouble (L1).** 90% of charged-off loans have a term that is
  not a multiple of 12, against 10% of loans cancelled before disbursement (which keep their approved terms), 16% of
  active and 18% of paid-off loans; 120-month loans show a charge-off share of 0.4% against 12.5% overall
  (`term_field_check.csv`, `term_charge_off_share_by_term.csv`). The term therefore carries information about the
  outcome. In the first fit it gave LightGBM an out-of-time AUC of 0.935 (`reports/p3/first_fit/`); without it, 0.737.
  The term is not used as a predictor or for censoring (deviation D1).
- **Payoff dates before mid-2005 are bulk-stamped.** 45,770 payoffs in the population are dated 2005-05, against
  about 2,600 a month from 2005-08. The risk window therefore starts in 2006-01, with loans disbursed earlier
  entering at their current age.
- **Charge-off is an administrative event (L7).** It comes a median of about four years after first disbursement
  and partly in batches: 1,935 charge-offs in 2008-05, 1,340 in 2015-08 and 505 in 2019-02, each more than twice the
  surrounding 13-month median (`chargeoff_batch_months.csv`).
- **Two fields change meaning across approval eras (L9).** The interest rate is missing for 94.5% of loans approved
  FY2000–FY2009 and the collateral flag's share of "Y" rises from 15% to 82–89%. Neither is used.
- **Business-age categories are coded differently across approval eras (L11).** "Startup" is under 1% of loans
  before FY2005 and 18–22% from FY2012; "new" is 24–31% through FY2008, 0% in FY2019 and 13–18% from FY2021;
  "unknown" is 12–14% in FY2018–19 and under 1% from FY2021 (`business_age_by_approval_fy.csv`). The field is a
  predictor in every model, so part of what its effects measure is approval era.

The annual charge-off rate of loans at risk peaks at 7.4% in 2010 and runs at 1.0–1.3% a year from 2019 to 2025
(`annual_chargeoffs.csv`).

## 3. Models

Unit: loan-month, 52,141,481 at risk. All charge-off and payoff months are kept with 5% of the other months; the
logit intercept is corrected by log 0.05 and evaluation weights non-events by 20. The 5% is drawn from all
loan-months by a deterministic hash, so each cause-specific model samples its non-events, including months in which
the competing event occurred, at the same rate, which is what the correction assumes (`scripts/41_p3_panel.py`).

| Model | Charge-off and payoff hazards |
|---|---|
| M0 lookup | Hazard by 12-month loan-age band × processing group |
| M1 logit (champion) | Cause-specific discrete-time logit: loan-age spline, log amount, guarantee share, processing group, business age and type, sector, franchise, month of year, state effects; charge-off adds the 4-quarter change in unemployment and the BBB–Treasury spread, lagged 12 months (lag chosen on the validation months from 0, 12, 24); payoff adds the prime rate and its 4-quarter change |
| M2 LightGBM | Same information; monotone increasing in the two macro terms for charge-off |
| Severity | S0 lookup (age band × processing), S1 fractional logit, S2 LightGBM, on charge-off amount ÷ original amount |

Expected loss of loan *i* in month *t* is S(t−1) × h_charge-off(t) × severity(t) × original amount, with S the
survival from both causes. The lender's share multiplies by one minus the guarantee share (7(a) losses are shared
pro rata). Hold-out states (MA, NJ, WI, MN, SC, KY, OK, AZ, WA) never enter a fit.

## 4. Does the challenger earn its complexity?

Fit 2006–2017, validation 2018–2019 (`eval_validation.csv`); then refit on 2006–2019 and score 2020-01..2026-09
(`eval_hazard.csv`).

| Charge-off hazard | AUC, in sample | AUC, out of time | Predicted ÷ actual, out of time |
|---|---|---|---|
| M0 lookup | 0.708 | 0.666 | 2.25 |
| M1 logit | 0.747 | 0.681 | 2.79 |
| M2 LightGBM | 0.821 | 0.737 | 2.43 |

- **Champion rule.** M2 replaces M1 only if it lowers the validation log loss by more than 1% and its validation
  predicted ÷ actual is within 0.90–1.10. M2 lowers log loss by 5.0% but over-predicts the validation months by
  1.66, so M1 stays champion (`champion.json`).
- **What the rule shows.** LightGBM ranks loans better, but every model over-predicts the level of charge-offs
  after 2019 by a factor of more than two. Ranking is not the problem; the level is.
- **Payoff hazard.** Out of time, predicted ÷ actual is 1.12 for M1, 1.06 for M2 and 0.88 for M0 (AUC 0.66–0.68).

## 5. Did the loans change? Vintage, not only the economy

Predicted ÷ actual charge-offs of the 2006–2019 champion, by approval cohort, in sample (`vintage_pred_vs_actual.csv`):

| Approval years | Predicted ÷ actual |
|---|---|
| FY2000–FY2005 | 0.82–1.09 |
| FY2006–FY2007 | 0.77, 0.69 |
| FY2009–FY2018 | 1.28–2.34 |

The model is calibrated overall in sample (1.00) because it averages a crisis whose losses came from loans approved
in FY2005–FY2008 with a decade of much safer loans. Its unemployment term was estimated on that one crisis. Adding
approval-era effects (deviation D4; `m1v_co_coefficients.csv`) halves the unemployment coefficient, from 0.215 to
0.103 per point of 4-quarter change (standard errors 0.004), and the era effects are large: relative to FY2000–04,
+0.18 for FY2005–08, −0.87 for FY2009–13 and −1.24 for FY2014–19 in log odds. About half of what the champion
reads as sensitivity to unemployment is the quality of the loans that happened to be on the books when unemployment
rose (L2).

## 6. Rolling backtest: does knowing the macro path help?

At each December origin the models are refitted on data to that date and project 24-month charge-off dollars for
the book at risk, with the realised macro path and with macro held at its value at the origin
(`backtest_rolling.csv`). Predicted ÷ actual dollars, realised macro:

| Origin | Book (loans) | Actual 24-month charge-offs | M0 | M1 | M1r | M1v |
|---|---|---|---|---|---|---|
| 2009-12 | 214,170 | $2.40bn | 0.94 | **0.98** | 1.04 | 0.94 |
| 2011-12 | 208,575 | $1.17bn | 2.41 | 1.77 | 1.79 | 1.51 |
| 2013-12 | 199,027 | $3.01bn | 0.90 | 0.56 | 0.37 | 0.35 |
| 2015-12 | 199,741 | $2.35bn | 1.53 | 1.09 | 1.08 | 0.64 |
| 2017-12 | 208,307 | $1.25bn | 2.97 | 2.31 | 1.63 | 1.13 |
| 2019-12 | 208,585 | $0.93bn | 3.88 | 7.24 | 3.84 | 2.14 |
| 2021-12 | 205,266 | $1.22bn | 3.10 | 1.94 | 0.80 | 0.90 |
| 2023-12 | 200,627 | $1.30bn | 2.92 | 2.00 | 1.39 | 0.82 |
| Within ±25% | | | 2 of 8 | 2 of 8 | 3 of 8 | 4 of 8 |

M1r is the champion scaled by actual ÷ predicted charge-offs of the previous 24 months; M1v adds approval-era
effects. The plan's pass rule (within ±25% at 6 or more of 8 origins) is not met by any model
(`backtest_summary.json`). The 2013 origin is pulled down by the 2015 write-off batch (§2).

**After remediation (§11).** With approval-era effects and the relief overlay O1, the same backtest is within ±25% at
4 of 8 origins (M1v) or 5 of 8 with trailing recalibration (R2a), and the typical error halves (mean absolute log
ratio 0.36 and 0.33, against 0.68 for the champion; `r2_backtest.csv`). The misses that remain alternate in sign at
the 2011, 2013 and 2015 origins (about 1.5, 0.3 and 0.7 times actual), when SBA wrote off the crisis-era loans it had
been liquidating: the book's 24-month charge-offs were $1.17bn after 2011-12 and $3.01bn after 2013-12, during an
expansion. No input known at the origin predicts that schedule.

**The macro path rarely matters (L4).** Realised and held-at-origin forecasts of the champion differ by less than 3%
at six of eight origins. The exceptions are 2009 (realised 0.98 against 1.23 held, the one origin where the path
helps) and 2019, where the path makes the forecast more than twice as wrong (§7).

## 7. When policy breaks the link: 2020–2021

From the 2019-12 origin, charge-off dollars in 2020-04..2021-12 were $0.82bn. The champion with the realised macro
path predicted 7.7 times that; with macro held at 2019-12, 3.1 times; with era effects, 2.2 and 1.5 times
(`pandemic_check.json`, `backtest_rolling_monthly.csv`). Unemployment rose by 9.2 points in a quarter (quarterly averages, 2020Q1 to Q2) while SBA made
borrowers' loan payments for several months under the CARES Act, so the historical link between unemployment and
write-offs did not hold (L3). The plan's rule (record a finding above 1.25) is met. A macro-driven loss model needs
an explicit overlay, or its macro terms switched off, when government support changes borrowers' payments. The
overlay O1 (deviation D5) holds unemployment and the BBB spread at their 2020Q1 values for 2020Q2–2021Q3, the quarters
in which SBA made the payments. With it, the era model's forecast for April 2020 to December 2021 falls from 2.2 to
1.6 times actual (`r2_backtest_monthly.csv`). O1 is a documented qualitative adjustment, not an estimate, and
results without it are reported beside it.

Two properties of O1 limit how its results can be read. First, the levels are held before the 4-quarter change is
computed, so O1 also changes the unemployment change for 2021Q4–2022Q3 (measured against the held levels) and,
through the 12-month lag, charge-off inputs until 2023Q3. Outside the relief months its net effect raises the era
model's out-of-time predicted ÷ actual from 0.74 to 0.81 (`r2_oot_eval.csv`), so M1v + O1 meets the 0.80–1.25 rule
because of the overlay. Second, a forecaster at 2019-12 did not know the relief programme; the 2019-origin results
with O1 are a policy-hindsight sensitivity, not a forecast that could have been made.

## 8. Severity

Charge-off dollars over original amount, by charge-off year, range from 0.58 (2017–2019) to 0.70 (2010)
(`annual_chargeoffs.csv`). Out of time, the fractional logit predicts total charge-off dollars within 2%
(predicted ÷ actual 1.02, mean absolute error 0.17 per loan; S0 1.06, S2 0.98; `eval_severity.csv`). Severity
depends mainly on the loan's age at charge-off and its guarantee share; its unemployment coefficient is −0.02
(`s1_severity_coefficients.csv`). The difference between scenarios therefore comes almost entirely from the
charge-off hazard. Severity is measured against the original amount because the data have no balances (L8).

## 9. Lifetime expected loss at 2026-09-30

Book: 232,292 active loans, $162.2bn approved, $128.2bn estimated outstanding (level-payment schedule from the
recorded rate and term). Final models fitted on 2006-01..2026-09. Horizon: each loan's remaining recorded term (at
least 12 months). Scenarios: Fed 2026 baseline and severely adverse from their 2026Q1 start to 2029Q1, then
reversion over four quarters to the 2006–2025 average (deviation D3). The estimate is of gross charge-off dollars as
SBA records them, before recoveries. Lifetime loss over estimated balance (`cecl_allowance.csv`):

| Model | Baseline | Severely adverse | Severe − baseline |
|---|---|---|---|
| M1v logit + approval eras | 3.4% | 4.1% | +0.7 pt |
| M1r logit, trailing recalibration (factor 0.81) | 4.6% | 6.9% | +2.2 pt |
| M1 logit (champion) | 5.7% | 8.5% | +2.7 pt |
| M2 LightGBM | 6.6% | 9.0% | +2.4 pt |
| M0 lookup (no macro) | 9.9% | 9.9% | 0 |

- **Model choice moves the allowance more than the scenario does (L5).** Across the four models with macro terms
  the baseline estimate runs from 3.4% to 6.6% of balance, a spread of 3.3 points; the severely adverse scenario
  adds 0.7 to 2.7 points. For comparison, the rolling backtest error of the best of them (M1v) is a factor of 1.5
  on a typical origin (mean absolute log ratio 0.39).
- **How much the scenario matters is itself a model choice.** M1, whose unemployment coefficient carries cohort
  quality (§5), adds 2.7 points under the severely adverse path; M1v adds 0.7.
- **Lender's share.** Losses are shared pro rata with SBA, so the lender's retained loss is the same percentage of
  its retained balance (M1v baseline $1.03bn of $4.32bn in total). This applies the guarantee percentage
  mechanically; a lender's allowance would also need its actual exposure and contract terms, expected recoveries and
  the accounting for the guarantee it holds.
- **Macro inputs for 2026Q1–Q3 (L12).** The Fed 2026 files end their history in 2025Q4, so in the estimation window
  2026Q1–Q3 carry baseline-scenario values (unemployment 4.6% against a published 4.1–4.3%; prime 6.2–6.6% against
  6.8%; `macro_2026_scenario_vs_published.csv`), which enter the payoff hazard directly. In the projection, each
  scenario runs from its own 2026Q1 start, so its 2026Q1–Q3 values (severely adverse: unemployment 5.9–8.2%)
  reach the charge-off hazard of 2027 through the 12-month lag. The estimates have not been re-run with published
  values for those quarters.
- **Horizon.** Allowing 24 months beyond the recorded term for charge-offs posted after maturity adds 0.25 points
  (M1v) to 0.46 points (M1) under the baseline.
- **After remediation.** With approval eras and the relief overlay, the candidates put lifetime loss at 3.4% (M1v),
  3.4% (R2b) and 4.4% (R2a) of balance under the baseline and 4.0%, 3.6% and 5.2% under the severely adverse
  scenario (`r2_cecl_allowance.csv`); the champion with the overlay stays at 6.0% and 8.5%. The range across
  remediated models, 1.0 point under the baseline, is still wider than the scenario effect for two of the three
  (+0.1 to +0.8 points).
- **Segments.** Express loans carry the highest estimated loss (M1v baseline 4.4% of balance against 3.4% for
  Preferred Lenders loans; `cecl_by_segment.csv`).

## 10. Monitoring: drift statistics against outcomes

The population stability index of the book's 24-month charge-off probability between successive origins is 0.02
between 2015-12 and 2017-12, while the champion's 2017-12 forecast turned out 2.3 times actual; it is above 2 only
across the pandemic origins, where it reacts to the macro inputs rather than to the loans (`backtest_psi.csv`).
As in the valuation study, drift statistics on inputs and scores did not flag the error that mattered (L6). An
outcome trigger is proposed: review the model when the trailing 24-month predicted ÷ actual charge-offs leaves
0.80–1.25.

## 11. Remediation: three rounds, each with its rule written before it ran

| Round | Candidates | Rule | Result |
|---|---|---|---|
| D4 | M1v approval eras; M1r trailing recalibration | Out of time 0.80–1.25 (excluding 2020-04..2021-12) and 24-month backtest within ±25% at ≥ 6 of 8 origins | Not met: 0.74 and 4/8; 0.93 and 3/8 |
| D5 | Eras plus recalibration (R2a), downturn exposure (R2b), both (R2c); relief overlay O1 on all | Same | Not met. Best: M1v + O1 0.81 and 4/8; R2a + O1 0.73 and 5/8 |
| D6 | Lifetime-to-date backtest of the book at six origins, 2009–2019 | Within ±25% at ≥ 5 of 6 | Not met: R2b 4/6, M1v and R2a 3/6, lookup 2/6, champion 0/6 |

Lifetime-to-date predicted ÷ actual charge-off dollars (`r3_lifetime_backtest.csv`):

| Origin | Lookup | Champion | M1v + O1 | R2a + O1 | R2b + O1 |
|---|---|---|---|---|---|
| 2009-12 | 0.55 | 0.39 | 0.40 | 0.42 | 0.96 |
| 2011-12 | 0.93 | 0.69 | 0.57 | 0.58 | 0.52 |
| 2013-12 | 1.01 | 0.65 | 0.38 | 0.30 | 0.37 |
| 2015-12 | 2.54 | 1.81 | 0.97 | 1.19 | 1.16 |
| 2017-12 | 3.41 | 2.47 | 1.11 | 1.14 | 1.22 |
| 2019-12 | 3.14 | 2.24 | 1.03 | 0.94 | 1.09 |

- **What the remediation fixed.** Out of time, outside the relief months, predicted ÷ actual moves from 1.75 for the
  champion to 0.72–0.83 across the second-round candidates with O1 (M1v 0.81, or 0.74 without O1; §7), and over the
  whole window from 2.79 to 0.91 for M1v + O1; the typical 24-month backtest error halves (mean absolute log ratio
  from 0.68 to 0.33 for R2a + O1 and 0.36 for M1v + O1); the 24-month forecast from 2019-12 falls from 7.2 to 1.6
  times actual.
- **No candidate meets both rules.** M1v + O1 and R2b + O1 meet the out-of-time rule but reach 4 of 8 backtest
  origins; R2a + O1 has the best backtest (5 of 8) but is at 0.73 out of time.
- **Where it holds.** Once the fit covers the crisis and the recovery (origins 2015, 2017, 2019), all three
  remediated models are within ±25% of the book's lifetime losses at every origin, where the champion overstates them
  1.8–2.5 times and the lookup 2.5–3.4 times.
- **Where it does not (L10).** Fitted on four to eight years (origins 2009–2013), the logits under-predict lifetime
  losses in 11 of 12 model-origin pairs (0.30–0.69; the downturn-exposure model is the exception at 2009): the fit has
  not yet seen how long crisis-era defaults take to be written off.
- **Decision.** No candidate meets a pre-set rule, so none is adopted and the vintage finding stays open.
  Recommendation, outside the pre-set rules and labelled as such: use M1v with O1 for the lifetime allowance only
  when it is fitted on at least one full credit cycle, as the 2026 fit is (2006–2026); report the range across M1v,
  R2a and R2b; do not use it for 24-month or quarterly charge-off forecasts.

## 12. Findings log

| # | Severity | Status | Finding | Use limit or remediation |
|---|---|---|---|---|
| L1 | High | Closed | Recorded term revised after default; any term-based feature leaks the outcome (challenger AUC 0.935 with it, 0.737 without) | Term not used as a predictor or for censoring |
| L2 | High | Open | Cohort quality is confounded with macro sensitivity: in-sample predicted ÷ actual 0.69–0.77 for FY2006–07 loans and 1.28–2.34 for FY2009–18; out of time 2.8 | Era effects halve the unemployment coefficient and bring the level to 0.74 (0.81 with O1); three rounds, no pre-set rule met (§11) |
| L3 | High | Open | Payment relief in 2020–21: 7.7-fold over-prediction with the realised macro path | Overlay O1 (documented, not estimated) brings the era model to 1.6 times |
| L4 | Medium | Open | The macro path changes 24-month forecasts by less than 3% at six of eight origins; helps once (2009), hurts once (2019) | Scenario-conditioned forecasts are assumptions more than estimates |
| L5 | Medium | Open | Model choice moves the lifetime estimate (3.4%–6.6% of balance, baseline; 3.4%–4.4% across remediated models) more than the scenario (+0.1 to +2.7 points) | Report the allowance as a range across models, not as one model's number |
| L6 | Medium | Open | PSI 0.02 while the forecast was 2.3 times actual (2017) | Outcome trigger on 24-month predicted ÷ actual |
| L7 | Medium | Info | Charge-off timing is administrative: four-year median lag, batches of up to 1,935 loans in a month | Read monthly backtests with the batches in view |
| L8 | Low | Open | Severity measured on original amount, not balance | Needs balance data |
| L9 | Low | Info | Interest rate missing FY2000–09; collateral flag recording changes | Not used |
| L10 | Medium | Open | Fitted on four to eight years, the logits under-predict the book's lifetime losses in 11 of 12 model-origin pairs (0.30–0.69 at the 2009–2013 origins) | Fit on at least one full credit cycle before use for an allowance |
| L11 | Low | Open | Business-age categories are coded differently across approval eras ("new" 24–31% to FY2008, 0% in FY2019; "unknown" 12–14% in FY2018–19) | Read business-age effects as partly era effects; test the models without the field |
| L12 | Low | Open | 2026Q1–Q3 macro inputs are scenario values, not published data (unemployment 4.6% against 4.1–4.3%) | Re-run with published values for elapsed quarters |

## 13. Use limits and limitations

- **Disposition.** Research and benchmark use only. No candidate met the pre-set rules, so none is approved for an
  allowance; the outputs are expected gross SBA write-offs under stated scenarios, not a lender's allowance.
- **Use.** The models rank loans by charge-off risk with modest discrimination (out-of-time AUC 0.68–0.74). For a
  lifetime allowance, the era model with the relief overlay is within ±25% of lifetime losses once it is fitted on a
  full cycle; it is not fit for 24-month forecasts, and its scenario sensitivity should be read as a model assumption.
- **Not tested.** Default timing, delinquency, balances and recoveries are not in the data; the guarantee is assumed
  to share losses pro rata; the book is SBA-guaranteed small-business lending, not a bank's own portfolio.
- **Data window.** One full credit cycle (2008–2011) and one policy-distorted shock (2020–21) are in the data.

## Reproduce

Everything here runs from public data: `scripts/00_download_sba.sh` (SBA 7(a) files and the Fed 2026 scenarios),
then scripts 40–47 in order (about two hours on a 32-core machine). The aggregate tables behind every number are in
`reports/p3/`.
