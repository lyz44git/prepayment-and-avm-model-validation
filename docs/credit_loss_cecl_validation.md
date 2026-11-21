# Small-business credit loss: charge-off hazard, severity and lifetime expected loss, with validation

Public SBA 7(a) loan data, as of 2026-09-30. Analysis plan written and hashed before any model was fitted
(`docs/credit_loss_plan.md`; hash in `reports/p3/plan_hash.json`); every change made afterwards is listed with
its reason in `docs/credit_loss_deviations.md` (D1–D6, including three remediation rounds whose rules were written before they ran).

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

- **The recorded term appears to be revised after a loan runs into trouble (L1).** 90% of charged-off loans have a term that is
  not a multiple of 12, against 10% of loans cancelled before disbursement (which keep their approved terms), 16% of
  active and 18% of paid-off loans; 120-month loans show a charge-off share of 0.4% against 12.5% overall
  (`term_field_check.csv`, `term_charge_off_share_by_term.csv`). The term therefore carries information about the
  outcome. In the first fit it gave LightGBM an out-of-time AUC of 0.935 (`reports/p3/first_fit/`); without it, 0.737.
  On the same rows, adding the term features back alone takes LightGBM from 0.737 to 0.935, and cutting the risk set
  at the recorded term without them gives 0.723 (`r6_term_ablation.csv`): the leak is the field, not the risk set.
  The term is not used as a predictor or for censoring (deviation D1).
- **Payoff dates before mid-2005 are bulk-stamped.** 45,770 payoffs in the population are dated 2005-05, against
  about 2,600 a month from 2005-08 (the plan, written before the population filters were final, counted 46,519). The risk window therefore starts in 2006-01, with loans disbursed earlier
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
the competing event occurred, at the same rate, which is what the correction assumes (`scripts/41_p3_panel.py`; the
design and correction follow [King and Zeng 2001](https://doi.org/10.1093/oxfordjournals.pan.a004868)).

| Model | Charge-off and payoff hazards |
|---|---|
| M0 lookup | Hazard by 12-month loan-age band × processing group |
| M1 logit (champion) | Cause-specific discrete-time logit: loan-age spline, log amount, guarantee share, processing group, business age and type, sector, franchise, month of year, state effects; charge-off adds the 4-quarter change in unemployment and the BBB–Treasury spread, lagged 12 months (lag chosen on the validation months from 0, 12, 24); payoff adds the prime rate, its 4-quarter change and the 4-quarter change in unemployment (unlagged; the plan listed only the first two, see the review notes in the deviations log) |
| M2 LightGBM | Same information; monotone increasing in the two macro terms for charge-off |
| Severity | S0 lookup (age band × processing), S1 fractional logit, S2 LightGBM, on charge-off amount ÷ original amount |

Expected loss of loan *i* in month *t* is S(t−1) × h_charge-off(t) × severity(t) × original amount, with S the
survival from both causes. The lender's share multiplies by one minus the guarantee share (7(a) losses are shared
pro rata). Hold-out states (MA, NJ, WI, MN, SC, KY, OK, AZ, WA) never enter a fit.

## 4. Does the challenger earn its complexity?

Fit 2006–2017, validation 2018–2019 (`eval_validation.csv`); then score 2020-01..2026-09 (`eval_hazard.csv`) with M0 and M1 refitted on 2006–2019. M2
keeps its 2006–2017 fit, with 2018–2019 used for early stopping, so the out-of-time comparison is not exactly
like for like.

| Charge-off hazard | AUC, in sample | AUC, out of time | Predicted ÷ actual, out of time |
|---|---|---|---|
| M0 lookup | 0.708 | 0.666 | 2.25 |
| M1 logit | 0.747 | 0.681 | 2.79 |
| M2 LightGBM | 0.821 | 0.737 | 2.43 |

- **Champion rule.** M2 replaces M1 only if it lowers the validation log loss by more than 1% and its validation
  predicted ÷ actual is within 0.90–1.10. M2 lowers log loss by 5.0% but over-predicts the validation months by
  1.66, so M1 stays champion by default (`champion.json`). The rule tests the challenger only: M1 itself
  over-predicts the validation months by 2.13, and out of time by 2.79 against M2's 2.43, both far from 1; only in
  the hold-out states in sample is M1 the closer of the two (1.09 against 0.71). It is retained for
  interpretability, not for accuracy.
- **What the rule shows.** LightGBM ranks loans better, but every model over-predicts the level of charge-offs
  after 2019 by a factor of more than two. Ranking is not the problem; the level is.
- **Payoff hazard.** Out of time, predicted ÷ actual is 1.12 for M1, 1.06 for M2 and 0.88 for M0 (AUC 0.66–0.68).
- **Hold-out states.** In the nine states no model saw, in sample, M1 is at AUC 0.734 and predicted ÷ actual 1.09, M2 at
  0.782 and 0.71, M0 at 0.702 and 1.16; out of time, M1 at 0.686 and 3.06, M2 at 0.731 and 1.77, M0 at 0.669 and 2.60.
  LightGBM ranks better in unseen states too but is miscalibrated there in sample. Severity transfers: S1 is at 1.04
  times actual dollars in the hold-out states (`eval_hazard.csv`, `eval_severity.csv`).

## 5. Did the loans change? Vintage, not only the economy

Predicted ÷ actual charge-offs of the 2006–2019 champion, by approval cohort, in sample (`vintage_pred_vs_actual.csv`):

| Approval years | Predicted ÷ actual |
|---|---|
| FY2000–FY2005 | 0.82–1.08 |
| FY2006–FY2008 | 0.77, 0.69, 0.82 |
| FY2009–FY2018 | 1.28–2.34 |

The model is calibrated overall in sample (1.00) because it averages a crisis whose losses came from loans approved
in FY2005–FY2008 with a decade of much safer loans. Its unemployment term was estimated on that one crisis. Adding
approval-era effects (deviation D4; `m1v_co_coefficients.csv`) halves the unemployment coefficient, from 0.215 to
0.103 per point of 4-quarter change (model-based standard errors 0.004, which ignore that the term varies only by
month and so overstate the precision), and the era effects are large: relative to FY2000–04,
+0.18 for FY2005–08, −0.87 for FY2009–13 and −1.24 for FY2014–19 in log odds. The forecast is therefore sensitive to
how the model represents the loans that were on the books during the recession (L2). The coefficient change is a
specification sensitivity, not a measured share of underwriting quality.

Two caveats bound that reading. Approval era, loan age and calendar time (which carries the macro terms) are
collinear, the age–period–cohort problem: a young era is observed only at young ages, so its effect rests on the
age profile estimated from older loans. And the era cut-points (FY2005, 2009, 2014, 2020) were chosen after the
vintage table above was seen, and other cut-points were not tried. With one downturn, the data cannot determine how
much of the coefficient change reflects underwriting quality; the size should be read as one decomposition among
several that fit the data.

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
effects. The plan's pass rule (within ±25%, that is 0.75–1.25, at 6 or more of 8 origins) is not met by any model
(`backtest_summary.json`). The 2013 origin is pulled down by the 2015 write-off batch (§2).

**After remediation (§11).** With approval-era effects and the relief overlay O1, the same backtest is within ±25% at
4 of 8 origins (M1v) or 5 of 8 with trailing recalibration (R2a), and the typical error halves (mean absolute log
ratio 0.36 and 0.33, against 0.68 for the champion; `r2_backtest.csv`). Half of that gain is the 2019 origin, where
the overlay uses hindsight: without that origin the error falls from 0.49 to 0.34 and 0.32, by about a third.
The 2019 origin itself stays outside the band (1.6 and 1.4 times actual). The other misses fall at the 2011, 2013 and
2015 origins (M1v about 1.5, 0.35 and 0.65 times actual; R2a 1.6, 0.28 and 0.79, inside the band at 2015), when SBA
wrote off the crisis-era loans it had been liquidating: the book's 24-month charge-offs were $1.17bn after 2011-12 and
$3.01bn after 2013-12, during an expansion. None of the tested predictors captures that schedule.

**The macro path rarely matters (L4).** Realised and held-at-origin forecasts of the champion differ by less than 3%
at six of eight origins. The exceptions are 2009 (realised 0.98 against 1.23 held, the one origin where the path
helps) and 2019, where the path makes the forecast more than twice as wrong (§7). Part of this is mechanical: with
the 12-month lag, the first 12 of the 24 forecast months use the same inputs under both paths.

## 7. When policy breaks the link: 2020–2021

From the 2019-12 origin, charge-off dollars in 2020-04..2021-12 were $0.82bn. The champion with the realised macro
path predicted 7.7 times that; with macro held at 2019-12, 3.1 times; with era effects, 2.2 and 1.5 times
(`pandemic_check.json`, `backtest_rolling_monthly.csv`). Unemployment rose by 9.2 points in a quarter (quarterly averages, 3.8% in 2020Q1 to 13.0% in Q2;
`r5_unemployment_2020.csv`) while SBA made
borrowers' loan payments for several months under the CARES Act, so the historical link between unemployment and
write-offs did not hold (L3). The plan's rule (record a finding above 1.25) is met. A macro-driven loss model needs
an explicit overlay, or its macro terms switched off, when government support changes borrowers' payments. The
overlay O1 (deviation D5) holds unemployment and the BBB spread at their 2020Q1 values for 2020Q2–2021Q3, the span
of SBA's payments under the CARES Act (from April 2020) and the Economic Aid Act (from February 2021); for most
loans the payments paused for a few months in between, which O1 does not model. With it, the era model's forecast for April 2020 to December 2021 falls from 2.2 to
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
(`annual_chargeoffs.csv`). Out of time, on loans that charged off, the fractional logit's
predicted severity × original amount is within 2% of the charge-off dollars with severity capped at [0, 1]
(predicted ÷ actual 1.02 against capped dollars, 1.01 against recorded dollars; mean absolute error 0.17 per loan; S0 1.06, S2 0.98; `eval_severity.csv`). This tests
severity given a charge-off; the backtests test the full chain against recorded dollars. Severity
depends mainly on the loan's age at charge-off and its guarantee share; its unemployment coefficient is −0.02
(`s1_severity_coefficients.csv`). The difference between scenarios therefore comes from the hazards, not from
severity: mainly the charge-off hazard, and partly the payoff hazard, which also responds to the macro path
(coefficients −0.107 on the unemployment change and +0.121 on the prime rate; `m1_pif_coefficients.csv`), so loans
live longer under the severe path. Split exactly into the three parts (Shapley values over all eight combinations of
baseline and severe inputs; §14), the era model's 0.41-point increment is 0.32 from the charge-off hazard, 0.08 from
slower payoffs and less than 0.01 from severity; for R2b it is −0.13, +0.07 and less than 0.01, so its severe estimate
ends below its baseline. Severity is measured against the original amount because the data have no balances (L8).

## 9. Lifetime expected loss at 2026-09-30

Book: the 232,292 eligible loans active at 2026-09 (in the risk window; loans first disbursed in 2026-09 enter the
risk set only the next month and are not in it, while loans that paid off in 2026-09 but are not yet recorded are, L13),
$162.2bn approved, $128.2bn estimated outstanding (level-payment schedule from the recorded rate and term, missing
rates set to the median). Final models fitted on 2006-01..2026-09. Horizon: each loan's remaining recorded term (at
least 12 months). Scenarios: Fed 2026 baseline and severely adverse from their 2026Q1 start to 2029Q1; after that the
levels revert in a straight line over four quarters to their 2006–2025 average and the 4-quarter changes are taken from
the reverted levels (deviation D3; L16). The scenarios are hypothetical paths, not
forecasts, and the interagency policy statement on allowances does not prescribe a four-quarter reversion or reverting
inputs rather than losses: both are this study's assumptions. The estimate is of gross charge-off dollars as SBA
records them, before recoveries. Lifetime loss over estimated balance (`cecl_allowance.csv`):

| Model | Baseline | Severely adverse | Severe − baseline |
|---|---|---|---|
| M1v logit + approval eras | 3.4% | 4.0% | +0.6 pt |
| M1r logit, trailing recalibration (factor 0.81) | 4.7% | 6.8% | +2.1 pt |
| M1 logit (champion) | 5.8% | 8.4% | +2.5 pt |
| M2 LightGBM | 6.9% | 8.6% | +1.6 pt |
| M0 lookup (no macro) | 9.9% | 9.9% | 0 |

*Update note (2026-11-19).* The 4-quarter macro changes after 2029Q1 now follow the reverted levels (L16); before,
they reverted to their own averages. The earlier runs put the candidates at 3.4–4.4% under the baseline, the severe
scenario at +0.12 to +0.82 points, and the champion at 5.7%; their tables are in `before_2026-11-19/`, and
`r5_allowance_update_bridge.csv` sets every estimate beside its update.

- **Model choice moves the allowance more than the scenario does (L5).** Across the four models with macro terms
  the baseline estimate runs from 3.4% to 6.9% of balance, a spread of 3.5 points; the severely adverse scenario
  adds 0.6 to 2.5 points. For comparison, the rolling backtest error of the best of them (M1v) is a factor of 1.5
  on a typical origin (mean absolute log ratio 0.39).
- **How much the scenario matters is itself a model choice.** M1, whose unemployment coefficient absorbs part of the
  cohort shift (§5), adds 2.5 points under the severely adverse path; M1v adds 0.6.
- **Lender's share.** Losses are shared pro rata with SBA loan by loan, so the lender's retained loss is about the
  same percentage of its retained balance (3.40% against 3.42% for the whole book; M1v baseline $1.05bn of
  $4.38bn in total); the two differ slightly because guarantee shares vary across loans. This applies the guarantee percentage
  mechanically; a lender's allowance would also need its actual exposure and contract terms, expected recoveries and
  the accounting for the guarantee it holds.
- **Macro inputs for 2026Q1–Q3 (L12).** The Fed 2026 files end their history in 2025Q4, so in the estimation window
  2026Q1–Q3 carry baseline-scenario values (unemployment 4.6% against a published 4.1–4.3%; prime 6.2–6.6% against
  6.8%; `macro_2026_scenario_vs_published.csv`), which enter the payoff hazard directly. In the projection, each
  scenario runs from its own 2026Q1 start, so its 2026Q1–Q3 values (severely adverse: unemployment 5.9–8.2%)
  reach the charge-off hazard of 2027 through the 12-month lag. Re-run with the published values for those quarters,
  shared by both scenarios, and each scenario starting in 2026Q4 (§14), the baseline estimates move by 0.01 points or
  less and the severe ones rise by 0.08–0.10 points.
- **Horizon.** Allowing 24 months beyond the recorded term for charge-offs posted after maturity adds 0.25 points
  (M1v) to 0.48 points (M1) under the baseline.
- **After remediation.** With approval eras and the relief overlay, the candidates put lifetime loss at 3.5% (M1v),
  3.5% (R2b) and 4.5% (R2a) of balance under the baseline and 3.9%, 3.4% and 5.0% under the severely adverse
  scenario (`r2_cecl_allowance.csv`); the champion with the overlay is at 6.2% and 8.2%. The range across
  candidates, 1.0 point under the baseline, is wider than the scenario effect of each (−0.05 to +0.53 points). For R2b
  and R2c the severe estimate is slightly below the baseline. After 2029Q1 unemployment rises back to its long-run
  average in the baseline and falls back to it from 8% in the severe path, so the reversion offsets part of the
  stress; and the downturn-exposure term of R2b has a negative coefficient (−0.076, s.e. 0.003;
  `r2b_co_coefficients.csv`), the opposite of its rationale in D5. R2b's scenario effect should be read as a
  sensitivity, not as a better-specified model, and the reversion assumption as part of the scenario.
- **Segments.** Express loans carry the highest estimated loss (M1v baseline 4.4% of balance against 3.4% for
  Preferred Lenders loans; `cecl_by_segment.csv`).

## 10. Monitoring: drift statistics against outcomes

The population stability index of the book's 24-month charge-off probability between successive origins is 0.02
between 2015-12 and 2017-12, while the champion's 2017-12 forecast turned out 2.3 times actual; it is above 2 only
across the pandemic origins, where it reacts to the macro inputs rather than to the loans (`backtest_psi.csv`).
The statistic flagged the 2011 and 2013 transitions (PSI 0.34 and 0.40, above the usual 0.25 alarm) but missed
the 2017 error and the 2023 error (PSI 0.009 while the 2023 forecast was 2.00 times actual). It is also not a
statistic anyone could have computed at the time: each origin's model is refitted on data to that origin and scored
with the realised macro path. As in the valuation study, drift statistics did not reliably flag the errors that
mattered (L6). An outcome trigger is proposed: review the model when the trailing 24-month predicted ÷ actual
charge-offs leaves 0.80–1.25. Back-tested with information available at each origin (§14), that trigger fired at
5 of 8 origins, each followed by a miss, and stayed quiet at the only other miss (2011); a PSI computed without
future data flagged 2 of the 6 misses.

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
- **What the lifetime test shows, and its limits.** At the three latest origins (2015, 2017, 2019) the three
  candidates are within ±25% of the book's losses to date (0.94–1.22), where the champion overstates them 1.8–2.5
  times and the lookup 2.5–3.4 times. This is an observation made after the results were seen, not a passed test:
  the three books overlap and share the 2020–26 outcome window, the losses are to date (81 months for the 2019
  origin), and every run in the table uses O1, which knows about the 2020–21 relief. Without O1 the counts are the same (best four of six) and the three candidates stay within
  ±25% at the three latest origins (0.87–1.21; `r3_lifetime_backtest_no_overlay.csv`), so the pattern does not come
  from the overlay.
- **Where it does not (L10).** Fitted on four to eight years (origins 2009–2013), the logits under-predict lifetime
  losses in all 12 model-origin pairs, in 11 of them by more than 30% (0.30–0.69; the downturn-exposure model at
  2009 is at 0.96): the fit has not yet seen how long crisis-era defaults take to be written off.
- **The era model is not calibrated by cohort out of time (L14).** Its whole-window 0.98 hides offsetting errors
  (`remediation_vintage.csv`): loans approved FY2014–19, 62% of out-of-time charge-offs, are at 0.70–0.82 of actual,
  FY2022–23 at 0.64–0.71, while FY2009–13 are at 1.27–1.78 and the oldest cohorts (FY2000–08) at 2.8–8.8, where loan
  age runs past the last spline knot. In the 2026 final fit the trailing factor of R2a is 1.29: the era model
  under-predicts its own last 24 months by about a fifth. The 3.5% lower end of the allowance range therefore assumes
  that this recent under-prediction does not persist; the 4.5% upper end assumes it does.
- **What the rules did not prevent.** Each round's rule was fixed before it ran, but each round's candidates were
  designed after the previous round's results, the relief overlay with knowledge of 2020–21, and the same 2020–26
  window was scored three times. The best-of-candidates figures (5 of 8, 4 of 6, 0.72–0.83) are therefore optimistic
  as measures of out-of-sample performance.
- **The backtests are retrospective.** Each refits a fixed specification at each origin, but the 12-month macro lag
  was chosen on the 2018–19 validation months and the macro history comes from the 2026 files, which carry later
  revisions. They test the specification on realised, revised inputs, not forecasts that could have been made at
  each origin. Choosing the lag at each origin from the data available then does not help (§14).
- **The lifetime test is not the allowance projection.** The lifetime backtest projects every loan to 2026-09; the
  allowance stops at each loan's remaining recorded term, a horizon rule that cannot be backtested because the term
  field is revised after default (L1).
- **Decision.** No candidate meets a pre-set rule, so none is adopted and the vintage finding stays open.
  Recommendation, outside the pre-set rules and labelled as such: if a lifetime estimate is needed for research or
  benchmarking, fit the era model on at least one full credit cycle, as the 2026 fit is (2006–2026), and report the
  range across M1v, R2a and R2b; do not use any of them for 24-month or quarterly charge-off forecasts.

## 12. Findings log

| # | Severity | Status | Finding | Use limit or remediation |
|---|---|---|---|---|
| L1 | High | Closed | Recorded term appears revised after default; any term-based feature leaks the outcome (challenger AUC 0.935 with it, 0.737 without) | Term not used as a predictor or for censoring. Closed for the hazard models only: the term still sets eligibility, the projection horizon and the estimated balance (§9) |
| L2 | High | Open | Cohort quality and macro sensitivity are confounded: in-sample predicted ÷ actual 0.69–0.82 for FY2006–08 loans and 1.28–2.34 for FY2009–18; out of time 2.8 | Era effects halve the unemployment coefficient and bring the level to 0.74 (0.81 with O1); three rounds, no pre-set rule met (§11) |
| L3 | High | Open | Payment relief in 2020–21: 7.7-fold over-prediction with the realised macro path | Overlay O1 (documented, not estimated) brings the era model to 1.6 times |
| L4 | Medium | Open | The macro path changes 24-month forecasts by less than 3% at six of eight origins (five with the payoff lag fixed, §14); helps once (2009), hurts once (2019) | Scenario-conditioned forecasts are assumptions more than estimates |
| L5 | Medium | Open | Model choice moves the lifetime estimate (3.4%–6.9% of balance, baseline; 3.5%–4.5% across remediated models) more than the scenario (−0.05 to +2.5 points) | Report the allowance as a range across models, not as one model's number |
| L6 | Medium | Open | PSI 0.02 while the forecast was 2.3 times actual (2017) | Outcome trigger on trailing 24-month predicted ÷ actual (back-tested in §14: fired before 5 of 6 misses, no false alarm) |
| L7 | Medium | Info | Charge-off timing is administrative: four-year median lag, batches of up to 1,935 loans in a month | Read monthly backtests with the batches in view |
| L8 | Low | Open | Severity measured on original amount, not balance | Needs balance data |
| L9 | Low | Info | Interest rate missing FY2000–09; collateral flag recording changes | Not used |
| L10 | Medium | Open | Fitted on four to eight years, the logits under-predict the book's lifetime losses in all 12 model-origin pairs, 11 by more than 30% (2009–2013 origins) | Fit on at least one full credit cycle before use for an allowance |
| L11 | Low | Open | Business-age categories are coded differently across approval eras ("new" 24–31% to FY2008, 0% in FY2019; "unknown" 12–14% in FY2018–19) | Read business-age effects as partly era effects; test the models without the field |
| L12 | Low | Closed | 2026Q1–Q3 macro inputs are scenario values, not published data (unemployment 4.6% against 4.1–4.3%) | Re-run with published values and the scenarios from 2026Q4 (§14): baseline within 0.01 points, severe up 0.08–0.10 |
| L13 | Low | Open | Recording lag: 2026-09 has no recorded payoffs (about 2,300 in a typical month), so loans that paid off that month are still in the active book | End the payoff window a month earlier; see §14 |
| L14 | Medium | Open | The era model under-predicts recent cohorts out of time (FY2014–19 at 0.70–0.82, FY2022–23 at 0.64–0.71) while the whole window looks calibrated (0.98) | Report the allowance as a range; read the lower end as optimistic |
| L15 | Low | Closed | Input dates: 16 charge-offs with no date were dated at the 2026-09 cutoff and 9 loans first disbursed more than a year before approval were kept (25 of 994,992 loans; 16 of 115,117 charge-offs in the risk window; `r5_input_date_checks.csv`) | Dropped in `scripts/40_p3_build.py`; rebuilt with the checks, no pass count changes and headline figures move in the second or third decimal (§14) |
| L16 | Medium | Closed | Scenario path: after 2029Q1 the unemployment and prime-rate changes reverted to their own averages, not with their levels (in the severe path the change fed to the models was near zero while the level fell 2.2–2.5 points; `r5_scenario_path_reconciliation.csv`) | Changes taken from the reverted levels; lifetime loss re-run (§9) |

## 13. Use limits and limitations

- **Disposition.** Research and benchmark use only. No candidate met the pre-set rules, so none is approved for an
  allowance; the outputs are scenario-conditional expected gross SBA write-offs, not a lender's allowance, and the
  range across specifications is not a confidence interval.
- **Use.** The models rank loans by charge-off risk with modest discrimination (out-of-time AUC 0.68–0.74). Their
  level is not reliable enough for 24-month forecasts, and their scenario sensitivity should be read as a model
  assumption. A lifetime estimate from the era model is a benchmark with a range, not a tested allowance, and its
  lower end rests on recent cohorts behaving better than the model has seen them behave (L14).
- **Not tested.** Default timing, delinquency, balances and recoveries are not in the data; the guarantee is assumed
  to share losses pro rata; the book is SBA-guaranteed small-business lending, not a bank's own portfolio.
- **Data window.** One full credit cycle (2008–2011) and one policy-distorted shock (2020–21) are in the data.

## 14. Robustness checks

These were run after the three remediation rounds. None changes a decision
above; each is reported beside, not in place of, the original result.

**24-month backtest with the payoff lag fixed** (`scripts/48_p3_backtest_v2.py`, `r4_backtest_rolling_v2.csv`,
`r4_backtest_summary_v2.json`, `r4_pandemic_check_v2.json`). Script 43 fitted the payoff logit with the unemployment
change at the charge-off lag and scored it unlagged; the rerun fits and scores it unlagged. Dollar ratios move
by at most 0.05 and no pass count changes (M1 2, M1v 4, M1r 3 of 8 origins). The champion's mean absolute log ratio
goes from 0.679 to 0.684 (0.493 to 0.499 without the 2019 origin), the pandemic-window ratio from 7.69 to 7.66, and the
number of origins where the realised and held macro paths differ by less than 3% from six to five (2021 at 4.1%).

**Monitoring with information available at each origin** (`scripts/49_p3_monitoring.py`, `r4_monitoring_trigger.csv`,
`r4_psi_exante.csv`).

| Signal | Fired and the next 24 months missed | Fired, no miss | Quiet, missed | Quiet, no miss |
|---|---|---|---|---|
| Trailing 24-month predicted ÷ actual charge-off counts outside 0.80–1.25 | 5 | 0 | 1 (2011) | 2 |
| Same on dollars | 4 | 1 (2015) | 2 (2011, 2017) | 1 |
| PSI of the 24-month score, previous model, macro held at the new origin, above 0.25 | 2 | 0 | 4 | 1 |

The count trigger is the better alarm, with two caveats: its trailing window is in sample at each origin, and at
2013 it fired in the opposite direction to the miss (trailing 1.49, next 24 months 0.56). The ex-ante PSI, unlike the
PSI in §10, did not flag the 2019 origin: the earlier 2.40 came from realised pandemic macro.

**Allowance sensitivities** (`scripts/50_p3_allowance_sensitivities.py`, `r4_allowance_sensitivities.csv`; one change
at a time; lifetime loss as % of estimated balance, baseline / severely adverse):

| Variant | M1v | R2a | R2b |
|---|---|---|---|
| As in §9 | 3.46 / 3.87 | 4.48 / 5.01 | 3.47 / 3.42 |
| Payoff hazard scaled to its out-of-time level (÷ 1.117) | 3.64 / 4.04 | 4.71 / 5.22 | 3.65 / 3.58 |
| Reversion over 8 quarters instead of 4 | 3.44 / 3.90 | 4.46 / 5.04 | 3.49 / 3.44 |
| R2a factor phased out after the scenario (2029Q1) | | 3.98 / 4.57 | |
| FY2020+ loans given the FY2014–19 era effect | 4.01 / 4.48 | 4.66 / 5.20 | |

- The reversion speed hardly matters (0.03 points or less); the payoff calibration adds about 0.2 points. The
  table uses the scenario path of §9; the earlier values are in `before_2026-11-19/r4_allowance_sensitivities.csv`.
- The two assumptions that move single estimates most are about recent cohorts. Loans approved FY2020 or later are
  80% of the active loans and 86% of the estimated balance (`r5_active_book_by_approval_era.csv`), and their
  fitted era effect is lower than FY2014–19's (−1.14 against −0.96 in log odds); giving them the earlier cohorts'
  effect raises the era model from 3.5% to 4.0%. Phasing out R2a's recalibration factor after the scenario lowers it
  from 4.5% to 4.0%. These are different changes applied to different models, so they do not show where the range is
  centred.
- Scenario decomposition: for M1v, 0.31 of the 0.41-point severe increment comes from switching the charge-off hazard
  and severity together and 0.10 through slower payoffs; for R2a 0.41 and 0.13; for R2b −0.13 and 0.08. Severity on
  its own contributes less than 0.01 points (below).
- Recording lag (L13): 2026-09 is the only month in 2006–2026 with no recorded payoffs (2,030 predicted). A typical
  month has about 2,290 payoffs, 0.99% of the active book. The payoff out-of-time ratio is 1.105 without that month
  (`r4_recording_lag.json`).
- Final-fit coefficients of M1v and R2b are in `r4_final_fit_coefficients.csv` (unemployment change 0.119 and 0.160;
  downturn exposure −0.077).

**Inference across months** (`scripts/51_p3_inference.py`, `r4_inference_month_clustered.csv`,
`r4_dur4_difference_bootstrap.json`). Standard errors clustered by calendar month are 4–7 times the model-based ones.
The unemployment-change coefficient stays significant (z = 8.8 in M1, 4.5 in M1v), the approval-era effects stay
significant (|z| 4.2–13.2 in M1v), the BBB spread is not (|z| below 1), and R2b's downturn-exposure term is at z = −3.7. A
100-replicate month-cluster bootstrap puts the drop in the unemployment coefficient when era effects are added at
0.112 (95% interval 0.095–0.166; no replicate at or below zero). Month clusters ignore correlation across months, so
these intervals may still be too narrow; the next check allows for it.

**Correlation between months** (`scripts/55_p3_inference_time_blocks.py`, `r6_inference_time_blocks.csv`,
`r6_dur4_difference_time_blocks.json`). Newey–West standard errors on the monthly sums of the score, with bandwidths of
12 and 24 months, are two to three times the month-clustered ones for the macro terms.

| Term | z, month clusters (above) | z, Newey–West 12 months | z, Newey–West 24 months |
|---|---|---|---|
| Unemployment change, M1 | 8.8 | 3.6 | 3.2 |
| Unemployment change, M1v | 4.5 | 2.2 | 2.0 |
| Approval-era effects, M1v (smallest absolute z) | 4.2 | 2.1 | 2.2 |

The era model's unemployment term is at the edge of conventional significance once neighbouring months are allowed to
be correlated. The drop in the coefficient when era effects are added stays clearly positive: 0.112 with a Newey–West
standard error of 0.035 (12 months) or 0.042 (24 months), and a 12-month moving-block bootstrap interval of 0.036–0.173
(0.15% of draws at or below zero).

**Published macro history and the scenario channels** (`scripts/54_p3_allowance_history_and_channels.py`,
`r6_allowance_published_history.csv`, `r6_scenario_channels.csv`). With the published 2026Q1–Q3 unemployment, prime
rate and Treasury yield shared by both scenarios and each scenario starting in 2026Q4 (L12), the models refitted and the
book re-projected:

| Model | Baseline, as in §9 / published history | Severely adverse, as in §9 / published history |
|---|---|---|
| M1v | 3.46 / 3.45 | 3.87 / 3.95 |
| R2a | 4.48 / 4.47 | 5.01 / 5.11 |
| R2b | 3.47 / 3.48 | 3.42 / 3.53 |

The baseline barely moves. The severe estimates rise because the shifted path puts more of the stress after the data
date, which raises the scenario effect from −0.05 to +0.53 points to +0.05 to +0.64. Model choice still moves the
estimate more. Exact decomposition of the severe-minus-baseline increment (Shapley values; the two one-at-a-time orders
differ from them by 0.015 points or less):

| Model | Charge-off hazard | Payoff hazard | Severity | Total |
|---|---|---|---|---|
| M1v | +0.32 | +0.08 | +0.006 | +0.41 |
| R2b | −0.13 | +0.07 | +0.005 | −0.05 |

**Severity, the projected book and the term field** (`scripts/56_p3_severity_book_term_checks.py`).
- *Severity cap* (`r6_severity_capping.csv`). Out of time, 411 of 14,808 charge-offs exceed the original amount, and
  capping severity at 1 removes 0.8% of recorded charge-off dollars. Against recorded dollars the fractional logit is at
  1.01 (1.02 against capped dollars, §8).
- *Book* (`r6_book_bridge.csv`). The 2026-09 projection leaves out 354 active loans first disbursed in 2026-09 (they
  enter the risk set the next month). It includes 5,047 loans past their recorded term, which have a zero estimated
  balance but are projected for 12 months, so they add loss but no balance, and 2,790 loans with no recorded rate,
  which take the median. Each backtest origin leaves out the 2,400–3,600 loans first disbursed in the origin month.
- *Term field* (`r6_term_ablation.csv`). LightGBM fitted as in §4 with and without the term features (recorded term,
  months to maturity), on the published risk set and on one cut at the recorded term: out-of-time AUC 0.737 and 0.935
  on the published rows, 0.723 and 0.926 on rows cut at the recorded term (a stricter risk set than the first fit's). On the same rows the
  term features alone move the AUC from 0.737 to 0.935, so the field itself carries the leak.
- *Ranking within months* (`r6_within_month_auc.csv`). Computed within each calendar month, so that differences in
  level between months do not count as ranking, the out-of-time AUC is 0.710 for the logit and 0.741 for LightGBM
  (pooled: 0.681 and 0.737).

**The lag chosen at each origin** (`scripts/53_p3_backtest_lag_by_origin.py`, `scripts/58_p3_backtest_lag_by_origin_r2.py`,
`r5_backtest_lag_by_origin.csv`, `r6_backtest_lag_by_origin_r2.csv`). The 12-month charge-off lag was chosen on the
2018–19 validation months, which an origin before 2019 could not have known. At each origin T the lag (0, 12 or 24
months) is here chosen by the event-weighted log loss of the champion fitted to T − 24 and scored on T − 23 .. T; the
models are then refitted to T with that lag and run as in the backtest with the payoff lag fixed (53) or the second round with the
relief overlay (58). Origins within ±25%, of 8:

| Model | Lag fixed at 12 | Lag chosen at each origin | Same, tables rebuilt with the L15 checks |
|---|---|---|---|
| M1 logit (champion) | 2 | 1 | 2 |
| M1v logit + approval eras | 4 | 3 | 4 |
| M1r trailing recalibration | 3 | 1 | 2 |
| M1v + O1 | 4 | 3 | 4 |
| R2a + O1 (recalibrated) | 5 | 4 | 5 |
| R2b + O1 (downturn exposure) | 4 | 3 | 3 |
| R2c + O1 | 4 | 4 | 4 |

Choosing the lag in real time does not help: no model gains an origin. Most of the loss comes from the 2009 origin, where
the choice rests on a fit of two years (2006–07). That choice is unstable: it is 24 months on the published tables and 12
on the rebuilt ones, which differ only by 25 loans with date errors. The fixed lag therefore does not explain the
failure to meet the rule, and the earliest origins cannot support a choice of lag.

**Rebuild with the input-date checks** (`scripts/59_p3_rebuild_l15.sh`, `scripts/59_p3_rebuild_comparison.py`,
`r6_rebuild_l15_comparison.csv`). Every credit-loss table was rebuilt from the public files with the L15 checks on
(994,967 loans; 115,101 charge-offs in the risk window), and the checks above re-run. No fixed-lag pass count and no adoption
decision changes. With the lag chosen at each origin, five of seven counts move by one (M1 1 to 2, M1v 3 to 4, M1r 1
to 2, M1v + O1 3 to 4, R2a + O1 4 to 5): dropping 25 loans renumbers the loans, and the 5% case-control sample is drawn
by a hash of the loan number, so these refits use a different sample. The figures quoted in this report move in the second or third decimal: out-of-time predicted ÷ actual 2.79 to
2.81 (1.75 to 1.77 outside the relief months); LightGBM's out-of-time AUC 0.737 to 0.736; the candidates' lifetime loss
3.46–4.48% to 3.45–4.45% of balance; the unemployment coefficients 0.215 and 0.103 to 0.216 and 0.104. The one result
that moves more is the lag chosen at the 2009 origin (above). The published tables are kept; the rebuilt ones are
produced by the script.

## 15. Related literature

Full citations are in `docs/references.md`.

- **Estimator.** The charge-off and payoff hazards are cause-specific discrete-time hazards estimated as logits, the
  approach of [Shumway (2001)](https://doi.org/10.1086/209665) and [Campbell, Hilscher, and Szilagyi (2008)](https://doi.org/10.1111/j.1540-6261.2008.01416.x).
- **Why the level fails out of time.** [Rajan, Seru, and Vig (2015)](https://doi.org/10.1016/j.jfineco.2014.09.012) link the failure of a default model to lenders'
  changing use of soft information after securitization. Here, too, forecasts deteriorate across cohorts (§5, §11),
  but the SBA data do not observe soft information, so they do not identify why the cohorts changed. [Demyanyk and Van
  Hemert (2011)](https://doi.org/10.1093/rfs/hhp033) document vintage effects in loan quality after controlling for observables; the approval-era effects
  here are a conditional analogue.
- **What observables miss.** [Das et al. (2007)](https://doi.org/10.1111/j.1540-6261.2007.01202.x) find default clustering beyond a model with
  observable intensities, and [Duffie et al. (2009)](https://doi.org/10.1111/j.1540-6261.2009.01495.x) model it with a common frailty factor. The SBA
  write-offs also cluster (§2), but administrative batching and a common shock to borrowers would both produce that,
  and these records cannot separate the two; no frailty term is estimated here.
- **Machine learning.** [Fuster et al. (2022)](https://doi.org/10.1111/jofi.13090) study machine-learning default models
  in mortgage lending. Here LightGBM ranks better than the logit and over-predicts less out of time (2.43 against 2.79,
  §4), but it still fails the calibration rule, so it does not address the level failure.
- **Policy regimes.** The 2020–21 payment relief broke the estimated link between unemployment and write-offs (§7),
  the forecast risk when policy changes loan payments that [Lucas (1976)](https://doi.org/10.1016/S0167-2231(76)80003-6) warns of; the size of the relief overlay is a
  modelling judgement, not an estimate of how borrowers or lenders responded.
- **Research practice.** The plan hashed before fitting, the deviations log and the reporting of failed rules are this
  study's response to the transparency [Harvey (2017)](https://doi.org/10.1111/jofi.12530) calls for; the adaptive design across rounds (§11) is the risk he
  warns about.
- **Guidance.** The report follows the structure of the 2011 model risk guidance (SR 11-7), since replaced by SR 26-2 /
  OCC Bulletin 2026-13 (April 2026). The lifetime-loss
  framing follows ASU 2016-13; the reversion beyond the scenario period is a choice the 2023 interagency policy
  statement on allowances leaves to management.

## Reproduce

Everything here runs from public data: `scripts/00_download_sba.sh` (SBA 7(a) files and the Fed 2026 scenarios),
then scripts 40–47 in order; the robustness checks are scripts 42e and 48–59. `scripts/40_p3_build.py` applies the
L15 input checks; set `LEGACY_FILTERS = True` there to rebuild the loan table exactly as for the published tables. The
aggregate tables behind the reported results are in `reports/p3/`.
