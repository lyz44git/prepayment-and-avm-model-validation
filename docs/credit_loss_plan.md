# Small-business loan losses: charge-off hazard, severity and lifetime expected loss — analysis plan

Written on 2026-11-18, before any model was fitted. It follows a data-quality profile of the files (status codes,
date coverage, field coverage by approval era, and the share of loans in each final status by approval year); no
loss rate by segment, month or model was computed before this plan. The SHA-256 of this file is recorded by the
model script before it fits anything (`reports/p3/plan_hash.json`).

## Purpose

Build and validate the parts of an allowance for credit losses (CECL) estimate on public loan-level data: a
monthly charge-off hazard (probability of default, timed at charge-off), a competing payoff hazard (which sets the
expected life), a severity model (charge-off amount over the original loan amount), and lifetime expected loss for
an active book under supervisory macro scenarios, with challengers, out-of-time and hold-out-state tests, a rolling
backtest and a findings log.

## Data

U.S. Small Business Administration, 7(a) FOIA loan files as of 2026-09-30 (FY2000–FY2009, FY2010–FY2019,
FY2020–present; U.S. Government Works). SHA-256 in `data_external/sba/SHA256SUMS.txt`. Macro inputs: Federal
Reserve 2026 supervisory scenarios (historic domestic file to 2025Q4; baseline and severely adverse from 2026Q1).

## Population

Term loans (RevolverStatus = N) with final status paid in full, charged off or active (EXEMPT); a first
disbursement date; term of at least 6 months; gross approval above zero; project in the 50 states or DC. Rows that
are identical in every field are kept once. Borrower names and addresses are dropped when the loan table is built
and never leave the working folder.

## Risk window and events

- Calendar months 2006-01 to 2026-09. Payoff dates before mid-2005 are bulk-stamped (46,519 payoffs dated 2005-05
  against about 2,600 a month from 2005-08), so earlier months cannot form a correct risk set.
- A loan is at risk from the later of its first full month after disbursement and 2006-01 (delayed entry, so loans
  disbursed earlier enter at their current age), until its event month or censoring at 2026-09 or 60 months after
  scheduled maturity, whichever comes first.
- Events: charge-off in the month of ChargeOffDate (the 20 dated 2026-10-01 are set to 2026-09), payoff in the
  month of PaidInFullDate. The charge-off date is when SBA wrote the loan off, after default and liquidation, so the
  hazard times charge-off, not default. This lag is described, not modelled.

## Covariates (fixed list)

- Loan age: natural cubic spline, knots at 6, 12, 24, 48, 84, 120 and 180 months.
- Months to scheduled maturity (clipped at 0) and an indicator for being past maturity.
- Term group: up to 84 months, 85–180, over 180.
- Log gross approval; SBA guarantee share of the loan.
- Processing group: Express (SBA Express, Community Express, Patriot Express, Export Express); Preferred Lenders
  (with or without EWCP); Standard (7a General, Certified Lenders, 7a with EWCP); Low Documentation; other.
- Business age group: startup (loan funds will open the business); new (2 years or less); existing (more than 2
  years, including change of ownership); unknown.
- Business type (corporation, individual, partnership); NAICS 2-digit sector (missing as its own level); franchise.
- Month of year; state fixed effects.
- Macro, national, quarterly values from the Fed files applied to each month of the quarter: 4-quarter change in
  the unemployment rate and the BBB corporate yield minus the 10-year Treasury yield, both lagged L months for
  charge-off, with L chosen from {0, 12, 24} on the validation months; the prime rate and its 4-quarter change
  (lag 0) for payoff.
- Not used, with the reason recorded as a finding: interest rate and fixed/variable flag (missing for loans approved
  FY2000–FY2008), collateral flag (its share of "Y" jumps from about 15% to over 80% between approval eras, a
  recording change), secondary-market sale flag (set after origination), jobs supported (self-reported), lender
  name (the current holder, not the originator).

## Models

- **M0, lookup.** Monthly hazard by 12-month loan-age band × term group × processing group, from the fit months.
- **M1, champion candidate.** Cause-specific discrete-time logit for charge-off and for payoff, on all events plus
  5% of non-event loan-months (selected by a deterministic hash of loan and month), intercept corrected by log 0.05.
- **M2, challenger.** LightGBM per cause on the same samples and features; charge-off monotone increasing in the
  unemployment change and the BBB spread; early stopping on the validation months; raw score shifted by log 0.05.
- **Severity.** s = charge-off amount ÷ gross approval, clipped to [0, 1]. S0: mean by loan-age band × term group.
  S1: fractional logit with the loan covariates, a spline in age at charge-off and the lagged macro terms. S2:
  LightGBM with a cross-entropy objective.
- **Expected loss** for loan i in month t: S(t−1) × h_chargeoff(t) × s(t) × gross approval, where S is survival
  from both causes. The lender's share multiplies this by (1 − guarantee share), since 7(a) losses are shared pro
  rata.

## Splits and champion rule

- Development months 2006-01 to 2019-12: fit on 2006-01 to 2017-12, validate on 2018-01 to 2019-12 (choice of L,
  M2 early stopping, champion). Then refit on 2006-01 to 2019-12.
- Out of time: 2020-01 to 2026-09.
- Hold-out states, excluded from every fit: MA, NJ, WI, MN, SC, KY, OK, AZ, WA (one per census division, as in the
  prepayment study); scored with the training-share-weighted state effect.
- M1 is champion unless M2 lowers the validation log loss of the charge-off hazard by more than 1% and its
  validation predicted ÷ actual is within 0.90–1.10.

## Backtests

- **Rolling origins.** T = December of 2009, 2011, 2013, 2015, 2017, 2019, 2021 and 2023. At each T, refit M0 and
  M1 (L fixed) on months up to T and project 24-month charge-off counts and dollars for the book active at T,
  (a) with the realised macro path and (b) with macro held at its value at T. Compare with actual charge-offs.
  Pass rule: (a) within ±25% of actual dollars at 6 or more of the 8 origins.
- **Pandemic period.** SBA paid the principal and interest of 7(a) borrowers for several months under the CARES Act
  while unemployment rose. Expected: over-prediction for 2020-04 to 2021-12. A finding is recorded if the
  2019-12 model with realised macro over-predicts charge-off dollars in those months by more than 25%.
- **Severity out of time.** Dollar predicted ÷ actual and mean absolute error for charge-offs in 2020-01 to 2026-09.

## Lifetime expected loss at 2026-09-30

- Book: active (EXEMPT) loans in the population at 2026-09, projected to scheduled maturity with both hazards.
- Macro: Fed 2026 supervisory baseline and severely adverse paths from 2026Q4 to 2028Q4 (the reasonable and
  supportable period), then straight-line reversion over 4 quarters to the 2006–2025 average of each input, flat
  after that.
- Reported as lifetime loss over gross approval and over estimated outstanding balance (level-payment schedule from
  the initial rate, where the rate is recorded), total and lender's share, by scenario and segment.

## Monitoring

Population stability index of the charge-off score distribution of the active book between consecutive origins.

## Publication

Only aggregate tables are published (no loan-level rows, no borrower or lender names). Any change to this plan
after the first model is fitted is recorded as a deviation in the report.
