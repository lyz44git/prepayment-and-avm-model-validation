# Prepayment hazard model: remediation plan for findings F9 and F2

Written 2026-11-18, before any model in this plan was fitted. The SHA-256 of this file is recorded in
`reports/p1/remediation/meta.json` by the script that runs it (`scripts/31_p1_remediation.py`).

## Why

The validation report lists ten findings. Two of them have a remediation that can be tested with the data in hand:

- **F9 (medium).** A −10% house-price path changes the projected CPR by −0.01 pp. Log appreciation and estimated
  LTV are functions of the same index and enter with offsetting coefficients, so the model is not usable for
  house-price stress. Listed remediation: one equity measure.
- **F2 (high).** The new-lien (refinance) hazard is under-predicted out of time by 26–30% after the September 2024
  rate dip. Listed remediation: lagged rate terms, and recalibration as part of monitoring.

## What is already known, and what that limits

The out-of-time window (2024-07..2025-06) was used to evaluate the first version of the models. Results on it for a
re-specified model are therefore not a clean test. Three things limit the damage:

1. The candidate specifications below are fixed here, from the findings log, before fitting.
2. The choice among them uses training months only (fit 2022-05..2023-12, validate 2024-01..2024-06).
3. The out-of-time and hold-out-state results are computed once for the selected specification and reported
   whatever they show.

Rate terms that vary only over time are identified from 26 training months. Adding more of them can fit noise; that
is why a richer specification must earn its place on the validation months.

## Data and splits

Unchanged: the 4% panel (22,885,356 property-months), train = 41 jurisdictions × 2022-05..2024-06, out of time =
same jurisdictions × 2024-07..2025-06, spatial = 9 hold-out states, spatial out of time. Same case-control scheme
(all events + 10% of non-events, intercept corrected by log 0.10).

## Step 0. Reproduce the published model

Refit the published logit (S0) with the published sampling seed. Pass: every coefficient within 1e-6 of
`m1_coefficients_*.csv`, and the out-of-time pool-CPR MAE and correlation equal to the published 0.427 pp and 0.900
to three decimals. If this fails, stop.

## Step 1. Candidate specifications

All use the published design (tenure spline, piecewise rate gap, PMMS level and 3-month change, property and tract
controls, month of year, state effects). They differ only in:

| ID | Equity | Added rate terms |
|---|---|---|
| S0 | log appreciation + estimated LTV (published) | none |
| S1 | estimated LTV only | none |
| S2 | estimated LTV only | drop from the 12-month peak: min(PMMS_t − max(PMMS_{t−12..t−1}), 0) |
| S3 | estimated LTV only | 6-month and 12-month PMMS change |
| S4 | estimated LTV only | all three |

The same specification is used for both causes (sale, new lien).

## Step 2. Selection on training months only

Fit S1–S4 on 2022-05..2023-12 and score 2024-01..2024-06 (training states). Criterion: log loss of the new-lien
hazard. S1 is the default. A richer specification replaces it only if it lowers that log loss by at least 0.1%;
among those that do, the lowest wins. The selected specification is called M1r and is refitted on all training
months, with county-clustered standard errors.

## Step 3. Fixed-origin evaluation (once)

For S0, S1 and M1r: AUC and predicted ÷ actual by cause and split; monthly pool CPR MAE and correlation for the
out-of-time and spatial-out-of-time months. The other candidates' out-of-time numbers are written to a separate
table marked "not used for selection".

## Step 4. Walk-forward re-estimation

A model that is maintained is re-estimated as data arrive. For each out-of-time month t, refit on training-state
rows dated t−2 or earlier (one month is skipped for recording lag; 90% of events are recorded within one month),
then predict month t. Done for the S0 and M1r specifications. Hold-out states never enter a fit; they are scored
with the training-share-weighted state effect. Covariates for month t are the realised rate and house-price index,
as in the fixed-origin backtest, so this is a conditional, pseudo-real-time forecast on final-vintage data.

Comparators already published: fixed-origin logit (MAE 0.43 pp, r 0.90), logit with trailing three-month
recalibration (0.30 pp, r 0.83), best rolling ARX benchmark (0.31 pp, r 0.64).

## Step 5. Stress test

Rerun the seven published scenarios for M1r, plus house prices −20% and +10% over 12 months with rates flat.
Count-based CPR and 12- and 24-month terminations.

## Closure rules, fixed now

- **F9 is closed** if, under M1r, house prices −10% lower the 24-month average CPR by at least 0.05 pp, −20% lowers
  it further, and +10% raises it; and S1 costs no more than 0.005 of out-of-time AUC on either cause and no more
  than 0.02 pp of out-of-time pool-CPR MAE against S0. Expected sign: less equity, fewer new liens.
- **F2 is reduced at fixed origin** if M1r's out-of-time new-lien predicted ÷ actual is at least 0.85 (published:
  0.70).
- **F2 is mitigated by maintenance** if the walk-forward M1r has out-of-time new-lien predicted ÷ actual between
  0.90 and 1.10 and pool-CPR MAE at or below 0.30 pp.
- Anything that fails stays open in the findings log, with the numbers.

## Not in scope

The LightGBM challenger is not refitted. The sale-side lock-in coefficient (F3) and the rate-proxy attenuation (F4)
need loan-level note rates and are not addressed.
