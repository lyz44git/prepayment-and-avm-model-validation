# Mortgage prepayment (turnover + new-lien) competing-risk hazard model

Model development, a self-administered validation and a rate-scenario stress test, structured like an SR 11-7 /
OCC 2011-12 validation report. The developer and the validator are the same person, so this is not an independent
validation in the SR 11-7 sense. Every number comes from `scripts/05–09, 16, 17, 21, 22` and the CSVs in
`reports/p1/`. Figure numbers in this report ("Fig. 1" etc.) refer to `reports/p1/fig*.png`, not to the README figures.

## 1. Purpose and scope

Prepayment is the behavioural input to MBS and whole-loan valuation, balance-sheet runoff and rate stress tests.
In a lifetime credit-loss estimate it sets how long each loan stays on the books, and therefore how many months of
default risk the allowance covers.
A mortgage prepays when the home is sold (turnover) or the loan is refinanced, or replaced by a cash-out or
other new first lien. This project estimates monthly hazards for both channels from 2022-05 to 2025-06, the
fastest rate rise in 40 years followed by a plateau. It then:
- backtests the model out of time and out of state,
- tests how the owner's rate relative to the market ("lock-in") shifts each hazard,
- projects 24-month termination rates and pool runoff under rate and house-price scenarios.

**Unit of analysis.** An owner-occupied single-family home or condo, as recorded by Vendor C in ≈ 2022-04,
with an arm's-length purchase before 2022-05.

**What is NOT observed.** The loan at risk. For a home that later sells or refinances, the file holds only the
successor loan, and the 2025 household record describes the buyer. The model is therefore a
property-level termination model; it is a proxy for loan prepayment, not a loan-level model. That limitation
shapes every design choice below.

## 2. Data and outcome construction

| Item | Value |
|---|---|
| Source | 348.9M rows of linked household, property and lien records (read-only) → 52.9M addresses → **16,060,240 at-risk properties** in 49 states + DC (VT excluded: no 2025-vintage sales) |
| Events 2022-05..2025-06 | sale 1,139,050 (7.09%); new first lien without a sale 747,759 (4.66%); other transfer (quitclaim, trustee deed, gift, affidavit) 45,031 (0.28%); none 14,128,400 |
| Sale rule | 2025-vintage transfer dated in the window, transaction type 1, deed not QC/TD/AF/GF, sale date ≠ old-vintage sale date. 89.1% also show a different owner surname between vintages (name compared inside DuckDB, never extracted) |
| New-lien rule | no transfer in the window and a Vendor A first lien dated in the window (event month = lien date). Captures rate refinances and cash-out / new first liens; they cannot be told apart |
| Recording lag | median 0 months, 90th percentile 1 month → window ends at 2025-06, two months before the 2025 vintage's last recordings |
| Modelling sample | 4% random sample of properties (hash of the de-identified id): 642,445 properties, **22,885,356 property-months** |
| Splits (fixed in advance) | train = 41 jurisdictions (40 states + DC) × 2022-05..2024-06 (13.08M rows; 27,707 sales, 16,071 new liens); **OOT** = same states × 2024-07..2025-06 (5.68M rows); **spatial** = 9 hold-out states (MA, NJ, WI, MN, SC, KY, OK, AZ, WA; one per census division) × train months (2.88M); **spatial-OOT** (1.25M) |

**Covariates (all symmetric between event and non-event properties).**
- *Pre-window property attributes:* purchase price indexed to 2022-04, age, size, beds, baths, lot, pool, condo.
- *Tract attributes:* density, owner share, median age.
- *State.*
- *Time-varying:* tenure since purchase; Freddie Mac PMMS 30-year rate; its 3-month change; month of year.
- *Rate gap:* PMMS(t) − PMMS(purchase month), the "lock-in" measure.
- *Equity:* log appreciation since purchase from the self-built repeat-sales index; estimated current LTV,
  assuming 80% LTV at purchase and 30-year level pay at the purchase-month rate.
- *Excluded on purpose:* Vendor A lien fields (rate, term, loan type, lender). They exist only for the
  non-event, pre-window lien, the informative-missingness trap behind finding F1. Race and ethnicity are not used.

## 3. Data integrity and measurement validation

| Check | Result |
|---|---|
| Rate proxy vs Vendor A "recorded" rate, stayers whose lien is the purchase lien (n = 3,941,387) | median spread 0.00 pp, MAE 0.146 pp, corr 0.935; 30-year fixed conventional (n = 2,193,177): MAE 0.027 pp, corr 0.995 |
| Is Vendor A's rate a recorded note rate? | **No.** It equals PMMS of the lien month for 30-year fixed conventional liens (MAE 0.018 pp, corr 0.996); 68.5% of all rates equal a monthly PMMS value |
| Stayers who refinanced before 2022 | 6,195,652 (61% of stayers with a lien). For them the purchase-month proxy is off by median −1.69 pp (MAE 2.06 pp). Large measurement error for these owners, so the gap coefficient is expected to be attenuated. "Min PMMS since purchase" is no better (corr 0.00) |
| Self-built repeat-sales index vs FHFA (CA) | annual-change correlation 0.88; 5-year 0.95; 2006-07→2012-01 −41.3% vs FHFA −40.9%; 2000→2006 boom understated (+119% vs +145%); the two indices diverge before 1998 |
| Owner-change confirmation of sales | 89.1% of sale events vs 3.1% of non-events |

## 4. Models (challengers to each other)

| ID | Model | Notes |
|---|---|---|
| M0 | Actuarial lookup | hazard by state × tenure bucket (credibility-weighted toward the national curve) × month-of-year factor |
| M1 | Cause-specific discrete-time logit hazard | natural cubic spline in tenure, piecewise-linear rate gap (gap+ / gap−), PMMS level and 3-month change, equity, property and tract controls, month-of-year and state FE. Fit on all events + 10% of non-events with the King–Zeng intercept correction; county-clustered SEs; 79 columns |
| M2 | LightGBM cause-specific hazards | same information plus categorical state; **monotone constraints**: hazard non-increasing in the rate gap (both causes) and in PMMS (new lien); early stopping on the last 6 training months (125 / 96 trees) |

For states not seen in training, M1 uses the training-share-weighted average state effect rather than the
reference category. The naive default over-predicted hold-out sales by 27% (predicted/actual 1.271 → 1.025).

## 5. Results

### 5.1 Monthly hazard discrimination and calibration-in-the-large (`eval_monthly_hazard_final.csv`)

| Cause | Split | AUC M0 / M1 / M2 | Predicted ÷ actual M0 / M1 / M2 |
|---|---|---|---|
| Sale | train | 0.647 / 0.650 / **0.704** | 1.00 / 1.00 / 1.00 |
| Sale | OOT | 0.628 / 0.639 / **0.642** | 1.06 / 1.04 / **1.01** |
| Sale | spatial | 0.632 / 0.642 / **0.647** | 1.08 / **1.02** / 0.94 |
| Sale | spatial-OOT | 0.620 / **0.634** / 0.633 | 1.09 / **1.02** / 0.90 |
| New lien | train | 0.604 / 0.629 / **0.704** | 1.00 / 1.00 / 1.00 |
| New lien | OOT | 0.550 / 0.592 / **0.593** | **0.83** / 0.70 / 0.74 |
| New lien | spatial | 0.574 / 0.608 / **0.611** | 1.00 / 0.98 / 1.03 |
| New lien | spatial-OOT | 0.534 / **0.582** / 0.578 | 0.77 / 0.64 / 0.73 |

M2's in-sample advantage (0.704 vs 0.650) mostly disappears out of time (0.642 vs 0.639). M2 is
over-fitting relative to its OOT value, and the parsimonious logit is a defensible champion.

### 5.2 Pool-level CPR backtest (train states; `monthly_cpr_backtest_summary.csv`, Fig. 1)

| Split | Actual mean CPR | Predicted M0 / M1 / M2 | MAE of monthly CPR, pp (M0 / M1 / M2) | Correlation (M0 / M1 / M2) |
|---|---|---|---|---|
| train (26 months) | 3.9% | 3.9 / 3.9 / 3.9 | 0.48 / 0.15 / 0.07 | 0.83 / 0.98 / 1.00 |
| OOT (12 months) | 3.9% | 3.7 / 3.5 / 3.5 | 0.53 / **0.43** / 0.48 | 0.85 / **0.90** / 0.79 |

New-lien activity is under-predicted in every out-of-time month: predicted ÷ actual is 0.79–0.85 in 2024-07..09,
then 0.61–0.75 after PMMS averaged 6.2% in 2024-09; the largest pool-CPR miss is 2025-04 (−0.94 pp). New-lien
activity rose faster than the models' contemporaneous-rate specification allows (finding F2).

### 5.3 Twelve-month termination forecast, properties at risk on 2024-07-01 (`cum12_backtest.csv`, Fig. 4)

Every property is rolled forward 12 months from its 2024-07 state along the realised rate and HPI paths.

| Split | Properties | Actual 12-month rate | AUC M0 / M1 / M2 | Predicted rate M1 / M2 | Top-decile lift M1 / M2 |
|---|---|---|---|---|---|
| OOT | 482,037 | 3.90% | 0.571 / 0.576 / **0.581** | 3.48% / 3.47% | 1.45 / 1.50 |
| spatial-OOT | 106,289 | 3.92% | 0.565 / 0.581 / **0.583** | 3.32% / 3.23% | 1.42 / 1.53 |

Out of time, observed rates rise in every decile and sit up to 0.7 pp above predicted (largest in the middle
deciles; logit, `cum12_calibration_oot_m1.csv`). In the hold-out states one decile breaks the ordering and the gap
reaches about 1.1 pp. Ranking holds better than level.

> An earlier version of this backtest summed hazards only over the months each property was observed. That
> measures exposure length, not risk, and inverted the ranking (AUC ≈ 0.19). It was caught, removed from
> `09`, and replaced by `17`. Recorded here as a validation lesson.

### 5.4 Rate sensitivity: lock-in and refinance incentive (M1, county-clustered SEs)

| Term | Sale | New lien |
|---|---|---|
| Rate gap above 0 (owner's rate below market), per +1 pp | OR 1.020 (0.991–1.049), n.s. | **OR 0.892 (0.859–0.928)** |
| Rate gap below 0 (owner in the money), per +1 pp | coef 0.048 (s.e. 0.022) | coef −0.010 (s.e. 0.024), n.s. |
| PMMS level, per +1 pp (owner's rate held fixed) | coef −0.149 (z −7.2) | coef −0.225 (z −8.5) |
| Estimated LTV | −0.15 (n.s.) | **−2.99 (z −6.1)**: less equity → fewer new liens |
| Average marginal effect of +1 pp gap | M1 +2.2%, M2 −1.0% (constraint-imposed) | M1 **−9.3%**, M2 −4.5% |

Each row of the sensitivity table changes one assumption (`sensitivity_lockin.csv`):

| Specification | Sale gap+ OR | New-lien gap+ OR |
|---|---|---|
| Baseline | 1.020 | 0.892 |
| Tenure knots (12, 60, 120, 240) | **0.968** (significant) | 0.882 |
| 2022-05..2023-06 only | 1.025 | 0.866 |
| 2023-07..2024-06 only | 1.013 (s.e. 0.30) | 0.927 |
| Case-control seed 1 | 1.016 | 0.890 |
| Note rate = cohort median recorded rate | 1.049 | 0.976 (n.s.) |
| Sale requires owner surname change | 1.024 | n/a |

**Interpretation.**
- *Refinance / new-lien sensitivity.* A higher market rate relative to the owner's purchase-cohort rate lowers
  the hazard by about 9% per percentage point. That is robust to tenure specification, sub-period and sampling;
  it attenuates under the cohort-median rate proxy.
- *Sale-side lock-in is not identified in this design.* Its sign flips with the tenure-spline knots. Purchase
  month fixes both tenure and the cohort rate (the age–period–cohort problem), and the 2023-07..2024-06
  sub-period has almost no rate variation (s.e. 0.30).
- *The large literature lock-in effect* (Fonseca & Liu) is estimated with loan-level note rates; this data does
  not have them. The finding is reported as a limitation, not hidden.
- *Market-rate level vs turnover.* The market-rate level has a strong negative association with turnover
  (−14% odds per pp). It is identified from 2022–25 time variation and may absorb other macro conditions; it is
  the effect the stress test uses.

**Inference across months (added at review).** The standard errors above cluster by county, but the PMMS level and
other market terms vary only by month, with about 26 training months. Refitting M1 with two-way (county × month)
clustered standard errors and a leave-one-month-out jackknife (`scripts/34_p1_inference.py`,
`r4_inference_twoway.csv`; coefficients and county-clustered errors reproduce the published ones exactly):
- The refinance rate-gap odds ratio keeps its interval under every method (0.89; 95% CI 0.86–0.93), because it is
  identified from differences between purchase cohorts within each month.
- The PMMS-level standard errors about double under the jackknife (sale ×1.78, new lien ×2.07); both terms stay
  significant (p = 0.0004).
- The 3-month rate change and the new-lien month-of-year effects are not significant once month-level correlation is
  allowed for, and the sale-side lock-in stays non-significant under every method.

### 5.5 Replication, stability, monitoring

- **Replication.** A separate full-data IRLS fit (13.08M rows, no subsampling) reproduces the six key case-control
  coefficients within 0.20 s.e. (sale) and 0.17 s.e. (new lien). All-coefficient correlation 0.999; prediction
  correlation 0.9993 / 0.9996.
- **Stability.** New-lien gap odds ratio 0.866 → 0.927 between the two training halves (stable sign). Sale
  coefficient unstable (see above).
- **PSI, train → OOT.**
  - Sale scores: 0.035 (M1), 0.048 (M2). Stable.
  - **New-lien scores (M1): 0.275**, above the 0.25 action threshold, consistent with finding F2.
  - Inputs: gap 0.038, log appreciation 0.048, estimated LTV 0.059. Tenure 0.31 reflects mechanical ageing of a
    closed cohort, and the PMMS rate level 4.78 reflects the move to a rate range partly outside training (F10).

### 5.6 Benchmarks and cash-flow valuation (`ts_benchmark.csv`, Fig. 8; `valuation_risk.csv`, Fig. 7)

- **Time-series benchmarks** for monthly pool CPR (scripts/22). A flat forecast matches the logit's out-of-time
  error (0.42 vs 0.43 pp) but not its shape (logit r = 0.90). The logit rescaled each month by its trailing
  three-month actual ÷ predicted ratio reaches 0.30 pp (r = 0.83); the best rolling one-month-ahead benchmark, ARX(1)
  with PMMS, reaches 0.31 pp (r = 0.64). With a two-month lag in the recalibration (to match recording lag) the error
  is 0.30 pp (r = 0.85), so the conclusion does not depend on the lag.
- **Illustrative cash-flow valuation** (scripts/21). A pass-through pool of 539,120 imputed loans (WAC 4.25%,
  WAM 237 months) discounted at a flat 5.88%: base price 85.7; effective duration 6.80 when prepayment responds to
  rates against 6.21 when it is held fixed (±100 bp), and convexity 36 against 65. At +200 bp the champion and
  challenger prices differ by about one point (74.8 vs 75.7).

## 6. Stress test (`stress_summary.csv`, `stress_paths.csv`, Fig. 6)

**Setup.**
- Pool: the 564,732 sample properties still at risk at 2025-06.
- Horizon: 24 months (2025-07..2027-06).
- Rates: shocks phased in linearly over 6 months from 6.82% (the 2025-06 PMMS).
- Two further paths: "realised PMMS" (actual FRED rates 2025-07..2026-10, then flat) and house prices −10% over
  12 months.
- Count-based rates need no balance assumption. The balance-weighted CPR uses an illustrative pool in which each
  property carries its purchase-money loan (80% LTV, 30-year, purchase-month PMMS).

| Scenario | 12-month terminations M1 | 24-month terminations M1 | Avg CPR, count, M1 | Δ CPR vs base, M1 / M2 (pp) | Avg CPR, balance-weighted, M1 |
|---|---|---|---|---|---|
| Base (flat 6.82%) | 3.28% | 6.33% | 3.22% | — | 3.35% |
| Realised PMMS path | 3.56% | 6.48% | 3.30% | +0.08 / +0.09 | 3.43% |
| −100 bp | 3.73% | 7.42% | 3.79% | +0.57 / +0.63 | 3.93% |
| −200 bp | 4.26% | 8.72% | 4.47% | +1.25 / +0.94 | 4.64% |
| +100 bp | 2.90% | 5.42% | 2.75% | −0.47 / −0.16 | 2.88% |
| +200 bp | 2.58% | 4.68% | 2.37% | −0.85 / −0.18 | 2.49% |
| HPI −10%, rates flat | 3.27% | 6.31% | 3.22% | −0.01 / −0.00 | 3.33% |

**Reading the table.**
- **Asymmetric rate convexity.** The response to rate cuts is larger than to rate rises (−200 bp: +1.25 pp;
  +200 bp: −0.85 pp, M1). This is the familiar prepayment convexity, here produced by the new-lien channel.
- **The challenger disagrees in the extrapolation region.** The training PMMS range was 5.22–7.62%. At +200 bp
  (8.8%) LightGBM is flat by construction (trees cannot extrapolate), so its −0.18 pp versus M1's −0.85 pp is a
  model-uncertainty band, not a precise estimate. The −200 bp (4.8%), +100 bp (7.8%) and +200 bp results end outside
  the development range and should carry an "outside development data" flag.
- **The house-price shock is inert.** The −10% HPI path changes CPR by −0.01 pp. Log appreciation and estimated
  LTV are both functions of the same index and enter the new-lien equation with offsetting partial effects
  (−0.49 and −2.99), so a price shock moves them in cancelling directions. The model is not fit for house-price
  stress until equity is re-specified with a single measure (finding F9).
- The horizon-truncated WAL (≈ 1.86 years in every scenario) is dominated by the balance treated as repaid at
  month 24 and is not informative here.

## 7. Findings log

| # | Severity | Status | Finding | Remediation |
|---|---|---|---|---|
| F1 | High | Open (accepted limitation) | Loan at risk unobserved for event properties; the household record for sold homes is the buyer's | Symmetric pre-window covariates only; property-level termination framed as a prepayment proxy; loan-level data needed for production |
| F2 | High | **Open** | New-lien hazard under-predicted OOT by 30% (logit) and 26% (LightGBM) (PSI 0.275), in every out-of-time month. Two parts: a gap that pre-dates the rate dip (0.80 in 2024-01..06 for the specification refitted through 2023-12, `remediation/step2_selection_validation.csv`; 0.815 in 2024-07..08 for the published model) and a widening to 0.68 from September 2024 (`r4_f2_by_month.csv`). Recorded new first liens per sale rose from 0.57 to 0.86 while the share of sales with a recorded buyer lien stayed at 0.72–0.73, so a change in lien-recording coverage does not explain it | §9: lagged rate terms were not supported on the validation months, and monthly re-estimation leaves predicted ÷ actual at 0.71. Trailing three-month recalibration (pool-CPR MAE 0.43 → 0.30 pp) is the monitoring control |
| F3 | Medium | Open | Sale lock-in coefficient not identified (sign flips across tenure specifications) | Do not use it for decisions; needs loan-level note rates or an exogenous rate shifter |
| F4 | Medium | Open | Rate proxy is attenuated for the 61% of stayers who refinanced before 2022 | Regression-calibration sensitivity reported; loan-level rate history needed |
| F5 | Medium | Open | M2 monotone constraints impose a lock-in sign the data do not support for sales | Keep M1 as champion; report M2 constrained and unconstrained in future |
| F6 | Low | Closed | Unseen-state default in M1 over-predicted hold-out sales by 27% | Training-share-weighted state effect (now 1.02) |
| F7 | Low | Closed | Exposure-biased 12-month cumulative backtest | Replaced by the roll-forward backtest |
| F8 | Info | Info | Vendor A mortgage "interest rate" is a PMMS lookup, not a recorded rate | Do not treat it as loan-level data |
| F9 | Medium | **Open** | House-price stress response is muted (−0.01 pp for HPI −10%) because two collinear equity terms offset | §9: with estimated LTV as the only equity term the response roughly doubles (−10%: −0.008 → −0.015 pp; −20%: −0.023 → −0.030; +10%: +0.003 → +0.015) at no cost in accuracy, but stays far below the 0.05 pp bar: the training data hold almost no homes above 80% LTV, so the model must not be used for house-price stress testing |
| F10 | Medium | Open | The −200, +100 and +200 bp scenarios end outside the training rate range (5.22–7.62%); M1 and M2 diverge (+200 bp: −0.85 vs −0.18 pp) | Report the M1–M2 range as model uncertainty; widen the development window when data allow |

## 8. Limitations

- Property-level proxy for loan prepayment. No loan balances, so the runoff pool is illustrative.
- Rate effects are identified from one rising-rate episode.
- The repeat-sales index is the model's own construction and is estimated partly from window sales. It is used
  as a realised-macro input (conditional forecast), not as a forecast.
- The new-lien event mixes rate refinances with cash-out and other new first liens. Vendor A lien coverage varies
  by state (state FE absorb the level).
- Selection: properties enter only if a household in the source records lived there in 2025 (the data's
  construction), so the risk set is not a random sample of all owner-occupied homes.

## 9. Remediation of F9 and F2 (2026-11-18)

The plan, candidate specifications and closure rules were written before any model in this section was fitted
(`docs/prepayment_model_remediation_plan.md`; its SHA-256 is stored in `reports/p1/remediation/meta.json`). The run script
recorded the hash when it started (plan saved 09:54, run started 09:55): this shows the plan was not edited
afterwards, not when it was written. Code:
`scripts/31_p1_remediation.py`, `scripts/32_p1_ltv_diagnostic.py`. Tables: `reports/p1/remediation/`.

**Step 0. Reproduction.** The published logit was refitted from the panel with the published sampling seed. Every
coefficient matched to within 5e-16 and the out-of-time pool-CPR MAE and correlation matched exactly (0.427 pp,
0.900), so the remediation starts from the published model.

**Step 2. Selection on training months only** (fit 2022-05..2023-12, score 2024-01..2024-06; new-lien log loss).

| Spec | Equity | Added rate terms | Validation log loss (new lien) |
|---|---|---|---|
| S0 (published) | log appreciation + estimated LTV | none | 0.0086102 |
| **S1 (selected)** | estimated LTV only | none | 0.0086112 |
| S2 | estimated LTV only | drop from 12-month peak | 0.0086997 |
| S3 | estimated LTV only | 6- and 12-month PMMS change | 0.0086416 |
| S4 | estimated LTV only | all three | 0.0086440 |

No rate term lowered validation log loss, so the plan's default S1 was kept.

**Step 3–4. Backtests** (`step3_4_cpr_backtest.csv`, `step4_eval_walkforward.csv`).

| Model | Out-of-time pool-CPR MAE (pp) | Correlation | New-lien predicted ÷ actual | Sale predicted ÷ actual |
|---|---|---|---|---|
| S0, fixed origin (published) | 0.427 | 0.900 | 0.70 | 1.04 |
| S1, fixed origin | 0.423 | 0.900 | 0.70 | 1.04 |
| S0, re-estimated each month | 0.433 | 0.912 | 0.70 | 1.04 |
| S1, re-estimated each month | 0.430 | 0.911 | 0.71 | 1.04 |
| S0 + trailing three-month recalibration (§5.6) | 0.30 | 0.83 | — | — |

- Dropping log appreciation costs nothing: out-of-time AUC moves by less than 0.001 for either cause.
- Hold-out states (spatial out of time) behave the same: S1 at fixed origin has pool-CPR MAE 0.60 pp, r 0.90 and
  new-lien predicted ÷ actual 0.64; re-estimated monthly, 0.60 pp, r 0.91 and 0.64, as for S0.
- Monthly re-estimation (refit on data to t − 2, predict t) does not remove the new-lien shortfall. The months after
  the September 2024 dip are a small part of each refit sample, and the rate terms that vary only over time cannot
  absorb a level shift that the 2022–24 months did not contain.
- *Not used for selection:* S3 would have cut the out-of-time new-lien shortfall (predicted ÷ actual 0.83, still short of the
  0.85 bar; pool-CPR MAE 0.32 pp), but it was worse on the validation months, which contain no rate dip. A rule that could only be
  tuned on the out-of-time window is not a validated fix; it is recorded here as evidence that the error is a
  regime the development window does not contain.

**Step 5. Stress test with S1** (`step5_stress_summary.csv`; 564,732 homes, 24 months, count-based CPR).

| Scenario | Δ CPR vs base, S0 (pp) | Δ CPR vs base, S1 (pp) |
|---|---|---|
| −200 bp | +1.25 | +1.26 |
| +200 bp | −0.85 | −0.86 |
| HPI −20%, rates flat | −0.023 | −0.030 |
| HPI −10%, rates flat | −0.008 | −0.015 |
| HPI +10%, rates flat | +0.003 | +0.015 |

**Why the house-price response stays small** (`diag_stress_pool_ltv.json`, `diag_hazard_by_ltv_band_train.csv`).
The stress pool's estimated LTV has a median of 0.30 and 91% of homes are below 0.5. After a 10% price fall 0.003%
of homes reach 80% LTV; after a 20% fall, 0.6%. In the training months only 499 property-months (0.004%) sit above
80% LTV, so the model has no data on the region where house-price shocks bite. Estimated LTV also assumes 80% at
purchase and no later borrowing, which understates leverage. The equity channel can be stress-tested only with
loan-level balances, or with a development window that includes falling prices.

**Closure (rules fixed in the plan, `closure.json`).**
- F9: **not closed.** The response keeps its sign and order, roughly doubles, and accuracy is unchanged, but −10%
  moves CPR by −0.015 pp against a 0.05 pp bar.
- F2: **not reduced at fixed origin** (0.70 against a 0.85 bar) and **not mitigated by re-estimation** (0.71 against
  0.90–1.10). Trailing recalibration stays the control.

**Decision (not part of the pre-registered plan).** S1 is proposed as champion: accuracy is unchanged (validation
log loss 0.0086112 vs 0.0086102; out-of-time AUC within 0.001) and it removes the two offsetting equity terms.
§5–6 describe S0; S1's results are in this section. S1 must not be used for house-price stress testing (F9), and
trailing recalibration remains a required monitoring step (F2).

## 10. Related literature

Full citations are in `docs/references.md`.

- **Design.** Prepayment and its competitor are modelled as cause-specific discrete-time hazards, in the
  competing-risk tradition of Deng, Quigley and Van Order (2000), with an empirical prepayment function as in Schwartz
  and Torous (1989). The estimator is the logit hazard of Shumway (2001).
- **Why the refinance forecast fails (F2).** Berger, Milbradt, Tourre and Vavra (2021) show that the prepayment
  response to a rate cut depends on the distribution of refinancing incentives left by the past rate path, and
  Andersen, Campbell, Nielsen and Ramadorai (2020) show that households act on an incentive with delays that differ
  across households. The training months here (2022-05 to 2024-06) contain almost no owners with an incentive, so the
  model cannot learn the response that the September 2024 dip produced; part of the gap was already present before
  the dip (§7, `r4_f2_by_month.csv`).
- **Heterogeneity.** Stanton (1995) and Deng et al. (2000) build borrower heterogeneity into prepayment; ignoring it
  biases prepayment estimates in Deng et al. The model here has none, and the vendor rate field is a market-rate lookup
  rather than the loan's rate (F8), so both the incentive and its heterogeneity are measured poorly.
