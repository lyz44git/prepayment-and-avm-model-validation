# Champion–challenger validation of third-party home-valuation models

**Question.** Lenders buy home-value fields from data vendors and use them as collateral and verification
inputs. How accurate, fresh and fair are those vendor values against recorded
arm's-length sale prices? And can an independently built challenger (gradient boosting plus a
neighbour-attention network over comparable sales) beat them out of time and out of geography?

**What this is not.** It is not a credit model, and no bank, servicer or credit-bureau data is involved. All outcomes are
public-record sale prices.

**Where it sits in a loss estimate.** For a collateralised loan, the collateral value enters the loan-to-value ratio:
with the loan amount fixed, a value overstated by a fraction e makes the estimated LTV 1/(1+e) times the true one. How
that carries into a loss also depends on lien position, costs, the balance at default and the recovery process, none of
which this study models. A median error of 11–16% is the size of the valuation uncertainty a loss model would inherit.

Numbers come from `scripts/10–15`, `19` and `23–29` and the CSVs in `reports/p2/`.

## 1. Data

| Item | Value |
|---|---|
| Ground truth | 2,418,184 arm's-length residential resales (Vendor C 2025 vintage, transaction type 1, deed in {WD, SW, GD, BA, WJ, DD}, $30k–$20M), 2022-05..2025-08, 49 states + DC. The price-disclosure screen meant to drop states where fewer than 90% of resales carry a price did not bind (§5, V10), so the sample covers every state with recorded prices |
| Features | Vendor C old vintage (≤ 2022-04): prior arm's-length sale and date, 2021 assessor values, property attributes; tract density / owner share / median age; lat/lon |
| Comparable-sale pool | 10,335,897 arm's-length sales 2012-01..2022-04, indexed to 2022-04 with the repeat-sales index (`r5_avm_data_counts.csv`) |
| Market index | Self-built Bailey–Muth–Nourse repeat-sales index. The valuation study uses a version estimated **only from pairs whose second sale is ≤ 2024-06, excluding the 9 hold-out states** (748,303 pairs, against 1,286,709 in the full index), frozen after 2024-06 |
| Vendor values (the champions) | Vendor A AVM (+ Vendor A's stated AVM error), Vendor B estimated home value, Vendor A market value, Vendor C 2025 assessor value. **Evaluated only, never used as features** |
| Splits (fixed in advance) | train ≤ 2024-03 (1,243,394); calibration 2024-04..06 (166,722); **OOT test 2024-07..2025-08 (591,687)**; **spatial hold-out** MA, NJ, WI, MN, SC, KY, OK, AZ, WA, all months (416,381) |
| Excluded from features | race / ethnicity tract shares (used only in the disparity audit); every 2025 household field |

## 2. Vendor due diligence (champions)

Out-of-time window, 591,687 sales (`vendor_accuracy_oot.csv`). Metrics:
- MdAPE: median |value/price − 1|.
- PPE10: share of sales within ±10%.
- COD and PRD: IAAO uniformity and regressivity measures.

| Vendor field | Coverage | Median value/price | MdAPE | PPE10 | COD | PRD |
|---|---|---|---|---|---|---|
| Vendor A AVM | 78.5% | 0.684 | 34.4% | 8.4% | 42.8 | 1.28 |
| Vendor B estimated value | 91.7% | 0.974 | 13.0% | 45.3% | 25.4 | 1.12 |
| Vendor A market value | 95.5% | 0.882 | 16.3% | 29.8% | 26.0 | 1.10 |
| Vendor C 2025 assessor value | 99.6% | 0.810 | 23.1% | 24.1% | 35.3 | 1.11 |

**Findings.**

1. **The Vendor A AVM sits far below sale prices and does not track the market.** Its median value is 0.68 of the
   sale price (34.4% median error). Its log error regressed on state-level market appreciation since 2022-04 has a
   slope of **−1.017 (s.e. 0.018)**: a slope of −1 is what a value fixed before 2022 produces, and an up-to-date
   value would give 0. The other vendor fields do not track appreciation either (slopes −1.35 to −1.39), so the slope
   alone does not single the AVM out; its level does. The test also gives a steep slope (−1.37) for the field that is anchored to the
   sale price, so it is not a clean test of staleness and is reported as descriptive only. If it were unbiased, its implied value date would be
   **≈ 2020-08**. The vendor-stated AVM error has no ranking
   power for realised error: Spearman 0.045, AUC 0.498 for |error| > 20%, n = 455,397.
2. **The Vendor B "estimated home value" is anchored to the recorded sale price, i.e. look-ahead.**
   - 29.5% of OOT sales match within 0.5%.
   - The median ratio is exactly 1.000 in every quarter through 2024Q3, with median error under 0.3%; in 2024Q3,
     52.5% of its values are within 0.5% of the sale price (`vendor_accuracy_by_quarter.csv`). It drifts away for sales
     from 2024Q4 onward (median error 6.4%, then 15% in 2025Q1).
   - The snapshot shows the value, not when it was produced or how; "anchored" describes how closely it tracks the
     recorded price, not a verified vendor method.
   - In the hold-out states its MdAPE is 0.3%.
   - It is therefore **not a valid benchmark**, and any model trained with it on sold homes would leak the
     target.
3. **Vendor A market value and the assessor value** behave like ~2022 assessed values: implied dates 2022-04 and
   2022-02. Both are regressive (PRD > 1.03).
4. **Mortgage-rate field.** The prepayment study found that Vendor A's "interest rate" equals the PMMS rate of the lien month.
   It is modelled, not recorded.

## 3. Challengers

| Model | Spec |
|---|---|
| B0 index AVM | prior arm's-length sale × repeat-sales index (covers only the 55% of sales with a prior sale) |
| B0′ comp index | median of the 20 nearest pre-2022 comps × index |
| B1 hedonic OLS | log price on property, comp, assessor and index features + state FE |
| C1 LightGBM | 33 features (no race variables), early stopping on the calibration window (837 trees) |
| C1q | LightGBM 10%/90% quantile models, split-conformal calibration on 2024-04..06 (conformalized quantile regression, [Romano, Patterson, and Candès 2019](https://arxiv.org/abs/1905.03222)) |
| C2 neighbour-attention NN | PyTorch on Apple MPS. 32 comp tokens (indexed price, size, age, beds/baths, condo, dx/dy, distance, comp age, size/age gaps) → 2-layer transformer encoder → subject cross-attention → residual over "comp median × index"; Huber loss; 10 epochs; best calibration MdAPE 11.07% (epoch 9) |
| C3 blend | equal-weight average of C1 and C2 in log price; weights not tuned, blend added after calibration-window (not test) results |

## 4. Results

**Out-of-time (591,687 sales; each estimator on its own coverage).**

| Estimator | Coverage | Median ratio | MdAPE | PPE10 | PPE20 | COD |
|---|---|---|---|---|---|---|
| Vendor A AVM (incumbent vendor value) | 78.5% | 0.684 | 34.4% | 8.4% | 20.3% | 42.8 |
| Vendor A market value (best non-anchored vendor) | 95.5% | 0.882 | 16.3% | 29.8% | 58.0% | 26.0 |
| B0 index AVM | 54.8% | 1.007 | 14.6% | 36.9% | 61.8% | 31.9 |
| B1 hedonic OLS | 100% | 0.980 | 14.5% | 36.9% | 62.7% | 28.0 |
| C1 LightGBM | 100% | 0.961 | 12.0% | 43.5% | 69.2% | 24.5 |
| C2 neighbour-attention NN | 100% | 0.988 | 12.0% | 43.4% | 68.9% | 26.0 |
| **C3 blend** | 100% | 0.974 | **11.4%** | **45.1%** | **70.5%** | 24.5 |

- **Confidence intervals.** County-cluster bootstrap (200 draws) on the 454,526 rows where both Vendor A fields
  exist: C3 MdAPE **11.3% (95% CI 10.9–11.7%)** vs Vendor A market value **16.7% (15.7–18.2%)**, a difference of
  **−5.4 pp (CI −6.7 to −4.4)**. Against the Vendor A AVM: 34.4% (33.5–35.2%).
- **Common support** (419,615 rows with all four vendor values): C3 11.1%, Vendor A market value 16.6%, Vendor A AVM 34.3%.
- **Excluding the 12 non-disclosure states** (464,080 rows): C3 10.8% vs Vendor A market value 16.3%. The result holds.
  Excluding instead the 12 states that fail the price-disclosure screen as it was meant to be computed (460,745 rows;
  §5, `r5_accuracy_by_screen_version.csv`): C3 10.7% vs 16.4%. Retrained without them as well (§9), the result is the
  same.
- **Segments.** With a prior recorded sale: C3 9.9%. Without one: 14.0%.

**Spatial hold-out (416,381 sales in 9 states the models never saw, index rebuilt without them).**

| Estimator | MdAPE | PPE10 | Median ratio |
|---|---|---|---|
| Vendor A AVM | 29.7% | 10.2% | 0.730 |
| Vendor A market value | 12.0% | 44.6% | 0.929 |
| C1 LightGBM | 13.5% | 38.7% | 0.935 |
| C2 neighbour-attention NN | 12.9% | 40.9% | 0.991 |
| C3 blend | 12.2% | 42.3% | 0.959 |

Out of geography, **the challenger does no better than Vendor A market value**: no measurable difference, bootstrap
difference +0.1 pp, CI −2.5 to +1.8 (no equivalence test was run, so this is not evidence that the two are equal). Most hold-out sales fall in 2022–24, when that vendor's ~2022 values were still current, and the
hold-out states get only a division-level index. The attention network transports best of the three challengers
(median ratio 0.991 vs 0.935 for GBM). Its embedding for an unseen state never receives a training example, so its value
is set by the initialisation. Re-scoring the trained network with that embedding replaced by zeros, by the mean of the
trained state embeddings or by three fresh random draws moves the blend's hold-out error from 12.2% to 12.3–12.7% over
all months, and from 12.8% to 12.9–13.5% in the out-of-time months (`r6_unseen_state_embedding.csv`). The all-month
comparison with the vendor (12.0%) therefore depends on an untrained parameter.

**Calibrated uncertainty (C1q, 80% nominal).**

| Window | Coverage (sale price inside the interval) | Median width, upper ÷ lower − 1 |
|---|---|---|
| Calibration | 80.0% | 60% |
| OOT | 77.5% | 62% |
| Spatial | 78.1% | 66% |

The intervals are C1q's, not the blend's. The slight under-coverage is consistent with drift, but drift is not its
only possible cause: the LightGBM tree count was early-stopped on the same calibration window (§5). Interval width
ranks large C1 errors (|error| > 20%) with **AUC 0.77** OOT; Vendor A's stated AVM error ranks the AVM's own large errors
with **0.498**. The two AUCs are on different targets, so they are not a head-to-head comparison. Calibrating with the
exact split-conformal rank ceil((n + 1) × 0.8) instead of an interpolated quantile changes nothing at this sample size
(166,722 calibration sales); in the hold-out states out of time, coverage is 76.1% (`r6_conformal_exact_rank.csv`).

**Drift.** With the model and index frozen at 2024-06, C3 monthly MdAPE rises from 10.6% (2024-07) to 12.4%
(2025-08), about 0.14 pp per month between the end points. Vendor A market value goes from 15.5% to 17.4%.

**Price tiers (upmarket).**
- Top within-state decile MdAPE: C3 16.3%, C2 16.1%, Vendor A market value 18.2%, Vendor A AVM 47.9%.
- Bottom decile (non-standard and distressed resales) is poor for every estimator (30–74%).
- $1M+ screen: C1 AUC 0.954 vs Vendor A market value 0.947. Recall 72.9% vs 67.4% at precision 88.5% vs 90.7%.
  The ranking gain is small; the recall gain comes with lower precision.

**Disparity audit.** Tract Black + Hispanic share quintiles, computed within price decile so that tier
regressivity is removed:

| Estimator | Within-tier value/price, Q5 − Q1 (log points × 100) |
|---|---|
| Vendor A AVM | **−9.1** (relative undervaluation in the highest-minority-share tracts) |
| Vendor C assessor | +13.2 (relative over-assessment) |
| Vendor B | +9.2 |
| C1 LightGBM | −0.8 |
| C2 attention | −3.2 |
| C3 blend | −2.0 |

On realised-price tiers the Vendor A AVM gap is the kind of pattern the 2024 interagency AVM quality-control rule and
the PAVE task force target, but it is not robust to tiers known before the sale (§9, V8). These are tract-level ratio
comparisons: a tract's minority share is not any borrower's protected status, and a valuation error is not a lending
decision, so this is not a fair-lending test.

**Explainability.**
- TreeSHAP share of mean |SHAP| for C1 (20k OOT rows): pre-2022 neighbour comps 39.6%, 2021 assessor values
  28.1%, location/tract 11.6%, property attributes 8.5%, indexed prior sale 7.7%, market index 4.5%.
- C2 attention weights are close to uniform across the 32 comps (3.2% for the nearest vs 3.1% on average). The
  modest preference goes to size-similar comps (Spearman −0.14 with the size gap) rather than to the nearest
  (−0.05). The network's gain comes from learned comp representations; the attention map is not a strong
  explanation by itself.

## 5. Limitations

- **Vendor values carry look-ahead** (they are dated after many test sales), so the vendor comparison favours
  the vendors on timing. The challengers win on OOT accuracy anyway; out of geography they do not beat Vendor A market value.
- **Sale prices in non-disclosure states** may be MLS-sourced, and the recorded ones may be a selected subsample.
  The disclosure screen in `scripts/10_p2_sales.py` was meant to drop states where fewer than 90% of resales carry a
  price, but it computed each share as an average that skips missing prices, so every state passed. Over all resales,
  12 states fall below 90% (AK, ID, IN, KS, ME, MO, MS, MT, NM, TX, UT, WY; Texas 0.64; `r5_disclosure_denominator_bridge.csv`).
  These states are therefore in the training and test samples. Scored without them, and with the whole pipeline
  re-run on the 38 jurisdictions that pass (§9), the results hold, and none of the nine hold-out states is among them
  (V10).
- **Unseen-state handling and field dates.** C2's embedding for an unseen state receives no gradient in training, so
  its value is set by the initialisation, and C3 inherits it; other values move the blend's hold-out error by up to half
  a point (§4). The snapshot does not show when each fallback property attribute was formed.
- **The comp pool and assessor features date from 2022-04.** A production AVM would refresh comps monthly; the
  frozen design deliberately understates what a maintained challenger could achieve.
- **The 50/50 blend weight was fixed a priori.** The blend was added after seeing calibration-window
  (not test) MdAPEs of C1 and C2.
- **No interior-condition, listing or MLS data.** Bottom-tier errors remain large for every estimator. In the lowest
  price decile the blend values homes at 1.41 times the sale price (median error 41.3%, against 30.6% for Vendor A market
  value; `accuracy_by_price_tier_oot.csv`). Over-valuation is the dangerous direction for collateral, but most of
  this tail comes from sorting on the outcome (next bullet and §9).
- **Tiers sort on the outcome.** Price deciles are built from the realised sale price, which builds in regression to the
  mean at both ends and conditions the within-tier disparity on the outcome. Tiers built on a value known before the
  sale (§9) remove most of the bottom-decile error and most of the Vendor A AVM's minority-share gap; neither tiering is
  neutral, because both ex-ante values are also challenger inputs.
- **Sold homes are not the collateral that matters at default.** The test set is arm's-length resales; distressed sales
  and unsold homes are not in it.
- **Bootstrap and conformal details.** The county-cluster bootstrap uses 200 draws (1,000 in §9 give the same result). The LightGBM tree count was
  early-stopped on the same calibration window later used for the conformal quantile, which weakens the split-conformal
  coverage guarantee; the out-of-time coverage of 77.5% is the relevant check.

## 6. Monitoring plan (proposed)

| Metric | Threshold → action |
|---|---|
| Monthly MdAPE vs calibration baseline | +1.5 pp for 2 consecutive months → re-index |
| 80% interval coverage | below 75% → recalibrate the conformal q̂ |
| PSI of prediction / key inputs | > 0.25 → investigate |
| Within-tier Q5 − Q1 gap | magnitude > 5% → fairness review |
| Vendor-field staleness slope | re-run quarterly, alongside vendor recertification |

## 7. Monitoring back-test

The scoring service's own PSI code was run on each out-of-time month against the training-window reference
(`scripts/23_p2_service_reference.py`; `psi_backfill_summary.csv`). Six inputs that shift by design are excluded from
the alarm (sale month, market index level, months since the prior sale, state code, latitude, longitude); their PSI is
5.9–9.6 in every month. The first row is the calibration window, the baseline for the outcome trigger
(`r4_monitoring_backtest.csv`).

| Month | Prediction PSI | Max PSI, stable inputs | Median error, LightGBM | Median error, blend |
|---|---|---|---|---|
| 2024-04..06 | 0.012 | 0.012 | 10.9% | 10.4% |
| 2024-07 | 0.011 | 0.011 | 11.1% | 10.6% |
| 2024-08 | 0.005 | 0.008 | 11.2% | 10.7% |
| 2024-09 | 0.002 | 0.005 | 11.5% | 11.0% |
| 2024-10 | 0.002 | 0.007 | 11.7% | 11.2% |
| 2024-11 | 0.001 | 0.006 | 11.8% | 11.3% |
| 2024-12 | 0.001 | 0.005 | 12.0% | 11.6% |
| 2025-01 | 0.003 | 0.006 | 12.2% | 11.7% |
| 2025-02 | 0.003 | 0.007 | 11.9% | 11.5% |
| 2025-03 | 0.009 | 0.010 | 11.9% | 11.4% |
| 2025-04 | 0.005 | 0.010 | 12.4% | 11.8% |
| 2025-05 | 0.008 | 0.011 | 12.5% | 11.9% |
| 2025-06 | 0.008 | 0.010 | 12.6% | 12.1% |
| 2025-07 | 0.004 | 0.008 | 12.8% | 12.2% |
| 2025-08 | 0.005 | 0.009 | 12.9% | 12.4% |

Prediction PSI and the PSI of every stable input stayed at or below 0.012 in all 14 months while the LightGBM
challenger's median error rose from 11.1% to 12.9% (`oot_monthly_MdAPE.csv`). Input drift statistics did not see the
deterioration that mattered. The outcome trigger in §6 (median error 1.5 points above the calibration window, where
LightGBM was at 10.9%) would have: it first fires in 2025-05, and under its two-consecutive-month rule in 2025-06. The
baseline is slightly optimistic, because the LightGBM tree count was early-stopped on the calibration window. The
same blind spot appears in the credit-loss study.

## 8. Findings log and use limits

There was no pre-set rule for replacing a vendor value with the challenger, so the comparison below is reported as
evidence, not as an adoption decision.

| # | Severity | Status | Finding | Use limit or remediation |
|---|---|---|---|---|
| V1 | High | Open | Vendor A AVM sits 32% below sale prices (median ratio 0.684, median error 34.4%); its stated error has no ranking power (AUC 0.498) | Not usable as a collateral value without recalibration; do not use the stated error |
| V2 | High | Open | Vendor B estimated value is anchored to the recorded sale price (29.5% within 0.5%) | Not a benchmark and never a model feature (target leakage) |
| V3 | Medium | Open | Vendor A market value and the 2025 assessor value behave like 2022 values and are regressive (PRD > 1.03) | Re-index before use; monitor staleness quarterly |
| V4 | Medium | Open | The challenger beats the best usable vendor out of time (11.4% against 16.3%). In nine unseen states there is no measurable difference over all months (12.2% against 12.0%; 12.3–12.7% for the blend with other values of its untrained unseen-state embedding), and that comparison splits by period: the vendor is better in 2022-05..2024-06, when its ~2022 values were current (9.3% against 11.9%), and the challenger out of time (12.8% against 18.5%) | Judge out-of-footprint use against a vendor value of the same vintage; do not read the all-month result as transferability |
| V5 | Medium | Open | Error rose from 11.1% to 12.9% over 14 months while prediction PSI and the PSI of the stable inputs stayed at or below 0.012 | Outcome trigger: median error 1.5 points above the calibration window for two consecutive months (first crossed in 2025-05, fires in 2025-06) |
| V6 | Medium | Open | 80% conformal intervals cover 77.5% out of time; tree count early-stopped on the calibration window | Recalibrate the conformal quantile when coverage falls below 75% |
| V7 | Low | Open | Bottom-tier accuracy depends on how tiers are built: on realised-price deciles the blend values the lowest decile at 1.41 times price (error 41.3%); on deciles of the indexed prior sale its error there is 16.6% (§9) | Monitor the tails on tiers known before the sale; no low-value collateral use without that check |
| V8 | Low | Open | The Vendor A AVM's within-tier minority gap (−9.1 log points on realised-price tiers) shrinks to −1.0 to −2.4 and is not significant on tiers known before the sale; the blend's gap changes sign, from −2.0 to +2.6 or +3.0 (§9) | No fairness conclusion from either tiering alone; report both |
| V9 | Info | Info | Vendor "interest rate" is the PMMS rate of the lien month (prepayment study, F8) | Not a recorded loan rate |
| V10 | Medium | Closed | The price-disclosure screen did not bind: missing prices dropped out of each state's share, so every state passed; over all resales 12 states fall below 90% (Texas 0.64; §5) | Re-run on the 38 jurisdictions that pass (§9): 10.7% against 16.4% out of time, 11.9% against 12.0% in unseen states; the published tables keep all states |

**Disposition.** Research and benchmark use. The challenger is a monitoring benchmark for vendor values in the states
it was trained on; it is not approved as a production collateral value, and neither is any vendor field without the
remediation above.

## 9. Robustness checks

**More bootstrap draws.** With 1,000 county-cluster draws (`scripts/26_p2_robustness.py`, `r4_bootstrap_b1000.json`)
the blend's out-of-time advantage over Vendor A market value is −5.5 points (95% CI −6.6 to −4.3), with every draw
below zero; the first 200 draws reproduce the published interval.

**Unseen states by period.** The all-month tie in the nine hold-out states (12.2% against 12.0%) hides two periods
(`r4_accuracy_spatial_oot.csv`). In 2022-05..2024-06, when the vendor's ~2022 values were current, Vendor A market value
is more accurate (9.3% against 11.9%). In the out-of-time months the blend is (12.8% against 18.5%; on rows with both
vendor fields, −6.1 points, 95% CI −7.8 to −4.2). Transfer to new geography is therefore not shown to fail; the
comparison is confounded with the vendor's value date.

**Price tiers known before the sale.** The tier tables above sort on the realised price. Rebuilt on within-state
deciles of two values known before the sale, the prior sale indexed to 2024-06 (54.8% of out-of-time sales) and the
2021 assessor market value (90.6%) (`scripts/25_p2_exante_tiers.py`, `r4_accuracy_by_exante_tier.csv`,
`r4_disparity_exante_tier.csv`, 500 county-cluster draws):

| Tier basis | Vendor A AVM, Q5 − Q1 (log points × 100, 95% CI) | Blend, Q5 − Q1 (95% CI) |
|---|---|---|
| Realised sale price (published) | −9.1 (−12.5 to −6.0) | −2.0 (−3.2 to −0.8) |
| Indexed prior sale | −1.0 (−4.0 to +2.6) | +2.6 (+1.5 to +3.5) |
| 2021 assessor value | −2.4 (−5.3 to +0.9) | +3.0 (+1.8 to +4.0) |

On the same homes, the sales with a recorded prior sale, the blend's median error in the bottom tier is 26.2% (median
ratio 1.23) when the realised sale price defines the tiers and 16.6% (0.97) when the indexed prior sale does
(`r4_accuracy_by_exante_tier.csv`); the 41.3% above is on all sales, so it mixes the tier definition with a different
sample. The same homes enter both tests, though different homes fall in the bottom tier. The blend uses the prior sale
as an input, and ex-ante tiers carry the opposite artefact (the ex-ante value ÷ price ratio rises across deciles), so
these are sensitivity checks with no mathematical bound between them. The Vendor A AVM's minority-share gap and the
extreme bottom-decile error are not robust findings (V7, V8).

**The pipeline re-run on the states that pass the price-disclosure screen** (`scripts/29_p2_screened_rerun.sh`,
`r6_screened_*`). With each state's share of priced resales computed over all resales (§5), 12 states fail and the
sample falls to 1,971,173 sales in 38 jurisdictions; every step from the sample to the bootstrap and the monitoring
back-test was re-run without them.

| Result | Published (all states) | Re-run (states that pass) |
|---|---|---|
| Out of time: blend vs Vendor A market value, median error | 11.4% vs 16.3% | 10.7% vs 16.4% |
| Out of time, paired, 1,000 county-cluster draws: difference (95% CI) | −5.5 (−6.6 to −4.3) | −6.5 (−7.9 to −5.0) |
| Unseen states, all months | 12.2% vs 12.0% | 11.9% vs 12.0% |
| Unseen states, out-of-time months | 12.8% vs 18.5% | 12.3% vs 18.5% |
| Monitoring back-test: LightGBM error, first to last month | 11.1% to 12.9% | 10.2% to 12.2% |
| Trigger: first month over 1.5 points; two-month rule | 2025-05; 2025-06 | 2025-04; 2025-05 |
| Largest PSI of predictions and stable inputs | 0.012 | 0.013 |

The conclusions do not depend on the non-disclosure states: the blend's advantage out of time is a point larger
without them, unseen states still show no measurable difference, and error still rises while PSI stays flat.

## 10. Related literature

Full citations are in `docs/references.md`.

- **Disparity measurement.** [Avenancio-León and Howard (2022)](https://doi.org/10.1093/qje/qjac009) document racial gaps in property assessments relative to
  market values. The vendor-AVM gap measured here is sensitive to how price tiers are built: −0.091 log points on tiers
  of the realised sale price over all sales; on the same sales, −0.061 and −0.079 on realised-price tiers against −0.010
  and −0.024, not significant, on tiers of a value known before the sale (§9).
  That sensitivity is this study's result, not a claim about the cited paper.
- **Machine learning in credit.** [Fuster et al. (2022)](https://doi.org/10.1111/jofi.13090) study how more flexible
  models change who gains from better prediction in mortgage lending; the fairness checks in §4 and §9 are a much
  narrower measurement exercise on valuation errors, by tract, not by borrower.
- **Methods.** The market index is a repeat-sales index ([Bailey, Muth, and Nourse 1963](https://doi.org/10.1080/01621459.1963.10480679)); the intervals are conformalized
  quantile regression ([Romano, Patterson, and Candès 2019](https://arxiv.org/abs/1905.03222)), whose coverage guarantee assumes exchangeable calibration and
  test data, so out of time and out of state only the measured coverage counts; COD and PRD follow the IAAO Standard on
  Ratio Studies, whose thresholds are for property tax assessment, not bank AVM approval.
- **Model risk.** The structure of this report (vendor due diligence, challengers, outcomes analysis, monitoring and use
  limits) follows the 2011 supervisory guidance on model risk management (SR 11-7 / OCC 2011-12), since replaced by
  revised guidance (SR 26-2 / OCC Bulletin 2026-13, April 2026).
