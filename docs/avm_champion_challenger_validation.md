# Champion–challenger validation of third-party home-valuation models

**Question.** Lenders buy home-value fields from data vendors and use them as collateral and verification
inputs. How accurate, fresh and fair are those vendor values against recorded
arm's-length sale prices? And can an independently built challenger (gradient boosting plus a
neighbour-attention network over comparable sales) beat them out of time and out of geography?

**What this is not.** It is not a credit model, and no bank, servicer or credit-bureau data is involved. All outcomes are
public-record sale prices.

Every number in this report comes from `scripts/10–15` and `19`, and the CSVs in `reports/p2/`.

## 1. Data

| Item | Value |
|---|---|
| Ground truth | 2,418,184 arm's-length residential resales (Vendor C 2025 vintage, transaction type 1, deed in {WD, SW, GD, BA, WJ, DD}, $30k–$20M), 2022-05..2025-08, 49 states + DC |
| Features | Vendor C old vintage (≤ 2022-04): prior arm's-length sale and date, 2021 assessor values, property attributes; tract density / owner share / median age; lat/lon |
| Comparable-sale pool | 10,335,897 arm's-length sales 2012-01..2022-04, indexed to 2022-04 with the repeat-sales index |
| Market index | Self-built Bailey–Muth–Nourse repeat-sales index (1,286,709 pairs). The valuation study uses a version estimated **only from pairs whose second sale is ≤ 2024-06, excluding the 9 hold-out states**, frozen after 2024-06 |
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
   - The median ratio is exactly 1.000 in every quarter through 2024Q3, with median error under 0.3%
     (`vendor_accuracy_by_quarter.csv`). It drifts away for sales from 2024Q4 onward (median error 6.4%, then 15%
     in 2025Q1).
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
| B1 hedonic OLS | log price on property, comp, assessor and index features + state FE (train R² 0.725) |
| C1 LightGBM | 33 features (no race variables), early stopping on the calibration window (837 trees) |
| C1q | LightGBM 10%/90% quantile models, split-conformal calibration on 2024-04..06 (CQR; q̂ = 0.024) |
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
- **Segments.** With a prior recorded sale: C3 9.9%. Without one: 14.0%.

**Spatial hold-out (416,381 sales in 9 states the models never saw, index rebuilt without them).**

| Estimator | MdAPE | PPE10 | Median ratio |
|---|---|---|---|
| Vendor A AVM | 29.7% | 10.2% | 0.730 |
| Vendor A market value | 12.0% | 44.6% | 0.929 |
| C1 LightGBM | 13.5% | 38.7% | 0.935 |
| C2 neighbour-attention NN | 12.9% | 40.9% | 0.991 |
| C3 blend | 12.2% | 42.3% | 0.959 |

Out of geography, **the challenger does no better than Vendor A market value**: no significant difference, bootstrap
difference +0.1 pp, CI −2.5 to +1.8. Most hold-out sales fall in 2022–24, when that vendor's ~2022 values were still current, and the
hold-out states get only a division-level index. The attention network transports best of the three challengers
(median ratio 0.991 vs 0.935 for GBM).

**Calibrated uncertainty (C1q, 80% nominal).**

| Window | Coverage | Median interval width |
|---|---|---|
| Calibration | 80.0% | 60% of price |
| OOT | 77.5% | 62% of price |
| Spatial | 78.1% | 66% of price |

The slight under-coverage reflects drift. Interval width ranks large errors (|error| > 20%) with **AUC 0.77**
OOT, against **0.498** for Vendor A's stated AVM error.

**Drift.** With the model and index frozen at 2024-06, C3 monthly MdAPE rises from 10.6% (2024-07) to 12.4%
(2025-08), about 0.14 pp per month. Vendor A market value goes from 15.5% to 17.4%.

**Price tiers (upmarket).**
- Top within-state decile MdAPE: C3 16.3%, C2 16.1%, Vendor A market value 18.2%, Vendor A AVM 47.9%.
- Bottom decile (non-standard and distressed resales) is poor for every estimator (30–74%).
- $1M+ screen: C1 AUC 0.954 vs Vendor A market value 0.947. Recall 72.9% vs 67.4% at precision 88.5% vs 90.7%.
  The ranking gain is small; the recall gain comes with lower precision.

**Disparity audit.** Tract Black + Hispanic share quintiles, computed within price decile so that tier
regressivity is removed:

| Estimator | Within-tier value/price, Q5 − Q1 (log) |
|---|---|
| Vendor A AVM | **−9.1%** (relative undervaluation in the highest-minority-share tracts) |
| Vendor C assessor | +13.2% (relative over-assessment) |
| Vendor B | +9.2% |
| C1 LightGBM | −0.8% |
| C2 attention | −3.2% |
| C3 blend | −2.0% |

The Vendor A AVM gap is the pattern the 2024 interagency AVM quality-control rule and the PAVE task force
target. These are tract-level ratio comparisons, not a fair-lending test of any lending decision.

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
- **Sale prices in non-disclosure states** may be MLS-sourced. The sensitivity above shows the result does not
  depend on them.
- **The comp pool and assessor features date from 2022-04.** A production AVM would refresh comps monthly; the
  frozen design deliberately understates what a maintained challenger could achieve.
- **The 50/50 blend weight was fixed a priori.** The blend was added after seeing calibration-window
  (not test) MdAPEs of C1 and C2.
- **No interior-condition, listing or MLS data.** Bottom-tier errors remain large for every estimator.

## 6. Monitoring plan (proposed)

| Metric | Threshold → action |
|---|---|
| Monthly MdAPE vs calibration baseline | +1.5 pp for 2 consecutive months → re-index |
| 80% interval coverage | below 75% → recalibrate the conformal q̂ |
| PSI of prediction / key inputs | > 0.25 → investigate |
| Within-tier Q5 − Q1 gap | magnitude > 5% → fairness review |
| Vendor-field staleness slope | re-run quarterly, alongside vendor recertification |
