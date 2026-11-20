# References

These are the works cited in the README and the three reports. Each entry was checked against bibliographic records
(journal, volume, pages, DOI) on 2026-11-19. The note after each entry says how this repository relates to the work.
It describes the work only by its stated aim or main finding and does not attribute this repository's results to it.

## Default and loss models

- **Shumway, T. (2001).** Forecasting bankruptcy more accurately: A simple hazard model. *Journal of Business*, 74(1),
  101–124. https://doi.org/10.1086/209665
  — A discrete-time hazard estimated as a logit on firm-periods. The credit-loss and prepayment models here are
  cause-specific versions of the same estimator.
- **Campbell, J. Y., Hilscher, J., & Szilagyi, J. (2008).** In search of distress risk. *Journal of Finance*, 63(6),
  2899–2939. https://doi.org/10.1111/j.1540-6261.2008.01416.x
  — A dynamic logit model of failure with accounting and market variables, from the same estimator family.
- **King, G., & Zeng, L. (2001).** Logistic regression in rare events data. *Political Analysis*, 9(2), 137–163.
  https://doi.org/10.1093/oxfordjournals.pan.a004868
  — Sampling all events and a fraction of non-events, with a correction to the intercept. Both hazard studies use this
  design. Here the fraction is drawn by hash, at the same rate for every cause, which the correction requires.
- **Das, S. R., Duffie, D., Kapadia, N., & Saita, L. (2007).** Common failings: How corporate defaults are correlated.
  *Journal of Finance*, 62(1), 93–117. https://doi.org/10.1111/j.1540-6261.2007.01202.x
  — Corporate defaults cluster more than a model with observable default intensities and the doubly stochastic
  assumption allows. The paper's test cannot tell contagion, frailty and misspecified intensities apart. The SBA
  write-offs also cluster, but administrative batching of write-offs and a common shock to borrowers would both
  produce that pattern, and these records cannot separate the two.
- **Duffie, D., Eckner, A., Horel, G., & Saita, L. (2009).** Frailty correlated default. *Journal of Finance*, 64(5),
  2089–2123. https://doi.org/10.1111/j.1540-6261.2009.01495.x
  — In US public non-financial firms, an unobserved common factor (frailty) explains default clustering beyond
  observable covariates. A frailty term is a candidate extension of the credit-loss model and is not attempted here.
- **Demyanyk, Y., & Van Hemert, O. (2011).** Understanding the subprime mortgage crisis. *Review of Financial Studies*,
  24(6), 1848–1880. https://doi.org/10.1093/rfs/hhp033
  — Loan quality by origination vintage, after adjusting for borrower and loan characteristics and macro conditions.
  The approval-era effects here are an analogue, but they are conditional associations: with age, period and cohort
  overlapping, they do not measure underwriting quality.
- **Rajan, U., Seru, A., & Vig, V. (2015).** The failure of models that predict failure: Distance, incentives, and
  defaults. *Journal of Financial Economics*, 115(2), 237–260. https://doi.org/10.1016/j.jfineco.2014.09.012
  — A default model fitted before securitization fails afterwards, because lenders' changing use of soft information
  alters the data-generating process. In the SBA data, forecasts also deteriorate across cohorts. The SBA data do not
  observe soft information, so they do not identify why the cohorts changed.
- **Fuster, A., Goldsmith-Pinkham, P., Ramadorai, T., & Walther, A. (2022).** Predictably unequal? The effects of
  machine learning on credit markets. *Journal of Finance*, 77(1), 5–47. https://doi.org/10.1111/jofi.13090
  — Traditional and machine-learning default models and their distributional effects in mortgage lending. The AUC and
  calibration comparisons here are this repository's results. The valuation study's tract-level gaps are not
  borrower-level fairness measures.

## Prepayment

- **Schwartz, E. S., & Torous, W. N. (1989).** Prepayment and the valuation of mortgage-backed securities. *Journal of
  Finance*, 44(2), 375–392. https://doi.org/10.1111/j.1540-6261.1989.tb05062.x
  — A prepayment probability that depends on the economic state, estimated as an empirical prepayment function. This
  is the reduced-form benchmark the study follows.
- **Stanton, R. (1995).** Rational prepayment and the valuation of mortgage-backed securities. *Review of Financial
  Studies*, 8(3), 677–708. https://doi.org/10.1093/rfs/8.3.677
  — Structural prepayment by borrowers who face heterogeneous transaction costs. That heterogeneity is not modelled
  here, so it is a candidate explanation for the new-lien shortfall.
- **Deng, Y., Quigley, J. M., & Van Order, R. (2000).** Mortgage terminations, heterogeneity and the exercise of
  mortgage options. *Econometrica*, 68(2), 275–307. https://doi.org/10.1111/1468-0262.00110
  — Default and prepayment as dependent competing risks with borrower heterogeneity. The prepayment study models
  competing exits too, but for proxy events (a sale and a new first lien). Its parameters are not comparable, and it
  does not show that missing heterogeneity causes the shortfall.
- **Andersen, S., Campbell, J. Y., Nielsen, K. M., & Ramadorai, T. (2020).** Sources of inaction in household finance:
  Evidence from the Danish mortgage market. *American Economic Review*, 110(10), 3184–3230.
  https://doi.org/10.1257/aer.20180865
  — Danish households delay refinancing, and the delay differs across households. This is one candidate explanation
  for the new-lien shortfall, which began before the 2024 rate dip.
- **Berger, D., Milbradt, K., Tourre, F., & Vavra, J. (2021).** Mortgage prepayment and path-dependent effects of
  monetary policy. *American Economic Review*, 111(9), 2829–2878. https://doi.org/10.1257/aer.20181857
  — The prepayment response to a rate cut depends on the incentives that the past rate path created. With few owners
  clearly in the money in the training months, the response to the September 2024 dip is hard to estimate. The
  measured incentive, the event definition and borrower differences remain competing explanations.
- **Fonseca, J., & Liu, L. (2024).** Mortgage lock-in, mobility, and labor reallocation. *Journal of Finance*, 79(6),
  3729–3772. https://doi.org/10.1111/jofi.13398
  — Lock-in reduces moving. The paper estimates it from individual credit records and variation in the timing of
  origination, with borrowers' rates estimated from balance, term and payment. The proxy here assigns a market rate by
  purchase cohort, and its sale-side estimate is not identified.

## Valuation and disparity measurement

- **Bailey, M. J., Muth, R. F., & Nourse, H. O. (1963).** A regression method for real estate price index
  construction. *Journal of the American Statistical Association*, 58(304), 933–942.
  https://doi.org/10.1080/01621459.1963.10480679
  — A price index from repeat sales of the same property, estimated by regression. This is the method of the self-built
  index; the smoothing, trimming and hold-out rules are this repository's.
- **Avenancio-León, C. F., & Howard, T. (2022).** The assessment gap: Racial inequalities in property taxation.
  *Quarterly Journal of Economics*, 137(3), 1383–1434. https://doi.org/10.1093/qje/qjac009
  — Racial gaps in property assessment relative to market values. The valuation study measures a related gap for
  vendor AVMs and shows that its size depends on how price tiers are defined (§9 of that report). That sensitivity is
  this repository's result, not a claim about the cited paper.
- **International Association of Assessing Officers (2013).** *Standard on ratio studies* (approved April 2013;
  revision pending).
  — The definitions of the coefficient of dispersion (COD) and the price-related differential (PRD) used for vendor
  values. The standard's thresholds are for property tax assessment, not for approving a bank AVM.

## Estimation and inference methods

- **Romano, Y., Patterson, E., & Candès, E. J. (2019).** Conformalized quantile regression. *Advances in Neural
  Information Processing Systems*, 32, 3543–3553. https://arxiv.org/abs/1905.03222
  — Conformal calibration of quantile-regression intervals. Its coverage guarantee assumes the calibration and test
  data are exchangeable, so out of time and out of state only the measured coverage counts.
- **Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).** Robust inference with multiway clustering. *Journal of
  Business & Economic Statistics*, 29(2), 238–249. https://doi.org/10.1198/jbes.2010.07136
  — Cluster-robust inference with non-nested clustering. This is the county × month standard errors of the prepayment
  study.
- **MacKinnon, J. G., Nielsen, M. Ø., & Webb, M. D. (2025).** Cluster-robust jackknife and bootstrap inference for
  logistic regression models. *Econometric Reviews*, advance online publication.
  https://doi.org/10.1080/07474938.2025.2515161
  — The cluster jackknife for logit, which the paper proposes because the standard cluster-robust variance can be
  unreliable. This is the leave-one-month-out check of the prepayment study.

## Research practice

- **Lucas, R. E. (1976).** Econometric policy evaluation: A critique. *Carnegie-Rochester Conference Series on Public
  Policy*, 1, 19–46. https://doi.org/10.1016/S0167-2231(76)80003-6
  — Estimated relationships need not survive a change of policy regime. The 2020–21 payment relief shows the forecast
  risk when policy changes loan payments directly. The size of the relief overlay is a modelling judgement, not an
  estimate of how borrowers or lenders responded.
- **Harvey, C. R. (2017).** Presidential address: The scientific outlook in financial economics. *Journal of Finance*,
  72(4), 1399–1440. https://doi.org/10.1111/jofi.12530
  — Multiple testing, unreported tests and a call for transparent research practice. The plans hashed before fitting,
  the deviations log and the reporting of failed rules are this repository's way of doing that, not something the
  paper prescribes. The adaptive design across remediation rounds is the risk it warns about.

## Supervisory and accounting guidance

- **Board of Governors of the Federal Reserve System & Office of the Comptroller of the Currency (2011).** Supervisory
  guidance on model risk management (SR 11-7 / OCC 2011-12).
  — The structure of the validation reports: conceptual soundness, outcomes analysis, ongoing monitoring, use limits.
  It was replaced in April 2026 (next entry).
- **Board of Governors of the Federal Reserve System, Office of the Comptroller of the Currency, & Federal Deposit
  Insurance Corporation (2026).** Revised guidance on model risk management (SR 26-2 / OCC Bulletin 2026-13 / FDIC
  FIL-15-2026), 17 April 2026.
  — Supersedes SR 11-7. It sets no enforceable standards, is aimed mainly at banks with more than $30 billion in assets,
  narrows the definition of a model, and leaves generative and agentic AI out of scope. The reports here follow the 2011
  structure.
- **Financial Accounting Standards Board (2016).** Accounting Standards Update 2016-13, Financial Instruments—Credit
  Losses (Topic 326).
  — The current expected credit loss (CECL) standard behind the lifetime-loss framing.
- **Board of Governors of the Federal Reserve System, Federal Deposit Insurance Corporation, National Credit Union
  Administration, & Office of the Comptroller of the Currency (2023).** Interagency policy statement on allowances for
  credit losses (revised April 2023). *Federal Register*, 88, 25479–25488.
  — Beyond the reasonable and supportable forecast period, estimates revert to historical loss information or a proxy.
  The policy statement does not prescribe a four-quarter reversion; that choice, and reverting the macro inputs rather
  than the losses, are this study's assumptions.
- **Board of Governors of the Federal Reserve System (2026).** *2026 stress test scenarios*, February 2026.
  — The baseline and severely adverse paths used in the lifetime-loss estimate. They are hypothetical scenarios, not
  forecasts.
