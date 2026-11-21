# References

These are the works cited in the README and the three reports. Each entry was checked against bibliographic records
(authors, journal, volume, pages, DOI) on 2026-11-20. The note after each entry says how this repository relates to
the work. It describes the work only by its stated aim or main finding and does not attribute this repository's results to it.

## Default and loss models

- Shumway, Tyler. 2001. "Forecasting Bankruptcy More Accurately: A Simple Hazard Model." *Journal of Business* 74 (1): 101–24. https://doi.org/10.1086/209665.
  — A discrete-time hazard estimated as a logit on firm-periods. The credit-loss and prepayment models here are
  cause-specific versions of the same estimator.
- Campbell, John Y., Jens Hilscher, and Jan Szilagyi. 2008. "In Search of Distress Risk." *Journal of Finance* 63 (6): 2899–2939. https://doi.org/10.1111/j.1540-6261.2008.01416.x.
  — A dynamic logit model of failure with accounting and market variables, from the same estimator family.
- King, Gary, and Langche Zeng. 2001. "Logistic Regression in Rare Events Data." *Political Analysis* 9 (2): 137–63. https://doi.org/10.1093/oxfordjournals.pan.a004868.
  — Sampling all events and a fraction of non-events, with a correction to the intercept. Both hazard studies use this
  design. Here the fraction is drawn by hash, at the same rate for every cause, which the correction requires.
- Das, Sanjiv R., Darrell Duffie, Nikunj Kapadia, and Leandro Saita. 2007. "Common Failings: How Corporate Defaults Are Correlated." *Journal of Finance* 62 (1): 93–117. https://doi.org/10.1111/j.1540-6261.2007.01202.x.
  — Corporate defaults cluster more than a model with observable default intensities and the doubly stochastic
  assumption allows. The paper's test cannot tell contagion, frailty and misspecified intensities apart. The SBA
  write-offs also cluster, but administrative batching of write-offs and a common shock to borrowers would both
  produce that pattern, and these records cannot separate the two.
- Duffie, Darrell, Andreas Eckner, Guillaume Horel, and Leandro Saita. 2009. "Frailty Correlated Default." *Journal of Finance* 64 (5): 2089–2123. https://doi.org/10.1111/j.1540-6261.2009.01495.x.
  — In US public non-financial firms, an unobserved common factor (frailty) explains default clustering beyond
  observable covariates. A frailty term is a candidate extension of the credit-loss model and is not attempted here.
- Demyanyk, Yuliya, and Otto Van Hemert. 2011. "Understanding the Subprime Mortgage Crisis." *Review of Financial Studies* 24 (6): 1848–80. https://doi.org/10.1093/rfs/hhp033.
  — Loan quality by origination vintage, after adjusting for borrower and loan characteristics and macro conditions.
  The approval-era effects here are an analogue, but they are conditional associations: with age, period and cohort
  overlapping, they do not measure underwriting quality.
- Rajan, Uday, Amit Seru, and Vikrant Vig. 2015. "The Failure of Models That Predict Failure: Distance, Incentives, and Defaults." *Journal of Financial Economics* 115 (2): 237–60. https://doi.org/10.1016/j.jfineco.2014.09.012.
  — A default model fitted before securitization fails afterwards, because lenders' changing use of soft information
  alters the data-generating process. In the SBA data, forecasts also deteriorate across cohorts. The SBA data do not
  observe soft information, so they do not identify why the cohorts changed.
- Fuster, Andreas, Paul Goldsmith-Pinkham, Tarun Ramadorai, and Ansgar Walther. 2022. "Predictably Unequal? The Effects of Machine Learning on Credit Markets." *Journal of Finance* 77 (1): 5–47. https://doi.org/10.1111/jofi.13090.
  — Traditional and machine-learning default models and their distributional effects in mortgage lending. The AUC and
  calibration comparisons here are this repository's results. The valuation study's tract-level gaps are not
  borrower-level fairness measures.

## Prepayment

- Schwartz, Eduardo S., and Walter N. Torous. 1989. "Prepayment and the Valuation of Mortgage-Backed Securities." *Journal of Finance* 44 (2): 375–92. https://doi.org/10.1111/j.1540-6261.1989.tb05062.x.
  — A prepayment probability that depends on the economic state, estimated as an empirical prepayment function. This
  is the reduced-form benchmark the study follows.
- Stanton, Richard. 1995. "Rational Prepayment and the Valuation of Mortgage-Backed Securities." *Review of Financial Studies* 8 (3): 677–708. https://doi.org/10.1093/rfs/8.3.677.
  — Structural prepayment by borrowers who face heterogeneous transaction costs. That heterogeneity is not modelled
  here, so it is a candidate explanation for the new-lien shortfall.
- Deng, Yongheng, John M. Quigley, and Robert Van Order. 2000. "Mortgage Terminations, Heterogeneity and the Exercise of Mortgage Options." *Econometrica* 68 (2): 275–307. https://doi.org/10.1111/1468-0262.00110.
  — Default and prepayment as dependent competing risks with borrower heterogeneity. The prepayment study models
  competing exits too, but for proxy events (a sale and a new first lien). Its parameters are not comparable, and it
  does not show that missing heterogeneity causes the shortfall.
- Andersen, Steffen, John Y. Campbell, Kasper Meisner Nielsen, and Tarun Ramadorai. 2020. "Sources of Inaction in Household Finance: Evidence from the Danish Mortgage Market." *American Economic Review* 110 (10): 3184–3230. https://doi.org/10.1257/aer.20180865.
  — Danish households delay refinancing, and the delay differs across households. This is one candidate explanation
  for the new-lien shortfall, which began before the 2024 rate dip.
- Berger, David, Konstantin Milbradt, Fabrice Tourre, and Joseph Vavra. 2021. "Mortgage Prepayment and Path-Dependent Effects of Monetary Policy." *American Economic Review* 111 (9): 2829–78. https://doi.org/10.1257/aer.20181857.
  — The prepayment response to a rate cut depends on the incentives that the past rate path created. With few owners
  clearly in the money in the training months, the response to the September 2024 dip is hard to estimate. The
  measured incentive, the event definition and borrower differences remain competing explanations.
- Fonseca, Julia, and Lu Liu. 2024. "Mortgage Lock-In, Mobility, and Labor Reallocation." *Journal of Finance* 79 (6): 3729–72. https://doi.org/10.1111/jofi.13398.
  — Lock-in reduces moving. The paper estimates it from individual credit records and variation in the timing of
  origination, with borrowers' rates estimated from balance, term and payment. The proxy here assigns a market rate by
  purchase cohort, and its sale-side estimate is not identified.

## Valuation and disparity measurement

- Bailey, Martin J., Richard F. Muth, and Hugh O. Nourse. 1963. "A Regression Method for Real Estate Price Index Construction." *Journal of the American Statistical Association* 58 (304): 933–42. https://doi.org/10.1080/01621459.1963.10480679.
  — A price index from repeat sales of the same property, estimated by regression. This is the method of the self-built
  index; the smoothing, trimming and hold-out rules are this repository's.
- Avenancio-León, Carlos F., and Troup Howard. 2022. "The Assessment Gap: Racial Inequalities in Property Taxation." *Quarterly Journal of Economics* 137 (3): 1383–1434. https://doi.org/10.1093/qje/qjac009.
  — Racial gaps in property assessment relative to market values. The valuation study measures a related gap for
  vendor AVMs and shows that its size depends on how price tiers are defined (§9 of that report). That sensitivity is
  this repository's result, not a claim about the cited paper.
- International Association of Assessing Officers. 2013. *Standard on Ratio Studies*. Kansas City, MO: International Association of Assessing Officers.
  — The definitions of the coefficient of dispersion (COD) and the price-related differential (PRD) used for vendor
  values. The standard's thresholds are for property tax assessment, not for approving a bank AVM.

## Estimation and inference methods

- Romano, Yaniv, Evan Patterson, and Emmanuel J. Candès. 2019. "Conformalized Quantile Regression." In *Advances in Neural Information Processing Systems 32*, 3543–53. Red Hook, NY: Curran Associates. https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html.
  — Conformal calibration of quantile-regression intervals. Its coverage guarantee assumes the calibration and test
  data are exchangeable, so out of time and out of state only the measured coverage counts.
- Cameron, A. Colin, Jonah B. Gelbach, and Douglas L. Miller. 2011. "Robust Inference with Multiway Clustering." *Journal of Business & Economic Statistics* 29 (2): 238–49. https://doi.org/10.1198/jbes.2010.07136.
  — Cluster-robust inference with non-nested clustering. This is the county × month standard errors of the prepayment
  study.
- MacKinnon, James G., Morten Ørregaard Nielsen, and Matthew D. Webb. 2025. "Cluster-Robust Jackknife and Bootstrap Inference for Logistic Regression Models." *Econometric Reviews*, published online ahead of print. https://doi.org/10.1080/07474938.2025.2515161.
  — The cluster jackknife for logit, which the paper proposes because the standard cluster-robust variance can be
  unreliable. This is the leave-one-month-out check of the prepayment study.

## Research practice

- Lucas, Robert E., Jr. 1976. "Econometric Policy Evaluation: A Critique." *Carnegie-Rochester Conference Series on Public Policy* 1: 19–46. https://doi.org/10.1016/S0167-2231(76)80003-6.
  — Estimated relationships need not survive a change of policy regime. The 2020–21 payment relief shows the forecast
  risk when policy changes loan payments directly. The size of the relief overlay is a modelling judgement, not an
  estimate of how borrowers or lenders responded.
- Harvey, Campbell R. 2017. "Presidential Address: The Scientific Outlook in Financial Economics." *Journal of Finance* 72 (4): 1399–1440. https://doi.org/10.1111/jofi.12530.
  — Multiple testing, unreported tests and a call for transparent research practice. The plans hashed before fitting,
  the deviations log and the reporting of failed rules are this repository's way of doing that, not something the
  paper prescribes. The adaptive design across remediation rounds is the risk it warns about.

## Supervisory and accounting guidance

- Board of Governors of the Federal Reserve System and Office of the Comptroller of the Currency. 2011. "Supervisory Guidance on Model Risk Management." SR 11-7 / OCC Bulletin 2011-12, April 4.
  — The structure of the validation reports: conceptual soundness, outcomes analysis, ongoing monitoring, use limits.
  It was replaced in April 2026 (next entry).
- Board of Governors of the Federal Reserve System, Office of the Comptroller of the Currency, and Federal Deposit Insurance Corporation. 2026. "Revised Guidance on Model Risk Management." SR 26-2 / OCC Bulletin 2026-13 / FDIC FIL-15-2026, April 17. https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm.
  — Supersedes SR 11-7. It sets no enforceable standards, is aimed mainly at banks with more than $30 billion in assets,
  narrows the definition of a model, and leaves generative and agentic AI out of scope. The reports here follow the 2011
  structure.
- Financial Accounting Standards Board. 2016. *Financial Instruments—Credit Losses (Topic 326): Measurement of Credit Losses on Financial Instruments*. Accounting Standards Update 2016-13. Norwalk, CT: Financial Accounting Standards Board.
  — The current expected credit loss (CECL) standard behind the lifetime-loss framing.
- Board of Governors of the Federal Reserve System, Federal Deposit Insurance Corporation, National Credit Union Administration, and Office of the Comptroller of the Currency. 2023. "Interagency Policy Statement on Allowances for Credit Losses (Revised April 2023)." *Federal Register* 88 (April 27): 25479–88. https://www.federalregister.gov/documents/2023/04/27/2023-08876/interagency-policy-statement-on-allowances-for-credit-losses-revised-april-2023.
  — Beyond the reasonable and supportable forecast period, estimates revert to historical loss information or a proxy.
  The policy statement does not prescribe a four-quarter reversion; that choice, and reverting the macro inputs rather
  than the losses, are this study's assumptions.
- Board of Governors of the Federal Reserve System. 2026. *2026 Stress Test Scenarios*. Washington, DC: Board of Governors of the Federal Reserve System, February. https://www.federalreserve.gov/publications/2026-stress-test-scenarios.htm.
  — The baseline and severely adverse paths used in the lifetime-loss estimate. They are hypothetical scenarios, not
  forecasts.
