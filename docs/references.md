# References

Works cited in the README and the three reports. Each entry was checked against bibliographic records (journal,
volume, pages; DOI where one was confirmed). The right-hand note says how this repository relates to the work; none of
the claims below is a claim about what these papers find beyond their published abstracts.

## Default and loss models

- **Shumway, T. (2001).** Forecasting bankruptcy more accurately: A simple hazard model. *Journal of Business*, 74(1),
  101–124. https://doi.org/10.1086/209665
  — A discrete-time hazard estimated as a logit on firm-periods. The credit-loss and prepayment models here are
  cause-specific versions of the same estimator.
- **Campbell, J. Y., Hilscher, J., & Szilagyi, J. (2008).** In search of distress risk. *Journal of Finance*, 63(6),
  2899–2939.
  — Dynamic logit model of failure with accounting and market variables; same estimator family.
- **Das, S. R., Duffie, D., Kapadia, N., & Saita, L. (2007).** Common failings: How corporate defaults are correlated.
  *Journal of Finance*, 62(1), 93–117. https://doi.org/10.1111/j.1540-6261.2007.01202.x
  — Defaults cluster more than shared observable intensity factors imply. The SBA write-off batches and cohort shifts
  here are consistent with that; no frailty term is modelled.
- **Duffie, D., Eckner, A., Horel, G., & Saita, L. (2009).** Frailty correlated default. *Journal of Finance*, 64(5),
  2089–2123. https://doi.org/10.1111/j.1540-6261.2009.01495.x
  — An unobserved common factor (frailty) explains default clustering beyond observable covariates. A natural next
  step for the credit-loss model, not attempted here.
- **Demyanyk, Y., & Van Hemert, O. (2011).** Understanding the subprime mortgage crisis. *Review of Financial Studies*,
  24(6), 1848–1880. https://doi.org/10.1093/rfs/hhp033
  — Loan quality by origination vintage, after adjusting for borrower and loan characteristics and macro conditions.
  The approval-era effects here are the SBA counterpart.
- **Rajan, U., Seru, A., & Vig, V. (2015).** The failure of models that predict failure: Distance, incentives, and
  defaults. *Journal of Financial Economics*, 115(2), 237–260. https://doi.org/10.1016/j.jfineco.2014.09.012
  — A default model estimated in one lending regime breaks down systematically in another, because the link between
  observable characteristics and default changes. The credit-loss study shows the same mechanism with the opposite
  sign: a model learned on crisis-era cohorts over-predicts losses on the safer cohorts that followed.
- **Fuster, A., Goldsmith-Pinkham, P., Ramadorai, T., & Walther, A. (2022).** Predictably unequal? The effects of
  machine learning on credit markets. *Journal of Finance*, 77(1), 5–47. https://doi.org/10.1111/jofi.13090
  — Machine-learning default models and their distributional effects in mortgage lending. Here LightGBM ranks loans
  better than the logit but is no better calibrated out of time.

## Prepayment

- **Schwartz, E. S., & Torous, W. N. (1989).** Prepayment and the valuation of mortgage-backed securities. *Journal of
  Finance*, 44(2), 375–392.
  — Empirical prepayment function of explanatory variables; the reduced-form benchmark this study follows.
- **Stanton, R. (1995).** Rational prepayment and the valuation of mortgage-backed securities. *Review of Financial
  Studies*, 8(3), 677–708. https://doi.org/10.1093/rfs/8.3.677
  — Structural prepayment with heterogeneous transaction costs and infrequent review of the option. Heterogeneity is
  not modelled here.
- **Deng, Y., Quigley, J. M., & Van Order, R. (2000).** Mortgage terminations, heterogeneity and the exercise of
  mortgage options. *Econometrica*, 68(2), 275–307. https://doi.org/10.1111/1468-0262.00110
  — Prepayment and default as dependent competing risks; ignoring borrower heterogeneity biases prepayment estimates.
  The competing-risk design here follows this; the missing heterogeneity is a candidate reason for the refinance gap.
- **Andersen, S., Campbell, J. Y., Nielsen, K. M., & Ramadorai, T. (2020).** Sources of inaction in household finance:
  Evidence from the Danish mortgage market. *American Economic Review*, 110(10), 3184–3230.
  — Households act on refinancing incentives with time- and state-dependent delay. The response to a new incentive
  therefore depends on who holds it, which a model fitted without incentives cannot learn.
- **Berger, D., Milbradt, K., Tourre, F., & Vavra, J. (2021).** Mortgage prepayment and path-dependent effects of
  monetary policy. *American Economic Review*, 111(9), 2829–2878. https://doi.org/10.1257/aer.20181857
  — The prepayment response to a rate cut depends on the distribution of rate incentives created by the past rate
  path. The prepayment model here was fitted in 2022–24, when almost no owner was in the money, which is why it could
  not forecast the response to the September 2024 dip.

## Valuation and disparity measurement

- **Avenancio-León, C. F., & Howard, T. (2022).** The assessment gap: Racial inequalities in property taxation.
  *Quarterly Journal of Economics*, 137(3), 1383–1434. https://doi.org/10.1093/qje/qjac009
  — Racial gaps in property assessment relative to market values. The valuation study measures a related gap for
  vendor AVMs and shows that its size depends on how price tiers are defined (§9 of that report); that sensitivity is
  this repository's result, not a claim about the cited paper.

## Research practice

- **Lucas, R. E. (1976).** Econometric policy evaluation: A critique. *Carnegie-Rochester Conference Series on Public
  Policy*, 1, 19–46. https://doi.org/10.1016/S0167-2231(76)80003-6
  — Estimated relationships need not survive a change of policy regime. The 2020–21 payment-relief programme is a
  clean example in the credit-loss data.
- **Harvey, C. R. (2017).** Presidential address: The scientific outlook in financial economics. *Journal of Finance*,
  72(4), 1399–1440. https://doi.org/10.1111/jofi.12530
  — Unreported tests, multiple testing and p-hacking, and a call for more transparent research practice. The analysis
  plans hashed before fitting, the deviations log and the reporting of failed rules here follow that advice; the
  adaptive design across remediation rounds is the forking-paths risk it describes.

## Supervisory and accounting guidance

- **Board of Governors of the Federal Reserve System & Office of the Comptroller of the Currency (2011).** Supervisory
  guidance on model risk management (SR 11-7 / OCC 2011-12).
  — The structure of the validation reports: conceptual soundness, outcomes analysis, ongoing monitoring, use limits.
- **Financial Accounting Standards Board (2016).** Accounting Standards Update 2016-13, Financial Instruments—Credit
  Losses (Topic 326).
  — The current expected credit loss (CECL) standard behind the lifetime-loss framing.
