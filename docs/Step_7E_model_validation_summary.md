# FAIR CRQ — Step 7E: Model Validation and Limitations

**Status:** Accepted for exploratory, portfolio-level security-investment analysis; **not** validated for production decision-making.  
**Scenario:** Third-party SaaS vendor compromise, stolen OAuth credentials, unauthorized Salesforce CRM access and customer-data exfiltration. Confidentiality impact only; no modeled downtime.

## Baseline risk model

- Campaign count: Poisson(3) per year.
- Campaign-level success probability: Beta(0.36, 4.48), mean approximately 7.44%.
- Expected successful incidents: approximately 0.223 per year; probability of one or more successful incidents approximately 20%.
- Conditional incident losses include primary response, secondary response/planning, fines and judgments, and customer/deal-related reputation losses.
- A representative 100,000-year baseline simulation (seed 20261025) gave AAL **$272,725**, annual 95% VaR **$1,730,117**, 99% VaR **$3,793,562**, 95% TVaR **$3,028,009**, and 99% TVaR **$5,285,004**. Exact Monte Carlo outputs depend on seed.

## Validation evidence

| Check | Finding | Status |
|---|---|---|
| 7A Monte Carlo convergence | Across 20 independent 50,000-year runs, AAL SD about $2,555 (0.93% of mean); 99% TVaR SD about $95,888 (1.86%). | Acceptable for exploratory analysis; tails remain noisier |
| 7B Input/output distributions | Observed campaign and incident frequencies, calibrated indicator rates, affected-customer distribution, churn rates and accounting reconciliations matched expected behavior within simulation noise. | Pass |
| 7C Dependence diagnostics | Severity related positively to costs and consequences; high-severity quintiles produced greater losses. | Pass |
| 7C Dependence stress | Mean annual loss remained near $275K while 99% TVaR ranged from approximately $4.18M (weaker) to $5.95M (stronger). | Material tail sensitivity; retain as risk |
| 7D-1 Frequency parameter uncertainty | Illustrative frequency-only AAL 5th–95th interval approximately $72,740–$602,302 in final combined notebook. | Complete, illustrative |
| 7D-2 Magnitude parameter uncertainty | Illustrative magnitude-only AAL 5th–95th interval approximately $182,939–$437,669. | Complete, approximate |
| 7D-3 Combined uncertainty | 20,000 draws: mean AAL $277,149, median $227,345, 5th–95th interval $66,104–$654,789, SD $200,568. | Complete, illustrative |

## Modeling assumptions and limitations

1. **Hypothetical firm and scenario.** An illustrative B2B SaaS company; risk figures are not calibrated to a specific real company.
2. **Limited evidence for frequency parameters.** Campaign intensity and success probabilities are assumptions. The Beta distribution describing variability *between campaigns* is distinct from the illustrative epistemic distribution over the unknown mean success probability.
3. **Loss parameter evidence.** Churn, exposure, contribution margin, legal/regulatory consequences and response costs require empirical validation. The modeled contribution margin is a one-year horizon, not lifetime value.
4. **Conditional dependence.** A common latent severity factor and severity/visibility-driven logistic models introduce dependence. Correlations are provisional latent-normal parameters, not observed dollar-loss correlations. Joint dependence stress changes multiple mechanisms simultaneously.
5. **Tail estimates.** Rare extreme-loss estimates are sensitive to dependence and Monte Carlo noise; annual VaR/TVaR parameter uncertainty was not separately quantified.
6. **Response-surface approximation.** Step 7D-2 used a quadratic surrogate with 4.68% mean and 11.18% maximum holdout absolute percentage error; the maximum exceeded the provisional 10% acceptance threshold. One of 20,000 magnitude draws was outside the training domain and was excluded in that analysis. The final combined analysis resampled only valid draws. Do not interpret the surrogate as production-validated.
7. **Epistemic intervals are illustrative.** Input uncertainty distributions were selected for exploration, not estimated as empirical posteriors. Parameter draws were assumed independent, which may not hold in practice.
8. **Double counting.** Maintain distinct response, fines, settlements, churn and lost-deal loss categories; verify no overlap when adding control effects. The scenario excludes operational downtime by design.
9. **No formal external validation.** No independent real-world incident-loss dataset, external benchmark reconciliation, or expert elicitation has been completed. Local pytest results were not independently verified from the notebook uploads.

## Readiness decision

**GO for Step 8: exploratory security-control effectiveness modeling.** Preserve the unchanged baseline as a reference. Represent each control's impact explicitly on attack frequency/success or conditional loss, prevent double counting, and use paired or common-random-number comparisons where practical. Report both expected loss avoided and tail-risk effects, and test plausible control-effectiveness ranges.

**NO-GO for production budgeting or board-level loss forecasts without additional empirical calibration, external validation and independent review.**

## Suggested project artifact location

`docs/07_model_validation_summary.md`
