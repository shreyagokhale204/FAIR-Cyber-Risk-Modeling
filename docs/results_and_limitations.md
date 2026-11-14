# Results, interpretation, and limitations

## Executive summary

NovaFlow is a hypothetical SaaS company facing a modeled confidentiality incident: a compromised third-party vendor's credentials/OAuth tokens enable Salesforce customer-data exfiltration. The FAIR-inspired Monte Carlo model estimates **annual loss exposure**, then evaluates five illustrative investments and their combined effects.

## Baseline simulation

**100,000 simulated years; seed 20261025.** Dollar amounts are approximate and rounded for readability.

| Measure | Result | Interpretation |
|---|---:|---|
| Annual Average Loss | $272,725 | Mean annual loss including zero-loss years |
| Mean LEF | 0.2216 | Simulated successful incidents per year |
| P(≥1 successful incident) | 19.88% | Frequency event, not necessarily positive dollar loss |
| VaR90 | $984,692 | 90th-percentile annual loss |
| VaR95 | $1,730,117 | 95th-percentile annual loss |
| VaR99 | $3,793,562 | 99th-percentile annual loss |
| TVaR95 | $3,028,009 | Average of worst 5% simulated years |
| TVaR99 | $5,285,004 | Average of worst 1% simulated years |
| P(annual loss > $1M) | 9.87% | Strict exceedance probability |
| P(annual loss > $2M) | 3.99% | Strict exceedance probability |
| P(annual loss > $5M) | 0.43% | Strict exceedance probability |
| P(annual loss > $10M) | 0.02% | Strict exceedance probability |

Theoretical mean LEF from configured frequency parameters is ~0.22314/year. Simulation results vary slightly by seed and sample size. Maximum simulated losses are sample maxima, **not physical upper bounds**.

## Loss composition

The modeled reputation category—including affected/unaffected customer churn and lost deals—accounts for approximately **64% of mean incident loss** in the baseline analysis. This is a consequence of the assumed churn/deal probabilities and one-year contribution values; it is not an observed empirical allocation of real breach costs.

## Security investment findings

The model evaluates five controls and all 32 combinations. Under **central illustrative annual costs**:

| Control | Gross AAL reduction (approx.) | Annual cost | Net annual benefit (approx.) |
|---|---:|---:|---:|
| C1 OAuth governance | $54,590 | $30,000 | +$24,590 |
| C2 Least privilege | $33,038 | $35,000 | −$1,962 |
| C3 SaaS detection | $40,943 | $60,000 | −$19,057 |
| C4 Data protection | $26,746 | $50,000 | −$23,254 |
| C5 IR readiness | $2,549 | $25,000 | −$22,451 |

With sufficient budget, **C1 alone** minimizes *residual AAL plus central annual cost* in the evaluated combinations. The combination minimizing residual AAL alone differs, especially with higher budgets. The all-five combination reduces modeled AAL by roughly $125,245 (~46%), to ~$147,706, but its central annual cost exceeds its single-scenario gross benefit. Standalone reductions must **not** be added together because control effects overlap.

The optimization uses a separate, common-seed evaluation with a baseline near **$272,950**. Minor differences from the 100,000-year risk-analysis baseline reflect simulation settings, not a change in the underlying threat definition.

## Parameter uncertainty findings

Frequency-only, magnitude-only, and combined parameter draws produce distributions of **estimated AAL**. The combined exploratory analysis reported approximately:

| Statistic | Combined parameter uncertainty |
|---|---:|
| Mean estimated AAL | $277,149 |
| Median estimated AAL | $227,345 |
| Standard deviation | $200,568 |
| 5th percentile | $66,104 |
| 95th percentile | $654,789 |

These values are from the development uncertainty analysis and may differ from another saved series or revised run selected in the dashboard. Frequency uncertainty generated more AAL dispersion than modeled magnitude uncertainty **under the chosen parameter ranges**. The magnitude component uses a fitted response surface and therefore inherits approximation error. The 5th–95th interval is **not** annual loss VaR or a predictive interval for a particular year's realized loss.

## Key limitations

1. **One scenario only.** Does not quantify aggregate enterprise cyber risk or benefits against other attack types.
2. **Hypothetical company.** Organizational profile and customer economics are illustrative.
3. **Unvalidated parameter inputs.** Attack rates, success probability, loss amounts, dependencies, churn, and lost-deal assumptions require external calibration.
4. **Unvalidated control economics.** Effectiveness and low/central/high annual costs are illustrative, not vendor quotations or measured causal impacts.
5. **Loss scope.** No downtime/availability loss; no multi-year customer lifetime value; limited legal/jurisdictional granularity.
6. **Dependence assumptions.** Gaussian-copula correlations and logistic odds ratios are analyst inputs, not fitted correlations or causal effects.
7. **Response surface.** Some parameter-uncertainty estimates approximate incident severity using a fitted response surface rather than full Monte Carlo for every draw; prior exploratory holdout checks reported mean error around 4.68% and maximum around 11.18%, subject to the specific validation sample.
8. **Optimization criterion.** Focuses on AAL and incremental annual costs; does not optimize VaR/TVaR, multi-scenario benefits, regulatory minimums, or implementation dependencies.
9. **Simulation uncertainty.** Estimates depend on seeds, iteration counts, and numerical precision, especially for rare tail events.
10. **Decision context.** Financially negative *within this single modeled scenario* does not mean a control is unnecessary for compliance, defense in depth, or protection against other threats.

## Appropriate use

Use the dashboard to compare modeled financial exposures, understand tail risk, demonstrate the FAIR workflow, and explore how assumptions change recommendations. Do **not** treat its values as audited forecasts, guaranteed risk reductions, or a replacement for professional security risk assessment.
