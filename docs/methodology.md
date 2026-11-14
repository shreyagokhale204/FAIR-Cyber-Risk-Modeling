# Model methodology and assumptions

## 1. Objective, boundary, and FAIR framing

The modeled loss event is **unauthorized access to NovaFlow's Salesforce customer data and exfiltration following compromise of a third-party SaaS vendor's credentials/OAuth tokens**. It is a confidentiality-only scenario, with a one-year financial horizon. A campaign that does not result in the modeled incident is not counted as a loss event. The model does not include service interruption, ransomware downtime, or other attack paths.

The conceptual FAIR decomposition is:

\[
\text{Annual Average Loss (AAL)} = \mathbb{E}[\text{loss events per year}] \times \mathbb{E}[\text{loss per event}].
\]

This identity holds for the modeled independent frequency/severity process. The simulation calculates annual losses directly rather than simply multiplying two point estimates.

## 2. Organization and valuation assumptions

NovaFlow is hypothetical: $900M annual revenue, 3,200 employees, 3,000 enterprise customers, $300,000 annual revenue per customer, and a 70% contribution margin. The modeled one-year financial loss per churned customer or lost deal is **$210,000** ($300,000 × 70%). This is not a multi-year customer lifetime value estimate.

## 3. Loss event frequency

- Annual attack campaign count: **Poisson(λ = 3)**.
- Each campaign draws a success probability from **Beta(α = 0.36, β = 4.48)**, followed by a Bernoulli success/failure trial.
- Mean success probability: **7.4380%**; expected successful loss events per year: **0.22314**.
- At most one modeled loss event per successful campaign.
- Frequency and severity are modeled as independent.

Because the Beta probability is redrawn independently for each campaign, the resulting annual successful-event count is a *thinned Poisson* process. This particular design does **not** introduce annual overdispersion merely by using a Beta probability. A shared annual probability or correlated campaigns would require a different model.

## 4. Loss event magnitude

The loss from a successful incident is the sum of four categories. Their constituent cost items and indicators are modeled using PERT, Gamma, Lognormal, Bernoulli/Binomial, and conditional logistic distributions.

### 4.1 Primary response loss

Technical response labor: PERT(100, 300, 800) hours at $103/hour. IT/security management: PERT(20, 60, 160) hours at $150/hour. Executive response: PERT(5, 15, 40) hours at $200/hour. External digital forensics and incident response (DFIR): PERT($13,000, $80,000, $250,000).

### 4.2 Secondary response and planning

Includes external/internal legal work for initial triage, notification, regulatory engagement, and remediation/contracts; customer notification coordination; customer support contacts and setup; internal/external public relations; and response planning. Selected stages occur only if corresponding notification, regulatory, or external-agency indicators are triggered.

Illustrative inputs include a lognormal outside-counsel hourly rate (median $650; log-scale σ=0.30), internal legal rate $118.80/hour, notification coordination PERT(24, 60, 120) hours, customer support setup PERT(16, 40, 80) hours, and planning PERT(4, 16, 40) hours. Electronic notification is modeled; physical mail is off in the baseline. See `config/model_config.py` for the full parameter table and gating logic.

### 4.3 Fines, judgments, and settlements

A regulatory fine is gated by regulatory engagement and a separately calibrated fine indicator; the amount conditional on a fine follows a lognormal distribution with median $1M and log-scale σ=0.80. A judgment/settlement indicator gates a lognormal amount with median $500k and σ=1.00. These are **illustrative loss distributions**, not estimates from NovaFlow cases. The overall target probability of a fine is 5% and of regulatory engagement is 10%; the fine target is calibrated jointly with its gate.

### 4.4 Reputational loss

The number of affected enterprise customers follows a modified beta-PERT distribution with min/mode/max **300 / 1,500 / 3,000**, rounded to an integer and capped by the total customer count. Unaffected customers equal 3,000 minus affected customers.

Conditional binomial outcomes model churn among affected customers (target probability 0.2%), churn among unaffected customers (0.02%), and lost prospective deals (1% target). The pool of otherwise-won opportunities is modeled as Binomial(180, 0.20). Each lost customer or deal contributes **$210,000** of one-year lost contribution. These probabilities are targets for calibrated conditional models, not guaranteed incident-specific probabilities.

## 5. Incident severity and dependence

Each successful incident receives a latent standard-normal severity draw. A **one-factor Gaussian copula** induces dependence across selected continuous loss drivers, such as exposed customers, response hours, and external DFIR cost. Illustrative latent-normal correlations include 0.60 for affected customers and technical response hours, 0.50 for IT/security management hours, and 0.40 for executive response hours. These are **latent correlations**, not observed correlations between dollar losses.

Several binary outcomes (visibility, notification, regulatory engagement, fines, judgment, churn, and lost deals) are generated through calibrated logistic probability models whose predictors may include severity, visibility, and/or regulatory engagement. The calibration procedure chooses intercepts to match configured target probabilities while preserving modeled dependence. Some events are gated by other indicators; therefore their marginal and conditional probabilities should not be confused.

## 6. Annual simulation and risk metrics

For each simulated year, draw the campaign count; determine successful campaigns; draw incident losses for successful events; and sum them. Years with no modeled losses remain in the annual sample. The primary baseline uses **100,000 simulated years** and seed **20261025**.

- **AAL:** mean of simulated annual total losses.
- **LEF:** mean count of successful incidents per year.
- **VaR95 / VaR99:** 95th / 99th percentiles of annual loss.
- **TVaR95 / TVaR99:** mean of the worst 5% / 1% simulated annual losses, using the model's finite-sample convention.
- **Loss exceedance curve:** empirical fraction of simulated years with annual loss *strictly greater* than each threshold.

**Annual-loss VaR is not the same as parameter uncertainty in estimated AAL.** The former describes random realized losses under fixed parameters; the latter describes uncertainty about the expected loss estimate itself.

## 7. Uncertainty and sensitivity analysis

### Monte Carlo (aleatory) variability

Compare simulations with different random seeds and simulated-year counts. Changing seed/iteration count changes numerical precision and random sampling, **not** the underlying risk assumptions. Small samples can be especially unstable for VaR99 and TVaR99.

### Alternative parameter assumptions

The dashboard can rerun the annual model after adjusting campaign rate, success probability, customer exposure mode, churn targets, lost-deal probability, and contribution margin. The baseline and alternative are compared using the same seed and simulation length. These interactive experiments are distinct from the precomputed optimization.

### Epistemic parameter uncertainty

The development notebooks sample **20,000 parameter combinations** for the frequency analysis and **20,000 combinations** for the loss-magnitude analysis. The combined uncertainty analysis constructs **20,000 paired draws**, yielding three estimated-AAL series:

- **Frequency-only:** varying campaign frequency and mean success probability while holding expected incident magnitude at baseline.
- **Magnitude-only:** varying modeled loss magnitude assumptions while holding expected LEF at baseline.
- **Combined:** varying both simultaneously.

For each draw, an AAL estimate is computed; the histogram is the distribution of these estimates, **not** the distribution of realized annual losses. The magnitude analysis uses a fitted **response-surface approximation** to estimate incident loss across parameter combinations. Its approximation error and chosen input ranges limit interpretation. The observed dominance of frequency uncertainty is specific to those assumed ranges and this model, not a universal finding.

## 8. Security controls and investment optimization

Five binary controls affect three distinct levers:

| Lever | Controls | Central effects |
|---|---|---|
| Successful campaign probability | C1 OAuth governance; C3 SaaS detection | 20% and 15% relative reductions |
| Affected-customer exposure | C2 least privilege; C4 data protection | 25% and 20% relative reductions |
| Selected response hours | C5 IR readiness | 20% relative reduction |

Where controls act on the same lever, the model multiplies the *remaining* factors (e.g., C1+C3 leaves `0.80 × 0.85 = 0.68`, a 32% reduction in success probability, not 35%). Control effects and costs are illustrative.

The optimizer evaluates all **2^5 = 32** combinations under annual cost scenarios and budgets. It can select the feasible combination minimizing either:

1. **Residual AAL** (risk-only objective); or
2. **Residual AAL + annual incremental control cost** (total expected cost objective).

Gross annual benefit = baseline AAL − residual AAL; net annual benefit = gross benefit − incremental annual control cost. These are *one-scenario* financial effects. Compliance obligations, operational dependencies, other threats, and post-control tail-risk constraints are not included in the optimizer.

## 9. Implementation references

- `config/model_config.py` — distribution definitions, calibrated-probability targets, and baseline assumptions.
- `src/incident_simulation.py` — incident-level loss assembly.
- `src/annual_loss_simulation.py` — frequency, annual aggregation, and annual risk summaries.
- `src/control_effectiveness.py` — modeled control levers and comparisons.
- `notebooks/10_model_validation_parameter_uncertainty.ipynb` and `11_model_validation_loss_parameter_uncertainty.ipynb` — parameter uncertainty analyses.
- `notebooks/13_security_investment_optimization.ipynb` — control combination enumeration and optimization.

File names reflect the development project and should be checked against the final GitHub repository layout.
