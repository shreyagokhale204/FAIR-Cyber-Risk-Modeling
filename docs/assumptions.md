# CRQ Model Assumptions

This document records the main **illustrative baseline assumptions** for a hypothetical U.S. B2B SaaS company. It is not a statement of observed losses, real vendor pricing, or empirically validated control effectiveness. The authoritative parameter values and implementation logic remain in `config/model_config.py` and the `src/` modules.

## Scenario and scope

- **Organization:** A hypothetical B2B SaaS provider; annual revenue $900 million; 3,200 employees; 3,000 enterprise customers.
- **Attack path:** A third-party SaaS vendor is compromised; stolen credentials or OAuth tokens are used to access the focal firm's Salesforce CRM and exfiltrate customer data.
- **Modeled harm:** Confidentiality-related financial consequences. No ransomware downtime, outage, service interruption, or other attack paths.
- **Horizon:** One year.
- **Customer valuation:** $300,000 annual revenue per enterprise customer and 70% contribution margin; one-year lost contribution of $210,000 per lost customer or otherwise-won deal. This is **not** lifetime value.

## Loss event frequency assumptions

| Parameter | Baseline | Meaning |
|---|---|---|
| Attack campaign arrival count | Poisson with λ = 3 per year | Number of campaigns that could produce the modeled compromise |
| Success probability per campaign | Beta(0.36, 4.48), redrawn per campaign | Heterogeneity in chance of a successful modeled loss event |
| Mean campaign success probability | 7.4380% | 0.36 / (0.36 + 4.48) |
| Theoretical expected LEF | 0.22314 per year | 3 × mean campaign success probability |
| Successful events per campaign | At most one | Model simplification |

The annual number of incidents is random; an average LEF below one does **not** mean an incident must occur on a fixed schedule. In this implementation, independently redrawing the Beta success probability for each campaign does not itself create annual overdispersion.

## Incident severity and dependence

- Each successful incident receives a latent technical severity factor drawn from a standard normal distribution.
- Selected continuous drivers, including affected customers, response effort, and forensic costs, share dependence through a one-factor Gaussian copula.
- Selected yes/no consequences are generated with calibrated conditional logistic probabilities. Examples include public visibility, notifications, regulatory engagement, fines, judgments, churn, and lost deals.
- A probability target is an **overall calibration target**, not a fixed probability for every incident. Conditional event probabilities vary with modeled severity and other conditions.
- The incident frequency process and severity process are treated as independent.

## Loss magnitude assumptions

### Primary response

| Component | Illustrative assumption |
|---|---|
| Technical response | PERT(100, 300, 800) hours at $103/hour |
| IT/security management | PERT(20, 60, 160) hours at $150/hour |
| Executive response | PERT(5, 15, 40) hours at $200/hour |
| External digital forensics / incident response | PERT($13,000, $80,000, $250,000) |

### Secondary response and planning

Includes legal triage and response, customer notifications and support, regulatory engagement, internal/external public relations, and response planning. Some costs occur only if corresponding events are triggered.

Illustrative inputs include outside counsel's lognormal hourly rate (median $650; log-scale σ 0.30), internal legal work at $118.80/hour, notification coordination PERT(24, 60, 120) hours, customer support setup PERT(16, 40, 80) hours, and planning PERT(4, 16, 40) hours. Electronic notification is assumed; physical mail is disabled in the baseline. Other inputs and event gates are specified in `config/model_config.py`.

### Fines, judgments, and settlements

- Regulatory engagement overall target probability: 10%.
- Regulatory fine overall target probability: 5%, subject to regulatory engagement and model calibration.
- Fine amount, when imposed: lognormal, median $1 million, log-scale σ 0.80.
- Judgment/settlement amount, when triggered: lognormal, median $500,000, log-scale σ 1.00.
- These values are exploratory assumptions, not regulatory predictions.

### Reputation and commercial losses

| Parameter | Illustrative assumption |
|---|---|
| Affected enterprise customers | Beta-PERT min/mode/max 300 / 1,500 / 3,000, rounded and capped at 3,000 |
| Unaffected customers | 3,000 minus affected customers |
| Affected-customer churn target | 0.2% |
| Unaffected-customer churn target | 0.02% |
| Lost-deal probability target | 1% |
| Otherwise-won prospective deals | Binomial(180, 0.20) |
| One-year lost contribution per customer/deal | $210,000 |

Customer churn and lost-deal targets feed conditional models, so actual simulated probabilities may differ across incidents.

## Security controls and illustrative investment assumptions

| ID | Control | Primary model effect | Central efficacy | Annual incremental cost: low / central / high |
|---|---|---|---:|---:|
| C1 | Third-party OAuth governance | Reduces successful-campaign probability | 20% | $15k / $30k / $75k |
| C2 | Least-privilege Salesforce access | Reduces affected-customer exposure | 25% | $20k / $35k / $90k |
| C3 | SaaS anomaly detection and token revocation | Reduces successful-campaign probability | 15% | $30k / $60k / $120k |
| C4 | Data minimization / exfiltration restrictions | Reduces affected-customer exposure | 20% | $25k / $50k / $110k |
| C5 | Incident response readiness | Reduces selected response labor hours | 20% | $10k / $25k / $50k |

Control effects are **relative** reductions, not percentage-point reductions. When two controls affect the same lever, the remaining-risk factors multiply. For example, C1 and C3 leave 0.80 × 0.85 = 0.68 of the baseline success probability. Controls are binary in the investment analysis, and these costs are not vendor quotes.

## Simulation and uncertainty assumptions

- Baseline annual loss simulation: **100,000 simulated years**, seed **20261025**.
- Parameter uncertainty: **20,000 frequency-only draws, 20,000 magnitude-only draws, and 20,000 paired combined draws** in the development analyses.
- Frequency-only analysis varies frequency inputs while holding expected incident loss at its baseline.
- Magnitude-only analysis varies loss magnitude assumptions while holding expected LEF fixed. A fitted response surface approximates incident loss under varied inputs.
- Combined analysis varies both. The histograms show distributions of **estimated AAL**, not distributions of realized annual losses.
- The investment optimizer evaluates all 32 combinations of five controls under budget and cost scenarios; its objective is residual AAL or residual AAL plus annual incremental control cost.
- Interactive changes to model assumptions in the dashboard do not automatically recompute precomputed investment recommendations.

## Interpretation limits

The results describe one hypothetical attack scenario. They omit additional security benefits, regulatory mandates, organizational implementation constraints, attacker adaptation, and most correlations between different attack scenarios. Sensitivity analysis explores uncertainty but does not validate the input distributions. Consult `docs/methodology.md` for model mechanics and `docs/RESULTS_AND_LIMITATIONS.md` for results and caveats.
