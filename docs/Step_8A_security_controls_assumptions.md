# FAIR CRQ — Step 8A: Security Controls and Modeling Assumptions

**Status:** Proposed assumptions for exploratory modeling; not validated efficacy estimates.  
**Scenario:** Compromised third-party SaaS vendor -> stolen OAuth credentials -> unauthorized Salesforce access -> exfiltration of the focal firm's enterprise-customer information. Confidentiality only; no outage.

## 1. Objective and boundary

Assess the *incremental* benefit and cost of five investments against this one modeled loss scenario. Controls do not change attacker campaign arrival rate (baseline 3/year). A prevented loss event is a campaign that fails to produce the modeled exfiltration incident; merely reducing the volume of exfiltrated data affects loss magnitude, not loss-event frequency. Baseline campaign success probability is the mean of Beta(0.36,4.48), approximately 0.07438, giving expected LEF ~0.22314/year. Baseline AAL is approximately $275k/year, subject to Monte Carlo variation.

## 2. Control catalog and assumptions

All amounts are hypothetical annual incremental *fully loaded* costs (licensing, labor, operations, amortized implementation), **not** vendor quotes. Effectiveness estimates are deliberately illustrative and require sensitivity analysis.

| ID | Investment | Specific mechanism | Primary modeled lever | Low / central / high efficacy | Illustrative annual cost |
|---|---|---|---|---|---|
| C1 | Third-party OAuth governance | Approve/review integrations; remove unused connections; restrict vendor authorizations; revoke risky access | Reduce probability of successful exfiltration per campaign | 10% / 20% / 35% relative reduction in campaign success probability | $30k–$120k |
| C2 | Least-privilege Salesforce integration access | Narrow OAuth scopes, permission sets, objects and records accessible to vendor identities | Scale affected-customer exposure PERT min/mode/max | 10% / 25% / 40% relative reduction in exposure | $40k–$150k |
| C3 | SaaS anomaly detection and token revocation | Detect suspicious API/export patterns and disable compromised integrations before reportable exfiltration | Reduce probability of successful exfiltration per campaign | 5% / 15% / 30% relative reduction in campaign success probability | $60k–$200k |
| C4 | Data minimization and exfiltration restrictions | Minimize sensitive records available to integrations; constrain exports | Scale affected-customer exposure PERT min/mode/max | 10% / 20% / 35% relative reduction in exposure | $50k–$180k |
| C5 | Salesforce/vendor incident response readiness | Rehearsed revocation, vendor escalation, legal/technical playbooks | Scale selected response effort hours (not all costs) | 10% / 20% / 35% relative reduction in eligible hours | $25k–$100k |

**Interpretation:** A 20% reduction in campaign success means p_new = 0.8 * p_baseline, *not* a 20-percentage-point reduction. A 25% exposure reduction scales the affected-customer PERT parameters by 0.75, respecting integer bounds and ordering. The effect on total incident loss is indirect, through the existing loss model. C5 reduces specified work hours, not automatically the external DFIR fixed cost, fines, churn, or judgment.

## 3. Exact model mapping

- **C1 and C3:** `success_alpha` and `success_beta` must be adjusted so the mean campaign success probability is multiplied by `(1 - effectiveness)`. For simple deterministic policy comparisons, retain Beta concentration `alpha+beta=4.84` and set `alpha_new = 4.84*p_new`, `beta_new = 4.84*(1-p_new)`; this preserves concentration, **not variance**, and is a modeling convention. Alternatively, use a per-campaign success multiplier while preserving the original Beta distribution; choose one convention and use it consistently. No changes to `campaign_rate`.
- **C2 and C4:** scale `cfg['reputation']['affected_customers']` PERT `min`, `mode`, `max` with the existing `make_magnitude_config('customer_exposure', factor, base_config=...)`. Scaling down remains valid for individual control scenarios; verify combined factor remains positive and PERT ordering holds. The distribution and binomial churn calculations then update downstream.
- **C5:** reduce *hours* in the existing primary technical response, IT/security management, and executive response distributions, and eligible secondary internal coordination/response hours. The precise config keys must be inspected before writing implementation. Do not scale external professional-service costs, hourly wage rates, legal settlement amounts, regulatory fines, or reputation losses without separate justification.

## 4. Portfolio interactions and attribution

For controls affecting the same parameter, use multiplicative remaining-risk factors:

`remaining_factor = product(1 - effectiveness_i)`.

For example, C1=20% and C3=15% yield 1 - (0.8*0.85) = 32% relative reduction in success probability, not 35%. C2=25% and C4=20% yield 40% combined exposure reduction, not 45%. This assumes independent proportional effects and is an approximation, not evidence of independence. Perform portfolio-level simulations rather than summing standalone AAL reductions. Controls may also have fixed shared costs, implementation dependencies, and non-additive benefits; record these as limitations until modeled.

## 5. Effectiveness uncertainty

Use three deterministic scenarios (low/central/high) for each control as a first-pass sensitivity analysis. Do **not** treat them as empirically derived confidence bounds or assign probability distributions without defensible data. Later, model continuous efficacy only if the application warrants it.

## 6. Cost accounting and investment decision

For each standalone control or portfolio calculate:

- `residual_AAL` using the same annual simulation assumptions as baseline;
- `gross_AAL_reduction = baseline_AAL - residual_AAL`;
- `net_annual_benefit = gross_AAL_reduction - incremental_annual_control_cost`;
- `benefit_cost_ratio = gross_AAL_reduction / incremental_annual_control_cost`;
- optional `ROSI = net_annual_benefit / incremental_annual_control_cost`;
- tail measures VaR95/99 and TVaR95/99, noting Monte Carlo uncertainty.

Use **common random numbers** (identical seeds and comparable simulation draws) when feasible to reduce noise in comparisons; do not assume the same seed guarantees exact pathwise coupling if random-number consumption differs.

**Important financial constraint:** This is a *single-scenario* analysis. Since baseline AAL is ~$275k/year, no control portfolio can save more than that annually on this modeled scenario. Costs may exceed standalone scenario savings even if controls make sense across many cyber risks or compliance obligations. Never invent those out-of-scope benefits.

## 7. Key assumptions and limitations

1. Hypothetical firm and incident scenario; campaign arrival, per-campaign success and loss parameters are exploratory.
2. Control effectiveness and costs are illustrative, not calibrated to vendor evaluations or operational measurements.
3. Prevention efficacy changes successful incident frequency; exposure controls affect magnitude. Avoid double counting a prevented event as both frequency and magnitude reduction.
4. Customer count is a proxy for exposure; sensitivity of exposed data, contractual terms, jurisdiction, and record counts are not explicitly modeled.
5. Existing model's latent severity dependence, logistic indicator calibration, and other loss assumptions are retained; shifting exposure may affect the model's effective dependence and expected losses.
6. C5 response-hours reductions need config-key verification and may not imply faster containment, lower customer churn or lower fines.
7. Simultaneous-control effects are approximated by multiplicative factors, without empirical interaction evidence.
8. Model validation 7A–7D supports exploratory analysis only; loss response surface had ~4.68% mean and 11.18% maximum holdout error. Step 8 should preferably use the original incident and annual simulator rather than the 7D-2 response surface for control comparisons.
9. No outage costs, attacker substitution, adversarial adaptation, compliance-only benefits, or protection of other attack scenarios are included.
10. Treat annual control cost ranges as placeholders; use a consistent annualized cost basis in the optimization.

## 8. Implementation plan / acceptance checks for Step 8B

1. Inspect `config/model_config.py` and `src/primary_loss.py`, `src/secondary_loss_response_planning.py` for exact parameter keys before coding C5.
2. Create a validated control catalog with IDs, low/central/high efficacy, annual costs, parameter levers.
3. Implement a pure function `apply_controls(base_config, selected_controls, scenario)` that does not mutate input config, plus a frequency multiplier or modified success Beta parameters.
4. Verify: zero controls reproduces baseline; increasing prevention efficacy lowers mean LEF; exposure scaling lowers mean affected customers; C5 scales only intended hours; all probabilities, PERT parameters and costs remain valid.
5. Run common-seed baseline vs each control, then selected portfolios, and report AAL, VaR, TVaR and net benefit with sensitivity results.
6. Clearly distinguish model-estimated savings from assumed effectiveness and annual costs.

**Step 8A decision:** Approved as a transparent *proposed* control catalog for Step 8B implementation. No claim that efficacy/cost assumptions are measured or externally validated.
