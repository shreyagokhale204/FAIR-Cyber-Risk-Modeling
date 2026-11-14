# NovaFlow — FAIR-Based Cyber Risk Quantification

**Monte Carlo cyber loss modeling · Security investment optimization · Interactive decision-support dashboard**

NovaFlow is a **hypothetical B2B SaaS company** used to demonstrate how cybersecurity risks can be translated into financial outcomes and security investment decisions. This project models a specific third-party SaaS compromise, estimates annual financial exposure using a FAIR-inspired approach, evaluates five security controls, and provides an interactive Streamlit dashboard for exploring results and assumptions.

> **Scope and disclosure:** This is an exploratory, educational model—not a production risk assessment or a forecast for a real company. Several probability, severity, control-effectiveness, and cost inputs are illustrative rather than empirically validated. Results cover **one cyberattack scenario**, not organization-wide cyber risk.

## Business question

**How much financial risk does a third-party SaaS credential compromise create, and which security investments provide the greatest modeled reduction in expected loss under a limited budget?**

The analysis connects:
1. Attack campaigns and the probability of successful compromise;
2. Financial consequences of a successful incident;
3. Uncertainty in events and assumptions;
4. Security controls that reduce frequency or severity;
5. Budget-constrained security investment decisions.

## Modeled organization and scenario

| Attribute | Assumption |
|---|---|
| Organization | NovaFlow, hypothetical U.S. B2B SaaS company |
| Annual revenue | $900 million |
| Employees | 3,200 |
| Enterprise customers | 3,000 |
| Average annual revenue per customer | $300,000 |
| Contribution margin | 70% |
| Contribution value of lost customer/deal | $210,000 (one-year horizon) |
| Time horizon | One year |

**Threat scenario:** Attackers compromise a third-party SaaS vendor, steal credentials or OAuth tokens, use that trusted access to reach NovaFlow's Salesforce CRM, and exfiltrate enterprise-customer data. The model captures **confidentiality losses only**; operational downtime and availability losses are excluded.

## Key results

The baseline 100,000-year simulation (seed `20261025`) produced:

| Metric | Result |
|---|---:|
| Annual Average Loss (AAL) | **$272,725** |
| Average successful loss events per year (LEF) | **0.2216** (simulated) |
| Probability of at least one incident | **19.88%** |
| Annual loss VaR95 | **$1.73 million** |
| Annual loss VaR99 | **$3.79 million** |
| Annual loss TVaR95 | **$3.03 million** |
| Annual loss TVaR99 | **$5.29 million** |
| Probability annual loss exceeds $1 million | **9.87%** |
| Probability annual loss exceeds $2 million | **3.99%** |
| Probability annual loss exceeds $5 million | **0.43%** |

The theoretical expected annual LEF is `3 × (0.36 / (0.36 + 4.48)) ≈ 0.22314`. Simulation estimates vary slightly with sample size and seed.

**Interpretation:** The model has many zero-loss years and a long right tail. Expected loss alone therefore does not capture the size of unusually costly years.

### Security investment findings

The analysis evaluates all **32 combinations of five binary security controls** using illustrative annual costs and efficacy assumptions. Under the **central-cost scenario**, the financially optimal modeled choice with a sufficient budget is **C1 (OAuth governance)**: about **$30,000 annual cost**, **$54,590 gross AAL reduction**, and **$24,590 net annual benefit**. A strategy that minimizes *residual risk regardless of spending* can select a different combination. These are scenario-specific model results, not universal purchasing recommendations.

## Security controls

| ID | Investment | Modeled effect | Central efficacy | Annual cost: low / central / high |
|---|---|---|---:|---:|
| C1 | Third-party OAuth governance | Lower successful compromise probability | 20% | $15k / $30k / $75k |
| C2 | Least-privilege Salesforce access | Reduce affected-customer exposure | 25% | $20k / $35k / $90k |
| C3 | SaaS anomaly detection and token revocation | Lower successful compromise probability | 15% | $30k / $60k / $120k |
| C4 | Data minimization and exfiltration restrictions | Reduce affected-customer exposure | 20% | $25k / $50k / $110k |
| C5 | Incident response readiness | Reduce selected response labor hours | 20% | $10k / $25k / $50k |

Overlapping effects on the same model lever are **multiplicative**, not additive. Costs represent illustrative *incremental annual* spending.

## Dashboard

The Streamlit dashboard provides five sections:

| Section | Function |
|---|---|
| **Executive Overview** | Key risk and investment indicators, with a stated reference investment case |
| **Cyber Risk Analysis** | AAL, LEF, VaR, TVaR, annual loss histogram, loss exceedance curve, incident frequency, and loss composition |
| **Security Controls** | Control descriptions, effects, costs, and individual financial comparisons |
| **Investment Optimizer** | Budget and cost-scenario exploration; compare minimizing residual AAL versus minimizing total expected cost |
| **Uncertainty & Assumptions** | Monte Carlo experiments, interactive alternative assumptions, frequency/magnitude parameter uncertainty, control sensitivity, and model reference |

**Two types of uncertainty are deliberately separated:**
- **Aleatory:** Random year-to-year outcomes under fixed assumptions, explored through simulated years, seeds, and loss distributions.
- **Epistemic:** Uncertainty about the model parameters themselves, explored through alternative assumptions and distributions of estimated AAL.

Interactive assumption experiments **do not automatically recalculate** the precomputed security investment optimization results.

## Methodology at a glance

1. **Frequency:** Campaign arrivals follow `Poisson(λ = 3/year)`. Each campaign draws a success probability from `Beta(0.36, 4.48)` and then a Bernoulli outcome.
2. **Magnitude:** Successful incidents generate primary response, secondary response/planning, fines and judgments, and reputation-related losses.
3. **Dependence:** A latent incident-severity factor and Gaussian-copula structure correlate selected loss drivers; calibrated logistic models determine conditional outcomes.
4. **Annual aggregation:** Sum all successful incident losses in each simulated year, including zero-loss years.
5. **Validation and sensitivity:** Assess convergence, distribution behavior, dependence, and epistemic parameter uncertainty.
6. **Investment analysis:** Apply illustrative control effects, calculate residual AAL and costs, and enumerate all 32 control combinations.

See [Model methodology](docs/MODEL_METHODOLOGY.md) and [Results and limitations](docs/RESULTS_AND_LIMITATIONS.md) for details.

## Run locally

### Prerequisites

- Python **3.13** (development environment used in this project)
- `numpy`, `pandas`, `scipy`, `matplotlib`, `plotly`, `streamlit`, and `jupyter` for the notebooks/dashboard
- Project source modules and the precomputed CSV outputs in their expected locations

From the **project root**:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app/dashboard.py
```

Open the local URL shown by Streamlit (typically `http://localhost:8501`). To stop the server, press **Ctrl+C** in the terminal.

> **Before publishing:** Confirm the final dependency versions in `requirements.txt` using the tested environment. The supplied documentation includes a dependency list, but does not assert that a clean-environment install has already passed.

### Expected layout

```text
novaflow-fair-crq/
├── app/
│   └── dashboard.py
├── config/
│   └── model_config.py
├── src/
│   ├── annual_loss_simulation.py
│   ├── incident_simulation.py
│   ├── control_effectiveness.py
│   └── ... additional loss, calibration, and validation modules
├── notebooks/
│   └── ... model development, validation, and optimization notebooks
├── outputs/
│   ├── 09_portfolio_risk.csv
│   ├── 09_portfolio_financial.csv
│   ├── control_effectiveness_baseline.csv
│   ├── control_cost_sensitivity.csv
│   ├── combined_parameter_uncertainty_draws.csv
│   ├── parameter_uncertainty_summary.csv
│   ├── control_effectiveness_sensitivity.csv
│   ├── control_financial_sensitivity.csv
│   ├── combined_controls_sensitivity.csv
│   └── simulations/
│       ├── dashboard_annual_losses.csv
│       └── dashboard_incident_losses.csv
├── docs/
│   ├── MODEL_METHODOLOGY.md
│   ├── RESULTS_AND_LIMITATIONS.md
│   └── REPRODUCIBILITY.md
├── tests/
├── requirements.txt
└── README.md
```

The dashboard reads precomputed outputs for the baseline charts and investment comparisons; some uncertainty experiments call the simulation modules on demand. For exact input requirements and regeneration notes, see [Reproducibility](docs/REPRODUCIBILITY.md).

## Validation and responsible interpretation

- Numerical results are **simulation outputs**, not empirical loss forecasts.
- Control efficacy and annual costs are **assumptions**, not vendor quotes.
- Customer churn and other event probabilities are **illustrative targets**, not causal estimates.
- The investment objective focuses on **expected annual loss**; it does not optimize VaR/TVaR or account for compliance mandates, operational constraints, or other attack scenarios.
- Monte Carlo seeds help reproducibility but do not establish external validity.
- Some loss-parameter uncertainty estimates use a fitted response surface rather than full simulation for every parameter draw.

## Future improvements

Potential extensions include real-world calibration of campaign rates and incident costs, richer attacker behavior, multi-scenario enterprise risk, control dependencies, uncertainty-aware optimization, and tail-risk constraints.

## Project status

**Modeling and dashboard development complete; documentation prepared.** Final clean-environment testing and GitHub publication are separate closeout tasks.

---

*This repository is an educational demonstration of FAIR-inspired cyber risk quantification. FAIR terminology is used descriptively; this project is not presented as an official FAIR certification or endorsement.*
