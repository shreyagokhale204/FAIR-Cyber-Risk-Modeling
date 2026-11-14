# Reproducibility, testing, and repository preparation

## Run the dashboard

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app/dashboard.py
```

The app expects `config/`, `src/`, and `outputs/` at the project root. Some charts load **precomputed CSVs**; live uncertainty experiments also import the source simulation modules.

## Required saved dashboard outputs

At minimum, the dashboard expects these files under `outputs/`:

- `09_portfolio_risk.csv` (32 control combinations)
- `09_portfolio_financial.csv`
- `control_effectiveness_baseline.csv`
- `control_cost_sensitivity.csv`
- `simulations/dashboard_annual_losses.csv`
- `simulations/dashboard_incident_losses.csv`

For the full uncertainty and control-sensitivity views, also include:

- `combined_parameter_uncertainty_draws.csv`
- `parameter_uncertainty_summary.csv`
- `control_effectiveness_sensitivity.csv`
- `control_financial_sensitivity.csv`
- `combined_controls_sensitivity.csv`

Do not assume that empty `.gitkeep` folders provide the required simulation data. The dashboard risk-data notebook can export the annual and incident simulation tables when needed.

## Suggested notebook order

The development notebooks cover model configuration and incident losses; annual loss validation/convergence; sensitivity and dependence checks; frequency uncertainty; magnitude and combined uncertainty; security control effectiveness; investment optimization; and dashboard-data exports. Follow the notebook sequence in the repository and inspect each notebook's imports and saved outputs before rerunning.

Known notebook entry points include:

- `07_model_validation_convergence.ipynb`
- `08_model_validation_distributions.ipynb`
- `09_model_validation_dependence.ipynb`
- `10_model_validation_parameter_uncertainty.ipynb`
- `11_model_validation_loss_parameter_uncertainty.ipynb`
- `12_security_control_effectiveness.ipynb`
- `13_security_investment_optimization.ipynb`
- `14_dashboard_risk_data.ipynb`

These are **development filenames**, not a guarantee that every notebook is self-contained or runnable out of order.

## Validation checklist

Before GitHub publication:

1. Start Streamlit in a fresh environment and visit all five pages.
2. Check the risk table, loss exceedance curve, loss histogram, and loss composition against saved simulation outputs.
3. Test both investment objectives, all cost scenarios, and several budgets.
4. Confirm the control reference table is present under **Security Controls**.
5. Test the Monte Carlo experiment with multiple seeds and sample sizes; check that the model runs only when requested.
6. In **Change model assumptions**, reset inputs and run the baseline using the same seed and sample size; differences should be zero or attributable to an identified implementation issue.
7. Change one frequency parameter, one magnitude parameter, and both together; confirm sensible directionality and explain that effects are simulated, not guaranteed monotonic in small samples.
8. Inspect all three parameter-uncertainty series and verify that the reported mean, median, standard deviation, and interval width use the selected series.
9. Run available unit tests (for example, `python -m pytest tests` if the repository uses pytest) and record results.
10. Ensure all precomputed outputs intended for publication are present and do not contain private data.

## Dependency management

A starter `requirements.txt` is supplied with package names only. **Pin versions after validating a clean installation** in the actual repository, ideally from the tested virtual environment. Do not commit `.venv/`, caches, secrets, temporary large outputs, or irrelevant exploratory copies. Retain the CSVs that the dashboard actually needs or supply a documented regeneration path.

## Suggested publication structure

```text
app/              # Streamlit dashboard
config/           # Model assumptions
src/              # Simulators, controls, calibration, validation
notebooks/        # Research and development analysis
outputs/          # Selected reproducible results used by dashboard
docs/             # Technical and interpretation documentation
tests/            # Automated tests
README.md
requirements.txt
.gitignore
```

Screenshots are optional, but a small set from the Executive Overview, loss exceedance curve, investment optimizer, and uncertainty explorer would make the repository easier to evaluate.

## Reproducibility boundaries

Seeds and saved outputs make the exploratory simulation *computationally reproducible under matching code and dependencies*. They do not make the assumptions empirically valid. Live parameter changes do not recompute the precomputed investment comparison tables; those must be regenerated separately if model assumptions are permanently changed.
