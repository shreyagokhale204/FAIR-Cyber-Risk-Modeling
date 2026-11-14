
# src/sensitivity_analysis_loss_magnitude.py

from copy import deepcopy

import numpy as np
import pandas as pd

from config.model_config import get_config
from src.annual_loss_simulation import (
    simulate_annual_losses,
    summarize_annual_losses,
)


BASELINE_CHURN = 0.002
BASELINE_MARGIN = 210_000

DEFAULT_CHURN_VALUES = (0.0005, 0.001, 0.002, 0.003, 0.005)
DEFAULT_MARGIN_VALUES = (100_000, 150_000, 210_000, 250_000, 300_000)
DEFAULT_EXPOSURE_MULTIPLIERS = (0.50, 0.75, 1.00, 1.25, 1.50)

DEFAULT_THRESHOLDS = (2_000_000, 5_000_000, 10_000_000, 15_000_000)

LOSS_COMPONENTS = (
    "primary_loss_usd",
    "secondary_response_planning_loss_usd",
    "secondary_fines_judgement_loss_usd",
    "secondary_reputation_loss_usd",
)


def make_magnitude_config(parameter, value, base_config=None):
    """
    Return an independent config with one magnitude assumption changed.
    Does not modify the original configuration.
    """
    config = deepcopy(get_config() if base_config is None else base_config)
    value = float(value)

    if not np.isfinite(value):
        raise ValueError("Scenario value must be finite.")

    if parameter == "affected_churn":
        if not 0 < value < 1:
            raise ValueError("Affected churn target must be between 0 and 1.")

        config["probabilities"]["affected_churn"]["target"] = value

    elif parameter == "contribution_margin":
        if value <= 0:
            raise ValueError("Contribution margin must be positive.")

        config["reputation"]["loss_per_churned_customer_usd"] = value
        config["reputation"]["loss_per_lost_deal_usd"] = value

    elif parameter == "customer_exposure":
        if value <= 0:
            raise ValueError("Exposure multiplier must be positive.")

        spec = config["reputation"]["affected_customers"]
        total_customers = int(config["organization"]["enterprise_customers"])

        # Scale the original PERT assumptions and enforce the customer cap.
        for key in ("min", "mode", "max"):
            spec[key] = min(float(spec[key]) * value, total_customers)

        if not (spec["min"] < spec["mode"] < spec["max"]):
            raise ValueError(
                "Exposure multiplier creates an invalid PERT distribution."
            )

    else:
        raise ValueError(f"Unknown sensitivity parameter: {parameter}")

    # Force fresh logistic calibration when scenario assumptions change.
    for probability_spec in config["probabilities"].values():
        if probability_spec.get("family") == "calibrated_logistic":
            probability_spec["intercept"] = None

    return config


def run_magnitude_sensitivity(
    n_years=100_000,
    seed=20261025,
    churn_values=DEFAULT_CHURN_VALUES,
    margin_values=DEFAULT_MARGIN_VALUES,
    exposure_multipliers=DEFAULT_EXPOSURE_MULTIPLIERS,
    exceedance_thresholds=DEFAULT_THRESHOLDS,
    base_config=None,
):
    """
    Returns
    -------
    results : pandas.DataFrame
        One row per scenario, with annual risk statistics.
    annual_losses_by_scenario : dict
        Keys: (parameter, value)
        Values: full annual simulation DataFrames.
    """
    if not isinstance(n_years, (int, np.integer)) or n_years <= 0:
        raise ValueError("n_years must be a positive integer.")

    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")

    config = get_config() if base_config is None else deepcopy(base_config)

    scenarios = {
        "affected_churn": churn_values,
        "contribution_margin": margin_values,
        "customer_exposure": exposure_multipliers,
    }

    rows = []
    annual_losses_by_scenario = {}

    for parameter, values in scenarios.items():
        for raw_value in values:
            value = float(raw_value)
            scenario_config = make_magnitude_config(
                parameter, value, config
            )

            annual = simulate_annual_losses(
                n_years=n_years,
                seed=seed,
                campaign_rate=3.0,
                success_alpha=0.36,
                success_beta=4.48,
                config=scenario_config,
            )

            summary = summarize_annual_losses(annual)

            row = {
                "parameter": parameter,
                "parameter_value": value,
                **summary,
            }

            total_losses = annual["total_annual_loss_usd"].to_numpy()

            for threshold in exceedance_thresholds:
                millions = threshold / 1_000_000
                row[f"p_loss_gt_{millions:g}m"] = float(
                    np.mean(total_losses > threshold)
                )

            for component in LOSS_COMPONENTS:
                annual_column = component.replace(
                    "_loss_usd", "_annual_loss_usd"
                )

                if annual_column in annual.columns:
                    row[f"aal_{component}"] = float(
                        annual[annual_column].mean()
                    )

            rows.append(row)
            annual_losses_by_scenario[(parameter, value)] = annual

    results = pd.DataFrame(rows)

    return results, annual_losses_by_scenario
