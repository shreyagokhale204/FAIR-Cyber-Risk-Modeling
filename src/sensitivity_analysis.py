
"""Frequency sensitivity analysis for NovaFlow FAIR cyber risk."""

import numpy as np
import pandas as pd

from src.annual_loss_simulation import (
    simulate_annual_losses,
    summarize_annual_losses,
)


BASELINE_CAMPAIGN_RATE = 3.0
BASELINE_ALPHA = 0.36
BASELINE_BETA = 4.48

BETA_CONCENTRATION = BASELINE_ALPHA + BASELINE_BETA

BASELINE_SUCCESS_PROBABILITY = (
    BASELINE_ALPHA / BETA_CONCENTRATION
)


def run_frequency_sensitivity(
    n_years=100_000,
    seed=20261025,
    campaign_rates=(1, 2, 3, 4, 5),
    success_probabilities=(0.03, 0.05, 0.10, 0.15),
    exceedance_thresholds=(2e6, 5e6, 10e6, 15e6),
):
    """Run one-at-a-time sensitivity analysis of incident frequency.

    Varies:
      1. Expected attack campaigns per year.
      2. Mean campaign success probability.

    All incident-loss assumptions remain unchanged.

    Returns
    -------
    results : pandas.DataFrame
        Summary metrics for each sensitivity scenario.

    annual_losses_by_scenario : dict
        Keys: (parameter, float(parameter_value))
        Values: NumPy arrays of annual losses, including zero years.
    """
    if (
        not isinstance(n_years, (int, np.integer))
        or isinstance(n_years, (bool, np.bool_))
        or n_years <= 0
    ):
        raise ValueError("n_years must be a positive integer.")

    for rate in campaign_rates:
        if not np.isfinite(rate) or rate < 0:
            raise ValueError(
                "Campaign rates must be finite and nonnegative."
            )

    for probability in success_probabilities:
        if not np.isfinite(probability) or not 0 < probability < 1:
            raise ValueError(
                "Success probabilities must be between 0 and 1."
            )

    for threshold in exceedance_thresholds:
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError(
                "Exceedance thresholds must be finite and nonnegative."
            )

    results = []
    annual_losses_by_scenario = {}

    def run_scenario(parameter, value, rate, alpha, beta):
        annual = simulate_annual_losses(
            n_years=n_years,
            seed=seed,
            campaign_rate=rate,
            success_alpha=alpha,
            success_beta=beta,
        )

        metrics = summarize_annual_losses(annual)

        losses = annual[
            "total_annual_loss_usd"
        ].to_numpy(copy=True)

        scenario_key = (parameter, float(value))
        annual_losses_by_scenario[scenario_key] = losses

        row = {
            "parameter": parameter,
            "parameter_value": float(value),
            "expected_lef": rate * alpha / (alpha + beta),
            "simulated_lef": metrics["mean_incidents_per_year"],
            "probability_any_incident":
                metrics["probability_any_incident"],
            "aal_usd": metrics["aal_usd"],
            "median_annual_loss_usd":
                metrics["median_annual_loss_usd"],
            "var_90_usd": metrics["var_90_usd"],
            "var_95_usd": metrics["var_95_usd"],
            "var_99_usd": metrics["var_99_usd"],
            "tvar_90_usd": metrics["tvar_90_usd"],
            "tvar_95_usd": metrics["tvar_95_usd"],
            "tvar_99_usd": metrics["tvar_99_usd"],
        }

        for threshold in exceedance_thresholds:
            threshold_m = threshold / 1_000_000
            column = f"p_loss_gt_{threshold_m:g}m"
            row[column] = np.mean(losses > threshold)

        results.append(row)

    # A. Campaign frequency scenarios
    for rate in campaign_rates:
        run_scenario(
            parameter="campaign_rate",
            value=rate,
            rate=rate,
            alpha=BASELINE_ALPHA,
            beta=BASELINE_BETA,
        )

    # B. Campaign success probability scenarios
    probabilities = sorted(
        set(success_probabilities)
        | {BASELINE_SUCCESS_PROBABILITY}
    )

    for probability in probabilities:
        alpha = probability * BETA_CONCENTRATION
        beta = (1 - probability) * BETA_CONCENTRATION

        run_scenario(
            parameter="success_probability",
            value=probability,
            rate=BASELINE_CAMPAIGN_RATE,
            alpha=alpha,
            beta=beta,
        )

    results_df = pd.DataFrame(results)

    return results_df, annual_losses_by_scenario
