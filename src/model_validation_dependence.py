
"""
Step 7C — Dependence stress testing.

Changes dependence assumptions while preserving:
- marginal probability targets
- underlying loss distributions
- annual campaign and success-rate assumptions

The existing simulation and calibration functions
perform the actual Monte Carlo calculations.
"""

from copy import deepcopy

import numpy as np
import pandas as pd

from config.model_config import get_config
from src.annual_loss_simulation import (
    simulate_annual_losses,
    summarize_annual_losses,
)


DEPENDENCE_SCENARIOS = {
    "Weaker": 0.50,
    "Baseline": 1.00,
    "Stronger": 1.25,
}

DEFAULT_THRESHOLDS = (
    2_000_000,
    5_000_000,
    10_000_000,
    15_000_000,
)


def make_dependence_config(
    multiplier,
    base_config=None,
):
    """
    Construct a dependence scenario without modifying
    the original configuration.

    Continuous copula correlations are multiplied
    by multiplier.

    Severity and visibility log-odds coefficients
    are multiplied by multiplier.

    All calibrated intercepts are reset to None
    so that the existing calibration routine can
    recompute them for the new dependence scenario.
    """

    if not np.isfinite(multiplier) or multiplier < 0:
        raise ValueError(
            "Multiplier must be finite and nonnegative."
        )

    if base_config is None:
        base_config = get_config()

    config = deepcopy(base_config)

    correlations = config[
        "dependence"
    ]["correlations_latent_normal"]

    for name, baseline_rho in correlations.items():

        new_rho = float(baseline_rho) * multiplier

        # The Gaussian-copula implementation requires
        # absolute correlation strictly below 1.
        correlations[name] = float(
            np.clip(new_rho, -0.999, 0.999)
        )

    for name, probability_spec in config[
        "probabilities"
    ].items():

        for key in ("severity_or", "visibility_or"):

            if key not in probability_spec:
                continue

            baseline_or = float(
                probability_spec[key]
            )

            if baseline_or <= 0:
                raise ValueError(
                    f"{name}.{key} must be positive."
                )

            # Scaling log(OR) by m is equivalent
            # to raising OR to the power m.
            probability_spec[key] = float(
                baseline_or ** multiplier
            )

        if "intercept" in probability_spec:
            probability_spec["intercept"] = None

    return config


def run_dependence_stress_test(
    n_years=100_000,
    seed=20261106,
    scenarios=None,
    thresholds=DEFAULT_THRESHOLDS,
    base_config=None,
):
    """
    Run annual simulations under different
    dependence assumptions.

    Returns
    -------
    results_df:
        Summary risk metrics for each scenario.

    annual_by_scenario:
        Dictionary mapping scenario names to
        their full annual simulation DataFrames.

    Notes
    -----
    The same seed is used across scenarios to
    improve reproducibility. However, this does
    not guarantee matched underlying random draws
    because changing model parameters may alter
    the simulator's random-number consumption.
    """

    if n_years <= 0:
        raise ValueError(
            "n_years must be positive."
        )

    if scenarios is None:
        scenarios = DEPENDENCE_SCENARIOS

    if base_config is None:
        base_config = get_config()

    results = []
    annual_by_scenario = {}

    for scenario_name, multiplier in scenarios.items():

        scenario_config = make_dependence_config(
            multiplier=multiplier,
            base_config=base_config,
        )

        annual = simulate_annual_losses(
            n_years=n_years,
            seed=seed,
            config=scenario_config,
        )

        summary = summarize_annual_losses(annual)

        annual_by_scenario[scenario_name] = annual

        losses = annual[
            "total_annual_loss_usd"
        ].to_numpy(dtype=float)

        row = {
            "scenario": scenario_name,
            "dependence_multiplier": multiplier,
            "n_years": n_years,
            **summary,
        }

        for threshold in thresholds:
            row[
                f"prob_loss_gt_{int(threshold)}"
            ] = float(
                np.mean(losses > threshold)
            )

        results.append(row)

    results_df = pd.DataFrame(results)

    baseline_rows = results_df.loc[
        results_df["scenario"] == "Baseline"
    ]

    if len(baseline_rows) == 1:

        baseline_aal = baseline_rows.iloc[0][
            "aal_usd"
        ]

        results_df["aal_change_vs_baseline_usd"] = (
            results_df["aal_usd"]
            - baseline_aal
        )

        results_df["aal_change_vs_baseline_pct"] = (
            results_df["aal_usd"]
            / baseline_aal - 1
        ) * 100 if baseline_aal > 0 else np.nan

    return results_df, annual_by_scenario
