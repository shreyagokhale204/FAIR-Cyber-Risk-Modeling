
"""
Step 7A — Monte Carlo convergence and numerical stability.

Uses the existing annual FAIR simulation without changing
any underlying risk-model assumptions.
"""

import numpy as np
import pandas as pd

from src.annual_loss_simulation import (
    simulate_annual_losses,
    summarize_annual_losses,
)


DEFAULT_SAMPLE_SIZES = (
    1_000,
    5_000,
    10_000,
    50_000,
    100_000,
    250_000,
)

METRICS = (
    "aal_usd",
    "var_95_usd",
    "var_99_usd",
    "tvar_95_usd",
    "tvar_99_usd",
)


def run_convergence_analysis(
    sample_sizes=DEFAULT_SAMPLE_SIZES,
    seed=20261025,
    config=None,
):
    """
    Run annual loss simulations at increasing sample sizes.

    Returns a DataFrame containing the risk metrics for
    each simulation size.

    The same seed is used at each size for reproducibility.
    These are not guaranteed to be nested samples.
    """

    rows = []

    for n_years in sample_sizes:

        n_years = int(n_years)

        if n_years <= 0:
            raise ValueError(
                "Simulation sizes must be positive."
            )

        annual = simulate_annual_losses(
            n_years=n_years,
            seed=seed,
            config=config,
        )

        summary = summarize_annual_losses(annual)

        rows.append({
            "n_years": n_years,
            **summary,
        })

    return pd.DataFrame(rows)


def run_repeated_simulations(
    n_years=100_000,
    n_repetitions=20,
    base_seed=20261102,
    config=None,
):
    """
    Run independent annual simulations using different seeds.

    Returns:
        repetitions_df:
            One row per independent simulation.

        uncertainty_df:
            Monte Carlo mean, standard deviation,
            standard error, and approximate 95% CI
            for each risk metric.

    Confidence intervals describe Monte Carlo uncertainty
    conditional on the model's fixed assumptions.
    """

    if n_years <= 0:
        raise ValueError("n_years must be positive.")

    if n_repetitions < 2:
        raise ValueError(
            "At least two repetitions are required."
        )

    rows = []

    for repetition in range(n_repetitions):

        seed = base_seed + repetition

        annual = simulate_annual_losses(
            n_years=n_years,
            seed=seed,
            config=config,
        )

        summary = summarize_annual_losses(annual)

        rows.append({
            "repetition": repetition + 1,
            "seed": seed,
            "n_years": n_years,
            **summary,
        })

    repetitions_df = pd.DataFrame(rows)

    uncertainty_rows = []

    for metric in METRICS:

        values = repetitions_df[
            metric
        ].to_numpy(dtype=float)

        mean = float(np.mean(values))

        sd = float(np.std(values, ddof=1))

        se = sd / np.sqrt(n_repetitions)

        # Normal-approximation interval for the mean
        # across independent simulation estimates.
        ci_lower = mean - 1.96 * se
        ci_upper = mean + 1.96 * se

        relative_sd = (
            sd / abs(mean)
            if mean != 0
            else np.nan
        )

        uncertainty_rows.append({
            "metric": metric,
            "mean_estimate": mean,
            "sd_across_runs": sd,
            "se_of_mean": se,
            "relative_sd": relative_sd,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "min_estimate": float(np.min(values)),
            "max_estimate": float(np.max(values)),
        })

    uncertainty_df = pd.DataFrame(
        uncertainty_rows
    )

    return repetitions_df, uncertainty_df
