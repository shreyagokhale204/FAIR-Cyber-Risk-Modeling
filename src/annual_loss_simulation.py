
"""Annual FAIR cyber loss simulation for NovaFlow."""

import numpy as np
import pandas as pd

from src.incident_simulation import (
    simulate_incidents,
    LOSS_COLUMNS,
)


DEFAULT_CAMPAIGN_RATE = 3.0
DEFAULT_SUCCESS_ALPHA = 0.36
DEFAULT_SUCCESS_BETA = 4.48


def simulate_annual_losses(
    n_years=100_000,
    seed=20261025,
    campaign_rate=DEFAULT_CAMPAIGN_RATE,
    success_alpha=DEFAULT_SUCCESS_ALPHA,
    success_beta=DEFAULT_SUCCESS_BETA,
    config=None,
):
    """Simulate annual losses, including years with zero incidents.

    Each campaign receives an independent Beta-distributed
    probability of success.

    All successful incidents are simulated using the existing
    incident-level FAIR loss model.
    """
    if (
        not isinstance(n_years, (int, np.integer))
        or isinstance(n_years, (bool, np.bool_))
        or n_years <= 0
    ):
        raise ValueError("n_years must be a positive integer.")

    if not np.isfinite(campaign_rate) or campaign_rate < 0:
        raise ValueError("campaign_rate must be finite and nonnegative.")

    if (
        not np.isfinite(success_alpha)
        or not np.isfinite(success_beta)
        or success_alpha <= 0
        or success_beta <= 0
    ):
        raise ValueError("Beta parameters must be finite and positive.")

    rng = np.random.default_rng(seed)

    # A. Number of campaigns in each simulated year.
    campaigns = rng.poisson(campaign_rate, size=n_years)

    # B. Generate independent campaign-level success probabilities.
    total_campaigns = int(campaigns.sum())

    success_probabilities = rng.beta(
        success_alpha,
        success_beta,
        size=total_campaigns,
    )

    successes = (
        rng.random(total_campaigns) < success_probabilities
    )

    # C. Assign each successful campaign to its year.
    campaign_years = np.repeat(
        np.arange(n_years),
        campaigns,
    )

    incident_years = campaign_years[successes]

    incident_counts = np.bincount(
        incident_years,
        minlength=n_years,
    )

    n_incidents = len(incident_years)

    # D. Initialize annual results, including zero-loss years.
    annual = pd.DataFrame({
        "year_id": np.arange(1, n_years + 1),
        "campaign_count": campaigns,
        "successful_incident_count": incident_counts,
    })

    for column in LOSS_COLUMNS:
        annual[column] = np.zeros(n_years, dtype=float)

    annual["total_annual_loss_usd"] = np.zeros(
        n_years, dtype=float
    )

    annual["maximum_single_event_loss_usd"] = np.zeros(
        n_years, dtype=float
    )   

    # E. Simulate losses for successful incidents
    if n_incidents > 0:
        incident_seed = int(
            rng.integers(0, np.iinfo(np.int32).max)
        )

        incidents = simulate_incidents(
            n_incidents=n_incidents,
            seed=incident_seed,
            config=config,
            include_details=False,
        )

        # F. Aggregate incident losses into their respective years
        for column in LOSS_COLUMNS:
            annual[column] = np.bincount(
                incident_years,
                weights=incidents[column].to_numpy(),
                minlength=n_years,
            )

        # G. Calculate total annual loss
        annual["total_annual_loss_usd"] = np.bincount(
            incident_years,
            weights=incidents[
                "total_incident_loss_usd"
            ].to_numpy(),
            minlength=n_years,
        )

        # H. Calculate maximum single-event loss per year
        incident_loss_df = pd.DataFrame({
            "year_index": incident_years,
            "incident_loss_usd": incidents[
                "total_incident_loss_usd"
            ].to_numpy(),
        })

        maximum_by_year = (
            incident_loss_df.groupby("year_index")[
                "incident_loss_usd"
            ].max()
        )

        annual.loc[
            maximum_by_year.index,
            "maximum_single_event_loss_usd",
        ] = maximum_by_year.to_numpy()

    return annual


def empirical_tvar(values, confidence):
    """Expected shortfall in the worst (1-confidence) fraction.

    Uses fractional weighting at the quantile boundary so that
    exactly the upper tail fraction is averaged.
    """
    x = np.sort(np.asarray(values, dtype=float))

    if x.ndim != 1 or len(x) == 0:
        raise ValueError("values must be a nonempty 1D array.")

    if not np.all(np.isfinite(x)):
        raise ValueError("values must be finite.")

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")

    tail_size = len(x) * (1 - confidence)
    whole = int(np.floor(tail_size))
    fraction = tail_size - whole

    tail_sum = (
        x[-whole:].sum() if whole > 0 else 0.0
    )

    if fraction > 0:
        tail_sum += fraction * x[-whole - 1]

    return tail_sum / tail_size


def summarize_annual_losses(annual):
    """Return annual loss risk metrics."""
    losses = annual["total_annual_loss_usd"].to_numpy()

    summary = {
        "simulated_years": len(annual),
        "mean_campaigns_per_year":
            annual["campaign_count"].mean(),
        "mean_incidents_per_year":
            annual["successful_incident_count"].mean(),
        "probability_any_incident":
            np.mean(annual["successful_incident_count"] > 0),
        "aal_usd": losses.mean(),
        "median_annual_loss_usd": np.median(losses),
        "var_90_usd": np.quantile(losses, 0.90),
        "var_95_usd": np.quantile(losses, 0.95),
        "var_99_usd": np.quantile(losses, 0.99),
        "tvar_90_usd": empirical_tvar(losses, 0.90),
        "tvar_95_usd": empirical_tvar(losses, 0.95),
        "tvar_99_usd": empirical_tvar(losses, 0.99),
        "maximum_annual_loss_usd": losses.max(),
    }

    return pd.Series(summary, name="value")


def exceedance_probabilities(
    annual,
    thresholds=(100_000, 500_000, 1_000_000,
                2_000_000, 5_000_000, 10_000_000),
):
    """Probability annual loss strictly exceeds each threshold."""
    losses = annual["total_annual_loss_usd"].to_numpy()

    return pd.DataFrame({
        "threshold_usd": thresholds,
        "probability_exceeding": [
            np.mean(losses > threshold)
            for threshold in thresholds
        ],
    })


"""
A note on computational efficiency
Rather than calling the incident simulator separately for each year, we simulate all successful incidents in one vectorized batch and aggregate their losses into the appropriate years.

With 100,000 years, we expect approximately 22,314 successful incidents. This is manageable for the existing simulator.

We'll use the same seed, 20261025, for reproducibility. The seed does not affect the theoretical assumptions.
"""