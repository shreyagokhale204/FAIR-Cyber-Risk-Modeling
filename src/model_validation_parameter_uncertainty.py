
"""
Step 7D-1 — Frequency parameter uncertainty.

Distinguishes uncertainty in the model parameters
from randomness in annual incident outcomes.

This module uses the analytical AAL identity:
    AAL = campaign_rate * mean_success_probability
          * expected_loss_per_incident
"""

import numpy as np
import pandas as pd


DEFAULT_CAMPAIGN_SHAPE = 9.0
DEFAULT_CAMPAIGN_SCALE = 1.0 / 3.0

DEFAULT_SUCCESS_ALPHA = 3.6
DEFAULT_SUCCESS_BETA = 44.8

BASELINE_CAMPAIGN_RATE = 3.0
BASELINE_SUCCESS_PROBABILITY = 0.36 / (0.36 + 4.48)


def run_frequency_parameter_uncertainty(
    expected_incident_loss_usd,
    n_draws=10_000,
    seed=20261107,
    campaign_shape=DEFAULT_CAMPAIGN_SHAPE,
    campaign_scale=DEFAULT_CAMPAIGN_SCALE,
    success_alpha=DEFAULT_SUCCESS_ALPHA,
    success_beta=DEFAULT_SUCCESS_BETA,
):
    """
    Draw uncertain campaign rates and uncertain
    mean success probabilities.

    Returns one row per parameter draw.

    These are epistemic parameter draws, NOT
    simulated annual cyberattack outcomes.
    """

    if expected_incident_loss_usd <= 0:
        raise ValueError(
            "Expected incident loss must be positive."
        )

    if n_draws <= 0:
        raise ValueError("n_draws must be positive.")

    parameters = (
        campaign_shape,
        campaign_scale,
        success_alpha,
        success_beta,
    )

    if any(x <= 0 or not np.isfinite(x) for x in parameters):
        raise ValueError(
            "Distribution parameters must be finite and positive."
        )

    rng = np.random.default_rng(seed)

    campaign_rates = rng.gamma(
        shape=campaign_shape,
        scale=campaign_scale,
        size=n_draws,
    )

    mean_success_probabilities = rng.beta(
        a=success_alpha,
        b=success_beta,
        size=n_draws,
    )

    expected_lef = (
        campaign_rates * mean_success_probabilities
    )

    aal = (
        expected_lef * expected_incident_loss_usd
    )

    return pd.DataFrame({
        "draw": np.arange(1, n_draws + 1),
        "campaign_rate": campaign_rates,
        "mean_success_probability":
            mean_success_probabilities,
        "expected_annual_lef": expected_lef,
        "expected_annual_loss_usd": aal,
    })


def summarize_parameter_uncertainty(
    draws,
    expected_incident_loss_usd,
):
    """
    Summarize the distribution of plausible AAL
    estimates caused by frequency parameter uncertainty.
    """

    aal = draws[
        "expected_annual_loss_usd"
    ].to_numpy(dtype=float)

    baseline_aal = (
        BASELINE_CAMPAIGN_RATE
        * BASELINE_SUCCESS_PROBABILITY
        * expected_incident_loss_usd
    )

    return {
        "baseline_aal_usd": baseline_aal,
        "mean_aal_usd": float(np.mean(aal)),
        "median_aal_usd": float(np.median(aal)),
        "aal_p05_usd": float(np.quantile(aal, 0.05)),
        "aal_p95_usd": float(np.quantile(aal, 0.95)),
        "aal_p025_usd": float(np.quantile(aal, 0.025)),
        "aal_p975_usd": float(np.quantile(aal, 0.975)),
        "aal_sd_usd": float(np.std(aal, ddof=1)),
    }
