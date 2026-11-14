
"""Tests for NovaFlow annual FAIR loss simulation."""

import numpy as np
import pandas as pd
import pytest

from src.annual_loss_simulation import (
    simulate_annual_losses,
    summarize_annual_losses,
    empirical_tvar,
    exceedance_probabilities,
)
from src.incident_simulation import LOSS_COLUMNS


@pytest.fixture(scope="module")
def annual():
    return simulate_annual_losses(
        n_years=20_000,
        seed=123,
    )


def test_number_of_years(annual):
    assert len(annual) == 20_000


def test_nonnegative_losses(annual):
    columns = LOSS_COLUMNS + ["total_annual_loss_usd"]

    for column in columns:
        assert np.all(annual[column] >= 0)


def test_annual_accounting(annual):
    expected = annual[LOSS_COLUMNS].sum(axis=1)

    np.testing.assert_allclose(
        annual["total_annual_loss_usd"],
        expected,
    )


def test_zero_incident_years_have_zero_loss(annual):
    zero_years = annual[
        "successful_incident_count"
    ] == 0

    assert np.all(
        annual.loc[
            zero_years, "total_annual_loss_usd"
        ] == 0
    )


def test_incidents_never_exceed_campaigns(annual):
    assert np.all(
        annual["successful_incident_count"]
        <= annual["campaign_count"]
    )


def test_frequency_calibration(annual):
    expected_rate = 3 * 0.36 / (0.36 + 4.48)

    observed = annual[
        "successful_incident_count"
    ].mean()

    assert observed == pytest.approx(
        expected_rate,
        abs=0.015,
    )


def test_probability_any_incident(annual):
    rate = 3 * 0.36 / (0.36 + 4.48)
    theoretical = 1 - np.exp(-rate)

    observed = np.mean(
        annual["successful_incident_count"] > 0
    )

    assert observed == pytest.approx(
        theoretical,
        abs=0.015,
    )


def test_reproducibility():
    a = simulate_annual_losses(1000, seed=456)
    b = simulate_annual_losses(1000, seed=456)

    pd.testing.assert_frame_equal(a, b)


def test_summary_quantile_order(annual):
    summary = summarize_annual_losses(annual)

    assert (
        summary["var_90_usd"]
        <= summary["var_95_usd"]
        <= summary["var_99_usd"]
    )

    assert summary["tvar_99_usd"] >= summary["var_99_usd"]


def test_empirical_tvar():
    values = np.arange(1, 101, dtype=float)

    # Worst 10% = 91, 92, ..., 100.
    assert empirical_tvar(values, 0.90) == pytest.approx(
        95.5
    )


def test_exceedance_probabilities(annual):
    result = exceedance_probabilities(annual)

    probabilities = result[
        "probability_exceeding"
    ].to_numpy()

    assert np.all(np.diff(probabilities) <= 0)


def test_zero_campaign_rate():
    annual = simulate_annual_losses(
        n_years=100,
        seed=123,
        campaign_rate=0,
    )

    assert annual["successful_incident_count"].sum() == 0
    assert annual["total_annual_loss_usd"].sum() == 0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        simulate_annual_losses(n_years=0)

    with pytest.raises(ValueError):
        simulate_annual_losses(success_alpha=-1)
