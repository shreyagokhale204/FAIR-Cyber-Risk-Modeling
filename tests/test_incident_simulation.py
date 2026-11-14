
"""Tests for combined incident-level FAIR loss simulation."""

import numpy as np
import pandas as pd
import pytest

from src.incident_simulation import (
    simulate_incidents,
    summarize_incident_losses,
    LOSS_COLUMNS,
)


@pytest.fixture(scope="module")
def simulation():
    return simulate_incidents(
        n_incidents=20_000,
        seed=123,
        include_details=True,
    )


def test_number_of_incidents(simulation):
    assert len(simulation) == 20_000


def test_all_losses_nonnegative(simulation):
    for column in LOSS_COLUMNS + ["total_incident_loss_usd"]:
        assert np.all(simulation[column] >= 0), column


def test_components_add_to_total(simulation):
    expected = simulation[LOSS_COLUMNS].sum(axis=1)

    np.testing.assert_allclose(
        simulation["total_incident_loss_usd"],
        expected,
    )


def test_no_fine_without_regulator(simulation):
    invalid = (
        simulation["indicator_regulatory_fine"]
        & ~simulation["indicator_regulatory_engagement"]
    )

    assert not invalid.any()


def test_shared_affected_customer_count(simulation):
    np.testing.assert_array_equal(
        simulation["affected_customers"],
        simulation["reputation_affected_customers"],
    )


def test_shared_visibility(simulation):
    np.testing.assert_array_equal(
        simulation["indicator_public_visibility"].astype(int),
        simulation["reputation_public_visibility_indicator"],
    )


def test_notification_gates(simulation):
    no_notification = ~simulation["indicator_notification"]

    assert np.all(
        simulation.loc[
            no_notification,
            "secondary_response_planning_loss_usd",
        ] >= 0
    )

    assert np.all(
        simulation.loc[
            no_notification,
            "response_notification_total_usd",
        ] == 0
    )

    assert np.all(
        simulation.loc[
            no_notification,
            "response_support_total_usd",
        ] == 0
    )

    assert np.all(
        simulation.loc[
            no_notification,
            "response_planning_total_usd",
        ] == 0
    )


def test_reproducibility():
    a = simulate_incidents(
        n_incidents=1000,
        seed=777,
        include_details=False,
    )

    b = simulate_incidents(
        n_incidents=1000,
        seed=777,
        include_details=False,
    )

    pd.testing.assert_frame_equal(a, b)


def test_invalid_incident_count():
    with pytest.raises(ValueError):
        simulate_incidents(n_incidents=0)

    with pytest.raises(ValueError):
        simulate_incidents(n_incidents=-10)


def test_summary_statistics(simulation):
    summary = summarize_incident_losses(simulation)

    assert len(summary) == 5
    assert np.all(summary["mean_usd"] >= 0)


def test_mean_primary_loss_reasonable(simulation):
    mean_primary = simulation["primary_loss_usd"].mean()

    # Theoretical expectation from Step 4A is about $147,217.
    assert mean_primary == pytest.approx(
        147_217,
        rel=0.04,
    )


def test_reputation_customer_counts(simulation):
    assert np.all(
        simulation["reputation_affected_customers"]
        + simulation["reputation_unaffected_customers"]
        == 3000
    )
