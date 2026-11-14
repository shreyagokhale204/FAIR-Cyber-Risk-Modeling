
"""Tests for FAIR secondary response and planning losses."""

import numpy as np
import pytest

from config.model_config import get_config
from src.secondary_loss_response_planning import (
    simulate_secondary_response_planning,
)


def make_inputs(n=10_000, seed=123):
    """Create sample incidents and shared indicators."""
    rng = np.random.default_rng(seed)
    cfg = get_config()

    severity = rng.standard_normal(n)
    affected = rng.integers(300, 3001, size=n)

    indicators = {
        "notification": rng.random(n) < 0.75,
        "regulatory_engagement": rng.random(n) < 0.10,
        "external_pr_agency": rng.random(n) < 0.30,
    }

    return rng, cfg, severity, affected, indicators


def run_model(n=10_000, seed=123):
    rng, cfg, z, affected, indicators = make_inputs(n, seed)

    result = simulate_secondary_response_planning(
        z, affected, indicators, cfg, rng
    )

    return result, z, affected, indicators


# ---------------------------------------------------------
# Original accounting and gate tests
# ---------------------------------------------------------

def test_all_costs_nonnegative():
    result, _, _, _ = run_model()

    for key, values in result.items():
        if key.endswith("_usd"):
            assert np.all(np.asarray(values) >= 0), key


def test_components_add_to_total():
    result, _, _, _ = run_model()

    expected = (
        result["legal_total_usd"]
        + result["notification_total_usd"]
        + result["support_total_usd"]
        + result["pr_total_usd"]
        + result["planning_total_usd"]
    )

    np.testing.assert_allclose(
        result["secondary_response_planning_loss_usd"],
        expected,
    )


def test_notification_gate():
    rng, cfg, z, affected, indicators = make_inputs()

    indicators["notification"] = np.zeros(len(z), dtype=bool)

    result = simulate_secondary_response_planning(
        z, affected, indicators, cfg, rng
    )

    for key in (
        "notification_total_usd",
        "support_total_usd",
        "planning_total_usd",
    ):
        assert np.all(result[key] == 0), key

    assert np.all(
        result["legal_customer_notification_outside_hours"] == 0
    )
    assert np.all(
        result["legal_customer_notification_internal_hours"] == 0
    )


def test_regulatory_legal_gate():
    rng, cfg, z, affected, indicators = make_inputs()

    indicators["regulatory_engagement"] = np.zeros(
        len(z), dtype=bool
    )

    result = simulate_secondary_response_planning(
        z, affected, indicators, cfg, rng
    )

    assert np.all(
        result["legal_regulatory_engagement_outside_hours"] == 0
    )
    assert np.all(
        result["legal_regulatory_engagement_internal_hours"] == 0
    )


def test_external_pr_gate():
    rng, cfg, z, affected, indicators = make_inputs()

    indicators["external_pr_agency"] = np.zeros(
        len(z), dtype=bool
    )

    result = simulate_secondary_response_planning(
        z, affected, indicators, cfg, rng
    )

    assert np.all(result["pr_external_agency_cost_usd"] == 0)


def test_email_count():
    rng, cfg, z, affected, indicators = make_inputs()

    indicators["notification"] = np.ones(len(z), dtype=bool)

    result = simulate_secondary_response_planning(
        z, affected, indicators, cfg, rng
    )

    np.testing.assert_array_equal(
        result["notification_emails_sent"],
        affected,
    )

    np.testing.assert_allclose(
        result["notification_email_cost_usd"],
        affected * 0.16 / 1000,
    )


def test_support_calls_require_notification():
    result, _, _, indicators = run_model()

    no_notification = ~indicators["notification"]

    assert np.all(
        result["support_contacts"][no_notification] == 0
    )

    assert np.all(
        result["support_total_call_minutes"][no_notification] == 0
    )


def test_output_shapes():
    result, _, _, _ = run_model(n=1000)

    for key, values in result.items():
        assert np.shape(values) == (1000,), key


def test_counsel_rate_shared_across_stages():
    result, _, _, _ = run_model()

    rate = result["outside_counsel_hourly_rate_usd"]
    hours = result["legal_total_outside_hours"]

    np.testing.assert_allclose(
        result["legal_outside_counsel_cost_usd"],
        rate * hours,
    )


# ---------------------------------------------------------
# New tests: dependence and validation
# ---------------------------------------------------------

def test_legal_hours_correlated_with_severity():
    """Verify the intended legal-hour copula is actually used."""
    result, z, _, _ = run_model(n=100_000)

    # Notification-stage hours are zero when notification is absent.
    # Restrict the correlation check to active incidents.
    active = result["legal_customer_notification_active"]

    hours = result[
        "legal_customer_notification_outside_hours"
    ][active]

    severity = z[active]

    correlation = np.corrcoef(severity, hours)[0, 1]

    assert correlation > 0.35


def test_support_pr_planning_correlated_with_severity():
    """Verify all three corrected copula mappings."""
    result, z, _, indicators = run_model(n=100_000)

    support_active = indicators["notification"]

    support_corr = np.corrcoef(
        z[support_active],
        result["support_setup_hours"][support_active],
    )[0, 1]

    pr_corr = np.corrcoef(
        z,
        result["pr_internal_hours"],
    )[0, 1]

    planning_corr = np.corrcoef(
        z[support_active],
        result["planning_hours"][support_active],
    )[0, 1]

    assert support_corr > 0.25
    assert pr_corr > 0.25
    assert planning_corr > 0.25


def test_invalid_affected_customer_count():
    rng, cfg, z, affected, indicators = make_inputs(n=100)

    affected[0] = 4000

    with pytest.raises(ValueError):
        simulate_secondary_response_planning(
            z, affected, indicators, cfg, rng
        )
