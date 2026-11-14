
"""Tests for primary loss calculations."""

import numpy as np
import pytest

from config.model_config import get_config
from src.primary_loss import simulate_primary_loss


def test_primary_loss_nonnegative():
    rng = np.random.default_rng(123)
    cfg = get_config()
    severity = rng.standard_normal(10_000)

    result = simulate_primary_loss(severity, cfg, rng)

    assert np.all(result["primary_loss_usd"] >= 0)


def test_primary_loss_components_add_up():
    rng = np.random.default_rng(123)
    cfg = get_config()
    severity = rng.standard_normal(10_000)

    result = simulate_primary_loss(severity, cfg, rng)

    expected = (
        result["technical_response_cost_usd"]
        + result["management_response_cost_usd"]
        + result["executive_response_cost_usd"]
        + result["external_dfir_cost_usd"]
    )

    np.testing.assert_allclose(
        result["primary_loss_usd"], expected
    )


def test_primary_loss_shapes():
    rng = np.random.default_rng(123)
    cfg = get_config()
    severity = rng.standard_normal(1000)

    result = simulate_primary_loss(severity, cfg, rng)

    for values in result.values():
        assert np.shape(values) == (1000,)


def test_primary_loss_increases_with_severity():
    rng = np.random.default_rng(123)
    cfg = get_config()

    low = simulate_primary_loss(
        np.full(50_000, -2.0), cfg, rng
    )
    high = simulate_primary_loss(
        np.full(50_000, 2.0), cfg, rng
    )

    assert (
        np.mean(high["primary_loss_usd"])
        > np.mean(low["primary_loss_usd"])
    )


def test_primary_loss_mean_reasonable():
    rng = np.random.default_rng(123)
    cfg = get_config()
    severity = rng.standard_normal(100_000)

    result = simulate_primary_loss(severity, cfg, rng)

    # Theoretical marginal expectations:
    # Technical hours: 350 * 103
    # Management hours: 70 * 150
    # Executive hours: 17.5 * 200
    # DFIR: 97,166.67
    expected = (
        350 * 103
        + 70 * 150
        + 17.5 * 200
        + (13_000 + 4 * 80_000 + 250_000) / 6
    )

    assert np.mean(result["primary_loss_usd"]) == pytest.approx(
        expected, rel=0.02
    )
