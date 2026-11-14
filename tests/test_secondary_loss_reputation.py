
"""Tests for FAIR secondary reputation loss."""

import numpy as np
import pytest

from config.model_config import get_config
from src.calibration import (
    calibrate_probabilities,
    draw_incident_indicators,
)
from src.severity import copula_transform
from src.secondary_loss_reputation import (
    simulate_secondary_reputation_loss,
)


def make_inputs(n=10_000, seed=123):
    rng = np.random.default_rng(seed)
    cfg = calibrate_probabilities(get_config())

    severity = rng.standard_normal(n)

    affected_spec = cfg["reputation"]["affected_customers"]
    rho = cfg["dependence"]["correlations_latent_normal"][
        "affected_customers"
    ]

    affected = copula_transform(
        severity, rho, affected_spec, rng
    )

    affected = np.clip(
        np.rint(affected),
        0,
        cfg["organization"]["enterprise_customers"],
    ).astype(int)

    indicators = draw_incident_indicators(
        severity, cfg, rng
    )

    return rng, cfg, severity, affected, indicators


def run_model(n=10_000, seed=123):
    rng, cfg, z, affected, indicators = make_inputs(n, seed)

    result = simulate_secondary_reputation_loss(
        z, affected, indicators, cfg, rng
    )

    return result, z, affected, indicators


def test_all_losses_nonnegative():
    result, _, _, _ = run_model()

    for key, values in result.items():
        if key.endswith("_usd"):
            assert np.all(np.asarray(values) >= 0), key


def test_customer_counts_add_to_total():
    result, _, _, _ = run_model()

    assert np.all(
        result["affected_customers"]
        + result["unaffected_customers"] == 3000
    )


def test_churn_counts_within_population():
    result, _, _, _ = run_model()

    assert np.all(
        result["affected_churn_count"]
        <= result["affected_customers"]
    )

    assert np.all(
        result["unaffected_churn_count"]
        <= result["unaffected_customers"]
    )


def test_lost_deals_do_not_exceed_opportunities():
    result, _, _, _ = run_model()

    assert np.all(
        result["lost_deals_count"]
        <= result["otherwise_won_opportunities"]
    )


def test_components_add_to_total():
    result, _, _, _ = run_model()

    expected = (
        result["affected_churn_loss_usd"]
        + result["unaffected_churn_loss_usd"]
        + result["lost_deals_loss_usd"]
    )

    np.testing.assert_allclose(
        result["secondary_reputation_loss_usd"],
        expected,
    )


def test_output_shapes():
    result, _, _, _ = run_model(n=1000)

    for key, values in result.items():
        assert np.shape(values) == (1000,), key


def test_visibility_increases_reputation_risk():
    rng, cfg, z, affected, indicators = make_inputs(
        n=50_000
    )

    invisible = {
        **indicators,
        "public_visibility": np.zeros(
            len(z), dtype=bool
        ),
    }

    visible = {
        **indicators,
        "public_visibility": np.ones(
            len(z), dtype=bool
        ),
    }

    low = simulate_secondary_reputation_loss(
        z, affected, invisible, cfg, rng
    )

    high = simulate_secondary_reputation_loss(
        z, affected, visible, cfg, rng
    )

    for key in (
        "affected_churn_probability",
        "unaffected_churn_probability",
        "lost_deal_probability",
    ):
        assert np.all(high[key] > low[key])


def test_customer_weighted_churn_targets():
    result, _, _, _ = run_model(n=200_000)

    affected_rate = (
        result["affected_churn_count"].sum()
        / result["affected_customers"].sum()
    )

    unaffected_rate = (
        result["unaffected_churn_count"].sum()
        / result["unaffected_customers"].sum()
    )

    assert affected_rate == pytest.approx(
        0.002, abs=0.00015
    )

    assert unaffected_rate == pytest.approx(
        0.0002, abs=0.00005
    )


def test_invalid_affected_count():
    rng, cfg, z, affected, indicators = make_inputs(
        n=100
    )

    affected[0] = 4000

    with pytest.raises(ValueError):
        simulate_secondary_reputation_loss(
            z, affected, indicators, cfg, rng
        )
