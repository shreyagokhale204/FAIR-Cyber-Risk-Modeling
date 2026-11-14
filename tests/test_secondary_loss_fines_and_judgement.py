
"""Tests for FAIR secondary fines and judgments."""

import numpy as np
import pytest

from config.model_config import get_config
from src.calibration import (
    calibrate_probabilities,
    draw_incident_indicators,
)
from src.secondary_loss_fines_and_judgement import (
    simulate_secondary_fines_and_judgement,
)


def make_inputs(n=10_000, seed=123):
    rng = np.random.default_rng(seed)

    config = calibrate_probabilities(get_config())
    severity = rng.standard_normal(n)

    indicators = draw_incident_indicators(
        severity, config, rng
    )

    return rng, config, severity, indicators


def test_all_costs_nonnegative():
    rng, cfg, z, indicators = make_inputs()

    result = simulate_secondary_fines_and_judgement(
        z, indicators, cfg, rng
    )

    assert np.all(result["regulatory_fine_loss_usd"] >= 0)
    assert np.all(
        result["judgment_or_settlement_loss_usd"] >= 0
    )
    assert np.all(
        result["secondary_fines_judgement_loss_usd"] >= 0
    )


def test_components_add_to_total():
    rng, cfg, z, indicators = make_inputs()

    result = simulate_secondary_fines_and_judgement(
        z, indicators, cfg, rng
    )

    expected = (
        result["regulatory_fine_loss_usd"]
        + result["judgment_or_settlement_loss_usd"]
    )

    np.testing.assert_allclose(
        result["secondary_fines_judgement_loss_usd"],
        expected,
    )


def test_no_fine_without_regulatory_engagement():
    rng, cfg, z, indicators = make_inputs()

    result = simulate_secondary_fines_and_judgement(
        z, indicators, cfg, rng
    )

    no_regulator = ~indicators["regulatory_engagement"]

    assert np.all(
        result["regulatory_fine_loss_usd"][no_regulator] == 0
    )


def test_no_occurrence_means_zero_loss():
    rng, cfg, z, indicators = make_inputs()

    indicators["regulatory_fine"] = np.zeros(
        len(z), dtype=bool
    )
    indicators["judgment_or_settlement"] = np.zeros(
        len(z), dtype=bool
    )

    result = simulate_secondary_fines_and_judgement(
        z, indicators, cfg, rng
    )

    assert np.all(
        result["secondary_fines_judgement_loss_usd"] == 0
    )


def test_invalid_fine_without_regulator():
    rng, cfg, z, indicators = make_inputs(n=100)

    indicators["regulatory_engagement"] = np.zeros(
        len(z), dtype=bool
    )
    indicators["regulatory_fine"] = np.ones(
        len(z), dtype=bool
    )

    with pytest.raises(ValueError):
        simulate_secondary_fines_and_judgement(
            z, indicators, cfg, rng
        )


def test_output_shapes():
    rng, cfg, z, indicators = make_inputs(n=1000)

    result = simulate_secondary_fines_and_judgement(
        z, indicators, cfg, rng
    )

    for key, values in result.items():
        assert np.shape(values) == (1000,), key


def test_fine_and_judgment_medians():
    rng, cfg, z, indicators = make_inputs(n=100_000)

    result = simulate_secondary_fines_and_judgement(
        z, indicators, cfg, rng
    )

    fine_values = result["regulatory_fine_loss_usd"]
    fine_values = fine_values[fine_values > 0]

    judgment_values = result["judgment_or_settlement_loss_usd"]
    judgment_values = judgment_values[judgment_values > 0]

    assert np.median(fine_values) == pytest.approx(
        1_000_000, rel=0.06
    )

    assert np.median(judgment_values) == pytest.approx(
        500_000, rel=0.06
    )
