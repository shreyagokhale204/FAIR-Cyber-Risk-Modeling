
"""Tests for logistic calibration and conditional fine occurrence."""

import numpy as np
import pytest

from config.model_config import get_config
from src.calibration import (
    calibrate_probabilities,
    draw_incident_indicators,
    reputation_probabilities,
)


@pytest.fixture(scope="module")
def calibrated():
    return calibrate_probabilities(get_config())


def test_all_intercepts_calibrated(calibrated):
    for model in calibrated["probabilities"].values():
        assert np.isfinite(model["intercept"])


def test_base_configuration_not_mutated():
    cfg = get_config()
    calibrate_probabilities(cfg)

    assert all(
        model["intercept"] is None
        for model in cfg["probabilities"].values()
    )


def test_indicator_marginal_probabilities(calibrated):
    rng = np.random.default_rng(123)
    z = rng.standard_normal(500_000)

    indicators = draw_incident_indicators(z, calibrated, rng)

    for name in (
        "public_visibility",
        "notification",
        "regulatory_engagement",
        "external_pr_agency",
        "regulatory_fine",
        "judgment_or_settlement",
    ):
        observed = np.mean(indicators[name])
        target = calibrated["probabilities"][name]["target"]
        assert observed == pytest.approx(target, abs=0.005)


def test_fines_require_regulatory_engagement(calibrated):
    rng = np.random.default_rng(456)
    z = rng.standard_normal(200_000)

    indicators = draw_incident_indicators(z, calibrated, rng)

    assert not np.any(
        indicators["regulatory_fine"]
        & ~indicators["regulatory_engagement"]
    )


def test_visibility_increases_with_severity(calibrated):
    rng = np.random.default_rng(789)

    low = draw_incident_indicators(
        np.full(100_000, -2.0), calibrated, rng
    )
    high = draw_incident_indicators(
        np.full(100_000, 2.0), calibrated, rng
    )

    assert np.mean(high["public_visibility"]) > np.mean(
        low["public_visibility"]
    )


def test_reputation_probabilities_increase_with_visibility(calibrated):
    z = np.zeros(1000)

    invisible = reputation_probabilities(z, 0, calibrated)
    visible = reputation_probabilities(z, 1, calibrated)

    for name in ("affected_churn", "unaffected_churn", "lost_deal"):
        assert np.all(visible[name] > invisible[name])
