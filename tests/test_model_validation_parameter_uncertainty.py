
import numpy as np
import pandas as pd
import pytest

from src.model_validation_parameter_uncertainty import (
    run_frequency_parameter_uncertainty,
    summarize_parameter_uncertainty,
)


EXPECTED_INCIDENT_LOSS = 1_233_714


@pytest.fixture(scope="module")
def draws():
    return run_frequency_parameter_uncertainty(
        expected_incident_loss_usd=EXPECTED_INCIDENT_LOSS,
        n_draws=20_000,
        seed=42,
    )


def test_output_length(draws):
    assert len(draws) == 20_000


def test_campaign_rates_positive(draws):
    assert (draws["campaign_rate"] > 0).all()


def test_success_probabilities_valid(draws):
    p = draws["mean_success_probability"]

    assert ((p >= 0) & (p <= 1)).all()


def test_aal_nonnegative(draws):
    assert (
        draws["expected_annual_loss_usd"] >= 0
    ).all()


def test_aal_identity(draws):
    expected = (
        draws["campaign_rate"]
        * draws["mean_success_probability"]
        * EXPECTED_INCIDENT_LOSS
    )

    assert np.allclose(
        draws["expected_annual_loss_usd"],
        expected,
    )


def test_campaign_prior_mean(draws):
    assert abs(
        draws["campaign_rate"].mean() - 3.0
    ) < 0.1


def test_success_prior_mean(draws):
    target = 0.36 / (0.36 + 4.48)

    assert abs(
        draws["mean_success_probability"].mean()
        - target
    ) < 0.003


def test_reproducibility():
    a = run_frequency_parameter_uncertainty(
        EXPECTED_INCIDENT_LOSS,
        n_draws=100,
        seed=123,
    )

    b = run_frequency_parameter_uncertainty(
        EXPECTED_INCIDENT_LOSS,
        n_draws=100,
        seed=123,
    )

    pd.testing.assert_frame_equal(a, b)


def test_summary_quantiles(draws):
    summary = summarize_parameter_uncertainty(
        draws,
        EXPECTED_INCIDENT_LOSS,
    )

    assert (
        summary["aal_p025_usd"]
        <= summary["aal_p05_usd"]
        <= summary["median_aal_usd"]
        <= summary["aal_p95_usd"]
        <= summary["aal_p975_usd"]
    )


def test_invalid_draw_count():
    with pytest.raises(ValueError):
        run_frequency_parameter_uncertainty(
            EXPECTED_INCIDENT_LOSS,
            n_draws=0,
        )


def test_invalid_incident_loss():
    with pytest.raises(ValueError):
        run_frequency_parameter_uncertainty(
            -1,
            n_draws=100,
        )
