
"""Tests for NovaFlow frequency sensitivity analysis."""

import numpy as np
import pandas as pd
import pytest

from src.sensitivity_analysis import (
    run_frequency_sensitivity,
    BASELINE_SUCCESS_PROBABILITY,
)


@pytest.fixture(scope="module")
def sensitivity_outputs():
    return run_frequency_sensitivity(
        n_years=5_000,
        seed=123,
        campaign_rates=(1, 3, 5),
        success_probabilities=(0.03, 0.15),
    )


def test_number_of_scenarios(sensitivity_outputs):
    results, losses = sensitivity_outputs

    assert len(results) == 6
    assert len(losses) == 6


def test_output_types(sensitivity_outputs):
    results, losses = sensitivity_outputs

    assert isinstance(results, pd.DataFrame)
    assert isinstance(losses, dict)


def test_stored_annual_loss_shapes(sensitivity_outputs):
    _, losses = sensitivity_outputs

    for values in losses.values():
        assert isinstance(values, np.ndarray)
        assert values.shape == (5_000,)
        assert np.all(values >= 0)


def test_campaign_rate_expected_lef(sensitivity_outputs):
    results, _ = sensitivity_outputs

    rows = results[
        results["parameter"] == "campaign_rate"
    ]

    for _, row in rows.iterrows():
        expected = (
            row["parameter_value"]
            * BASELINE_SUCCESS_PROBABILITY
        )

        assert row["expected_lef"] == pytest.approx(expected)


def test_success_probability_expected_lef(sensitivity_outputs):
    results, _ = sensitivity_outputs

    rows = results[
        results["parameter"] == "success_probability"
    ]

    for _, row in rows.iterrows():
        assert row["expected_lef"] == pytest.approx(
            3 * row["parameter_value"]
        )


def test_baseline_lef(sensitivity_outputs):
    results, _ = sensitivity_outputs

    baseline = results[
        (results["parameter"] == "campaign_rate")
        & (results["parameter_value"] == 3)
    ].iloc[0]

    assert baseline["expected_lef"] == pytest.approx(
        3 * 0.36 / (0.36 + 4.48)
    )


def test_nonnegative_risk_metrics(sensitivity_outputs):
    results, _ = sensitivity_outputs

    columns = [
        "expected_lef",
        "simulated_lef",
        "aal_usd",
        "var_90_usd",
        "var_95_usd",
        "var_99_usd",
        "tvar_90_usd",
        "tvar_95_usd",
        "tvar_99_usd",
    ]

    assert np.all(results[columns].to_numpy() >= 0)


def test_probability_bounds(sensitivity_outputs):
    results, _ = sensitivity_outputs

    probability_columns = [
        "probability_any_incident",
        "p_loss_gt_2m",
        "p_loss_gt_5m",
        "p_loss_gt_10m",
        "p_loss_gt_15m",
    ]

    for column in probability_columns:
        assert results[column].between(0, 1).all()


def test_var_and_tvar_order(sensitivity_outputs):
    results, _ = sensitivity_outputs

    assert np.all(
        results["var_90_usd"]
        <= results["var_95_usd"]
    )

    assert np.all(
        results["var_95_usd"]
        <= results["var_99_usd"]
    )

    assert np.all(
        results["tvar_99_usd"]
        >= results["var_99_usd"]
    )


def test_exceedance_probabilities_decrease(sensitivity_outputs):
    results, _ = sensitivity_outputs

    columns = [
        "p_loss_gt_2m",
        "p_loss_gt_5m",
        "p_loss_gt_10m",
        "p_loss_gt_15m",
    ]

    probabilities = results[columns].to_numpy()

    assert np.all(np.diff(probabilities, axis=1) <= 0)


def test_exceedance_matches_raw_losses(sensitivity_outputs):
    results, losses = sensitivity_outputs

    for _, row in results.iterrows():
        key = (
            row["parameter"],
            float(row["parameter_value"]),
        )

        actual = np.mean(losses[key] > 2_000_000)

        assert row["p_loss_gt_2m"] == pytest.approx(actual)


def test_aal_matches_raw_losses(sensitivity_outputs):
    results, losses = sensitivity_outputs

    for _, row in results.iterrows():
        key = (
            row["parameter"],
            float(row["parameter_value"]),
        )

        assert row["aal_usd"] == pytest.approx(
            losses[key].mean()
        )


def test_baseline_scenarios_match(sensitivity_outputs):
    results, losses = sensitivity_outputs

    campaign_key = ("campaign_rate", 3.0)
    success_key = (
        "success_probability",
        BASELINE_SUCCESS_PROBABILITY,
    )

    np.testing.assert_array_equal(
        losses[campaign_key],
        losses[success_key],
    )


def test_invalid_inputs():
    with pytest.raises(ValueError):
        run_frequency_sensitivity(n_years=0)

    with pytest.raises(ValueError):
        run_frequency_sensitivity(
            n_years=100,
            campaign_rates=(-1,),
        )

    with pytest.raises(ValueError):
        run_frequency_sensitivity(
            n_years=100,
            success_probabilities=(1.5,),
        )
