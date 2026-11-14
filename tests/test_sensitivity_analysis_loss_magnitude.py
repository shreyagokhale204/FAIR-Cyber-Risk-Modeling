
# tests/test_sensitivity_analysis_loss_magnitude.py

from copy import deepcopy

import numpy as np
import pytest

from config.model_config import get_config
from src.sensitivity_analysis_loss_magnitude import (
    make_magnitude_config,
    run_magnitude_sensitivity,
)


def test_churn_target_changes():
    config = make_magnitude_config("affected_churn", 0.005)
    assert config["probabilities"]["affected_churn"]["target"] == 0.005


def test_margin_changes_both_values():
    config = make_magnitude_config("contribution_margin", 300_000)
    assert config["reputation"]["loss_per_churned_customer_usd"] == 300_000
    assert config["reputation"]["loss_per_lost_deal_usd"] == 300_000


def test_exposure_scales_pert():
    config = make_magnitude_config("customer_exposure", 0.5)
    spec = config["reputation"]["affected_customers"]

    assert spec["min"] == 150
    assert spec["mode"] == 750
    assert spec["max"] == 1500


def test_exposure_respects_customer_cap():
    config = make_magnitude_config("customer_exposure", 1.5)
    spec = config["reputation"]["affected_customers"]

    assert spec["max"] <= 3000
    assert spec["min"] < spec["mode"] < spec["max"]


def test_original_config_unchanged():
    original = get_config()
    snapshot = deepcopy(original)

    make_magnitude_config("affected_churn", 0.005, original)

    assert original == snapshot


def test_invalid_parameters():
    with pytest.raises(ValueError):
        make_magnitude_config("unknown", 1)

    with pytest.raises(ValueError):
        make_magnitude_config("affected_churn", 1.5)

    with pytest.raises(ValueError):
        make_magnitude_config("contribution_margin", -100)


@pytest.fixture(scope="module")
def small_sensitivity_run():
    return run_magnitude_sensitivity(
        n_years=2000,
        seed=123,
        churn_values=(0.001, 0.002),
        margin_values=(150_000, 210_000),
        exposure_multipliers=(0.75, 1.0),
    )


def test_scenario_count(small_sensitivity_run):
    results, annual = small_sensitivity_run
    assert len(results) == 6
    assert len(annual) == 6


def test_annual_loss_values_nonnegative(small_sensitivity_run):
    _, annual = small_sensitivity_run

    for frame in annual.values():
        assert len(frame) == 2000
        assert np.all(frame["total_annual_loss_usd"] >= 0)


def test_aal_matches_raw_losses(small_sensitivity_run):
    results, annual = small_sensitivity_run

    for _, row in results.iterrows():
        key = (row["parameter"], row["parameter_value"])
        actual = annual[key]["total_annual_loss_usd"].mean()

        assert np.isclose(row["aal_usd"], actual)


def test_exceedance_probabilities_valid(small_sensitivity_run):
    results, _ = small_sensitivity_run

    for threshold in (2, 5, 10, 15):
        values = results[f"p_loss_gt_{threshold}m"]
        assert values.between(0, 1).all()


def test_identical_baseline_scenarios(small_sensitivity_run):
    _, annual = small_sensitivity_run

    baseline_margin = annual[
        ("contribution_margin", 210000.0)
    ]["total_annual_loss_usd"].to_numpy()

    baseline_exposure = annual[
        ("customer_exposure", 1.0)
    ]["total_annual_loss_usd"].to_numpy()

    np.testing.assert_array_equal(
        baseline_margin,
        baseline_exposure,
    )
