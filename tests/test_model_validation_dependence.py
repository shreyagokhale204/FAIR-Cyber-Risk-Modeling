
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from config.model_config import get_config

from src.model_validation_dependence import (
    make_dependence_config,
    run_dependence_stress_test,
)


def test_baseline_configuration_unchanged():
    baseline = get_config()

    modified = make_dependence_config(
        1.0,
        base_config=baseline,
    )

    assert modified == baseline


def test_original_configuration_not_mutated():
    original = get_config()
    original_copy = deepcopy(original)

    make_dependence_config(
        0.5,
        base_config=original,
    )

    assert original == original_copy


def test_weaker_copula_correlations():
    baseline = get_config()
    weaker = make_dependence_config(
        0.5,
        base_config=baseline,
    )

    baseline_rho = baseline[
        "dependence"
    ]["correlations_latent_normal"]

    weaker_rho = weaker[
        "dependence"
    ]["correlations_latent_normal"]

    for name in baseline_rho:
        assert np.isclose(
            weaker_rho[name],
            baseline_rho[name] * 0.5,
        )


def test_stronger_copula_correlations():
    baseline = get_config()
    stronger = make_dependence_config(
        1.25,
        base_config=baseline,
    )

    baseline_rho = baseline[
        "dependence"
    ]["correlations_latent_normal"]

    stronger_rho = stronger[
        "dependence"
    ]["correlations_latent_normal"]

    for name in baseline_rho:
        assert np.isclose(
            stronger_rho[name],
            baseline_rho[name] * 1.25,
        )


def test_odds_ratio_scaling():
    baseline = get_config()

    weaker = make_dependence_config(
        0.5,
        base_config=baseline,
    )

    assert np.isclose(
        weaker["probabilities"][
            "public_visibility"
        ]["severity_or"],
        np.sqrt(2.0),
    )


def test_probability_targets_preserved():
    baseline = get_config()

    stronger = make_dependence_config(
        1.25,
        base_config=baseline,
    )

    for name in baseline["probabilities"]:

        assert (
            stronger["probabilities"][name]["target"]
            == baseline["probabilities"][name]["target"]
        )

        assert (
            stronger["probabilities"][name]["target_type"]
            == baseline["probabilities"][name]["target_type"]
        )


def test_intercepts_reset():
    stronger = make_dependence_config(1.25)

    for spec in stronger["probabilities"].values():
        assert spec["intercept"] is None


def test_invalid_multiplier():
    with pytest.raises(ValueError):
        make_dependence_config(-0.5)


def test_stress_test_output():
    results, annual = run_dependence_stress_test(
        n_years=500,
        seed=123,
    )

    assert len(results) == 3

    assert set(annual.keys()) == {
        "Weaker",
        "Baseline",
        "Stronger",
    }

    for name, df in annual.items():
        assert len(df) == 500

        assert (
            df["total_annual_loss_usd"] >= 0
        ).all()


def test_baseline_aal_difference_zero():
    results, _ = run_dependence_stress_test(
        n_years=500,
        seed=123,
    )

    baseline = results.loc[
        results["scenario"] == "Baseline"
    ].iloc[0]

    assert np.isclose(
        baseline["aal_change_vs_baseline_usd"],
        0,
    )


def test_stress_test_reproducibility():
    a, _ = run_dependence_stress_test(
        n_years=500,
        seed=123,
    )

    b, _ = run_dependence_stress_test(
        n_years=500,
        seed=123,
    )

    pd.testing.assert_frame_equal(a, b)
