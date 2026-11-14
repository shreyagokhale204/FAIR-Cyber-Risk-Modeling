
import numpy as np
import pandas as pd
import pytest

from config.model_config import get_config

from src.model_validation_loss_parameter_uncertainty import (
    BASELINE_LEF,
    design_matrix,
    generate_training_design,
    make_joint_magnitude_config,
    predict_incident_loss,
    propagate_loss_uncertainty,
    sample_loss_parameters,
    sample_pert,
    summarize_loss_uncertainty,
)


def test_parameter_draw_count():
    draws = sample_loss_parameters(1000, seed=42)
    assert len(draws) == 1000


def test_parameter_draws_reproducible():
    a = sample_loss_parameters(100, seed=42)
    b = sample_loss_parameters(100, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_parameter_ranges():
    draws = sample_loss_parameters(5000, seed=42)

    assert draws["affected_churn"].between(0, 1).all()
    assert draws["contribution_margin"].between(
        100000, 350000
    ).all()
    assert draws["customer_exposure"].between(
        0.5, 1.5
    ).all()


def test_parameter_means():
    draws = sample_loss_parameters(100000, seed=42)

    assert abs(draws["affected_churn"].mean() - .002) < .00005
    assert abs(draws["customer_exposure"].mean() - 1) < .01
    assert abs(draws["contribution_margin"].mean() - 215000) < 1500


def test_original_config_unchanged():
    cfg = get_config()

    modified = make_joint_magnitude_config(
        .003, 250000, 1.2, base_config=cfg
    )

    assert cfg["probabilities"]["affected_churn"]["target"] == .002
    assert cfg["reputation"]["loss_per_churned_customer_usd"] == 210000
    assert modified["probabilities"]["affected_churn"]["target"] == .003


def test_both_margin_fields_updated():
    modified = make_joint_magnitude_config(
        .002, 250000, 1.0
    )

    reputation = modified["reputation"]

    assert reputation["loss_per_churned_customer_usd"] == 250000
    assert reputation["loss_per_lost_deal_usd"] == 250000


def test_calibration_intercepts_reset():
    modified = make_joint_magnitude_config(
        .003, 250000, 1.2
    )

    assert all(
        spec["intercept"] is None
        for spec in modified["probabilities"].values()
        if spec["family"] == "calibrated_logistic"
    )


def test_training_design_baseline():
    design = generate_training_design(20)

    assert len(design) == 20
    assert design.iloc[0]["affected_churn"] == .002
    assert design.iloc[0]["contribution_margin"] == 210000
    assert design.iloc[0]["customer_exposure"] == 1


def test_design_matrix_shape():
    design = generate_training_design(20)
    assert design_matrix(design).shape == (20, 10)


def test_prediction_with_constant_model():
    design = generate_training_design(20)
    coefficients = np.array([1_200_000] + [0] * 9)

    predicted = predict_incident_loss(
        design, coefficients
    )

    assert np.allclose(predicted, 1_200_000)


def test_propagation_identity():
    draws = sample_loss_parameters(100, seed=42)
    coefficients = np.array([1_200_000] + [0] * 9)

    results = propagate_loss_uncertainty(
        draws, coefficients
    )

    assert np.allclose(
        results["expected_annual_loss_usd"],
        1_200_000 * BASELINE_LEF
    )


def test_summary_quantiles():
    draws = sample_loss_parameters(100, seed=42)
    coefficients = np.array([1_200_000] + [0] * 9)

    results = propagate_loss_uncertainty(
        draws, coefficients
    )
    summary = summarize_loss_uncertainty(results)

    assert summary["aal_p05_usd"] <= summary["aal_p95_usd"]


def test_invalid_sample_count():
    with pytest.raises(ValueError):
        sample_loss_parameters(0)
