
"""
Step 7D-2: Epistemic uncertainty in loss magnitude.

Fits a quadratic response surface to direct incident
simulations, then propagates parameter uncertainty.

Does not modify the original model configuration.
"""

import numpy as np
import pandas as pd

from config.model_config import get_config
from src.incident_simulation import simulate_incidents
from src.sensitivity_analysis_loss_magnitude import (
    make_magnitude_config,
)


BASELINE_CHURN = 0.002
BASELINE_MARGIN = 210_000.0
BASELINE_EXPOSURE = 1.0
BASELINE_LEF = 3.0 * 0.36 / (0.36 + 4.48)

# The training domain must cover nearly all sampled
# parameter combinations.
BOUNDS = {
    "affected_churn": (0.0001, 0.008),
    "contribution_margin": (100_000.0, 350_000.0),
    "customer_exposure": (0.5, 1.5),
}


def sample_pert(rng, minimum, mode, maximum, size,
                shape_lambda=4.0):
    if not minimum < mode < maximum:
        raise ValueError("PERT bounds must be ordered.")

    alpha = 1 + shape_lambda * (
        mode - minimum
    ) / (maximum - minimum)

    beta = 1 + shape_lambda * (
        maximum - mode
    ) / (maximum - minimum)

    return minimum + (
        maximum - minimum
    ) * rng.beta(alpha, beta, size=size)


def sample_loss_parameters(n_draws=20_000, seed=20261109):
    if n_draws <= 0:
        raise ValueError("n_draws must be positive.")

    rng = np.random.default_rng(seed)

    return pd.DataFrame({
        "affected_churn": rng.beta(
            4.0, 1996.0, size=n_draws
        ),
        "contribution_margin": sample_pert(
            rng, 100_000, 210_000, 350_000, n_draws
        ),
        "customer_exposure": sample_pert(
            rng, 0.5, 1.0, 1.5, n_draws
        ),
    })


def make_joint_magnitude_config(
    affected_churn,
    contribution_margin,
    customer_exposure,
    base_config=None,
):
    config = (
        get_config()
        if base_config is None
        else base_config
    )

    for parameter, value in (
        ("affected_churn", affected_churn),
        ("contribution_margin", contribution_margin),
        ("customer_exposure", customer_exposure),
    ):
        config = make_magnitude_config(
            parameter, value, base_config=config
        )

    return config


def design_matrix(parameters):
    """
    Quadratic response surface with interactions.

    Uses dimensionless, baseline-centered inputs.
    """
    x = (
        parameters["affected_churn"].to_numpy()
        / BASELINE_CHURN - 1
    )
    y = (
        parameters["contribution_margin"].to_numpy()
        / BASELINE_MARGIN - 1
    )
    z = (
        parameters["customer_exposure"].to_numpy()
        - BASELINE_EXPOSURE
    )

    return np.column_stack([
        np.ones(len(parameters)),
        x, y, z,
        x*x, y*y, z*z,
        x*y, x*z, y*z,
    ])


def simulate_parameter_design(
    parameters,
    n_incidents=10_000,
    seed=20261110,
    base_config=None,
):
    """
    Estimate expected incident loss at each design point.

    Each point uses a separate reproducible seed.
    """
    if n_incidents <= 0:
        raise ValueError("n_incidents must be positive.")

    rows = []

    for i, point in parameters.reset_index(
        drop=True
    ).iterrows():

        config = make_joint_magnitude_config(
            affected_churn=point["affected_churn"],
            contribution_margin=point["contribution_margin"],
            customer_exposure=point["customer_exposure"],
            base_config=base_config,
        )

        incidents = simulate_incidents(
            n_incidents=n_incidents,
            seed=seed + i,
            config=config,
            include_details=False,
        )

        losses = incidents[
            "total_incident_loss_usd"
        ].to_numpy(dtype=float)

        rows.append({
            **point.to_dict(),
            "mean_incident_loss_usd": losses.mean(),
            "mc_se_usd": losses.std(ddof=1)
                         / np.sqrt(n_incidents),
        })

    return pd.DataFrame(rows)


def generate_training_design(n_points=45, seed=20261111):
    """
    Stratified-random design across the model domain.
    Includes the exact baseline point.
    """
    if n_points < 12:
        raise ValueError("Use at least 12 design points.")

    rng = np.random.default_rng(seed)
    result = {}

    for name, (low, high) in BOUNDS.items():
        strata = (
            np.arange(n_points - 1)
            + rng.random(n_points - 1)
        ) / (n_points - 1)

        rng.shuffle(strata)

        result[name] = low + strata * (high - low)

    design = pd.DataFrame(result)

    baseline = pd.DataFrame([{
        "affected_churn": BASELINE_CHURN,
        "contribution_margin": BASELINE_MARGIN,
        "customer_exposure": BASELINE_EXPOSURE,
    }])

    return pd.concat(
        [baseline, design],
        ignore_index=True,
    )


def fit_response_surface(training_results):
    X = design_matrix(training_results)

    y = training_results[
        "mean_incident_loss_usd"
    ].to_numpy(dtype=float)

    coefficients, _, rank, _ = np.linalg.lstsq(
        X, y, rcond=None
    )

    if rank < X.shape[1]:
        raise ValueError("Training design is rank deficient.")

    return coefficients


def predict_incident_loss(parameters, coefficients):
    prediction = design_matrix(parameters) @ coefficients

    # Negative predictions indicate a failed approximation;
    # do not silently turn them into valid loss estimates.
    if not np.all(np.isfinite(prediction)):
        raise ValueError("Non-finite response surface output.")

    return prediction


def validate_response_surface(
    validation_results,
    coefficients,
):
    result = validation_results.copy()

    result["predicted_incident_loss_usd"] = (
        predict_incident_loss(result, coefficients)
    )

    result["absolute_pct_error"] = (
        np.abs(
            result["predicted_incident_loss_usd"]
            - result["mean_incident_loss_usd"]
        )
        / result["mean_incident_loss_usd"]
        * 100
    )

    return result


def propagate_loss_uncertainty(
    parameter_draws,
    coefficients,
    expected_lef=BASELINE_LEF,
):
    result = parameter_draws.copy()

    if expected_lef <= 0:
        raise ValueError("Expected LEF must be positive.")

    predictions = predict_incident_loss(
        result, coefficients
    )

    if np.any(predictions <= 0):
        raise ValueError(
            "Response surface predicts nonpositive losses. "
            "Revise the training domain or model."
        )

    result["expected_incident_loss_usd"] = predictions
    result["expected_annual_loss_usd"] = (
        predictions * expected_lef
    )

    return result


def summarize_loss_uncertainty(results):
    values = results[
        "expected_annual_loss_usd"
    ].to_numpy(dtype=float)

    return {
        "mean_aal_usd": float(np.mean(values)),
        "median_aal_usd": float(np.median(values)),
        "aal_p05_usd": float(np.quantile(values, .05)),
        "aal_p95_usd": float(np.quantile(values, .95)),
        "aal_p025_usd": float(np.quantile(values, .025)),
        "aal_p975_usd": float(np.quantile(values, .975)),
        "aal_sd_usd": float(np.std(values, ddof=1)),
    }


'''
Why a quadratic response surface?
We could fit a simple linear model, but the three inputs interact.
For example, increasing churn probability has a larger financial effect when contribution margin is high. Increasing customer exposure also changes the number of unaffected customers, which can affect losses elsewhere in the model.
The quadratic response surface includes pairwise interactions and curvature. However, its accuracy must be checked against direct simulations; the quadratic form is not guaranteed to represent the simulator exactly.
'''