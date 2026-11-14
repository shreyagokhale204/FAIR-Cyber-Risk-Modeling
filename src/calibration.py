
"""Calibrate severity-dependent logistic probabilities.

Uses deterministic Gauss-Hermite quadrature and Brent root finding.
No random simulation is required during calibration.
"""

from copy import deepcopy

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import brentq
from scipy.special import expit, ndtr

from src.distributions import inverse_cdf


def logistic_probability(intercept, severity, severity_or=1.0,
                         visibility=0, visibility_or=1.0):
    """Calculate logistic probability for severity and visibility."""
    linear_predictor = (
        intercept
        + np.log(severity_or) * np.asarray(severity)
        + np.log(visibility_or) * np.asarray(visibility)
    )
    return expit(linear_predictor)


def normal_quadrature(n=40):
    """Nodes and weights for expectations under N(0,1)."""
    nodes, weights = hermgauss(n)
    return np.sqrt(2) * nodes, weights / np.sqrt(np.pi)


def solve_intercept(target, expectation_fn):
    """Solve E[probability] = target for the logistic intercept."""
    if not 0 < target < 1:
        raise ValueError("Calibration target must be between 0 and 1.")

    def objective(a):
        return expectation_fn(a) - target

    return brentq(objective, -50, 50, xtol=1e-12)


def calibrate_probabilities(config, n_nodes=40):
    """Return a copy of config with calibrated logistic intercepts.

    Calibration order:
      1. Public visibility
      2. Notification and regulator engagement
      3. PR agency, judgments, and lost deals
      4. Regulatory fine with regulator gate
      5. Customer-weighted affected/unaffected churn
    """
    cfg = deepcopy(config)
    models = cfg["probabilities"]
    z, w = normal_quadrature(n_nodes)

    def severity_probability(name, intercept):
        m = models[name]
        return logistic_probability(
            intercept, z, m["severity_or"]
        )

    # Models depending only on technical severity.
    for name in ("public_visibility", "notification",
                 "regulatory_engagement"):
        m = models[name]
        m["intercept"] = solve_intercept(
            m["target"],
            lambda a, name=name: np.dot(
                w, severity_probability(name, a)
            )
        )

    # Visibility probability conditional on severity.
    p_visibility = severity_probability(
        "public_visibility", models["public_visibility"]["intercept"]
    )

    def visibility_averaged_probability(name, intercept):
        """E[I | Z] averaged over both possible visibility states."""
        m = models[name]

        p_not_visible = logistic_probability(
            intercept, z, m["severity_or"],
            visibility=0, visibility_or=m["visibility_or"]
        )
        p_visible = logistic_probability(
            intercept, z, m["severity_or"],
            visibility=1, visibility_or=m["visibility_or"]
        )

        return (
            (1 - p_visibility) * p_not_visible
            + p_visibility * p_visible
        )

    # Models influenced by both severity and public visibility.
    for name in ("external_pr_agency", "judgment_or_settlement",
                 "lost_deal"):
        m = models[name]
        m["intercept"] = solve_intercept(
            m["target"],
            lambda a, name=name: np.dot(
                w, visibility_averaged_probability(name, a)
            )
        )

    # Fine can occur ONLY if regulatory engagement occurs.
    p_regulator = severity_probability(
        "regulatory_engagement",
        models["regulatory_engagement"]["intercept"]
    )
    fine_model = models["regulatory_fine"]

    fine_model["intercept"] = solve_intercept(
        fine_model["target"],
        lambda a: np.dot(
            w,
            p_regulator * severity_probability("regulatory_fine", a)
        )
    )

    # Customer-weighted churn calibration.
    #
    # Affected customers depend on severity through the Gaussian copula.
    # Integrate over both severity and an independent copula residual.
    rho = cfg["dependence"]["correlations_latent_normal"][
        "affected_customers"
    ]
    customer_total = cfg["organization"]["enterprise_customers"]
    affected_spec = cfg["reputation"]["affected_customers"]

    residual_nodes = z
    residual_weights = w

    latent_affected = (
        rho * z[:, None]
        + np.sqrt(1 - rho**2) * residual_nodes[None, :]
    )

    u = np.clip(ndtr(latent_affected), 1e-12, 1 - 1e-12)
    affected = inverse_cdf(affected_spec, u)

    # Match the eventual incident simulation's rounding rule.
    affected = np.clip(np.rint(affected), 0, customer_total)

    expected_affected_given_z = affected @ residual_weights
    expected_unaffected_given_z = (
        customer_total - expected_affected_given_z
    )

    for name, customer_weights in (
        ("affected_churn", expected_affected_given_z),
        ("unaffected_churn", expected_unaffected_given_z),
    ):
        m = models[name]
        denominator = np.dot(w, customer_weights)

        m["intercept"] = solve_intercept(
            m["target"],
            lambda a, name=name, cw=customer_weights, den=denominator:
                np.dot(
                    w, cw * visibility_averaged_probability(name, a)
                ) / den
        )

    return cfg


def draw_incident_indicators(severity, calibrated_config, rng):
    """Draw shared event indicators for one incident or an array.

    Regulatory fine is gated by regulatory engagement.
    Public visibility is drawn once and reused by dependent events.
    """
    models = calibrated_config["probabilities"]
    severity = np.asarray(severity, dtype=float)

    if any(m["intercept"] is None for m in models.values()):
        raise ValueError("Calibrate probabilities before drawing indicators.")

    def probability(name, visibility=0):
        m = models[name]
        return logistic_probability(
            m["intercept"],
            severity,
            severity_or=m["severity_or"],
            visibility=visibility,
            visibility_or=m["visibility_or"],
        )

    indicators = {}

    for name in ("public_visibility", "notification",
                 "regulatory_engagement"):
        indicators[name] = (
            rng.random(size=severity.shape) < probability(name)
        )

    visibility = indicators["public_visibility"].astype(int)

    for name in ("external_pr_agency", "judgment_or_settlement"):
        indicators[name] = (
            rng.random(size=severity.shape)
            < probability(name, visibility)
        )

    # Fine cannot occur without regulatory engagement.
    indicators["regulatory_fine"] = (
        indicators["regulatory_engagement"]
        & (rng.random(size=severity.shape) < probability("regulatory_fine"))
    )

    return indicators


def reputation_probabilities(severity, visibility, calibrated_config):
    """Return calibrated churn and lost-deal probabilities."""
    models = calibrated_config["probabilities"]
    result = {}

    for name in ("affected_churn", "unaffected_churn", "lost_deal"):
        m = models[name]
        if m["intercept"] is None:
            raise ValueError(f"{name} has not been calibrated.")

        result[name] = logistic_probability(
            m["intercept"],
            severity,
            severity_or=m["severity_or"],
            visibility=visibility,
            visibility_or=m["visibility_or"],
        )

    return result
