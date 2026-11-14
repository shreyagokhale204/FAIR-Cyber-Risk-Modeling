
"""FAIR secondary loss: customer churn and lost business opportunities."""

import numpy as np

from src.distributions import sample_distribution
from src.severity import copula_transform
from src.calibration import reputation_probabilities


def simulate_secondary_reputation_loss(
    severity, affected_customers, indicators, config, rng
):
    """Simulate reputation-related loss per successful incident.

    Severity, affected customer counts, and visibility must refer
    to the same incidents across all loss modules.

    Returns one-dimensional arrays, one observation per incident.
    """
    z = np.atleast_1d(np.asarray(severity, dtype=float))
    affected_raw = np.atleast_1d(np.asarray(affected_customers))

    if z.ndim != 1 or affected_raw.shape != z.shape:
        raise ValueError(
            "Severity and affected customers must be matching 1D arrays."
        )

    if not np.all(np.isfinite(z)):
        raise ValueError("Severity values must be finite.")

    if not np.all(np.isfinite(affected_raw)):
        raise ValueError("Affected customer counts must be finite.")

    reputation = config["reputation"]
    organization = config["organization"]

    total_customers = organization["enterprise_customers"]

    if np.any(affected_raw != np.floor(affected_raw)):
        raise ValueError("Affected customer counts must be integers.")

    if np.any((affected_raw < 0) | (affected_raw > total_customers)):
        raise ValueError("Affected customer counts are out of range.")

    affected = affected_raw.astype(int)
    unaffected = total_customers - affected
    n = len(z)

    if "public_visibility" not in indicators:
        raise KeyError("Missing public_visibility indicator.")

    visibility = np.asarray(indicators["public_visibility"])

    if visibility.ndim == 0:
        visibility = np.full(n, visibility.item())

    if visibility.shape != (n,):
        raise ValueError(
            f"public_visibility must have shape ({n},)."
        )

    if not np.all((visibility == 0) | (visibility == 1)):
        raise ValueError("Visibility indicator must contain only 0/1.")

    visibility = visibility.astype(int)

    # Calibrated probabilities from Step 3.
    probabilities = reputation_probabilities(
        z, visibility, config
    )

    p_affected = np.asarray(
        probabilities["affected_churn"], dtype=float
    )
    p_unaffected = np.asarray(
        probabilities["unaffected_churn"], dtype=float
    )
    p_lost_deal = np.asarray(
        probabilities["lost_deal"], dtype=float
    )

    # ---------------------------------------------------------
    # A. AFFECTED CUSTOMER CHURN
    # ---------------------------------------------------------
    affected_churn_count = rng.binomial(
        n=affected,
        p=p_affected,
    )

    # ---------------------------------------------------------
    # B. UNAFFECTED CUSTOMER CHURN
    # ---------------------------------------------------------
    unaffected_churn_count = rng.binomial(
        n=unaffected,
        p=p_unaffected,
    )

    total_churn_count = (
        affected_churn_count + unaffected_churn_count
    )

    # ---------------------------------------------------------
    # C. LOST BUSINESS OPPORTUNITIES
    # ---------------------------------------------------------
    opportunities = reputation["otherwise_won_opportunities"]

    otherwise_won = rng.binomial(
        n=opportunities["n"],
        p=opportunities["p"],
        size=n,
    )

    lost_deals = rng.binomial(
        n=otherwise_won,
        p=p_lost_deal,
    )

    # ---------------------------------------------------------
    # D. FINANCIAL LOSS
    # ---------------------------------------------------------
    loss_per_customer = reputation[
        "loss_per_churned_customer_usd"
    ]

    loss_per_deal = reputation[
        "loss_per_lost_deal_usd"
    ]

    affected_churn_loss = (
        affected_churn_count * loss_per_customer
    )

    unaffected_churn_loss = (
        unaffected_churn_count * loss_per_customer
    )

    lost_deal_loss = lost_deals * loss_per_deal

    total_reputation_loss = (
        affected_churn_loss
        + unaffected_churn_loss
        + lost_deal_loss
    )

    return {
        "affected_customers": affected,
        "unaffected_customers": unaffected,
        "public_visibility_indicator": visibility,
        "affected_churn_probability": p_affected,
        "unaffected_churn_probability": p_unaffected,
        "lost_deal_probability": p_lost_deal,
        "affected_churn_count": affected_churn_count,
        "unaffected_churn_count": unaffected_churn_count,
        "total_churn_count": total_churn_count,
        "otherwise_won_opportunities": otherwise_won,
        "lost_deals_count": lost_deals,
        "affected_churn_loss_usd": affected_churn_loss,
        "unaffected_churn_loss_usd": unaffected_churn_loss,
        "lost_deals_loss_usd": lost_deal_loss,
        "secondary_reputation_loss_usd": total_reputation_loss,
    }