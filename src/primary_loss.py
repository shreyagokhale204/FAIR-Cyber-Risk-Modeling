
"""Simulate primary loss for successful NovaFlow incidents."""

import numpy as np

from src.severity import copula_transform


def simulate_primary_loss(severity, config, rng):
    """Calculate primary loss for one or multiple incidents.

    Parameters
    ----------
    severity : float or numpy array
        Shared technical severity scores, one per incident.
    config : dict
        Model configuration.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    dict
        Individual cost components and total primary loss.
    """
    primary = config["primary_response"]
    correlations = config["dependence"]["correlations_latent_normal"]

    def draw(name):
        return copula_transform(
            severity=severity,
            rho=correlations[name],
            spec=primary[name],
            rng=rng,
        )

    technical_hours = draw("technical_response_hours")
    management_hours = draw("it_security_management_hours")
    executive_hours = draw("executive_response_hours")
    dfir_cost = draw("external_dfir_cost_usd")

    technical_cost = (
        technical_hours * primary["technical_hourly_rate_usd"]
    )
    management_cost = (
        management_hours
        * primary["it_security_management_hourly_rate_usd"]
    )
    executive_cost = (
        executive_hours * primary["executive_hourly_rate_usd"]
    )

    total = (
        technical_cost
        + management_cost
        + executive_cost
        + dfir_cost
    )

    return {
        "technical_response_hours": technical_hours,
        "management_response_hours": management_hours,
        "executive_response_hours": executive_hours,
        "technical_response_cost_usd": technical_cost,
        "management_response_cost_usd": management_cost,
        "executive_response_cost_usd": executive_cost,
        "external_dfir_cost_usd": dfir_cost,
        "primary_loss_usd": total,
    }
