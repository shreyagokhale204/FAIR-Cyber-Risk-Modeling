
"""Combined FAIR incident-level financial loss simulation."""

import numpy as np
import pandas as pd

from config.model_config import get_config
from src.calibration import (
    calibrate_probabilities,
    draw_incident_indicators,
)
from src.severity import copula_transform

from src.primary_loss import simulate_primary_loss

from src.secondary_loss_response_planning import (
    simulate_secondary_response_planning,
)

from src.secondary_loss_fines_and_judgement import (
    simulate_secondary_fines_and_judgement,
)

from src.secondary_loss_reputation import (
    simulate_secondary_reputation_loss,
)


LOSS_COLUMNS = [
    "primary_loss_usd",
    "secondary_response_planning_loss_usd",
    "secondary_fines_judgement_loss_usd",
    "secondary_reputation_loss_usd",
]


def simulate_incidents(
    n_incidents=10_000,
    seed=42,
    config=None,
    include_details=True,
):
    """Simulate financial losses conditional on successful incidents.

    Parameters
    ----------
    n_incidents : int
        Number of successful incidents to simulate.
    seed : int
        Random seed for reproducibility.
    config : dict, optional
        Model configuration. Defaults to get_config().
    include_details : bool
        Whether to retain detailed intermediate outputs.

    Returns
    -------
    pandas.DataFrame
        One row per successful incident.
    """
    if not isinstance(n_incidents, (int, np.integer)):
        raise ValueError("n_incidents must be an integer.")

    if n_incidents <= 0:
        raise ValueError("n_incidents must be positive.")

    if config is None:
        config = get_config()

    # Calibrate the logistic intercepts.
    calibrated_config = calibrate_probabilities(config)

    rng = np.random.default_rng(seed)
    n = n_incidents

    # ---------------------------------------------------------
    # A. SHARED TECHNICAL SEVERITY
    # ---------------------------------------------------------
    severity = rng.standard_normal(n)

    # ---------------------------------------------------------
    # B. SHARED AFFECTED CUSTOMER COUNT
    # ---------------------------------------------------------
    affected_spec = calibrated_config[
        "reputation"
    ]["affected_customers"]

    rho = calibrated_config[
        "dependence"
    ]["correlations_latent_normal"]["affected_customers"]

    affected = copula_transform(
        severity=severity,
        rho=rho,
        spec=affected_spec,
        rng=rng,
    )

    total_customers = calibrated_config[
        "organization"
    ]["enterprise_customers"]

    affected = np.clip(
        np.rint(affected),
        0,
        total_customers,
    ).astype(int)

    # ---------------------------------------------------------
    # C. SHARED OCCURRENCE INDICATORS
    # ---------------------------------------------------------
    indicators = draw_incident_indicators(
        severity,
        calibrated_config,
        rng,
    )

    # ---------------------------------------------------------
    # D. PRIMARY LOSS
    # ---------------------------------------------------------
    primary = simulate_primary_loss(
        severity,
        calibrated_config,
        rng,
    )

    # ---------------------------------------------------------
    # E. SECONDARY RESPONSE AND PLANNING
    # ---------------------------------------------------------
    secondary_response = simulate_secondary_response_planning(
        severity,
        affected,
        indicators,
        calibrated_config,
        rng,
    )

    # ---------------------------------------------------------
    # F. FINES AND JUDGMENTS
    # ---------------------------------------------------------
    fines_judgments = simulate_secondary_fines_and_judgement(
        severity,
        indicators,
        calibrated_config,
        rng,
    )

    # ---------------------------------------------------------
    # G. REPUTATION LOSS
    # ---------------------------------------------------------
    reputation = simulate_secondary_reputation_loss(
        severity,
        affected,
        indicators,
        calibrated_config,
        rng,
    )

    # ---------------------------------------------------------
    # H. COMBINE FINANCIAL LOSSES
    # ---------------------------------------------------------
    total_loss = (
        primary["primary_loss_usd"]
        + secondary_response[
            "secondary_response_planning_loss_usd"
        ]
        + fines_judgments[
            "secondary_fines_judgement_loss_usd"
        ]
        + reputation["secondary_reputation_loss_usd"]
    )

    data = {
        "incident_id": np.arange(1, n + 1),
        "technical_severity": severity,
        "affected_customers": affected,
        "primary_loss_usd": primary["primary_loss_usd"],
        "secondary_response_planning_loss_usd":
            secondary_response[
                "secondary_response_planning_loss_usd"
            ],
        "secondary_fines_judgement_loss_usd":
            fines_judgments[
                "secondary_fines_judgement_loss_usd"
            ],
        "secondary_reputation_loss_usd":
            reputation["secondary_reputation_loss_usd"],
        "total_incident_loss_usd": total_loss,
    }

    # Include the shared occurrence indicators.
    for name, values in indicators.items():
        data[f"indicator_{name}"] = values

    if include_details:
        # Prefix each detailed output to prevent name collisions.
        for prefix, result in (
            ("primary", primary),
            ("response", secondary_response),
            ("fines", fines_judgments),
            ("reputation", reputation),
        ):
            for name, values in result.items():
                if name not in LOSS_COLUMNS:
                    data[f"{prefix}_{name}"] = values

    return pd.DataFrame(data)


def summarize_incident_losses(df):
    """Calculate summary statistics for incident-level losses."""
    required = LOSS_COLUMNS + ["total_incident_loss_usd"]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    summary = []

    for column in required:
        values = df[column].to_numpy()

        summary.append({
            "component": column,
            "mean_usd": np.mean(values),
            "median_usd": np.median(values),
            "p90_usd": np.percentile(values, 90),
            "p95_usd": np.percentile(values, 95),
            "p99_usd": np.percentile(values, 99),
            "maximum_usd": np.max(values),
        })

    return pd.DataFrame(summary)
