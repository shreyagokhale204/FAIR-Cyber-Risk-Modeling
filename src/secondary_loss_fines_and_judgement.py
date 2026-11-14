
"""FAIR secondary loss: regulatory fines and legal judgments."""

import numpy as np

from src.distributions import sample_distribution


def simulate_secondary_fines_and_judgement(
    severity, indicators, config, rng
):
    """Simulate regulatory fines and judgments per successful incident.

    Parameters
    ----------
    severity : float or 1D array
        Shared technical severity scores.
    indicators : dict
        Shared binary indicators from Step 3.
    config : dict
        Model configuration.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    dict
        Fine, judgment, and total secondary financial losses.
        All returned values are 1D NumPy arrays.
    """
    z = np.atleast_1d(np.asarray(severity, dtype=float))

    if z.ndim != 1 or not np.all(np.isfinite(z)):
        raise ValueError("Severity must be a finite scalar or 1D array.")

    n = len(z)
    fine_config = config["fines_and_judgments"]

    def get_indicator(name):
        if name not in indicators:
            raise KeyError(f"Missing incident indicator: {name}")

        values = np.asarray(indicators[name])

        if values.ndim == 0:
            values = np.full(n, values.item())

        if values.shape != (n,):
            raise ValueError(
                f"Indicator '{name}' must have shape ({n},)."
            )

        if not np.all((values == 0) | (values == 1)):
            raise ValueError(
                f"Indicator '{name}' must contain only 0/1 values."
            )

        return values.astype(bool)

    regulatory = get_indicator("regulatory_engagement")
    fine = get_indicator("regulatory_fine")
    judgment = get_indicator("judgment_or_settlement")

    # Fine occurrence must always be a subset of regulatory engagement.
    if np.any(fine & ~regulatory):
        raise ValueError(
            "A regulatory fine cannot occur without "
            "regulatory engagement."
        )

    # Conditional fine and settlement amounts.
    fine_amount = np.asarray(
        sample_distribution(
            fine_config["regulatory_fine"]["amount_if_fined_usd"],
            rng,
            size=n,
        ),
        dtype=float,
    )

    judgment_amount = np.asarray(
        sample_distribution(
            fine_config["judgment_or_settlement"]["amount_if_incurred_usd"],
            rng,
            size=n,
        ),
        dtype=float,
    )

    # Apply occurrence indicators.
    fine_loss = np.where(fine, fine_amount, 0.0)
    judgment_loss = np.where(judgment, judgment_amount, 0.0)

    total = fine_loss + judgment_loss

    return {
        "regulatory_engagement_indicator": regulatory,
        "regulatory_fine_indicator": fine,
        "judgment_or_settlement_indicator": judgment,
        "regulatory_fine_loss_usd": fine_loss,
        "judgment_or_settlement_loss_usd": judgment_loss,
        "secondary_fines_judgement_loss_usd": total,
    }
