
"""Shared technical severity and one-factor Gaussian copula."""

import numpy as np
from scipy.stats import norm

from src.distributions import inverse_cdf


def draw_severity(rng, size=None):
    """Draw standard-normal technical severity scores."""
    return rng.standard_normal(size=size)


def copula_transform(severity, rho, spec, rng, eps=1e-12):
    """Draw a continuous variable correlated with technical severity.

    rho: correlation between latent standard-normal scores, not
         necessarily Pearson correlation of the transformed values.

    Independent residuals must be drawn separately for each variable.
    """
    if not np.isfinite(rho) or not -1 <= rho <= 1:
        raise ValueError("rho must be finite and between -1 and 1.")

    z_s = np.asarray(severity, dtype=float)

    if np.any(~np.isfinite(z_s)):
        raise ValueError("Severity scores must be finite.")

    residual = rng.standard_normal(size=z_s.shape)

    z_k = rho * z_s + np.sqrt(1 - rho**2) * residual

    u = norm.cdf(z_k)
    u = np.clip(u, eps, 1 - eps)

    return inverse_cdf(spec, u)


def draw_correlated_variables(severity, variable_specs, correlations, rng):
    """Draw multiple continuous variables using one shared severity score.

    variable_specs: dict mapping variable names to distribution specs.
    correlations: dict mapping variable names to latent correlations.

    Each variable gets its own independent residual.
    """
    missing = set(variable_specs) - set(correlations)
    if missing:
        raise KeyError(f"Missing correlations: {sorted(missing)}")

    return {
        name: copula_transform(
            severity=severity,
            rho=correlations[name],
            spec=spec,
            rng=rng,
        )
        for name, spec in variable_specs.items()
    }
