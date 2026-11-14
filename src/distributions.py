
"""Probability distributions for the FAIR Monte Carlo model.

Continuous distributions support random sampling, inverse CDF and means.
Discrete distributions support random sampling and means.

Gamma uses shape/scale.
Lognormal uses median/log-standard-deviation.
PERT uses a scaled Beta distribution.
"""

import numpy as np
from scipy import stats


def _get_distribution(spec):
    """Build a SciPy distribution from a configuration dictionary."""
    family = spec["family"]

    if family == "pert":
        low = spec["min"]
        mode = spec["mode"]
        high = spec["max"]
        lam = spec.get("shape_lambda", 4.0)

        if not (low <= mode <= high):
            raise ValueError("PERT requires min <= mode <= max.")
        if lam <= 0:
            raise ValueError("PERT shape_lambda must be positive.")
        if low == high:
            return None

        alpha = 1 + lam * (mode - low) / (high - low)
        beta_param = 1 + lam * (high - mode) / (high - low)

        return stats.beta(
            a=alpha,
            b=beta_param,
            loc=low,
            scale=high - low
        )

    if family == "gamma":
        if spec["shape"] <= 0 or spec["scale"] <= 0:
            raise ValueError("Gamma shape and scale must be positive.")
        return stats.gamma(a=spec["shape"], scale=spec["scale"])

    if family == "lognormal":
        if spec["median"] <= 0 or spec["sigma"] <= 0:
            raise ValueError("Lognormal median and sigma must be positive.")
        return stats.lognorm(
            s=spec["sigma"],
            scale=spec["median"]
        )

    if family == "beta":
        if spec["alpha"] <= 0 or spec["beta"] <= 0:
            raise ValueError("Beta parameters must be positive.")
        return stats.beta(a=spec["alpha"], b=spec["beta"])

    if family == "poisson":
        if spec["lam"] < 0:
            raise ValueError("Poisson lambda cannot be negative.")
        return stats.poisson(mu=spec["lam"])

    if family == "binomial":
        n = spec["n"]
        p = spec["p"]
        if int(n) != n or n < 0 or not (0 <= p <= 1):
            raise ValueError("Invalid Binomial n or p.")
        return stats.binom(n=int(n), p=p)

    if family == "bernoulli":
        p = spec["p"]
        if not 0 <= p <= 1:
            raise ValueError("Bernoulli p must be between 0 and 1.")
        return stats.bernoulli(p)

    if family == "standard_normal":
        return stats.norm()

    if family == "fixed":
        return None

    raise ValueError(f"Unsupported distribution: {family}")


def sample_distribution(spec, rng, size=None):
    """Generate random values using an explicit NumPy random generator.

    size=None returns a scalar; size may also be an integer or tuple.
    """
    family = spec["family"]

    if family == "fixed":
        return np.full(size, spec["value"]) if size is not None else spec["value"]

    if family == "pert" and spec["min"] == spec["max"]:
        return (np.full(size, spec["min"])
                if size is not None else spec["min"])

    dist = _get_distribution(spec)
    return dist.rvs(size=size, random_state=rng)


def inverse_cdf(spec, u):
    """Convert uniform probabilities to values from a continuous distribution.

    Used by the Gaussian copula. Input u must lie in [0, 1].
    """
    u = np.asarray(u, dtype=float)

    if np.any(~np.isfinite(u)) or np.any((u < 0) | (u > 1)):
        raise ValueError("CDF probabilities must be finite and in [0, 1].")

    family = spec["family"]

    if family == "fixed":
        result = np.full_like(u, spec["value"], dtype=float)

    elif family == "pert" and spec["min"] == spec["max"]:
        result = np.full_like(u, spec["min"], dtype=float)

    elif family in ("pert", "gamma", "lognormal",
                    "beta", "standard_normal"):
        dist = _get_distribution(spec)
        result = dist.ppf(u)

    else:
        raise ValueError(
            f"Inverse CDF is restricted to continuous variables: {family}"
        )

    return result.item() if result.ndim == 0 else result


def distribution_mean(spec):
    """Return the theoretical mean of a configured distribution."""
    family = spec["family"]

    if family == "fixed":
        return float(spec["value"])

    if family == "pert" and spec["min"] == spec["max"]:
        return float(spec["min"])

    dist = _get_distribution(spec)
    return float(dist.mean())
