
"""Tests for NovaFlow distribution utilities."""

import numpy as np
import pytest

from config.model_config import pert, gamma, lognormal, beta
from src.distributions import (
    sample_distribution,
    inverse_cdf,
    distribution_mean,
)


def test_pert_theoretical_mean():
    spec = pert(100, 300, 800)
    assert distribution_mean(spec) == pytest.approx(350.0)


def test_pert_bounds():
    rng = np.random.default_rng(123)
    spec = pert(100, 300, 800)
    draws = sample_distribution(spec, rng, size=100_000)

    assert np.all(draws >= 100)
    assert np.all(draws <= 800)


def test_pert_sample_mean():
    rng = np.random.default_rng(123)
    spec = pert(100, 300, 800)
    draws = sample_distribution(spec, rng, size=100_000)

    assert np.mean(draws) == pytest.approx(350, abs=3)


def test_gamma_mean():
    spec = gamma(4, 25)
    assert distribution_mean(spec) == pytest.approx(100)


def test_beta_mean():
    spec = beta(0.36, 4.48)
    expected = 0.36 / (0.36 + 4.48)
    assert distribution_mean(spec) == pytest.approx(expected)


def test_lognormal_median():
    spec = lognormal(1_000_000, 0.80)
    assert inverse_cdf(spec, 0.5) == pytest.approx(1_000_000)


def test_inverse_cdf_shape():
    spec = pert(100, 300, 800)
    u = np.array([0.1, 0.5, 0.9])
    values = inverse_cdf(spec, u)

    assert values.shape == (3,)
    assert np.all(np.diff(values) > 0)


def test_inverse_cdf_invalid_probability():
    with pytest.raises(ValueError):
        inverse_cdf(pert(100, 300, 800), 1.2)


def test_reproducibility():
    spec = gamma(4, 25)

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    draws1 = sample_distribution(spec, rng1, size=1000)
    draws2 = sample_distribution(spec, rng2, size=1000)

    assert np.array_equal(draws1, draws2)


def test_fixed_distribution():
    spec = {"family": "fixed", "value": 103.0}
    rng = np.random.default_rng(42)

    draws = sample_distribution(spec, rng, size=10)

    assert np.all(draws == 103.0)
    assert distribution_mean(spec) == 103.0
