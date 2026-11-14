
"""Tests for shared severity and Gaussian copula."""

import numpy as np
import pytest
from scipy.stats import norm, spearmanr

from config.model_config import pert
from src.severity import (
    draw_severity,
    copula_transform,
    draw_correlated_variables,
)


def test_severity_is_standard_normal():
    rng = np.random.default_rng(123)
    z = draw_severity(rng, size=100_000)

    assert np.mean(z) == pytest.approx(0, abs=0.02)
    assert np.std(z) == pytest.approx(1, abs=0.02)


def test_copula_preserves_pert_mean():
    rng = np.random.default_rng(123)
    z = draw_severity(rng, size=100_000)

    x = copula_transform(z, 0.6, pert(100, 300, 800), rng)

    assert np.mean(x) == pytest.approx(350, abs=3)
    assert np.all((x >= 100) & (x <= 800))


def test_positive_dependence():
    rng = np.random.default_rng(123)
    z = draw_severity(rng, size=100_000)

    x = copula_transform(z, 0.6, pert(100, 300, 800), rng)

    rank_corr = spearmanr(z, x).statistic

    assert 0.50 < rank_corr < 0.65


def test_zero_rho_independence():
    rng = np.random.default_rng(123)
    z = draw_severity(rng, size=100_000)

    x = copula_transform(z, 0.0, pert(100, 300, 800), rng)

    assert abs(np.corrcoef(z, x)[0, 1]) < 0.02


def test_shared_severity_correlates_variables():
    rng = np.random.default_rng(123)
    z = draw_severity(rng, size=100_000)

    specs = {
        "hours": pert(100, 300, 800),
        "cost": pert(13_000, 80_000, 250_000),
    }
    rhos = {"hours": 0.6, "cost": 0.6}

    values = draw_correlated_variables(z, specs, rhos, rng)

    correlation = np.corrcoef(values["hours"], values["cost"])[0, 1]

    # Expected latent correlation is 0.6 * 0.6 = 0.36.
    # Transformed-value Pearson correlation is not exactly 0.36.
    assert 0.25 < correlation < 0.45


def test_invalid_rho():
    rng = np.random.default_rng(123)

    with pytest.raises(ValueError):
        copula_transform(0.0, 1.5, pert(100, 300, 800), rng)
