
import numpy as np
import pandas as pd
import pytest

from src.model_validation_convergence import (
    METRICS,
    run_convergence_analysis,
    run_repeated_simulations,
)


@pytest.fixture(scope="module")
def convergence_results():
    return run_convergence_analysis(
        sample_sizes=(100, 300, 500),
        seed=42,
    )


@pytest.fixture(scope="module")
def repeated_results():
    return run_repeated_simulations(
        n_years=500,
        n_repetitions=4,
        base_seed=100,
    )


def test_convergence_returns_dataframe(
    convergence_results,
):
    assert isinstance(
        convergence_results,
        pd.DataFrame,
    )


def test_convergence_sample_sizes(
    convergence_results,
):
    assert convergence_results[
        "n_years"
    ].tolist() == [100, 300, 500]


def test_convergence_metrics_present(
    convergence_results,
):
    for metric in METRICS:
        assert metric in convergence_results.columns


def test_convergence_metrics_nonnegative(
    convergence_results,
):
    for metric in METRICS:
        assert (
            convergence_results[metric] >= 0
        ).all()


def test_convergence_reproducibility():
    a = run_convergence_analysis(
        sample_sizes=(100,),
        seed=123,
    )

    b = run_convergence_analysis(
        sample_sizes=(100,),
        seed=123,
    )

    pd.testing.assert_frame_equal(a, b)


def test_invalid_sample_size():
    with pytest.raises(ValueError):
        run_convergence_analysis(
            sample_sizes=(0,),
        )


def test_repetitions_count(
    repeated_results,
):
    repetitions, _ = repeated_results

    assert len(repetitions) == 4


def test_repetitions_have_unique_seeds(
    repeated_results,
):
    repetitions, _ = repeated_results

    assert repetitions["seed"].nunique() == 4


def test_uncertainty_metrics_present(
    repeated_results,
):
    _, uncertainty = repeated_results

    assert set(uncertainty["metric"]) == set(
        METRICS
    )


def test_uncertainty_intervals_valid(
    repeated_results,
):
    _, uncertainty = repeated_results

    assert (
        uncertainty["ci_95_lower"]
        <= uncertainty["mean_estimate"]
    ).all()

    assert (
        uncertainty["mean_estimate"]
        <= uncertainty["ci_95_upper"]
    ).all()


def test_uncertainty_standard_deviations_nonnegative(
    repeated_results,
):
    _, uncertainty = repeated_results

    assert (
        uncertainty["sd_across_runs"] >= 0
    ).all()


def test_invalid_repetition_count():
    with pytest.raises(ValueError):
        run_repeated_simulations(
            n_years=100,
            n_repetitions=1,
        )


def test_repeated_simulation_reproducibility():
    a_runs, a_uncertainty = run_repeated_simulations(
        n_years=100,
        n_repetitions=3,
        base_seed=123,
    )

    b_runs, b_uncertainty = run_repeated_simulations(
        n_years=100,
        n_repetitions=3,
        base_seed=123,
    )

    pd.testing.assert_frame_equal(
        a_runs,
        b_runs,
    )

    pd.testing.assert_frame_equal(
        a_uncertainty,
        b_uncertainty,
    )
