
import pytest

from config.model_config import get_config
from src.control_effectiveness import make_control_config, expected_lef


def test_baseline_unchanged():
    original = get_config()
    config, meta = make_control_config([])

    assert config == original
    assert meta["annual_control_cost_usd"] == 0


def test_frequency_reduction():
    base = expected_lef(get_config())

    cfg, _ = make_control_config([
        "C1_oauth_governance",
        "C3_saas_detection",
    ])

    assert expected_lef(cfg) == pytest.approx(
        base * 0.8 * 0.85
    )

    assert cfg["frequency"]["campaigns_per_year"]["lam"] == 3


def test_exposure_reduction():
    cfg, _ = make_control_config([
        "C2_least_privilege",
        "C4_data_protection",
    ])

    assert cfg["reputation"]["affected_customers"]["mode"] == (
        pytest.approx(1500 * 0.75 * 0.8)
    )

    assert cfg["probabilities"]["affected_churn"]["intercept"] is None


def test_response_hours_only():
    cfg, _ = make_control_config(["C5_ir_readiness"])

    assert cfg["primary_response"]["technical_response_hours"]["mode"] == (
        pytest.approx(240)
    )

    assert cfg["primary_response"]["external_dfir_cost_usd"]["mode"] == 80000

    assert (
        cfg["secondary_response"]["legal"]["stages"]["initial_triage"]
        ["outside_hours"]["scale"]
        == 25
    )


def test_invalid_control_and_duplicate():
    with pytest.raises(ValueError):
        make_control_config(["unknown"])

    with pytest.raises(ValueError):
        make_control_config(["C1_oauth_governance"] * 2)
