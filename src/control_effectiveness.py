"""Step 8B: illustrative security-control scenarios for NovaFlow FAIR model."""
from copy import deepcopy
from math import isfinite

from config.model_config import get_config, validate_config

CONTROL_ASSUMPTIONS = {
    "C1_oauth_governance": {"parameter": "success", "low": .10, "central": .20, "high": .35, "annual_cost_usd": 75000},
    "C2_least_privilege": {"parameter": "exposure", "low": .10, "central": .25, "high": .40, "annual_cost_usd": 90000},
    "C3_saas_detection": {"parameter": "success", "low": .05, "central": .15, "high": .30, "annual_cost_usd": 120000},
    "C4_data_protection": {"parameter": "exposure", "low": .10, "central": .20, "high": .35, "annual_cost_usd": 110000},
    "C5_ir_readiness": {"parameter": "response", "low": .10, "central": .20, "high": .30, "annual_cost_usd": 50000},
}

# Keep the initial IR scope conservative: effort hours, not DFIR fees or penalties.
RESPONSE_PATHS = (
    ("primary_response", "technical_response_hours"),
    ("primary_response", "it_security_management_hours"),
    ("primary_response", "executive_response_hours"),
    ("secondary_response", "notification", "coordination_hours"),
    ("secondary_response", "customer_support", "setup_hours"),
    ("secondary_response", "planning", "hours"),
)


def _scale_pert(spec, multiplier):
    if spec["family"] != "pert":
        raise ValueError("Expected a PERT distribution")
    for key in ("min", "mode", "max"):
        spec[key] *= multiplier


def make_control_config(controls=(), level="central", base_config=None):
    """Return (new_config, metadata). All control effects are multiplicative."""
    if level not in ("low", "central", "high"):
        raise ValueError("level must be low, central, or high")
    if isinstance(controls, str):
        controls = [controls]
    controls = tuple(controls)
    if len(set(controls)) != len(controls):
        raise ValueError("Duplicate control selected")
    if any(c not in CONTROL_ASSUMPTIONS for c in controls):
        raise ValueError("Unknown control")
    cfg = deepcopy(get_config() if base_config is None else base_config)
    multipliers = {"success": 1.0, "exposure": 1.0, "response": 1.0}
    cost = 0.0
    for control in controls:
        item = CONTROL_ASSUMPTIONS[control]
        reduction = item[level]
        if not isfinite(reduction) or not 0 <= reduction < 1:
            raise ValueError("Invalid control effectiveness")
        multipliers[item["parameter"]] *= 1 - reduction
        cost += item["annual_cost_usd"]

    # Scaling alpha with fixed alpha+beta preserves the original Beta concentration.
    spec = cfg["frequency"]["campaign_success_probability"]
    concentration = spec["alpha"] + spec["beta"]
    old_mean = spec["alpha"] / concentration
    new_mean = old_mean * multipliers["success"]
    spec["alpha"] = concentration * new_mean
    spec["beta"] = concentration * (1 - new_mean)

    exposure = cfg["reputation"]["affected_customers"]
    _scale_pert(exposure, multipliers["exposure"])
    # Customer-weighted churn calibration must be redone after exposure changes.
    for probability in cfg["probabilities"].values():
        if probability.get("family") == "calibrated_logistic":
            probability["intercept"] = None

    for path in RESPONSE_PATHS:
        node = cfg
        for part in path:
            node = node[part]
        _scale_pert(node, multipliers["response"])

    validate_config(cfg)
    return cfg, {**multipliers, "annual_control_cost_usd": cost, "controls": controls, "level": level}


def expected_lef(cfg):
    campaigns = cfg["frequency"]["campaigns_per_year"]["lam"]
    success = cfg["frequency"]["campaign_success_probability"]
    return campaigns * success["alpha"] / (success["alpha"] + success["beta"])


def evaluate_controls(controls=(), level="central", n_incidents=20000, seed=20261112):
    """Estimate AAL using simulated incident mean x analytical expected LEF."""
    from src.incident_simulation import simulate_incidents
    cfg, meta = make_control_config(controls, level)
    incident_df = simulate_incidents(n_incidents=n_incidents, seed=seed, config=cfg, include_details=False)
    mean_loss = float(incident_df["total_incident_loss_usd"].mean())
    lef = expected_lef(cfg)
    return {"controls": ", ".join(meta["controls"]) or "Baseline",
            "level": level, "expected_lef": lef,
            "mean_incident_loss_usd": mean_loss,
            "aal_usd": lef * mean_loss,
            "annual_control_cost_usd": meta["annual_control_cost_usd"]}
