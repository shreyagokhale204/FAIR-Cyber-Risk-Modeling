
"""FAIR Monte Carlo model: centralized, declarative configuration.

Step 1 only: no random draws, logistic calibration, or loss calculations occur here.
Distribution parameterizations:
  pert: min, mode, max, shape_lambda (modified beta PERT)
  gamma: shape, scale (NOT rate)
  lognormal: median, sigma (natural-log standard deviation)
  beta: alpha, beta
  poisson: lam
  binomial: n, p
All monetary values are USD, hours are person-hours, and rates are USD/hour.
Odds ratios refer to a +1 SD increase in latent technical severity, or to a
binary predictor changing from 0 to 1. Logistic intercepts must be calibrated.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite


def pert(low: float, mode: float, high: float, shape_lambda: float = 4.0) -> dict:
    return {"family": "pert", "min": low, "mode": mode, "max": high,
            "shape_lambda": shape_lambda}


def gamma(shape: float, scale: float) -> dict:
    return {"family": "gamma", "shape": shape, "scale": scale}


def lognormal(median: float, sigma: float) -> dict:
    return {"family": "lognormal", "median": median, "sigma": sigma}


def beta(alpha: float, beta_: float) -> dict:
    return {"family": "beta", "alpha": alpha, "beta": beta_}


def fixed(value: float) -> dict:
    return {"family": "fixed", "value": value}


def logistic(target: float, severity_or: float = 1.0,
             visibility_or: float = 1.0, regulator_or: float = 1.0,
             target_type: str = "unconditional") -> dict:
    return {"family": "calibrated_logistic", "target": target,
            "target_type": target_type, "severity_or": severity_or,
            "visibility_or": visibility_or, "regulator_or": regulator_or,
            "intercept": None}


CONFIG = {
    "metadata": {
        "model_name": "Saas firm third-party token compromise / Salesforce exfiltration",
        "version": "1.0-step1",
        "currency": "USD",
        "time_horizon": "one_year",
        "scenario": "Vendor credential/OAuth-token theft, Salesforce access, customer-data exfiltration",
        "scope": "Confidentiality only; no operational downtime or availability loss",
        "assumptions_status": "Illustrative research model; not empirically fitted",
    },
    "organization": {
        "annual_revenue_usd": 900_000_000,
        "employees": 3_200,
        "enterprise_customers": 3_000,
        "average_annual_customer_revenue_usd": 300_000,
        "contribution_margin": 0.70,
        "contribution_per_lost_customer_or_deal_usd": 210_000,
    },
    "simulation": {
        "n_years": 100_000,
        "random_seed": 20261025,
        "chunk_size_years": 10_000,
        "quantiles": [0.90, 0.95, 0.99, 0.995],
        "retain_incident_level_results": True,
        "zero_incident_year_loss_usd": 0.0,
    },
    "frequency": {
        "campaigns_per_year": {"family": "poisson", "lam": 3.0},  
        "campaign_success_probability": beta(0.36, 4.48),    #E = 0.36 / (0.36 + 4.48) ≈ 7.5%
        "success_probability_draw_scope": "each_campaign", 
        "successful_incidents_per_campaign_max": 1,
        "frequency_severity_independent": True,
        "note": "Per-campaign independent Beta draws followed by one Bernoulli do not induce annual count overdispersion; the marginal annual count is Poisson-thinned.",
    },
    "dependence": {
        "severity_latent": {"family": "standard_normal", "draw_scope": "each_successful_incident"},
        "copula": "one_factor_gaussian",
        "independent_residual_per_continuous_variable": True,
        "correlations_latent_normal": {
            "affected_customers": 0.60,
            "technical_response_hours": 0.60,
            "it_security_management_hours": 0.50,
            "executive_response_hours": 0.40,
            "external_dfir_cost_usd": 0.60,
            "legal_initial_outside_hours": 0.50,
            "legal_initial_internal_hours": 0.50,
            "legal_notification_outside_hours": 0.50,
            "legal_notification_internal_hours": 0.50,
            "legal_regulator_outside_hours": 0.50,
            "legal_regulator_internal_hours": 0.50,
            "legal_remediation_outside_hours": 0.50,
            "legal_remediation_internal_hours": 0.50,
            "notification_coordination_hours": 0.40,
            "support_setup_hours": 0.40,
            "pr_internal_hours": 0.40,
            "secondary_planning_hours": 0.40,
        },
        "note": "Rhos are provisional latent-normal correlations, not correlations of realized dollar costs.",
    },
    "primary_response": {
        "technical_response_hours": pert(100, 300, 800),
        "technical_hourly_rate_usd": 103.00,
        "it_security_management_hours": pert(20, 60, 160),
        "it_security_management_hourly_rate_usd": 150.00,
        "executive_response_hours": pert(5, 15, 40),
        "executive_hourly_rate_usd": 200.00,
        "external_dfir_cost_usd": pert(13_000, 80_000, 250_000),
    },
    "secondary_response": {
        "legal": {
            "outside_counsel_hourly_rate_usd": lognormal(650, 0.30),
            "outside_counsel_rate_draw_scope": "once_per_incident",
            "internal_hourly_rate_usd": 118.80,
            "stages": {
                "initial_triage": {
                    "occurrence": {"family": "bernoulli", "p": 0.95},
                    "outside_hours": gamma(4, 25),
                    "internal_hours": gamma(4, 20),
                },
                "customer_notification": {
                    "gate": "notification",
                    "outside_hours": gamma(3, 20),
                    "internal_hours": gamma(3, 10),
                },
                "regulatory_engagement": {
                    "gate": "regulatory_engagement",
                    "outside_hours": gamma(1.5, 200),
                    "internal_hours": gamma(2, 50),
                },
                "remediation_and_contracts": {
                    "occurrence": {"family": "bernoulli", "p": 0.90},
                    "outside_hours": gamma(3, 15),
                    "internal_hours": gamma(3, 15),
                },
            },
        },
        "notification": {
            "gate": "notification",
            "coordination_hours": pert(24, 60, 120),
            "coordination_hourly_rate_usd": 84.28,
            "email_cost_per_1000_messages_usd": 0.16,
            "emails_per_notified_customer": 1,
            "customer_count_basis": "affected_customers",
            "physical_mail_in_base_case": False,
            "physical_mail_cost_per_customer_usd": pert(0.95, 2.72, 4.60),
        },
        "customer_support": {
            "gate": "notification",
            "contact_probability": beta(2, 18),
            "contact_count_distribution": "binomial_affected_customers",
            "call_duration_minutes": gamma(4, 5),
            "call_duration_draw_scope": "per_contact",
            "setup_hours": pert(16, 40, 80),
            "hourly_rate_usd": 48.85,
            "setup_gate": "notification",
        },
        "public_relations": {
            "internal_hours": gamma(3, 20),
            "internal_hourly_rate_usd": 100.04,
            "internal_hours_gate": "none_provisional",
            "external_agency_gate": "external_pr_agency",
            "external_agency_fee_usd": lognormal(15_000, 0.50),
        },
        "planning": {
            "gate": "notification",
            "hours": pert(4, 16, 40),
            "hourly_rate_usd": 103.07,
        },
    },
    "fines_and_judgments": {
        "regulatory_fine": {
            "gate": "regulatory_engagement",
            "occurrence_indicator": "regulatory_fine",
            "amount_if_fined_usd": lognormal(1_000_000, 0.80),
        },
        "judgment_or_settlement": {
            "gate": "judgment_or_settlement",
            "amount_if_incurred_usd": lognormal(500_000, 1.00),
        },
        "amounts_severity_dependent": False,
    },
    "reputation": {
        "affected_customers": pert(300, 1_500, 3_000),
        "affected_count_rounding": "round_to_nearest_integer_and_clip_0_to_customer_total",
        "unaffected_customers": "enterprise_customers_minus_affected_customers",
        "churn": {
            "affected": {"distribution": "binomial", "trials": "affected_customers", "probability_model": "affected_churn"},
            "unaffected": {"distribution": "binomial", "trials": "unaffected_customers", "probability_model": "unaffected_churn"},
        },
        "otherwise_won_opportunities": {
            "family": "binomial", "n": 180, "p": 0.20,
        },
        "lost_deals": {"distribution": "binomial", "trials": "otherwise_won_opportunities", "probability_model": "lost_deal"},
        "loss_per_churned_customer_usd": 210_000,
        "loss_per_lost_deal_usd": 210_000,
        "valuation_horizon": "one_year_contribution_margin_only",
        "note": "No additional multi-year customer lifetime value or double counting of lost deals and churn.",
    },
    "probabilities": {
        "public_visibility": logistic(0.30, severity_or=2.0),
        "notification": logistic(0.75, severity_or=2.0),
        "regulatory_engagement": logistic(0.10, severity_or=2.0),
        "external_pr_agency": logistic(0.30, severity_or=1.5, visibility_or=3.0),
        "regulatory_fine": logistic(0.05, severity_or=2.0,
                                    target_type="unconditional_with_regulatory_engagement_gate"),
        "judgment_or_settlement": logistic(0.05, severity_or=2.0, visibility_or=3.0),
        "affected_churn": logistic(0.002, severity_or=1.5, visibility_or=2.0,
                                   target_type="customer_weighted"),
        "unaffected_churn": logistic(0.0002, severity_or=1.3, visibility_or=2.0,
                                     target_type="customer_weighted"),
        "lost_deal": logistic(0.01, severity_or=1.5, visibility_or=3.0),
    },
    "calibration": {
        "method": "numerical_root_finding_with_normal_quadrature",
        "logistic_intercepts_are_pre_simulation_outputs": True,
        "fine_calibration": "E[I_reg * logistic(a_f + ln(2)*Zs)] = 0.05",
        "affected_churn_calibration": "E[A * p_affected(Zs,V)] / E[A] = 0.002",
        "unaffected_churn_calibration": "E[(3000-A) * p_unaffected(Zs,V)] / E[3000-A] = 0.0002",
        "visibility_integrated_over_conditional_distribution": True,
        "numerical_probability_tolerance": 1e-5,
        "note": "Intercepts None until Step 3; 50% is only the average fine probability conditional on regulatory engagement, not a fixed per-incident rate.",
    },
    "sensitivity": {
        "severity_correlation_multipliers": [0.0, 1.0, 1.5],
        "cap_absolute_correlation_below_one": True,
        "visibility_unconditional_probability": [0.15, 0.30, 0.60],
        "visibility_severity_odds_ratio": [1.0, 2.0, 3.0],
        "visibility_churn_odds_ratio_both_groups": [1.0, 2.0, 3.0],
        "unaffected_visibility_odds_ratio_alternative": 3.0,
        "physical_mail_toggle": [False, True],
        "frequency_underreporting_adjustments": "to_be_specified_separately",
    },
    "open_decisions": {
        "pr_internal_hours_gate": "Currently always incurred per incident; confirm whether this should be gated by publicity/PR response.",
        "support_setup_and_contacts": "Currently notification-gated; confirm whether some support arises without formal notification.",
        "regulator_fine_scope": "Assumes every modeled fine requires formal regulatory engagement; review against intended definition.",
        "regulatory_fine_target_feasibility": "5% overall fine rate with 10% engagement implies 50% average fine rate among engaged incidents; validate externally.",
        "rho_values": "Provisional; perform sensitivity analysis.",
        "logistic_odds_ratios": "Provisional analyst assumptions; perform sensitivity analysis.",
        "customer_churn_and_lost_deal_rates": "Illustrative targets, not estimated causal effects.",
        "notification_and_regulator_joint_logic": "Currently both severity-dependent but not logically nested; consider jurisdictional rules if needed.",
    },
}


def get_config() -> dict:
    """Return a deep copy so scenarios cannot mutate the base configuration."""
    return deepcopy(CONFIG)


def validate_config(cfg: dict | None = None) -> None:
    """Check internal parameter consistency; not empirical model validation."""
    cfg = CONFIG if cfg is None else cfg
    assert cfg["organization"]["enterprise_customers"] > 0
    assert cfg["simulation"]["n_years"] > 0
    assert cfg["organization"]["contribution_per_lost_customer_or_deal_usd"] == (
        cfg["organization"]["average_annual_customer_revenue_usd"] *
        cfg["organization"]["contribution_margin"]
    )
    rhos = cfg["dependence"]["correlations_latent_normal"]
    assert all(isfinite(rho) and -1 <= rho <= 1 for rho in rhos.values())
    probs = cfg["probabilities"]
    assert all(0 < p["target"] < 1 for p in probs.values())
    assert all(p[k] > 0 for p in probs.values() for k in
               ("severity_or", "visibility_or", "regulator_or"))
    assert probs["regulatory_fine"]["target"] <= probs["regulatory_engagement"]["target"], (
        "A gated fine cannot be more frequent than regulatory engagement."
    )
    assert cfg["fines_and_judgments"]["regulatory_fine"]["gate"] == "regulatory_engagement"
    assert cfg["reputation"]["affected_customers"]["max"] <= cfg["organization"]["enterprise_customers"]
    assert cfg["secondary_response"]["legal"]["stages"]["regulatory_engagement"]["gate"] == "regulatory_engagement"
    for q in cfg["simulation"]["quantiles"]:
        assert 0 < q < 1


if __name__ == "__main__":
    validate_config()
    q = CONFIG["frequency"]["campaign_success_probability"]
    expected_lef = CONFIG["frequency"]["campaigns_per_year"]["lam"] * q["alpha"] / (q["alpha"] + q["beta"])
    print("Configuration validated.")
    print(f"Expected annual successful incident frequency: {expected_lef:.6f}")
    print(f"Configured probability models: {len(CONFIG['probabilities'])}")
    print(f"Configured copula-linked variables: {len(CONFIG['dependence']['correlations_latent_normal'])}")
    print("Note: logistic intercepts will be calibrated in Step 3.")
