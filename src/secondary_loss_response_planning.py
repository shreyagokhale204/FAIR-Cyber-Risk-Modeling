
"""FAIR secondary loss: legal, notification, support, PR, and planning."""

import numpy as np

from src.distributions import sample_distribution
from src.severity import copula_transform


# Explicit mapping from legal stage names to copula correlation keys.
LEGAL_CORRELATION_PREFIXES = {
    "initial_triage": "legal_initial",
    "customer_notification": "legal_notification",
    "regulatory_engagement": "legal_regulator",
    "remediation_and_contracts": "legal_remediation",
}


def simulate_secondary_response_planning(
    severity, affected_customers, indicators, config, rng
):
    """Simulate secondary response/planning costs per successful incident.

    All inputs refer to the SAME incidents. Severity, affected customers,
    and indicators must be generated once and shared across loss modules.

    Returns a dictionary of NumPy arrays, one value per incident.
    """
    z = np.atleast_1d(np.asarray(severity, dtype=float))
    affected_raw = np.atleast_1d(np.asarray(affected_customers))

    if z.ndim != 1 or affected_raw.shape != z.shape:
        raise ValueError(
            "Severity and affected customers must be matching 1D arrays."
        )

    if not np.all(np.isfinite(z)):
        raise ValueError("Severity values must be finite.")

    if not np.all(np.isfinite(affected_raw)):
        raise ValueError("Affected customer counts must be finite.")

    total_customers = config["organization"]["enterprise_customers"]

    if np.any(affected_raw != np.floor(affected_raw)):
        raise ValueError("Affected customer counts must be integers.")

    if np.any((affected_raw < 0) | (affected_raw > total_customers)):
        raise ValueError("Affected customer counts are out of range.")

    affected = affected_raw.astype(int)
    n = len(z)

    secondary = config["secondary_response"]
    correlations = config["dependence"]["correlations_latent_normal"]

    def indicator(name):
        """Retrieve a shared binary indicator with shape validation."""
        if name not in indicators:
            raise KeyError(f"Missing incident indicator: {name}")

        values = np.asarray(indicators[name], dtype=bool)

        if values.ndim == 0:
            return np.full(n, bool(values), dtype=bool)

        if values.shape != (n,):
            raise ValueError(
                f"Indicator '{name}' must have shape ({n},)."
            )

        return values

    def draw(spec, correlation_key=None):
        """Sample a distribution, using copula dependence when specified."""
        if correlation_key is not None:
            if correlation_key not in correlations:
                raise KeyError(
                    f"Missing copula correlation: {correlation_key}"
                )

            values = copula_transform(
                severity=z,
                rho=correlations[correlation_key],
                spec=spec,
                rng=rng,
            )
        else:
            values = sample_distribution(spec, rng, size=n)

        return np.asarray(values, dtype=float)

    def resolve_gate(gate):
        """Translate a configuration gate into a boolean array."""
        if gate in (None, "none", "none_provisional"):
            return np.ones(n, dtype=bool)

        return indicator(gate)

    result = {}

    # =========================================================
    # A. LEGAL RESPONSE
    # =========================================================
    legal = secondary["legal"]

    # One counsel rate per incident, shared across all legal stages.
    counsel_rate = draw(legal["outside_counsel_hourly_rate_usd"])
    internal_rate = legal["internal_hourly_rate_usd"]

    total_outside_hours = np.zeros(n, dtype=float)
    total_internal_hours = np.zeros(n, dtype=float)

    for stage_name, stage in legal["stages"].items():

        if "gate" in stage:
            active = resolve_gate(stage["gate"])

        elif "occurrence" in stage:
            active = draw(stage["occurrence"]).astype(bool)

        else:
            active = np.ones(n, dtype=bool)

        prefix = LEGAL_CORRELATION_PREFIXES[stage_name]

        outside_hours = draw(
            stage["outside_hours"],
            f"{prefix}_outside_hours",
        )

        internal_hours = draw(
            stage["internal_hours"],
            f"{prefix}_internal_hours",
        )

        # Potential workload is drawn first, then occurrence is applied.
        outside_hours = np.where(active, outside_hours, 0.0)
        internal_hours = np.where(active, internal_hours, 0.0)

        result[f"legal_{stage_name}_active"] = active
        result[f"legal_{stage_name}_outside_hours"] = outside_hours
        result[f"legal_{stage_name}_internal_hours"] = internal_hours

        total_outside_hours += outside_hours
        total_internal_hours += internal_hours

    outside_legal_cost = total_outside_hours * counsel_rate
    internal_legal_cost = total_internal_hours * internal_rate

    legal_total = outside_legal_cost + internal_legal_cost

    result.update({
        "outside_counsel_hourly_rate_usd": counsel_rate,
        "legal_total_outside_hours": total_outside_hours,
        "legal_total_internal_hours": total_internal_hours,
        "legal_outside_counsel_cost_usd": outside_legal_cost,
        "legal_internal_cost_usd": internal_legal_cost,
        "legal_total_usd": legal_total,
    })

    # =========================================================
    # B. CUSTOMER NOTIFICATION
    # =========================================================
    notification = secondary["notification"]
    notification_active = resolve_gate(notification["gate"])

    coordination_hours = draw(
        notification["coordination_hours"],
        "notification_coordination_hours",
    )

    coordination_hours = np.where(
        notification_active, coordination_hours, 0.0
    )

    coordination_cost = (
        coordination_hours
        * notification["coordination_hourly_rate_usd"]
    )

    emails_sent = np.where(
        notification_active,
        affected * notification["emails_per_notified_customer"],
        0,
    )

    email_cost = (
        emails_sent
        * notification["email_cost_per_1000_messages_usd"]
        / 1000.0
    )

    physical_mail_cost = np.zeros(n, dtype=float)

    if notification["physical_mail_in_base_case"]:
        mail_unit_cost = draw(
            notification["physical_mail_cost_per_customer_usd"]
        )
        physical_mail_cost = np.where(
            notification_active,
            affected * mail_unit_cost,
            0.0,
        )

    notification_total = (
        coordination_cost + email_cost + physical_mail_cost
    )

    result.update({
        "notification_coordination_hours": coordination_hours,
        "notification_emails_sent": emails_sent,
        "notification_coordination_cost_usd": coordination_cost,
        "notification_email_cost_usd": email_cost,
        "notification_physical_mail_cost_usd": physical_mail_cost,
        "notification_total_usd": notification_total,
    })

    # =========================================================
    # C. CUSTOMER SUPPORT
    # =========================================================
    support = secondary["customer_support"]
    support_active = resolve_gate(support["gate"])

    # One customer-contact probability per incident.
    contact_probability = draw(support["contact_probability"])

    contacts = rng.binomial(
        n=affected,
        p=contact_probability,
    )

    contacts = np.where(support_active, contacts, 0)

    duration = support["call_duration_minutes"]

    if duration["family"] != "gamma":
        raise ValueError(
            "Aggregated call duration requires a Gamma distribution."
        )

    # Sum of N independent Gamma(k, theta) variables:
    # Gamma(N*k, theta).
    total_call_minutes = np.zeros(n, dtype=float)
    has_contacts = contacts > 0

    total_call_minutes[has_contacts] = rng.gamma(
        shape=duration["shape"] * contacts[has_contacts],
        scale=duration["scale"],
    )

    call_cost = (
        total_call_minutes / 60.0
        * support["hourly_rate_usd"]
    )

    setup_hours = draw(
        support["setup_hours"],
        "support_setup_hours",
    )

    setup_active = resolve_gate(support["setup_gate"])

    setup_hours = np.where(
        setup_active, setup_hours, 0.0
    )

    setup_cost = setup_hours * support["hourly_rate_usd"]
    support_total = setup_cost + call_cost

    result.update({
        "support_contact_probability": contact_probability,
        "support_contacts": contacts,
        "support_total_call_minutes": total_call_minutes,
        "support_call_cost_usd": call_cost,
        "support_setup_hours": setup_hours,
        "support_setup_cost_usd": setup_cost,
        "support_total_usd": support_total,
    })

    # =========================================================
    # D. PUBLIC RELATIONS
    # =========================================================
    pr = secondary["public_relations"]

    internal_pr_hours = draw(
        pr["internal_hours"],
        "pr_internal_hours",
    )

    internal_pr_active = resolve_gate(pr["internal_hours_gate"])

    internal_pr_hours = np.where(
        internal_pr_active, internal_pr_hours, 0.0
    )

    internal_pr_cost = (
        internal_pr_hours * pr["internal_hourly_rate_usd"]
    )

    external_pr_active = resolve_gate(pr["external_agency_gate"])

    agency_fee = draw(pr["external_agency_fee_usd"])

    external_pr_cost = np.where(
        external_pr_active, agency_fee, 0.0
    )

    pr_total = internal_pr_cost + external_pr_cost

    result.update({
        "pr_internal_hours": internal_pr_hours,
        "pr_internal_cost_usd": internal_pr_cost,
        "pr_external_agency_cost_usd": external_pr_cost,
        "pr_total_usd": pr_total,
    })

    # =========================================================
    # E. PLANNING
    # =========================================================
    planning = secondary["planning"]
    planning_active = resolve_gate(planning["gate"])

    planning_hours = draw(
        planning["hours"],
        "secondary_planning_hours",
    )

    planning_hours = np.where(
        planning_active, planning_hours, 0.0
    )

    planning_cost = (
        planning_hours * planning["hourly_rate_usd"]
    )

    result.update({
        "planning_hours": planning_hours,
        "planning_total_usd": planning_cost,
    })

    # =========================================================
    # F. TOTAL SECONDARY RESPONSE / PLANNING LOSS
    # =========================================================
    total = (
        legal_total
        + notification_total
        + support_total
        + pr_total
        + planning_cost
    )

    result["secondary_response_planning_loss_usd"] = total

    return result
