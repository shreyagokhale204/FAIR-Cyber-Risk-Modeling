
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NovaFlow | Cyber Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

CONTROL_NAMES = {
    "C1_oauth_governance": "C1 · OAuth Governance",
    "C2_least_privilege": "C2 · Least Privilege",
    "C3_saas_detection": "C3 · SaaS Detection",
    "C4_data_protection": "C4 · Data Protection",
    "C5_ir_readiness": "C5 · IR Readiness",
}

CONTROL_COSTS = {
    "C1_oauth_governance": {
        "low": 15000, "central": 30000, "high": 75000
    },
    "C2_least_privilege": {
        "low": 20000, "central": 35000, "high": 90000
    },
    "C3_saas_detection": {
        "low": 30000, "central": 60000, "high": 120000
    },
    "C4_data_protection": {
        "low": 25000, "central": 50000, "high": 110000
    },
    "C5_ir_readiness": {
        "low": 10000, "central": 25000, "high": 50000
    },
}


def money(value):
    return f"${value:,.0f}"


def signed_money(value):
    return f"{value:+,.0f} USD"


def short_controls(value):
    if pd.isna(value) or value == "None":
        return "No controls"

    return " + ".join(
        item.strip().split("_")[0]
        for item in str(value).split(",")
    )


# --------------------------------------------------
# 2. LOAD AND VALIDATE MODEL OUTPUTS
# --------------------------------------------------

@st.cache_data
def load_data():
    files = {
        "risk": "09_portfolio_risk.csv",
        "financial": "09_portfolio_financial.csv",
        "controls": "control_effectiveness_baseline.csv",
        "cost_sensitivity": "control_cost_sensitivity.csv",
    }

    data = {}

    for key, filename in files.items():
        path = OUTPUTS / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

        data[key] = pd.read_csv(path)

    return data


try:
    data = load_data()
except (FileNotFoundError, pd.errors.ParserError) as exc:
    st.error(str(exc))
    st.stop()

risk_df = data["risk"]
financial_df = data["financial"]
controls_df = data["controls"]
cost_df = data["cost_sensitivity"]

required_risk = {
    "portfolio_id",
    "controls",
    "control_count",
    "residual_aal_usd",
}

if not required_risk.issubset(risk_df.columns):
    st.error(
        "Portfolio risk CSV is missing expected columns: "
        + str(required_risk - set(risk_df.columns))
    )
    st.stop()

if len(risk_df) != 32:
    st.error("Expected 32 control combinations in the saved optimization output.")
    st.stop()

baseline = risk_df.loc[
    risk_df["control_count"] == 0,
    "residual_aal_usd",
].iloc[0]


# --------------------------------------------------
# 3. LIVE PORTFOLIO OPTIMIZATION
# --------------------------------------------------

def get_portfolios(cost_level):
    df = risk_df.copy()

    def portfolio_cost(controls):
        if controls == "None" or pd.isna(controls):
            return 0

        ids = [
            c.strip()
            for c in str(controls).split(",")
        ]

        return sum(
            CONTROL_COSTS[c][cost_level]
            for c in ids
        )

    df["annual_control_cost_usd"] = (
        df["controls"].apply(portfolio_cost)
    )

    df["total_expected_cost_usd"] = (
        df["residual_aal_usd"]
        + df["annual_control_cost_usd"]
    )

    df["risk_reduction_usd"] = (
        baseline - df["residual_aal_usd"]
    )

    df["net_annual_benefit_usd"] = (
        df["risk_reduction_usd"]
        - df["annual_control_cost_usd"]
    )

    df["risk_reduction_pct"] = (
        100 * df["risk_reduction_usd"] / baseline
    )

    df["portfolio_name"] = (
        df["controls"].apply(short_controls)
    )

    return df


def optimize_portfolio(df, budget, objective):
    feasible = df[
        df["annual_control_cost_usd"] <= budget
    ].copy()

    if objective == "Minimize total expected cost":
        metric = "total_expected_cost_usd"
    else:
        metric = "residual_aal_usd"

    return feasible.sort_values(
        [metric, "annual_control_cost_usd", "portfolio_id"]
    ).iloc[0]


# --------------------------------------------------
# 4. SIDEBAR
# --------------------------------------------------

st.sidebar.title("🛡️ NovaFlow")
st.sidebar.caption("Cyber Risk Decision Intelligence")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Cyber Risk Analysis",
        "Security Controls",
        "Investment Optimizer",
        "Uncertainty & Assumptions",
    ],
)

# Investment settings are displayed only on the two pages where
# users can explore security-control costs and optimization choices.
# The Executive Overview uses a fixed, clearly disclosed reference case.
cost_level = "central"
budget = 100000
objective = "Minimize total expected cost"

if page in ("Security Controls", "Investment Optimizer"):
    st.sidebar.divider()
    st.sidebar.subheader("Investment assumptions")

    cost_level = st.sidebar.selectbox(
        "Annual control cost scenario",
        ["central", "low", "high"],
        index=0,
        key="investment_cost_level",
    )

    if page == "Investment Optimizer":
        budget = st.sidebar.slider(
            "Annual investment budget ($)",
            min_value=0,
            max_value=450000,
            value=100000,
            step=5000,
            format="$%d",
            key="investment_budget",
        )
        objective = st.sidebar.radio(
            "Optimization objective",
            [
                "Minimize total expected cost",
                "Minimize residual cyber risk",
            ],
            key="investment_objective",
        )

portfolios = get_portfolios(cost_level)

selected = optimize_portfolio(
    portfolios, budget, objective
)

st.sidebar.caption(
    "Model: FAIR-based Monte Carlo simulation\n\n"
    "Scenario: Third-party OAuth compromise"
)


# --------------------------------------------------
# 5. EXECUTIVE OVERVIEW
# --------------------------------------------------

if page == "Executive Overview":

    st.title("Cyber Risk Intelligence")
    st.caption(
        "NovaFlow | Third-party OAuth compromise "
        "| Annual financial exposure"
    )

    st.info(
        "Decision scenario: An attacker compromises a "
        "third-party SaaS vendor, obtains OAuth credentials, "
        "accesses NovaFlow's Salesforce CRM, and exfiltrates "
        "customer data. The model assumes no operational downtime."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Baseline Annual Risk",
        money(baseline),
    )

    c2.metric(
        "Residual Annual Risk",
        money(selected["residual_aal_usd"]),
        delta=(
            f"-{selected['risk_reduction_pct']:.1f}%"
        ),
        delta_color="normal",
    )

    c3.metric(
        "Annual Investment",
        money(selected["annual_control_cost_usd"]),
    )

    c4.metric(
        "Net Annual Benefit",
        money(selected["net_annual_benefit_usd"]),
    )

    st.divider()

    left, right = st.columns([1.4, 1])

    with left:
        st.subheader("Annual Expected Loss")

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=["Baseline", "Selected Portfolio"],
            y=[
                baseline,
                selected["residual_aal_usd"],
            ],
            text=[
                money(baseline),
                money(selected["residual_aal_usd"]),
            ],
            textposition="auto",
            marker_color=["#64748b", "#0d9488"],
        ))

        fig.update_layout(
            yaxis_title="Annual Average Loss (USD)",
            showlegend=False,
            height=360,
        )

        st.plotly_chart(
            fig, width="stretch"
        )

    with right:
        st.subheader("Investment Recommendation")

        st.success(
            f"Recommended: {selected['portfolio_name']}"
        )

        st.write(
            f"**Annual investment:** "
            f"{money(selected['annual_control_cost_usd'])}"
        )

        st.write(
            f"**Expected annual risk reduction:** "
            f"{money(selected['risk_reduction_usd'])}"
        )

        st.write(
            f"**Net annual benefit:** "
            f"{signed_money(selected['net_annual_benefit_usd'])}"
        )

        st.write(
            f"**Risk reduction:** "
            f"{selected['risk_reduction_pct']:.1f}%"
        )

        st.caption(
            "Executive Overview uses a fixed reference case: "
            "central annual control costs, a $100,000 budget, "
            "and minimizing total expected cost. "
            "Use Investment Optimizer to explore other assumptions."
        )

    st.divider()

    st.subheader("Decision Context")

    st.write(
        "The financially optimal control combination minimizes "
        "expected annual cyber losses plus annual control "
        "spending. The risk-minimizing combination instead "
        "prioritizes the lowest residual expected loss "
        "within the available budget."
    )


    st.warning(
        "This is an exploratory, single-scenario CRQ model. "
    )


# --------------------------------------------------
# 6. CYBER RISK ANALYSIS (BASELINE SIMULATION)
# --------------------------------------------------

elif page == "Cyber Risk Analysis":
    st.title("Cyber Risk Analysis")
    st.caption("Baseline third-party OAuth compromise | Monte Carlo annual loss simulation")

    @st.cache_data
    def load_risk_simulations():
        folder = OUTPUTS / "simulations"
        annual_file = folder / "dashboard_annual_losses.csv"
        incident_file = folder / "dashboard_incident_losses.csv"
        missing = [str(p) for p in (annual_file, incident_file) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing simulation CSV(s): " + ", ".join(missing)
                + ". Run the export cell in notebooks/14_dashboard_risk_data.ipynb first."
            )
        return pd.read_csv(annual_file), pd.read_csv(incident_file)

    try:
        annual_data, incident_data = load_risk_simulations()
    except (FileNotFoundError, pd.errors.ParserError) as exc:
        st.error(str(exc))
        st.stop()

    component_labels = {
        "primary_loss_usd": "Primary response",
        "secondary_response_planning_loss_usd": "Secondary response & planning",
        "secondary_fines_judgement_loss_usd": "Fines & judgments",
        "secondary_reputation_loss_usd": "Reputational loss",
    }
    required_annual = {
        "total_annual_loss_usd", "successful_incident_count",
        "maximum_single_event_loss_usd",
    }
    required_incident = set(component_labels) | {"total_incident_loss_usd"}
    if not required_annual.issubset(annual_data.columns):
        st.error(f"Annual CSV missing columns: {sorted(required_annual - set(annual_data.columns))}")
        st.stop()
    if not required_incident.issubset(incident_data.columns):
        st.error(f"Incident CSV missing columns: {sorted(required_incident - set(incident_data.columns))}")
        st.stop()

    losses = pd.to_numeric(annual_data["total_annual_loss_usd"], errors="coerce")
    counts = pd.to_numeric(annual_data["successful_incident_count"], errors="coerce")
    if losses.isna().any() or counts.isna().any() or len(losses) == 0:
        st.error("Annual simulation data contains missing or invalid risk values.")
        st.stop()

    sorted_losses = np.sort(losses.to_numpy(dtype=float))
    n_years = len(sorted_losses)
    aal = float(losses.mean())
    var = {p: float(np.quantile(sorted_losses, p)) for p in (0.90, 0.95, 0.99)}
    tvar = {
        p: float(sorted_losses[-int(np.ceil(n_years * (1 - p))):].mean())
        for p in (0.90, 0.95, 0.99)
    }
    exceedance_thresholds = [1_000_000, 2_000_000, 5_000_000, 10_000_000]
    exceedance = {
        threshold: float(np.mean(sorted_losses > threshold))
        for threshold in exceedance_thresholds
    }

    st.subheader("Annual Financial Risk")
    a, b, c, d = st.columns(4)
    a.metric("Annual Average Loss", money(aal))
    b.metric("Probability of ≥1 Incident", f"{(counts > 0).mean():.2%}")
    c.metric("Average LEF", f"{counts.mean():.4f} / year")
    d.metric("99% VaR", money(var[0.99]))

    st.markdown("#### Full Risk Metrics")
    metrics = [
        ("Simulated Years", f"{n_years:,}"),
        ("Maximum Single Event Loss", money(annual_data["maximum_single_event_loss_usd"].max())),
        ("Maximum Total Annual Loss", money(losses.max())),
        ("Average LEF", f"{counts.mean():.4f}"),
        ("P(At Least One Incident)", f"{(counts > 0).mean():.2%}"),
        ("Annual Average Loss (AAL)", money(aal)),
        ("Median Annual Loss", money(losses.median())),
    ]
    for p in (0.90, 0.95, 0.99):
        metrics.append((f"VaR {p:.0%}", money(var[p])))
    for p in (0.90, 0.95, 0.99):
        metrics.append((f"TVaR {p:.0%}", money(tvar[p])))
    for threshold in exceedance_thresholds:
        metrics.append((f"P(Annual Loss > {money(threshold)})", f"{exceedance[threshold]:.2%}"))

    metrics_df = pd.DataFrame(metrics, columns=["Risk Metric", "Value"])
    st.dataframe(metrics_df, hide_index=True, width="stretch")
    st.download_button(
        "Download Risk Metrics CSV", metrics_df.to_csv(index=False),
        file_name="novaflow_baseline_risk_metrics.csv", mime="text/csv",
    )
    st.caption(
        "All figures are calculated from the saved baseline annual simulation. "
    )

    st.divider()
    st.subheader("Annual Loss Exceedance Curve (LEC)")
    st.write(
        "For each loss threshold, the curve shows the fraction of simulated years "
        "in which **total annual loss is strictly greater than that amount**. "
        "A lower curve means less annual loss exposure."
    )
    view = st.radio(
        "Curve view", ["Full range", "Up to 99th-percentile loss"],
        horizontal=True, key="lec_view",
    )
    upper = float(sorted_losses[-1]) if view == "Full range" else float(var[0.99])
    # Include zero and all four decision thresholds; use a compact grid for responsive plotting.
    grid = np.linspace(0.0, upper, 400)
    thresholds = np.unique(np.concatenate([
        grid, [0.0],
        [float(t) for t in exceedance_thresholds if t <= upper],
    ]))
    # searchsorted(side='right') counts values <= threshold, giving P(loss > threshold).
    probs = (n_years - np.searchsorted(sorted_losses, thresholds, side="right")) / n_years
    lec = go.Figure()
    lec.add_trace(go.Scatter(
        x=thresholds, y=100 * probs, mode="lines",
        name="Annual loss exceedance", line=dict(color="#0d9488", width=3),
        hovertemplate="Threshold: $%{x:,.0f}<br>P(Annual loss > threshold): %{y:.2f}%<extra></extra>",
    ))
    for threshold in exceedance_thresholds:
        if threshold <= upper:
            lec.add_trace(go.Scatter(
                x=[threshold], y=[100 * exceedance[threshold]],
                mode="markers", marker=dict(size=9),
                name=f"> {money(threshold)}",
                hovertemplate=f"Threshold: {money(threshold)}<br>Exceedance: {exceedance[threshold]:.2%}<extra></extra>",
            ))
    lec.update_layout(
        xaxis_title="Annual loss threshold (USD)",
        yaxis_title="Probability of exceeding threshold (%)",
        yaxis=dict(range=[0, max(21, float(probs.max() * 100) * 1.05)]),
        xaxis=dict(tickprefix="$", tickformat=",.0f"),
        height=450, legend_title_text="Series / thresholds",
        hovermode="closest",
    )
    st.plotly_chart(lec, width="stretch")
    st.caption(
        "This curve shows the probability that annual financial losses exceed "
        "a given amount, based on 100,000 simulated years, including years with "
        "no losses. For example, a 5% exceedance probability means annual losses "
        "exceed that amount in approximately 1 out of every 20 years."
    )

    st.divider()
    st.subheader("Annual Loss Distribution")
    st.write(
        f"**{(losses == 0).mean():.1%} of simulated years have zero loss.** "
        "The histogram shows only positive-loss years for readability, "
        "but all years are included in the risk metrics and LEC."
    )
    positive = losses[losses > 0]
    histogram = px.histogram(
        x=positive, nbins=70,
        labels={"x": "Annual loss (USD)", "y": "Simulated years"},
    )
    histogram.add_vline(x=var[0.95], line_dash="dash", line_color="orange",
                        annotation_text="VaR95")
    histogram.add_vline(x=var[0.99], line_dash="dash", line_color="red",
                        annotation_text="VaR99")
    histogram.update_layout(height=420, xaxis_title="Annual loss (USD)",
                            yaxis_title="Simulated years")
    st.plotly_chart(histogram, width="stretch")

    st.divider()
    st.subheader("Successful Incidents per Year")
    frequency = counts.value_counts().sort_index().rename_axis("Incident count").reset_index(name="Years")
    frequency["Incident count"] = frequency["Incident count"].astype(str)
    frequency_fig = px.bar(
        frequency, x="Incident count", y="Years",
        title="Simulated annual incident counts",
    )
    frequency_fig.update_layout(height=370)
    st.plotly_chart(frequency_fig, width="stretch")
    st.write(f"**Expected successful incidents per year:** {counts.mean():.4f}")

    st.divider()
    st.subheader("Loss Composition per Successful Incident")
    component_means = incident_data[list(component_labels)].mean()
    composition = pd.DataFrame({
        "Component": [component_labels[key] for key in component_means.index],
        "Mean Loss (USD)": component_means.values,
    })
    composition["Share (%)"] = (
        100 * composition["Mean Loss (USD)"] / composition["Mean Loss (USD)"].sum()
    )
    pie = px.pie(
        composition, names="Component", values="Mean Loss (USD)",
        hole=0.45, title="Mean incident loss by component",
    )
    st.plotly_chart(pie, width="stretch")
    st.dataframe(
        composition, width="stretch", hide_index=True,
        column_config={
            "Mean Loss (USD)": st.column_config.NumberColumn(format="$%.0f"),
            "Share (%)": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
    st.caption(
        "Component shares are calculated conditional on a successful incident. "
        "Annual loss metrics also account for incident frequency."
    )
    with st.expander("How to interpret the risk measures"):
        st.markdown(
            "- **AAL:** Mean annual loss, including zero-loss years.\n"
            "- **LEF:** Average number of successful loss events per year.\n"
            "- **VaR90/95/99:** 90th/95th/99th percentile of annual loss.\n"
            "- **TVaR90/95/99:** Mean annual loss in the worst 10%/5%/1% "
            "of simulated years, respectively.\n"
            "- **Loss exceedance:** Probability that annual loss is strictly "
            "greater than a chosen dollar threshold.\n"
            "- **Maximum loss:** Largest observed simulated loss, not a "
            "theoretical upper bound."
        )


# --------------------------------------------------
# 7. INVESTMENT OPTIMIZER
# --------------------------------------------------

elif page == "Investment Optimizer":

    st.title("Security Investment Optimizer")

    st.write(
        "Evaluate all 32 possible combinations of "
        "five cybersecurity controls."
    )

    st.subheader("Recommended Control Combination")

    st.success(selected["portfolio_name"])

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Investment",
        money(selected["annual_control_cost_usd"])
    )
    c2.metric(
        "Residual AAL",
        money(selected["residual_aal_usd"])
    )
    c3.metric(
        "Risk Reduction",
        f"{selected['risk_reduction_pct']:.1f}%"
    )
    c4.metric(
        "Net Annual Benefit",
        money(selected["net_annual_benefit_usd"])
    )

    st.divider()

    st.subheader("Budget vs. Residual Cyber Risk")

    budget_values = sorted(
        set(
            list(range(0, 450001, 10000))
            + [budget]
        )
    )

    curve_rows = []

    for b in budget_values:
        for obj in [
            "Minimize total expected cost",
            "Minimize residual cyber risk",
        ]:
            p = optimize_portfolio(
                portfolios, b, obj
            )

            curve_rows.append({
                "Budget": b,
                "Residual AAL": p["residual_aal_usd"],
                "Objective": obj,
            })

    curve_df = pd.DataFrame(curve_rows)

    fig = px.line(
        curve_df,
        x="Budget",
        y="Residual AAL",
        color="Objective",
        markers=True,
        title="Optimal Residual Risk by Budget",
    )

    fig.update_layout(height=420)

    st.plotly_chart(
        fig, width="stretch"
    )

    st.subheader("All Feasible Control Combinations")

    feasible = portfolios[
        portfolios["annual_control_cost_usd"] <= budget
    ].sort_values("total_expected_cost_usd")

    display_df = feasible[
        [
            "portfolio_name",
            "annual_control_cost_usd",
            "residual_aal_usd",
            "risk_reduction_pct",
            "net_annual_benefit_usd",
            "total_expected_cost_usd",
        ]
    ].copy()

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "portfolio_name": "Control combination",
            "annual_control_cost_usd":
                st.column_config.NumberColumn(
                    "Annual Cost", format="$%.0f"
                ),
            "residual_aal_usd":
                st.column_config.NumberColumn(
                    "Residual AAL", format="$%.0f"
                ),
            "risk_reduction_pct":
                st.column_config.NumberColumn(
                    "Risk Reduction", format="%.1f%%"
                ),
            "net_annual_benefit_usd":
                st.column_config.NumberColumn(
                    "Net Benefit", format="$%.0f"
                ),
            "total_expected_cost_usd":
                st.column_config.NumberColumn(
                    "Total Expected Cost", format="$%.0f"
                ),
        },
    )

    st.download_button(
        "Download Feasible Portfolios",
        data=display_df.to_csv(index=False),
        file_name="novaflow_feasible_portfolios.csv",
        mime="text/csv",
    )


# --------------------------------------------------
# 8. SECURITY CONTROLS
# --------------------------------------------------

elif page == "Security Controls":

    st.title("Security Control Effectiveness")

    st.write(
        "Compare the expected annual financial benefits "
        "of individual cybersecurity investments."
    )

    st.subheader("Security controls considered")
    st.write(
        "Each control targets a specific driver in the FAIR model. "
        "Effectiveness and annual incremental costs are illustrative; "
        "the selected cost scenario is shown in the last column."
    )
    control_catalog = [
        ("C1", "Third-party OAuth governance", "Loss event frequency",
         "Reduces success probability per campaign", "20%", "C1_oauth_governance"),
        ("C2", "Least-privilege Salesforce access", "Loss magnitude",
         "Reduces affected-customer exposure", "25%", "C2_least_privilege"),
        ("C3", "SaaS anomaly detection / token revocation", "Loss event frequency",
         "Reduces success probability per campaign", "15%", "C3_saas_detection"),
        ("C4", "Data minimization / exfiltration restrictions", "Loss magnitude",
         "Reduces affected-customer exposure", "20%", "C4_data_protection"),
        ("C5", "Incident response readiness", "Loss magnitude",
         "Reduces selected response labor hours", "20%", "C5_ir_readiness"),
    ]
    catalog = pd.DataFrame([{
        "ID": code, "Control": name, "Risk dimension": dimension,
        "Modeled effect": effect, "Central effect": pct,
        "Low annual cost": CONTROL_COSTS[key]["low"],
        "Central annual cost": CONTROL_COSTS[key]["central"],
        "High annual cost": CONTROL_COSTS[key]["high"],
        f"Selected ({cost_level}) cost": CONTROL_COSTS[key][cost_level],
    } for code, name, dimension, effect, pct, key in control_catalog])
    st.dataframe(catalog, width="stretch", hide_index=True)
    st.caption(
        "C1 and C3 act on the probability of a successful incident. "
        "C2 and C4 act on affected-customer exposure. C5 acts on selected "
        "response labor hours. Combined effects can overlap and are not additive."
    )
    st.divider()

    individual = portfolios[
        portfolios["control_count"] == 1
    ].copy()

    individual = individual.sort_values(
        "risk_reduction_usd",
        ascending=False
    )

    fig = px.bar(
        individual,
        x="portfolio_name",
        y="risk_reduction_usd",
        title="Gross Annual Risk Reduction by Control",
        labels={
            "portfolio_name": "Security Control",
            "risk_reduction_usd": "Risk Reduction (USD)",
        },
    )

    st.plotly_chart(
        fig, width="stretch"
    )

    fig2 = px.bar(
        individual,
        x="portfolio_name",
        y="net_annual_benefit_usd",
        title=(
            "Net Annual Benefit — "
            f"{cost_level.title()} Cost Assumptions"
        ),
        labels={
            "portfolio_name": "Security Control",
            "net_annual_benefit_usd":
                "Net Annual Benefit (USD)",
        },
    )

    fig2.add_hline(y=0, line_color="gray")

    st.plotly_chart(
        fig2, width="stretch"
    )

    st.subheader("Individual Control Comparison")

    st.dataframe(
        individual[
            [
                "portfolio_name",
                "annual_control_cost_usd",
                "risk_reduction_usd",
                "risk_reduction_pct",
                "net_annual_benefit_usd",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Individual control benefits cannot be added "
        "directly because controls can overlap."
    )


# --------------------------------------------------
# 9. UNCERTAINTY & ASSUMPTIONS
# --------------------------------------------------

elif page == "Uncertainty & Assumptions":
    st.title("Risk Model Explorer")
    st.caption("Monte Carlo variability, parameter uncertainty, and assumption sensitivity")
    #st.info(
    #    "The controls on this page explore the *baseline* threat scenario. "
    #    "They do not update the separately precomputed 32-portfolio investment optimizer."
    #)

    @st.cache_data(show_spinner=False)
    def load_uncertainty_data():
        files = {
            "summary": "parameter_uncertainty_summary.csv",
            "draws": "combined_parameter_uncertainty_draws.csv",
            "effectiveness": "control_effectiveness_sensitivity.csv",
            "financial": "control_financial_sensitivity.csv",
            "combined_controls": "combined_controls_sensitivity.csv",
        }
        return {
            key: pd.read_csv(OUTPUTS / filename)
            if (OUTPUTS / filename).is_file() else pd.DataFrame()
            for key, filename in files.items()
        }

    @st.cache_data(show_spinner=False, max_entries=24)
    def run_explorer(n_years, seed, campaign_rate, success_mean,
                     affected_mode, affected_churn, unaffected_churn,
                     lost_deal, contribution_margin):
        # Import only on demand. ROOT is the project directory.
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from config.model_config import get_config
        from src.annual_loss_simulation import simulate_annual_losses

        cfg = get_config()
        cfg["reputation"]["affected_customers"]["mode"] = affected_mode
        cfg["probabilities"]["affected_churn"]["target"] = affected_churn
        cfg["probabilities"]["unaffected_churn"]["target"] = unaffected_churn
        cfg["probabilities"]["lost_deal"]["target"] = lost_deal
        cfg["organization"]["contribution_margin"] = contribution_margin
        contribution = (cfg["organization"]["average_annual_customer_revenue_usd"]
                        * contribution_margin)
        cfg["organization"]["contribution_per_lost_customer_or_deal_usd"] = contribution
        cfg["reputation"]["loss_per_churned_customer_usd"] = contribution
        cfg["reputation"]["loss_per_lost_deal_usd"] = contribution
        # Preserve original Beta concentration; vary only its mean.
        concentration = 0.36 + 4.48
        return simulate_annual_losses(
            n_years=int(n_years), seed=int(seed),
            campaign_rate=float(campaign_rate),
            success_alpha=float(success_mean * concentration),
            success_beta=float((1 - success_mean) * concentration),
            config=cfg,
        )

    def explorer_metrics(df):
        x = df["total_annual_loss_usd"].to_numpy(dtype=float)
        return {
            "AAL": float(np.mean(x)),
            "LEF": float(df["successful_incident_count"].mean()),
            "P(≥1 incident)": float((df["successful_incident_count"] > 0).mean()),
            "VaR95": float(np.quantile(x, .95)),
            "VaR99": float(np.quantile(x, .99)),
            "TVaR95": float(np.mean(np.sort(x)[-max(1, int(np.ceil(len(x)*.05))):])),
        }

    aleatory_tab, epistemic_tab, combined_tab, control_tab, assumptions_tab = st.tabs([
        "Monte Carlo variability", "Change model assumptions",
        "Parameter uncertainty", "Control sensitivity", "Model reference",
    ])

    # Reset callbacks restore the displayed inputs and clear results from old runs.
    def reset_mc_inputs():
        st.session_state.update({"mc_years": 10000, "mc_seed": 20261025})
        st.session_state.pop("novaflow_mc", None)
        st.session_state.pop("novaflow_mc_settings", None)

    def reset_scenario_inputs():
        st.session_state.update({
            "scenario_campaign_rate": 3.0,
            "scenario_success_pct": 100 * 0.36 / (0.36 + 4.48),
            "scenario_affected_mode": 1500,
            "scenario_affected_churn": 0.20,
            "scenario_unaffected_churn": 0.020,
            "scenario_lost_deal": 1.0,
            "scenario_margin": 70,
            "scenario_years": 10000,
            "scenario_seed": 20261025,
        })
        st.session_state.pop("novaflow_scenario", None)
        st.session_state.pop("novaflow_scenario_settings", None)

    with aleatory_tab:
        st.subheader("Random outcomes and Monte Carlo precision")
        st.write(
            "Even with identical risk assumptions, different simulated years produce different "
            "numbers of incidents and different financial losses. A **random seed** selects a "
            "reproducible sequence of random outcomes. Increasing **simulated years** generally "
            "makes the estimated average and tail metrics more stable; it does not change the "
            "underlying cyber risk."
        )
        with st.form("aleatory_form"):
            ca, cb = st.columns(2)
            with ca:
                n_mc = st.select_slider(
                    "Simulated years", options=[1000, 5000, 10000, 25000, 50000, 100000],
                    value=10000, key="mc_years")
            with cb:
                seed_mc = st.number_input(
                    "Random seed", min_value=0, max_value=2_147_483_647,
                    value=20261025, step=1, key="mc_seed")
            go_mc = st.form_submit_button("Run Monte Carlo experiment", type="primary")
        st.button("Reset Monte Carlo inputs", key="reset_mc", on_click=reset_mc_inputs)
        if go_mc:
            try:
                with st.spinner("Running the FAIR annual loss simulator…"):
                    st.session_state["novaflow_mc"] = run_explorer(
                        n_mc, seed_mc, 3., 0.36 / 4.84, 1500, .002, .0002, .01, .70)
                st.session_state["novaflow_mc_settings"] = (n_mc, seed_mc)
            except Exception as exc:
                st.error(f"Simulation failed: {exc}")
        if "novaflow_mc" in st.session_state:
            df_mc = st.session_state["novaflow_mc"]
            mm = explorer_metrics(df_mc)
            st.caption(
                f"Last completed run: {len(df_mc):,} simulated years; "
                f"seed {st.session_state.get('novaflow_mc_settings', ('?', '?'))[1]}"
            )
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("AAL", money(mm["AAL"]))
            k2.metric("VaR95", money(mm["VaR95"]))
            k3.metric("VaR99", money(mm["VaR99"]))
            k4.metric("Average LEF", f"{mm['LEF']:.4f}")
            x = df_mc["total_annual_loss_usd"].to_numpy(dtype=float)
            checkpoints = np.unique(np.clip(np.geomspace(100, len(x), 110).astype(int), 1, len(x)))
            cumulative = np.cumsum(x)
            convergence = pd.DataFrame({
                "Simulated years": checkpoints,
                "Cumulative AAL": cumulative[checkpoints - 1] / checkpoints,
            })
            fig = px.line(
                convergence, x="Simulated years", y="Cumulative AAL",
                title="How the estimated AAL stabilizes as more years are simulated")
            fig.update_layout(yaxis_tickprefix="$", height=390)
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Rare, expensive incidents may be absent from small samples. Tail measures "
                "such as VaR99 often need substantially more simulations than AAL."
            )
            with st.expander("Show simulation output summary"):
                st.dataframe(pd.DataFrame([mm]).T.rename(columns={0: "Estimate"}), width="stretch")
        else:
            st.caption("Choose a seed and number of years, then run the experiment.")

    with epistemic_tab:
        st.subheader("Explore alternative risk assumptions")
        st.write(
            "This is a **what-if analysis**: choose specific values for the assumptions, "
            "then run the original simulation again. The results are compared with the "
            "default assumptions using the **same seed and number of simulated years**. "
            "You can change frequency assumptions, loss magnitude assumptions, or both."
        )
        with st.form("epistemic_form"):
            st.markdown("#### A. Loss event frequency assumptions")
            f1, f2 = st.columns(2)
            with f1:
                campaign_rate = st.slider(
                    "Annual attack campaigns (Poisson λ)", 0.5, 10.0, 3.0, 0.5,
                    key="scenario_campaign_rate")
            with f2:
                success_pct = st.number_input(
                    "Mean success probability per campaign (%)",
                    min_value=1.0, max_value=25.0,
                    value=100 * 0.36 / (0.36 + 4.48),
                    step=0.1, format="%.6f", key="scenario_success_pct")
            st.markdown(
                "**Campaign rate (λ):** Expected number of relevant attacker campaigns "
                "in one year, not successful breaches.\n\n"
                "**Success probability:** Average chance a campaign produces a "
                "successful incident. The model draws a probability from a Beta "
                "distribution for each campaign; this input changes its mean "
                "while holding concentration fixed.\n\n"
                "**Expected LEF:** Campaign rate multiplied by mean success "
                "probability; this is the expected number of successful incidents per year."
            )
            st.divider()
            st.markdown("#### B. Loss magnitude assumptions")
            m1, m2 = st.columns(2)
            with m1:
                affected_mode = st.slider(
                    "Most likely affected customers (PERT mode)", 300, 3000, 1500, 100,
                    key="scenario_affected_mode")
                affected_churn_pct = st.slider(
                    "Affected customer churn target (%)", 0.05, 1.0, 0.20, 0.05,
                    key="scenario_affected_churn")
                unaffected_churn_pct = st.slider(
                    "Unaffected customer churn target (%)", 0.005, 0.10, 0.020, 0.005,
                    format="%.3f", key="scenario_unaffected_churn")
            with m2:
                lost_deal_pct = st.slider(
                    "Lost-deal probability target (%)", 0.2, 5.0, 1.0, 0.2,
                    key="scenario_lost_deal")
                margin_pct = st.slider(
                    "Contribution margin (%)", 30, 90, 70, 5,
                    key="scenario_margin")
            st.markdown(
                "**Affected customers:** Most likely number of customer accounts "
                "exposed in a successful incident, within a PERT range of 300–3,000.\n\n"
                "**Affected churn:** Target probability of losing a customer "
                "whose account was exposed.\n\n"
                "**Unaffected churn:** Target probability of losing a customer "
                "whose account was not exposed but who may react to the breach.\n\n"
                "**Lost deals:** Target probability of losing a prospective deal.\n\n"
                "**Contribution margin:** Share of annual customer revenue counted "
                "as lost contribution, rather than treating all revenue as profit.\n\n"
                "These probability targets feed conditional event models; "
                "they are not fixed outcomes for every incident."
            )
            st.divider()
            st.markdown("#### C. Simulation settings")
            r1, r2 = st.columns(2)
            with r1:
                n_ep = st.select_slider(
                    "Simulated years", options=[1000, 5000, 10000, 25000, 50000],
                    value=10000, key="scenario_years")
            with r2:
                seed_ep = st.number_input(
                    "Scenario random seed", min_value=0, max_value=2_147_483_647,
                    value=20261025, key="scenario_seed")
            go_ep = st.form_submit_button("Run assumption scenario", type="primary")
        st.button("Reset all assumption inputs", key="reset_scenario", on_click=reset_scenario_inputs)
        if go_ep:
            try:
                with st.spinner("Recalculating losses under the selected assumptions…"):
                    baseline_success_pct = 100 * 0.36 / (0.36 + 4.48)
                    # When displayed inputs match defaults, use exactly the same
                    # parameter value in both simulations, avoiding rounding drift.
                    effective_success_mean = (0.36 / 4.84 if abs(success_pct - baseline_success_pct) < 1e-6
                                              else success_pct / 100)
                    scenario = run_explorer(
                        n_ep, seed_ep, campaign_rate, effective_success_mean,
                        affected_mode, affected_churn_pct / 100,
                        unaffected_churn_pct / 100, lost_deal_pct / 100, margin_pct / 100)
                    baseline_experiment = run_explorer(
                        n_ep, seed_ep, 3., 0.36 / 4.84, 1500, .002, .0002, .01, .70)
                st.session_state["novaflow_scenario"] = (scenario, baseline_experiment)
                st.session_state["novaflow_scenario_settings"] = (n_ep, seed_ep)
            except Exception as exc:
                st.error(f"Scenario simulation failed: {exc}")
        if "novaflow_scenario" in st.session_state:
            scenario, base_run = st.session_state["novaflow_scenario"]
            current = explorer_metrics(scenario)
            reference = explorer_metrics(base_run)
            keys = ["AAL", "LEF", "P(≥1 incident)", "VaR95", "VaR99", "TVaR95"]
            compare = pd.DataFrame({
                "Metric": keys,
                "Baseline": [reference[k] for k in keys],
                "Adjusted": [current[k] for k in keys],
            })
            compare["Change (%)"] = np.where(
                compare["Baseline"] != 0,
                100 * (compare["Adjusted"] / compare["Baseline"].replace(0, np.nan) - 1),
                np.nan)
            st.caption(
                f"Last completed comparison: {st.session_state['novaflow_scenario_settings'][0]:,} "
                f"simulated years; seed {st.session_state['novaflow_scenario_settings'][1]}"
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Adjusted AAL", money(current["AAL"]),
                      delta=f"{current['AAL'] - reference['AAL']:+,.0f} USD vs baseline")
            c2.metric("Adjusted LEF", f"{current['LEF']:.4f}")
            c3.metric("Adjusted VaR95", money(current["VaR95"]))
            st.dataframe(compare, width="stretch", hide_index=True,
                         column_config={"Change (%)": st.column_config.NumberColumn(format="%.1f%%")})
            fig = px.bar(pd.DataFrame({
                "Assumptions": ["Default", "Adjusted"],
                "Annual Average Loss": [reference["AAL"], current["AAL"]]}),
                x="Assumptions", y="Annual Average Loss",
                title="Annual average loss under default vs adjusted assumptions")
            fig.update_layout(yaxis_tickprefix="$", height=350)
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "The same seed and sample size reduce comparison noise, but they do not "
                "eliminate Monte Carlo variability. These changes do not update the "
                "separately calculated investment recommendations."
            )
        else:
            st.caption("Change one or more assumptions, then run the scenario.")

    with combined_tab:
        st.subheader("Uncertainty about the true Annual Average Loss")
        st.write(
            "The previous tab tests **one chosen alternative**. Here we explore **many plausible "
            "parameter combinations**. For each draw, an AAL estimate is calculated and placed "
            "in a histogram. The spread shows how uncertain our estimate of the *expected* annual "
            "loss is because we do not know the true input parameters. It is **not** the "
            "distribution of actual yearly losses, nor an annual VaR chart."
        )
        st.markdown("#### What do the uncertainty categories mean?")
        st.markdown(
            "- **Frequency parameters:** vary assumptions governing how often successful incidents "
            "occur, such as the campaign arrival rate and campaign success probability, while "
            "holding loss magnitude assumptions at their reference values.\n"
            "- **Magnitude parameters:** vary assumptions governing the financial consequences "
            "of a successful incident, while holding frequency assumptions at their reference values.\n"
            "- **Combined:** vary frequency and magnitude assumptions together, showing their "
            "joint impact on estimated AAL."
        )
        st.caption(
            "The saved distributions were generated by the project's parameter-uncertainty "
            "analysis. This view reads those saved results; it does not run a new simulation "
            "when you switch between series. The precise parameters and ranges are those "
            "used to create the saved analysis."
        )
        uncertainty = load_uncertainty_data()
        summary = uncertainty["summary"]
        if not summary.empty:
            with st.expander("Show parameter-uncertainty summary"):
                st.dataframe(summary, width="stretch", hide_index=True)
        draws = uncertainty["draws"]
        if not draws.empty:
            aal_cols = [
                c for c in draws.columns
                if "aal" in c.lower() and pd.api.types.is_numeric_dtype(draws[c])
            ]
            if aal_cols:
                default = next((c for c in aal_cols if "combined" in c.lower()), aal_cols[0])
                col = st.selectbox(
                    "Estimated AAL series", aal_cols, index=aal_cols.index(default),
                    key="uncertainty_aal_series")
                st.button(
                    "Reset selected AAL series", key="reset_aal_series",
                    on_click=lambda: st.session_state.update({"uncertainty_aal_series": default}),
                )
                series_name = col.lower()
                if "combined" in series_name:
                    st.info("**Combined:** both frequency and magnitude parameters vary across draws.")
                elif "freq" in series_name:
                    st.info("**Frequency-only:** incident frequency assumptions vary; magnitude assumptions are held fixed.")
                elif "mag" in series_name or "severity" in series_name:
                    st.info("**Magnitude-only:** incident loss assumptions vary; frequency assumptions are held fixed.")
                else:
                    st.caption("The selected series is shown using its saved column name; see the summary for context.")
                v = pd.to_numeric(draws[col], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan).dropna()
                if len(v):
                    p05, p50, p95 = v.quantile([0.05, 0.50, 0.95])
                    mean_aal = v.mean()
                    std_aal = v.std(ddof=1)
                    interval_width = p95 - p05

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Mean estimated AAL", money(mean_aal))
                    c2.metric("Median estimated AAL", money(p50))
                    c3.metric("Standard deviation", money(std_aal))

                    c4, c5, c6 = st.columns(3)
                    c4.metric("5th percentile", money(p05))
                    c5.metric("95th percentile", money(p95))
                    c6.metric("90% interval width", money(interval_width))
                    fig = px.histogram(
                        x=v, nbins=65, title="Estimated AAL across parameter draws",
                        labels={"x": "Estimated AAL (USD)", "y": "Parameter draws"})
                    fig.add_vline(x=p05, line_dash="dash", line_color="orange")
                    fig.add_vline(x=p95, line_dash="dash", line_color="red")
                    fig.update_layout(height=410)
                    st.plotly_chart(fig, width="stretch")
                    st.caption(
                        f"Series: {col}; {len(v):,} valid draws. The 5th–95th percentile "
                        "range describes uncertainty in estimated AAL, not losses in a single year."
                    )
            else:
                st.warning("No numeric AAL column found in the saved parameter draws.")
        else:
            st.warning("Missing outputs/combined_parameter_uncertainty_draws.csv")
        st.markdown(
            "**Interpretation:** In the saved analysis, frequency uncertainty contributed "
            "more to the spread of estimated AAL than the modeled magnitude uncertainty. "
            "That conclusion depends on the ranges assigned to the uncertain parameters."
        )

    with control_tab:
        st.subheader("Control effectiveness and cost uncertainty")
        st.write("These charts use saved sensitivity analyses, rather than rerunning the model when you change the selection.")
        uncertainty = load_uncertainty_data()
        eff = uncertainty["effectiveness"]
        if {"control", "effectiveness", "gross_risk_reduction_usd"}.issubset(eff.columns):
            fig = px.bar(eff, x="control", y="gross_risk_reduction_usd", color="effectiveness",
                         barmode="group", title="Gross annual risk reduction across effectiveness assumptions")
            st.plotly_chart(fig, width="stretch")
            with st.expander("Effectiveness data"):
                st.dataframe(eff, width="stretch", hide_index=True)
        else:
            st.warning("Effectiveness CSV missing or has unexpected column names.")
        fin = uncertainty["financial"]
        if {"control", "effectiveness", "cost_level", "net_benefit_usd"}.issubset(fin.columns):
            control = st.selectbox("Control for cost/effectiveness matrix", sorted(fin["control"].dropna().unique()))
            matrix = fin.loc[fin["control"] == control].pivot_table(
                index="effectiveness", columns="cost_level", values="net_benefit_usd", aggfunc="first")
            matrix = matrix.reindex(index=["low", "central", "high"], columns=["low", "central", "high"])
            fig = px.imshow(matrix, text_auto=",.0f", aspect="auto", color_continuous_scale="RdBu",
                            color_continuous_midpoint=0,
                            labels={"x": "Cost assumption", "y": "Effectiveness assumption", "color": "Net annual benefit"},
                            title="Net annual benefit by cost and effectiveness")
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Financial sensitivity CSV missing or has unexpected column names.")
        if not uncertainty["combined_controls"].empty:
            with st.expander("All-five-controls sensitivity"):
                st.dataframe(uncertainty["combined_controls"], width="stretch", hide_index=True)

    with assumptions_tab:
        st.subheader("Model reference: how the cyber loss model works")
        st.write(
            "This model estimates the annual financial consequences of one defined cyberattack "
            "scenario for a hypothetical B2B SaaS business. It combines **how often an incident "
            "occurs** with **how much each successful incident costs**. The assumptions below "
            "are illustrative and should be calibrated with organization-specific evidence."
        )
        st.markdown("#### Scenario and boundaries")
        st.markdown(
            "An attacker compromises a third-party SaaS vendor, obtains usable OAuth credentials, "
            "accesses NovaFlow's Salesforce CRM, and exfiltrates customer data. The modeled harm "
            "is a **confidentiality breach**. There is no service outage or operational downtime "
            "loss in this scenario. The reference organization has **3,000 enterprise customers**."
        )
        with st.expander("1. How often can a loss event happen?", expanded=True):
            st.markdown(
                "**Campaign arrivals:** The number of relevant attacker campaigns in a year "
                "follows a **Poisson distribution**, with an average of **3 campaigns/year**. "
                "Some years have none; other years have several.\n\n"
                "**Campaign success:** For each campaign, a success probability is independently "
                "drawn from **Beta(0.36, 4.48)**, with mean approximately **7.44%**. A "
                "Bernoulli trial then determines whether that campaign succeeds.\n\n"
                "**Loss event frequency (LEF):** A successful campaign becomes one modeled "
                "incident. Expected LEF is about **3 × 7.44% = 0.223 successful incidents/year**. "
                "The probability of one or more incidents in a year is about **20%**. "
                "The Beta draw is independent for each campaign; this setup does not by itself "
                "introduce annual incident-count overdispersion."
            )
        with st.expander("2. What determines the severity of an incident?"):
            st.markdown(
                "A successful incident is assigned a shared **technical severity score**, "
                "drawn from a standard normal distribution. Higher-severity incidents tend "
                "to produce more consequential outcomes.\n\n"
                "The model links severity to **affected-customer counts** and multiple "
                "cost-related outcomes through a **Gaussian copula** and calibrated "
                "conditional probability models. This captures dependence: an unusually "
                "severe breach can simultaneously affect more customers and trigger more "
                "secondary consequences. The score is a latent simulation driver, not a "
                "published incident-severity rating."
            )
        with st.expander("3. How many customers are affected?"):
            st.markdown(
                "The affected-customer count uses a **modified beta-PERT distribution** "
                "with **minimum 300, most likely 1,500, maximum 3,000**. The sampled count "
                "is rounded and constrained to the company's 3,000 enterprise customers. "
                "Technical severity is correlated with this count, so customer exposure "
                "is not simulated independently of incident severity."
            )
        with st.expander("4. What financial loss categories are included?"):
            st.markdown(
                "The model adds four financial components for every successful incident:\n\n"
                "**A. Primary loss:** Direct first-party costs of responding to and investigating "
                "the incident, calculated by the primary-loss module.\n\n"
                "**B. Secondary response and planning:** Additional response, notification, "
                "and planning-related costs, conditional on incident characteristics and "
                "secondary outcomes.\n\n"
                "**C. Fines and judgments:** Potential regulatory fines and legal judgments, "
                "including conditional events that may or may not occur in a given incident.\n\n"
                "**D. Reputational loss:** Lost contribution from customer churn and "
                "prospective deals that do not close. The model distinguishes customers "
                "whose data was affected from those not directly affected.\n\n"
                "**Total incident loss** is the sum of these four components. **Total annual "
                "loss** is the sum of losses from every successful incident in that year; "
                "years with no successful incidents have zero modeled loss."
            )
            st.caption(
                "The detailed labor-hour, legal-cost, and other component distributions "
                "are defined in the underlying loss modules. This page summarizes the "
                "verified structure rather than inventing values for those distributions."
            )
        with st.expander("5. How are customer churn and lost business valued?"):
            st.markdown(
                "**Affected-customer churn target:** **0.20%**. "
                "**Unaffected-customer churn target:** **0.020%**. "
                "**Lost-deal target:** **1.0%**. These are baseline probability targets "
                "used by calibrated conditional models; realized outcomes can vary with "
                "incident severity and other modeled indicators.\n\n"
                "Each lost customer or deal is valued at **$210,000 of one-year contribution** "
                "in the reference assumptions. This is contribution rather than gross "
                "revenue; it represents the value remaining after relevant variable costs. "
                "The assumption explorer lets you adjust the contribution margin and "
                "recalculate this value from average annual customer revenue."
            )
        with st.expander("6. Which events and costs move together?"):
            st.markdown(
                "A shared technical severity variable drives correlated outcomes. "
                "The incident simulator first draws severity and affected-customer count, "
                "then conditional event indicators (for example, notification, regulatory "
                "action, external publicity, judgments, churn, and lost deals). The four "
                "loss components use these common incident characteristics. This avoids "
                "treating every financial consequence as an unrelated random event."
            )
        with st.expander("7. How are annual risk measures calculated?"):
            st.markdown(
                "The annual simulator generates many hypothetical years. For each year it "
                "draws campaign counts, determines successful incidents, simulates each "
                "incident's financial losses, and sums them.\n\n"
                "**AAL:** mean loss across *all* years, including zero-loss years. "
                "**VaR95 / VaR99:** annual loss at the 95th / 99th percentile. "
                "**TVaR:** mean annual loss in the worst tail of simulated years. "
                "**Loss exceedance curve:** probability that annual loss exceeds each "
                "dollar threshold. **LEF:** mean number of successful incidents per year."
            )
        with st.expander("8. What do the three uncertainty analyses tell us?"):
            st.markdown(
                "**Monte Carlo variability (aleatory):** outcomes vary from year to year "
                "even when assumptions are fixed; seeds and iteration counts affect the "
                "simulation estimate, not the underlying risk.\n\n"
                "**Alternative assumptions (what-if):** change one or several parameters "
                "and rerun the model to compare a chosen scenario with the default.\n\n"
                "**Parameter uncertainty (epistemic):** evaluate many plausible parameter "
                "draws and examine the resulting spread of estimated AAL. This measures "
                "uncertainty about the expected loss, not annual loss variability."
            )
        with st.expander("9. Security controls and important limitations"):
            st.markdown(
                "Five illustrative security controls target either campaign success "
                "probability, affected-customer exposure, or selected response labor hours. "
                "Their effects and annual costs are assumptions, not empirically validated "
                "performance estimates.\n\n"
                "**Limitations:** This is one hypothetical organization and one cyberattack "
                "scenario. The estimates exclude other attack types, operational downtime, "
                "and benefits of controls outside the modeled scenario. The investment "
                "recommendations are based on separately precomputed scenarios and do "
                "not automatically update when inputs are changed in the explorer. "
                "Small simulation samples may produce unstable tail estimates."
            )


# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "FAIR-Based CRQ | Exploratory Project | "
    "Illustrative assumptions — not validated vendor estimates"
)
