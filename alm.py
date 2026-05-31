# ============================================================
#  SA Two-Pot ALM Decision-Support Platform
#  A story-driven Asset-Liability Management dashboard
# ============================================================
#  Architecture
#  ------------
#  PAGE CONFIG
#  COLOUR PALETTE
#  GLOBAL CONSTANTS
#  HELPER FUNCTIONS  (fund / liquidity / investment / member / plotly)
#  NAVIGATION
#  PAGE 1 - INTRO
#  PAGE 2 - FUND
#  PAGE 3 - LIQUIDITY
#  PAGE 4 - INVESTMENT
#  PAGE 5 - MEMBER OUTCOMES
#  PAGE 6 - AI ADVISORY
#  FOOTER
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── PyPortfolioOpt ────────────────────────────────────────────────────
from pypfopt import EfficientFrontier

# ─────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SA Two-Pot ALM Platform",
    page_icon="🇿🇦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────
C_GOLD   = "#F0B429"
C_GREEN  = "#39D353"
C_RED    = "#FF6B6B"
C_BLUE   = "#58A6FF"
C_PURPLE = "#BC8CFF"
C_ORANGE = "#FF9A3C"
C_TEAL   = "#26C6DA"

C_BG     = "#0D1117"   # page background
C_PANEL  = "#1C2128"   # chart panel
C_GRID   = "#334155"
C_AXIS   = "#E6EDF3"
C_TEXT   = "#E6EDF3"

ASSET_COLORS = [C_GREEN, C_BLUE, C_PURPLE, C_ORANGE, C_TEAL, "#FFFFFF"]

# Light CSS polish for an executive-dashboard feel
st.markdown("""
<style>

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0B0F14;
}

section[data-testid="stSidebar"] * {
    color: #E6EDF3 !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 14px 16px;
}

/* Metric label text */
div[data-testid="stMetricLabel"] p {
    color: #E6EDF3 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}

/* Metric value text */
div[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

/* Metric delta text */
div[data-testid="stMetricDelta"] {
    color: #58A6FF !important;
}

/* Main page headings */
h1, h2, h3 {
    color: #0D1117 !important;
    letter-spacing: 0.2px;
}

/* Info / warning / success speech bubbles */
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] div,
div[data-testid="stAlert"] span {
    color: #0D1117 !important;
    font-weight: 500 !important;
}

/* Plotly legend and axis text */
.legendtext,
.xtick text,
.ytick text,
.xtitle,
.ytitle,
.gtitle {
    fill: #E6EDF3 !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Force Streamlit metric label/title text to be lighter */
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] *,
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] label *,
div[data-testid="stMetric"] [data-testid="stMarkdownContainer"] p {
    color: #E6EDF3 !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

/* Prevent faded/truncated metric label text */
div[data-testid="stMetricLabel"] {
    opacity: 1 !important;
}

</style>
""", unsafe_allow_html=True)


def style_fig(fig, height=420, title=None, legend_bottom=True):
    """Apply the consistent dark theme to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C_BG,
        plot_bgcolor=C_PANEL,
        font=dict(color=C_TEXT, family="monospace", size=12),
        title=dict(text=title, font=dict(color="#FFFFFF", size=15)) if title else None,
        height=height,
        margin=dict(l=60, r=30, t=60 if title else 30, b=60),
        hoverlabel=dict(bgcolor=C_PANEL, font_size=12, font_family="monospace"),
    )
    fig.update_xaxes(gridcolor=C_GRID, zerolinecolor=C_GRID, linecolor="#30363D")
    fig.update_yaxes(gridcolor=C_GRID, zerolinecolor=C_GRID, linecolor="#30363D")
    if legend_bottom:
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#E6EDF3", size=12))
        )
    return fig


PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
    "responsive": True,
}

# ─────────────────────────────────────────────────────────────────────
#  GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────
SAVINGS_SPLIT = 1 / 3
RETIRE_SPLIT  = 2 / 3
GLOBAL_SEED   = 42

ASSET_CLASSES = ["SA Equities", "SA Bonds", "Global Equities",
                 "SA Listed Property", "Inflation-Linked", "Cash & MM"]
N_ASSETS = len(ASSET_CLASSES)

EXPECTED_RETURNS = np.array([0.1350, 0.1050, 0.1200, 0.1100, 0.0950, 0.0850])
VOLATILITIES     = np.array([0.18,   0.08,   0.22,   0.16,   0.07,   0.01])

CORR_MATRIX = np.array([
    [1.00, 0.05, 0.55, 0.55, 0.00, 0.00],
    [0.05, 1.00, 0.05, 0.20, 0.75, 0.35],
    [0.55, 0.05, 1.00, 0.40, 0.00, 0.00],
    [0.55, 0.20, 0.40, 1.00, 0.10, 0.00],
    [0.00, 0.75, 0.00, 0.10, 1.00, 0.20],
    [0.00, 0.35, 0.00, 0.00, 0.20, 1.00],
])
COV_MATRIX = np.outer(VOLATILITIES, VOLATILITIES) * CORR_MATRIX

# Strategic asset allocation used as the fund's current liquidity profile
ASSET_ALLOCATION = {
    "SA Equities": 0.35, "SA Bonds": 0.25, "Global Equities": 0.20,
    "SA Listed Property": 0.095, "Inflation-Linked": 0.10, "Cash & MM": 0.005,
}
LIQUIDITY_SCORES = {
    "SA Equities": 0.70, "SA Bonds": 0.80, "Global Equities": 0.60,
    "SA Listed Property": 0.45, "Inflation-Linked": 0.65, "Cash & MM": 1.00,
}

# Savings-pot Reg-28-style constraints (indices into ASSET_CLASSES)
SAV_CONSTRAINTS = [
    lambda w: w[5] >= 0.15,   # Cash & MM >= 15%
    lambda w: w[0] <= 0.20,   # SA Equities <= 20%
    lambda w: w[2] <= 0.20,   # Global Equities <= 20%
]
RET_CONSTRAINTS = [
    lambda w: w[2] <= 0.45,            # Offshore <= 45%
    lambda w: w[0] + w[2] + w[3] <= 0.75,  # Total equity <= 75%
]


# ═════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════

# ── Fund module ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_fund(n_members, contribution_rate, seed=GLOBAL_SEED):
    """Simulate a realistic SA DC pension fund membership and balances."""
    rng = np.random.default_rng(seed)
    ages             = rng.uniform(22, 62, n_members)
    years_of_service = ages - 22
    monthly_salaries = np.clip(rng.lognormal(np.log(35000), 0.65, n_members), 8000, 500000)
    annual_salaries  = monthly_salaries * 12
    annual_contribs  = annual_salaries * contribution_rate

    def fv_annuity(pmt, r, n):
        if r == 0 or n == 0:
            return pmt * n
        return pmt * ((1 + r) ** n - 1) / r

    r = 0.07
    vested_balances = np.array([
        fv_annuity(annual_contribs[i], r, max(0, years_of_service[i] - 1))
        for i in range(n_members)
    ])
    savings_balances = annual_contribs * SAVINGS_SPLIT * 0.5
    retire_balances  = annual_contribs * RETIRE_SPLIT  * 0.5

    df = pd.DataFrame({
        "member_id": range(1, n_members + 1),
        "age": ages, "years_of_service": years_of_service,
        "monthly_salary": monthly_salaries, "annual_salary": annual_salaries,
        "annual_contribution": annual_contribs,
        "vested_balance": vested_balances, "savings_balance": savings_balances,
        "retire_balance": retire_balances,
    })
    df["total_balance"] = df["vested_balance"] + df["savings_balance"] + df["retire_balance"]
    return df


@st.cache_data(show_spinner=False)
def fund_metrics(n_members, contribution_rate):
    """Headline fund metrics, computed independently of any page."""
    df = generate_fund(n_members, contribution_rate)
    total_aum = df["total_balance"].sum()
    return {
        "df": df,
        "total_aum": total_aum,
        "total_savings": df["savings_balance"].sum(),
        "savings_pct": df["savings_balance"].sum() / total_aum * 100,
        "avg_age": df["age"].mean(),
        "median_salary": df["monthly_salary"].median(),
    }


# ── Liquidity module ──────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_stress(pct, n_members, contribution_rate, months,
               cash_alloc, market_impact_factor, seed=GLOBAL_SEED):
    """Stress test a single mass-withdrawal scenario."""
    df = generate_fund(n_members, contribution_rate)
    total_aum = df["total_balance"].sum()
    rng = np.random.default_rng(seed)

    eligible    = df[df["savings_balance"] >= 2000]
    withdrawers = eligible.sample(frac=pct, random_state=seed)
    total_w     = withdrawers["savings_balance"].sum()
    monthly_w   = rng.dirichlet(np.ones(months)) * total_w
    liquid_buf  = total_aum * cash_alloc
    cumulative  = np.cumsum(monthly_w)
    forced      = np.maximum(0, cumulative - liquid_buf)
    impact      = forced * market_impact_factor
    shortfall   = np.maximum(0, cumulative - liquid_buf)
    return {
        "monthly_withdrawals": monthly_w,
        "cumulative": cumulative,
        "forced_equity_sales": forced,
        "liquidity_shortfall": shortfall,
        "total_withdrawal": total_w,
        "liquid_buffer": liquid_buf,
        "total_market_impact": impact[-1] / 1e6,
    }


@st.cache_data(show_spinner=False)
def run_all_stress(n_members, contribution_rate, months, cash_alloc,
                   market_impact_factor, scenarios_tuple):
    """Run every scenario; returns dict keyed by scenario name."""
    return {
        name: run_stress(pct, n_members, contribution_rate, months,
                         cash_alloc, market_impact_factor)
        for name, pct in scenarios_tuple
    }


# ── Investment module ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def optimise_pot(constraints_key, risk_free):
    """Max-Sharpe optimisation for a pot. constraints_key in {'savings','retire'}."""
    mu  = pd.Series(EXPECTED_RETURNS, index=ASSET_CLASSES)
    cov = pd.DataFrame(COV_MATRIX, index=ASSET_CLASSES, columns=ASSET_CLASSES)
    ef  = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
    cons = SAV_CONSTRAINTS if constraints_key == "savings" else RET_CONSTRAINTS
    for c in cons:
        ef.add_constraint(c)
    ef.max_sharpe(risk_free_rate=risk_free)
    weights = ef.clean_weights()
    perf    = ef.portfolio_performance(risk_free_rate=risk_free)  # (ret, vol, sharpe)
    return dict(weights), perf


@st.cache_data(show_spinner=False)
def scan_frontier(constraints_key, risk_free, n_points=120):
    """Trace an efficient frontier by scanning target returns."""
    mu  = pd.Series(EXPECTED_RETURNS, index=ASSET_CLASSES)
    cov = pd.DataFrame(COV_MATRIX, index=ASSET_CLASSES, columns=ASSET_CLASSES)
    cons = SAV_CONSTRAINTS if constraints_key == "savings" else RET_CONSTRAINTS

    rets_out, vols_out = [], []
    for target in np.linspace(mu.min() * 0.9, mu.max() * 1.1, n_points):
        try:
            ef = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
            for c in cons:
                ef.add_constraint(c)
            ef.efficient_return(target_return=target)
            p = ef.portfolio_performance(risk_free_rate=risk_free)
            rets_out.append(p[0]); vols_out.append(p[1])
        except Exception:
            pass
    rets = np.array(rets_out); vols = np.array(vols_out)
    sharpe = (rets - risk_free) / vols if len(vols) else np.array([])
    order = np.argsort(vols) if len(vols) else []
    return rets[order], vols[order], sharpe[order]


def reg28_utilisation(weights):
    """Equity exposure vs the 75% Regulation 28 growth cap."""
    eq = weights.get("SA Equities", 0) + weights.get("Global Equities", 0) \
         + weights.get("SA Listed Property", 0)
    return eq / 0.75 * 100


# ── Member outcome module ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def simulate_member(withdraw_savings, n_sims, years, savings_mu, savings_sigma,
                    retire_mu, retire_sigma, contribution_rate, salary_growth,
                    monthly_salary, vested_start=450_000, savings_start=7_000,
                    retire_start=14_000, seed=GLOBAL_SEED):
    """Monte Carlo of a single member's retirement wealth (final balances)."""
    rng = np.random.default_rng(seed)
    annual_salary = monthly_salary * 12
    final_balances = np.empty(n_sims)
    for i in range(n_sims):
        vested, savings, retire = vested_start, savings_start, retire_start
        salary = annual_salary
        sav_rets  = rng.lognormal(np.log(1 + savings_mu) - 0.5 * savings_sigma ** 2, savings_sigma, years)
        ret_rets  = rng.lognormal(np.log(1 + retire_mu)  - 0.5 * retire_sigma ** 2,  retire_sigma,  years)
        vest_rets = rng.lognormal(np.log(1 + retire_mu)  - 0.5 * retire_sigma ** 2,  retire_sigma,  years)
        for y in range(years):
            savings += salary * contribution_rate * savings_split_const()
            retire  += salary * contribution_rate * (1 - savings_split_const())
            savings *= sav_rets[y]; retire *= ret_rets[y]; vested *= vest_rets[y]
            if withdraw_savings and savings >= 2000:
                savings = 0
            salary *= (1 + salary_growth)
        final_balances[i] = vested + retire + (savings if not withdraw_savings else 0)
    return final_balances


def savings_split_const():
    return SAVINGS_SPLIT


@st.cache_data(show_spinner=False)
def simulate_member_paths(n_paths, years, savings_mu, savings_sigma, retire_mu,
                          retire_sigma, contribution_rate, salary_growth,
                          monthly_salary, member_age, retirement_age,
                          vested_start=450_000, savings_start=7_000,
                          retire_start=14_000, seed=GLOBAL_SEED):
    """Year-by-year wealth paths for a fan chart (preserve & withdraw)."""
    rng = np.random.default_rng(seed)
    annual_salary = monthly_salary * 12
    preserve = np.zeros((n_paths, years + 1))
    withdraw = np.zeros((n_paths, years + 1))
    for flag, storage in [(False, preserve), (True, withdraw)]:
        for sim in range(n_paths):
            v, s, r, sal = vested_start, savings_start, retire_start, annual_salary
            storage[sim, 0] = v + s + r
            sr = rng.lognormal(np.log(1 + savings_mu) - 0.5 * savings_sigma ** 2, savings_sigma, years)
            rr = rng.lognormal(np.log(1 + retire_mu)  - 0.5 * retire_sigma ** 2,  retire_sigma,  years)
            vr = rng.lognormal(np.log(1 + retire_mu)  - 0.5 * retire_sigma ** 2,  retire_sigma,  years)
            for y in range(years):
                s += sal * contribution_rate * SAVINGS_SPLIT
                r += sal * contribution_rate * RETIRE_SPLIT
                s *= sr[y]; r *= rr[y]; v *= vr[y]
                if flag and s >= 2000:
                    s = 0
                sal *= (1 + salary_growth)
                storage[sim, y + 1] = v + r + (s if not flag else 0)
    years_axis = np.arange(member_age, member_age + years + 1)
    return preserve, withdraw, years_axis


def insight(text, kind="info"):
    """Trustee-insight callout helper."""
    body = f"**🧭 Trustee Insight** — {text}"
    {"info": st.info, "success": st.success, "warning": st.warning}[kind](body)


# ═════════════════════════════════════════════════════════════════════
#  NAVIGATION
# ═════════════════════════════════════════════════════════════════════
st.sidebar.title("🇿🇦 Two-Pot ALM Platform")
st.sidebar.caption("Asset-Liability Management decision-support")

PAGES = [
    "🏠 Intro & Two-Pot Context",
    "👥 Understanding the Fund",
    "🚨 Liquidity Risk",
    "📈 Investment Strategy",
    "👤 Member Outcomes",
    "🤖 AI Trustee Advisory",
]
page = st.sidebar.radio("Dashboard Navigation", PAGES)
st.sidebar.divider()

# ── Page-specific sidebar controls ───────────────────────────────────
# Sensible global defaults so every page can compute independently.
N_MEMBERS         = 1000
CONTRIBUTION_RATE = 0.15
RISK_FREE         = 0.085
member_age        = 35
retirement_age    = 65
monthly_salary    = 35000
N_SIMS            = 2000
stress_horizon    = 12
withdraw_rate     = 0.45
cash_alloc        = ASSET_ALLOCATION["Cash & MM"]
market_impact     = 0.05
salary_growth     = 0.06
api_key_input     = ""
run_ai            = False

if page == "👥 Understanding the Fund":
    st.sidebar.header("Fund Parameters")
    N_MEMBERS         = st.sidebar.slider("Number of Members", 500, 5000, 1000, step=100)
    CONTRIBUTION_RATE = st.sidebar.slider("Contribution Rate (%)", 10, 25, 15) / 100

elif page == "🚨 Liquidity Risk":
    st.sidebar.header("Liquidity Stress Controls")
    stress_horizon = st.sidebar.slider("Stress Horizon (months)", 3, 24, 12)
    withdraw_rate  = st.sidebar.slider("Headline Participation Rate (%)", 10, 90, 45) / 100
    cash_alloc     = st.sidebar.slider("Cash Allocation / Buffer (%)", 0.5, 15.0, 0.5, step=0.5) / 100
    market_impact  = st.sidebar.slider("Market Impact Factor (%)", 1, 15, 5) / 100

elif page == "📈 Investment Strategy":
    st.sidebar.header("Investment Assumptions")
    RISK_FREE = st.sidebar.slider("Risk-Free Rate (Repo, %)", 6.0, 10.0, 8.5) / 100

elif page == "👤 Member Outcomes":
    st.sidebar.header("Member Profile")
    member_age     = st.sidebar.slider("Current Age", 25, 55, 35)
    retirement_age = st.sidebar.slider("Retirement Age", 60, 70, 65)
    monthly_salary = st.sidebar.slider("Monthly Salary (R)", 10000, 150000, 35000, step=1000)
    salary_growth  = st.sidebar.slider("Salary Growth (%)", 0, 12, 6) / 100
    N_SIMS         = st.sidebar.slider("Monte Carlo Paths", 500, 5000, 2000, step=500)

elif page == "🤖 AI Trustee Advisory":
    st.sidebar.header("🤖 Claude AI Advisory")
    api_key_input = st.sidebar.text_input("Anthropic API Key (optional)", type="password",
                                          placeholder="sk-ant-...")
    run_ai = st.sidebar.button("Run AI Advisory Report", type="primary")

st.sidebar.divider()
st.sidebar.caption("Python · PyPortfolioOpt · Plotly · Streamlit · Claude API")


# ═════════════════════════════════════════════════════════════════════
#  PAGE 1 — INTRO
# ═════════════════════════════════════════════════════════════════════
if page == "🏠 Intro & Two-Pot Context":
    st.title("🇿🇦 The Two-Pot Story")
    st.subheader("An Asset-Liability Management decision-support platform for SA pension funds")

    st.markdown("""
    **Once upon a time**, South African pension fund members contributed steadily towards
    retirement, trusting decades of compounding to deliver long-term security.

    On **1 September 2024**, the **Two-Pot Retirement System** was introduced, fundamentally changing 
    how retirement savings can be accessed by allowing members to withdraw a portion of their savings before retirement. 

    This shift introduced a new reality for pension funds: **liquidity risk from potential large-scale withdrawals**, 
    requiring trustees to rethink how cash flows are managed.
    At the same time, **investment strategies must now balance competing objectives** — maintaining sufficient 
    liquidity to meet withdrawals while still preserving long-term growth for retirement adequacy.
    For members, this creates a behavioural trade-off: **frequent early withdrawals can significantly reduce long-term 
    retirement outcomes due to the loss of compounding over time**.

    In response, trustees now need an **integrated decision-support framework** to manage liquidity, optimise portfolios, 
    and protect member outcomes. *That is what this platform is.*
    """)

    st.divider()
    st.markdown("#### How the Two-Pot system splits every future contribution")

    # Simple process-flow / split visual
    flow = go.Figure()
    flow.add_trace(go.Bar(
        x=[33.3], y=["Future contribution"], orientation="h", name="Savings Pot (1/3)",
        marker_color=C_GOLD, hovertemplate="Savings Pot: 1/3 — accessible before retirement<extra></extra>"))
    flow.add_trace(go.Bar(
        x=[66.7], y=["Future contribution"], orientation="h", name="Retirement Pot (2/3)",
        marker_color=C_BLUE, hovertemplate="Retirement Pot: 2/3 — preserved to retirement<extra></extra>"))
    flow.update_layout(barmode="stack")
    style_fig(flow, height=200)
    flow.update_xaxes(title="Share of contribution (%)", range=[0, 100])
    st.plotly_chart(flow, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("""
    - **Savings Pot — 1/3 of future contributions.** Withdrawable before retirement (subject to tax/rules). *This is the new liquidity exposure.*
    - **Retirement Pot — 2/3 of future contributions.** Preserved until retirement.
    - **Vested Pot.** Existing accumulated savings under the previous rules.
    """)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Effective date", "1 Sep 2024")
    c2.metric("Savings pot share", "33%")
    c3.metric("Core new risk", "Liquidity")
    c4.metric("Decision-makers", "Trustees")

    insight("This platform answers one board-level question: *How should SA pension funds "
            "adapt liquidity management and investment strategy under the Two-Pot system?* "
            "Walk the modules left to right — each chapter builds on the last.", "info")

    st.divider()
    st.markdown("#### 🧭 How to navigate")
    st.markdown("""
    | Chapter | Page | The question it answers |
    |---|---|---|
    | 1 | 👥 **Understanding the Fund** | Who are the members behind the numbers? |
    | 2 | 🚨 **Liquidity Risk** | What happens if members start withdrawing? |
    | 3 | 📈 **Investment Strategy** | How should the fund adapt? |
    | 4 | 👤 **Member Outcomes** | What does this mean for retirement security? |
    | 5 | 🤖 **AI Trustee Advisory** | What should trustees do next? |
    """)


# ═════════════════════════════════════════════════════════════════════
#  PAGE 2 — FUND
# ═════════════════════════════════════════════════════════════════════
elif page == "👥 Understanding the Fund":
    st.title("👥 Who Are the Members Behind the Numbers?")
    st.markdown("""
    Before we can talk about risk, we need to meet the fund. This chapter simulates a realistic
    South African defined-contribution membership — their ages, salaries, and how their wealth
    is split across the **Vested**, **Savings**, and **Retirement** pots. The size of that
    Savings Pot is the seed of every liquidity question that follows.
    """)

    fm = fund_metrics(N_MEMBERS, CONTRIBUTION_RATE)
    fund_df = fm["df"]
    total_aum = fm["total_aum"]

    # Executive metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total AUM", f"R{total_aum/1e9:.2f}bn")
    c2.metric("Active Members", f"{N_MEMBERS:,}")
    c3.metric("Average Age", f"{fm['avg_age']:.1f} yrs")
    c4.metric("Savings Pot % of Assets", f"{fm['savings_pct']:.2f}%")

    st.divider()
    left, right = st.columns(2)

    with left:
        fig = px.histogram(fund_df, x="age", nbins=30, color_discrete_sequence=[C_BLUE])
        fig.add_vline(x=fm["avg_age"], line_dash="dash", line_color=C_GOLD,
                      annotation_text=f"Mean {fm['avg_age']:.1f}")
        fig.update_traces(hovertemplate="Age %{x:.0f}<br>%{y} members<extra></extra>")
        style_fig(fig, title="Age Distribution", legend_bottom=False)
        fig.update_xaxes(title="Age"); fig.update_yaxes(title="Members")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with right:
        sal_k = fund_df["monthly_salary"] / 1000
        fig = px.histogram(sal_k, nbins=40, color_discrete_sequence=[C_GREEN])
        fig.add_vline(x=fund_df["monthly_salary"].median()/1000, line_dash="dash",
                      line_color=C_GOLD,
                      annotation_text=f"Median R{fund_df['monthly_salary'].median()/1000:.0f}k")
        fig.update_traces(hovertemplate="R%{x:.0f}k<br>%{y} members<extra></extra>")
        style_fig(fig, title="Monthly Salary Distribution", legend_bottom=False)
        fig.update_xaxes(title="Monthly Salary (Rk)"); fig.update_yaxes(title="Members")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    left, right = st.columns(2)

    with left:
        age_bins   = [22, 30, 40, 50, 62]
        age_labels = ["22-30", "31-40", "41-50", "51-62"]
        fund_df = fund_df.assign(age_group=pd.cut(fund_df["age"], bins=age_bins, labels=age_labels))
        grouped = fund_df.groupby("age_group", observed=True)[
            ["vested_balance", "savings_balance", "retire_balance"]].mean() / 1000
        fig = go.Figure()
        fig.add_bar(x=age_labels, y=grouped["vested_balance"], name="Vested Pot", marker_color=C_PURPLE)
        fig.add_bar(x=age_labels, y=grouped["retire_balance"], name="Retirement Pot", marker_color=C_BLUE)
        fig.add_bar(x=age_labels, y=grouped["savings_balance"], name="Savings Pot", marker_color=C_GOLD)
        fig.update_layout(barmode="stack")
        fig.update_traces(hovertemplate="%{fullData.name}<br>Age %{x}<br>R%{y:.0f}k<extra></extra>")
        style_fig(fig, title="Average Balance by Age Group")
        fig.update_xaxes(title="Age Group"); fig.update_yaxes(title="Avg Balance (Rk)")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with right:
        pot_totals = [fund_df["vested_balance"].sum(), fund_df["retire_balance"].sum(),
                      fund_df["savings_balance"].sum()]
        fig = go.Figure(go.Pie(
            labels=["Vested Pot", "Retirement Pot", "Savings Pot"], values=pot_totals,
            marker_colors=[C_PURPLE, C_BLUE, C_GOLD], hole=0.45,
            hovertemplate="%{label}<br>R%{value:,.0f}<br>%{percent}<extra></extra>"))
        style_fig(fig, title="Fund Asset Composition")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    left, right = st.columns(2)

    with left:
        fig = px.scatter(
            fund_df, x="age", y=fund_df["savings_balance"]/1000, color=fund_df["monthly_salary"]/1000,
            color_continuous_scale="YlOrRd", opacity=0.6,
            labels={"color": "Salary (Rk)", "y": "Savings Balance (Rk)", "age": "Age"})
        fig.update_traces(marker=dict(size=6),
                          hovertemplate="Age %{x:.0f}<br>Savings R%{y:.1f}k<extra></extra>")
        style_fig(fig, title="Savings Pot Balance vs Age", legend_bottom=False)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with right:
        deciles  = np.percentile(fund_df["annual_salary"], np.arange(10, 110, 10)) / 1000
        sav_flow = deciles * CONTRIBUTION_RATE * SAVINGS_SPLIT
        ret_flow = deciles * CONTRIBUTION_RATE * RETIRE_SPLIT
        dlabels  = [f"D{i}" for i in range(1, 11)]
        fig = go.Figure()
        fig.add_bar(x=dlabels, y=ret_flow, name="Retirement (2/3)", marker_color=C_BLUE)
        fig.add_bar(x=dlabels, y=sav_flow, name="Savings (1/3)", marker_color=C_GOLD)
        fig.update_layout(barmode="stack")
        fig.update_traces(hovertemplate="%{fullData.name}<br>%{x}<br>R%{y:.0f}k<extra></extra>")
        style_fig(fig, title="Annual Contribution Flow by Salary Decile")
        fig.update_xaxes(title="Salary Decile"); fig.update_yaxes(title="Annual Contrib (Rk)")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    insight(f"The Savings Pot represents **{fm['savings_pct']:.2f}%** of total fund assets today, "
            f"about **R{fm['total_savings']/1e6:.1f}m**. It is small relative to AUM but it is the "
            f"slice members can withdraw — so it drives the liquidity risk explored next.", "info")


# ═════════════════════════════════════════════════════════════════════
#  PAGE 3 — LIQUIDITY
# ═════════════════════════════════════════════════════════════════════
elif page == "🚨 Liquidity Risk":
    st.title("🚨 What Happens If Members Start Withdrawing?")
    st.markdown("""
    Now the plot thickens. The Savings Pot can be withdrawn — so what if many members do so at
    once? This chapter is a **treasury risk dashboard**: it stress-tests mass-withdrawal
    scenarios against the fund's cash buffer, then measures the forced asset sales and market
    impact costs that follow when withdrawals outrun available liquidity. Adjust the scenario
    controls in the sidebar to explore your own stress.
    """)

    scenarios = (
        ("Base Case (20%)", 0.20),
        (f"Selected Stress ({withdraw_rate*100:.0f}%)", withdraw_rate),
        ("Severe Stress (70%)", 0.70),
    )
    scenario_colors = {scenarios[0][0]: C_GREEN, scenarios[1][0]: C_GOLD, scenarios[2][0]: C_RED}

    results = run_all_stress(N_MEMBERS, CONTRIBUTION_RATE, stress_horizon,
                             cash_alloc, market_impact, scenarios)
    fm = fund_metrics(N_MEMBERS, CONTRIBUTION_RATE)
    total_aum = fm["total_aum"]
    sel_name  = scenarios[1][0]
    sel       = results[sel_name]
    buffer    = sel["liquid_buffer"]
    funding_gap = max(0, sel["total_withdrawal"] - buffer)

    # Executive metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected Withdrawals", f"R{sel['total_withdrawal']/1e6:.1f}m")
    c2.metric("Liquidity Buffer", f"R{buffer/1e6:.1f}m")
    c3.metric("Funding Gap", f"R{funding_gap/1e6:.1f}m")
    c4.metric("Market Impact Cost", f"R{sel['total_market_impact']:.2f}m")

    st.divider()
    months_x = np.arange(1, stress_horizon + 1)
    left, right = st.columns(2)

    with left:
        fig = go.Figure()
        for name, r in results.items():
            fig.add_trace(go.Scatter(
                x=months_x, y=r["monthly_withdrawals"]/1e6, mode="lines+markers", name=name,
                line=dict(color=scenario_colors[name], width=2),
                hovertemplate="Month %{x}<br>Withdrawals R%{y:.2f}m<extra>"+name+"</extra>"))
        style_fig(fig, title="Monthly Withdrawal Cashflows")
        fig.update_xaxes(title="Month"); fig.update_yaxes(title="Withdrawals (Rm)")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with right:
        fig = go.Figure()
        buf_m = buffer / 1e6
        for name, r in results.items():
            fig.add_trace(go.Scatter(
                x=months_x, y=r["cumulative"]/1e6, mode="lines", name=name,
                line=dict(color=scenario_colors[name], width=2),
                customdata=r["liquidity_shortfall"]/1e6,
                hovertemplate="Month %{x}<br>Cumulative R%{y:.1f}m<br>Shortfall R%{customdata:.1f}m<extra>"+name+"</extra>"))
        fig.add_hline(y=buf_m, line_dash="dash", line_color="white",
                      annotation_text=f"Cash buffer R{buf_m:.0f}m")
        style_fig(fig, title="Cumulative Withdrawals vs Liquidity Buffer")
        fig.update_xaxes(title="Month"); fig.update_yaxes(title="Cumulative (Rm)")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    left, right = st.columns(2)

    with left:
        fig = go.Figure()
        for name, r in results.items():
            fig.add_trace(go.Scatter(
                x=months_x, y=r["forced_equity_sales"]/1e6, mode="lines", name=name,
                line=dict(color=scenario_colors[name], width=2), fill="tozeroy",
                fillcolor="rgba(255,107,107,0.06)",
                hovertemplate="Month %{x}<br>Forced sales R%{y:.1f}m<extra>"+name+"</extra>"))
        style_fig(fig, title="Cumulative Forced Asset Sales")
        fig.update_xaxes(title="Month"); fig.update_yaxes(title="Forced Sales (Rm)", rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with right:
        anames = list(ASSET_ALLOCATION.keys())
        avals  = [total_aum * v / 1e6 for v in ASSET_ALLOCATION.values()]
        lcolors = [C_GREEN if LIQUIDITY_SCORES[a] >= 0.75 else C_GOLD if LIQUIDITY_SCORES[a] >= 0.60
                   else C_RED for a in anames]
        fig = go.Figure(go.Bar(
            x=avals, y=anames, orientation="h", marker_color=lcolors,
            customdata=[LIQUIDITY_SCORES[a] for a in anames],
            hovertemplate="%{y}<br>R%{x:.0f}m<br>Liquidity score %{customdata:.2f}<extra></extra>"))
        style_fig(fig, title="Asset Liquidity Map (Green=High · Amber=Med · Red=Low)",
                  legend_bottom=False)
        fig.update_xaxes(title="Market Value (Rm)")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # Market impact bar
    names = list(results.keys())
    impacts = [results[n]["total_market_impact"] for n in names]
    fig = go.Figure(go.Bar(
        x=[n.split("(")[0].strip() for n in names], y=impacts,
        marker_color=[scenario_colors[n] for n in names],
        text=[f"R{v:.2f}m" for v in impacts], textposition="outside",
        hovertemplate="%{x}<br>Impact R%{y:.2f}m<extra></extra>"))
    style_fig(fig, title="Estimated Market Impact Cost by Scenario", legend_bottom=False)
    fig.update_yaxes(title="Impact Cost (Rm)")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # Trustee insight — find breach month for severe scenario
    sev = results["Severe Stress (70%)"]
    breach = np.argmax(sev["cumulative"] > buffer) + 1 if np.any(sev["cumulative"] > buffer) else None
    if funding_gap > 0:
        insight(f"Under the selected **{withdraw_rate*100:.0f}%** scenario, withdrawals of "
                f"R{sel['total_withdrawal']/1e6:.1f}m exceed the R{buffer/1e6:.1f}m cash buffer by "
                f"**R{funding_gap/1e6:.1f}m**, implying ~R{sel['total_market_impact']:.2f}m of "
                f"market-impact cost from forced sales.", "warning")
    else:
        insight(f"Under the selected **{withdraw_rate*100:.0f}%** scenario the cash buffer absorbs "
                f"all withdrawals — no forced sales. But test the severe case before relaxing.", "success")
    if breach:
        insight(f"Under **severe stress (70%)**, cumulative withdrawals exceed available cash after "
                f"**Month {breach}**, suggesting the fund may need to liquidate growth assets during "
                f"potentially stressed markets. A larger cash/near-cash buffer is the first lever.", "warning")


# ═════════════════════════════════════════════════════════════════════
#  PAGE 4 — INVESTMENT
# ═════════════════════════════════════════════════════════════════════
elif page == "📈 Investment Strategy":
    st.title("📈 How Should the Fund Adapt?")
    st.markdown("""
    Liquidity risk understood, the question becomes investment design. The two pots have
    different jobs: the **Savings Pot** must stay liquid and calm, while the **Retirement Pot**
    chases long-term growth within Regulation 28. Using PyPortfolioOpt we trace each
    **efficient frontier** and locate the **max-Sharpe** portfolio for each pot. Move the
    risk-free rate in the sidebar to see the frontier respond live.
    """)

    sav_weights, sav_perf = optimise_pot("savings", RISK_FREE)
    ret_weights, ret_perf = optimise_pot("retire", RISK_FREE)
    with st.spinner("Tracing efficient frontiers..."):
        sav_fr, sav_fv, sav_sh = scan_frontier("savings", RISK_FREE)
        ret_fr, ret_fv, ret_sh = scan_frontier("retire", RISK_FREE)

    # Executive metrics (retirement pot headline + reg28)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retirement Exp. Return", f"{ret_perf[0]*100:.1f}%")
    c2.metric("Retirement Volatility", f"{ret_perf[1]*100:.1f}%")
    c3.metric("Retirement Sharpe", f"{ret_perf[2]:.2f}")
    c4.metric("Reg 28 Utilisation", f"{reg28_utilisation(ret_weights):.0f}%")

    st.divider()

    # Centrepiece: interactive efficient frontier
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sav_fv*100, y=sav_fr*100, mode="markers+lines", name="Savings frontier",
        line=dict(color=C_GOLD, width=2),
        marker=dict(size=6, color=sav_sh, colorscale="YlOrRd",
                    colorbar=dict(title="Sharpe", x=1.02)),
        hovertemplate="Vol %{x:.1f}%<br>Return %{y:.1f}%<extra>Savings</extra>"))
    fig.add_trace(go.Scatter(
        x=ret_fv*100, y=ret_fr*100, mode="markers+lines", name="Retirement frontier",
        line=dict(color=C_BLUE, width=2),
        marker=dict(size=6, color=ret_sh, colorscale="Blues"),
        hovertemplate="Vol %{x:.1f}%<br>Return %{y:.1f}%<extra>Retirement</extra>"))
    # Optimal stars
    fig.add_trace(go.Scatter(
        x=[sav_perf[1]*100], y=[sav_perf[0]*100], mode="markers", name="Savings optimal",
        marker=dict(symbol="star", size=20, color=C_GOLD, line=dict(color="white", width=1)),
        hovertemplate=f"Savings optimal<br>Ret {sav_perf[0]*100:.1f}%<br>Vol {sav_perf[1]*100:.1f}%<br>Sharpe {sav_perf[2]:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[ret_perf[1]*100], y=[ret_perf[0]*100], mode="markers", name="Retirement optimal",
        marker=dict(symbol="star", size=20, color=C_BLUE, line=dict(color="white", width=1)),
        hovertemplate=f"Retirement optimal<br>Ret {ret_perf[0]*100:.1f}%<br>Vol {ret_perf[1]*100:.1f}%<br>Sharpe {ret_perf[2]:.2f}<extra></extra>"))
    # Asset markers
    fig.add_trace(go.Scatter(
        x=VOLATILITIES*100, y=EXPECTED_RETURNS*100, mode="markers+text", name="Asset classes",
        text=ASSET_CLASSES, textposition="top center", textfont=dict(size=9, color=C_TEXT),
        marker=dict(symbol="diamond", size=11, color=ASSET_COLORS),
        hovertemplate="%{text}<br>Vol %{x:.1f}%<br>Return %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=RISK_FREE*100, line_dash="dash", line_color="white",
                  annotation_text=f"Risk-free {RISK_FREE*100:.1f}%")
    style_fig(fig, height=560, title="Mean-Variance Efficient Frontiers — Savings vs Retirement Pot")
    fig.update_xaxes(title="Volatility (%)"); fig.update_yaxes(title="Expected Return (%)")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    insight("Drag to zoom, hover any frontier point for its risk/return, and click legend items "
            "to isolate a pot. The gold star sits lower-left (safe, liquid); the blue star sits "
            "upper-right (more growth, more volatility) — exactly the Two-Pot trade-off.", "info")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.markdown(f"#### 🏦 Savings Pot — Max Sharpe")
        st.caption(f"Return {sav_perf[0]*100:.2f}% · Vol {sav_perf[1]*100:.2f}% · Sharpe {sav_perf[2]:.3f}")
        vals = {k: v for k, v in sav_weights.items() if v > 0.005}
        fig = go.Figure(go.Pie(labels=list(vals.keys()), values=list(vals.values()), hole=0.45,
                               marker_colors=ASSET_COLORS,
                               hovertemplate="%{label}<br>%{percent}<extra></extra>"))
        style_fig(fig, title=None, legend_bottom=False)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with right:
        st.markdown(f"#### 🔒 Retirement Pot — Max Sharpe")
        st.caption(f"Return {ret_perf[0]*100:.2f}% · Vol {ret_perf[1]*100:.2f}% · Sharpe {ret_perf[2]:.3f}")
        vals = {k: v for k, v in ret_weights.items() if v > 0.005}
        fig = go.Figure(go.Pie(labels=list(vals.keys()), values=list(vals.values()), hole=0.45,
                               marker_colors=ASSET_COLORS,
                               hovertemplate="%{label}<br>%{percent}<extra></extra>"))
        style_fig(fig, title=None, legend_bottom=False)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # Sharpe along frontier
    fig = go.Figure()
    if len(sav_sh):
        fig.add_trace(go.Scatter(x=sav_fr*100, y=sav_sh, mode="lines", name="Savings",
                                 line=dict(color=C_GOLD, width=2),
                                 hovertemplate="Return %{x:.1f}%<br>Sharpe %{y:.2f}<extra>Savings</extra>"))
    if len(ret_sh):
        fig.add_trace(go.Scatter(x=ret_fr*100, y=ret_sh, mode="lines", name="Retirement",
                                 line=dict(color=C_BLUE, width=2),
                                 hovertemplate="Return %{x:.1f}%<br>Sharpe %{y:.2f}<extra>Retirement</extra>"))
    style_fig(fig, title="Sharpe Ratio Along the Frontier")
    fig.update_xaxes(title="Expected Return (%)"); fig.update_yaxes(title="Sharpe Ratio")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    insight(f"The optimal retirement portfolio targets **{ret_perf[0]*100:.1f}%** expected return "
            f"versus **{sav_perf[0]*100:.1f}%** for the savings pot — higher growth, but at "
            f"**{ret_perf[1]*100:.1f}%** volatility vs {sav_perf[1]*100:.1f}%. The savings pot "
            f"deliberately trades return for the liquidity the Two-Pot system demands.", "info")


# ═════════════════════════════════════════════════════════════════════
#  PAGE 5 — MEMBER OUTCOMES
# ═════════════════════════════════════════════════════════════════════
elif page == "👤 Member Outcomes":
    st.title("👤 What Does This Mean for Retirement Security?")
    st.markdown("""
    Every withdrawal is a member's decision with a decades-long consequence. This chapter runs a
    **Monte Carlo simulation** of one member's journey to retirement under thousands of market
    paths, comparing a member who **preserves** their Savings Pot against one who **withdraws**
    it each year. The gap between the two is the true, compounded cost of early access. Adjust
    the member profile in the sidebar to test different cases.
    """)

    years_to_retire = max(1, retirement_age - member_age)

    # Pull optimal portfolio stats independently (cross-page safe)
    _, sav_perf = optimise_pot("savings", RISK_FREE)
    _, ret_perf = optimise_pot("retire", RISK_FREE)
    sav_mu, sav_sig = sav_perf[0], sav_perf[1]
    ret_mu, ret_sig = ret_perf[0], ret_perf[1]

    with st.spinner(f"Running {N_SIMS:,} Monte Carlo simulations..."):
        preserve = simulate_member(False, N_SIMS, years_to_retire, sav_mu, sav_sig,
                                   ret_mu, ret_sig, CONTRIBUTION_RATE, salary_growth, monthly_salary)
        withdraw = simulate_member(True, N_SIMS, years_to_retire, sav_mu, sav_sig,
                                   ret_mu, ret_sig, CONTRIBUTION_RATE, salary_growth, monthly_salary)

    median_p = np.median(preserve); median_w = np.median(withdraw)
    cost = (median_p - median_w) / 1e6
    target = median_p  # treat preserve-median as the "target" wealth
    prob_target = (withdraw >= target).mean() * 100
    # crude "years lost": fraction of wealth lost x years to retire
    years_lost = (1 - median_w / median_p) * years_to_retire if median_p else 0

    # Executive metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median Wealth — Preserve", f"R{median_p/1e6:.2f}m")
    c2.metric("Cost of Withdrawals", f"R{cost:.2f}m", delta=f"-{cost:.2f}m", delta_color="inverse")
    c3.metric("P(Withdraw ≥ Preserve median)", f"{prob_target:.0f}%")
    c4.metric("Effective Years Lost", f"{years_lost:.1f} yrs")

    st.divider()
    left, right = st.columns(2)

    with left:
        hi = max(preserve.max(), withdraw.max())/1e6
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=preserve/1e6, name="Preserve", marker_color=C_GREEN,
                                   opacity=0.6, xbins=dict(start=0, end=hi, size=hi/60)))
        fig.add_trace(go.Histogram(x=withdraw/1e6, name="Withdraw", marker_color=C_RED,
                                   opacity=0.6, xbins=dict(start=0, end=hi, size=hi/60)))
        fig.add_vline(x=median_p/1e6, line_dash="dash", line_color=C_GREEN)
        fig.add_vline(x=median_w/1e6, line_dash="dash", line_color=C_RED)
        fig.update_layout(barmode="overlay")
        style_fig(fig, title="Distribution of Retirement Balances")
        fig.update_xaxes(title="Total Retirement Wealth (Rm)"); fig.update_yaxes(title="Frequency")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with right:
        pcts = [10, 25, 50, 75, 90]
        plabels = ["P10 (Bad)", "P25", "P50 (Median)", "P75", "P90 (Good)"]
        pp = [np.percentile(preserve, p)/1e6 for p in pcts]
        wp = [np.percentile(withdraw, p)/1e6 for p in pcts]
        fig = go.Figure()
        fig.add_bar(x=plabels, y=pp, name="Preserve", marker_color=C_GREEN,
                    hovertemplate="%{x}<br>R%{y:.2f}m<extra>Preserve</extra>")
        fig.add_bar(x=plabels, y=wp, name="Withdraw", marker_color=C_RED,
                    hovertemplate="%{x}<br>R%{y:.2f}m<extra>Withdraw</extra>")
        fig.update_layout(barmode="group")
        style_fig(fig, title="Retirement Balance by Percentile")
        fig.update_yaxes(title="Total Retirement Wealth (Rm)")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # Fan chart
    n_fan = min(300, N_SIMS)
    pres_paths, with_paths, years_axis = simulate_member_paths(
        n_fan, years_to_retire, sav_mu, sav_sig, ret_mu, ret_sig,
        CONTRIBUTION_RATE, salary_growth, monthly_salary, member_age, retirement_age)

    fig = go.Figure()
    def add_band(paths, color, label):
        lo = np.percentile(paths, 10, axis=0)/1e6
        hi = np.percentile(paths, 90, axis=0)/1e6
        med = np.median(paths, axis=0)/1e6
        rgba = {C_GREEN: "rgba(57,211,83,0.15)", C_RED: "rgba(255,107,107,0.15)"}[color]
        fig.add_trace(go.Scatter(x=years_axis, y=hi, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=years_axis, y=lo, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=rgba, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=years_axis, y=med, mode="lines", name=f"{label} (median)",
                                 line=dict(color=color, width=2.5),
                                 hovertemplate="Age %{x}<br>R%{y:.2f}m<extra>"+label+"</extra>"))
    add_band(pres_paths, C_GREEN, "Preserve")
    add_band(with_paths, C_RED, "Withdraw")
    style_fig(fig, height=460, title="Wealth Accumulation Fan Chart (10th-90th percentile)")
    fig.update_xaxes(title="Age"); fig.update_yaxes(title="Total Retirement Wealth (Rm)")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    insight(f"Repeated annual withdrawals reduce median retirement wealth by "
            f"**R{cost:.2f}m** — from R{median_p/1e6:.2f}m to R{median_w/1e6:.2f}m for this "
            f"member, roughly **{years_lost:.1f} years** of lost accumulation. The flexibility of "
            f"the Savings Pot has a real, compounding long-term price.", "warning")


# ═════════════════════════════════════════════════════════════════════
#  PAGE 6 — AI ADVISORY
# ═════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Trustee Advisory":
    st.title("🤖 What Should Trustees Do Next?")
    st.markdown("""
    The story's final chapter turns analysis into action. This page recomputes the fund,
    liquidity, investment, and member-outcome results **independently** — so it works even if you
    open it first — and hands them to Claude to synthesise into a structured, board-level ALM
    advisory report. Enter your Anthropic API key in the sidebar and run the report.
    """)

    # ── Recompute every required input independently (cross-page safe) ──
    fm = fund_metrics(N_MEMBERS, CONTRIBUTION_RATE)
    fund_df, total_aum = fm["df"], fm["total_aum"]

    ai_scenarios = (("Base Case (20%)", 0.20),
                    ("Moderate Stress (45%)", 0.45),
                    ("Severe Stress (70%)", 0.70))
    results = run_all_stress(N_MEMBERS, CONTRIBUTION_RATE, stress_horizon,
                             cash_alloc, market_impact, ai_scenarios)

    _, sav_perf = optimise_pot("savings", RISK_FREE)
    _, ret_perf = optimise_pot("retire", RISK_FREE)

    years_to_retire = max(1, retirement_age - member_age)
    preserve = simulate_member(False, N_SIMS, years_to_retire, sav_perf[0], sav_perf[1],
                               ret_perf[0], ret_perf[1], CONTRIBUTION_RATE, salary_growth, monthly_salary)
    withdraw = simulate_member(True, N_SIMS, years_to_retire, sav_perf[0], sav_perf[1],
                               ret_perf[0], ret_perf[1], CONTRIBUTION_RATE, salary_growth, monthly_salary)
    cost = (np.median(preserve) - np.median(withdraw)) / 1e6
    buffer = total_aum * cash_alloc

    # Snapshot for the board
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total AUM", f"R{total_aum/1e9:.2f}bn")
    c2.metric("Severe Withdrawals", f"R{results['Severe Stress (70%)']['total_withdrawal']/1e6:.0f}m")
    c3.metric("Retirement Sharpe", f"{ret_perf[2]:.2f}")
    c4.metric("Cost of Withdrawals", f"R{cost:.2f}m")

    st.divider()

    fund_summary = f"""
FUND PROFILE:
- Active members: {N_MEMBERS:,}
- Total AUM: R{total_aum/1e9:.2f}bn
- Total savings pot: R{fm['total_savings']/1e6:.1f}m ({fm['savings_pct']:.1f}% of AUM)
- Average member age: {fm['avg_age']:.1f} years
- Median monthly salary: R{fm['median_salary']:,.0f}

LIQUIDITY STRESS TEST (horizon {stress_horizon} months, cash buffer R{buffer/1e6:.1f}m):
- Base case (20% withdraw): R{results['Base Case (20%)']['total_withdrawal']/1e6:.1f}m withdrawn
- Moderate stress (45% withdraw): R{results['Moderate Stress (45%)']['total_withdrawal']/1e6:.1f}m withdrawn
- Severe stress (70% withdraw): R{results['Severe Stress (70%)']['total_withdrawal']/1e6:.1f}m withdrawn
- Severe market-impact cost: R{results['Severe Stress (70%)']['total_market_impact']:.2f}m
- Severe scenario exceeds liquid buffer: {'YES' if results['Severe Stress (70%)']['total_withdrawal'] > buffer else 'NO'}

OPTIMAL INVESTMENT STRATEGY (PyPortfolioOpt - Max Sharpe, risk-free {RISK_FREE*100:.1f}%):
- Savings pot: Return {sav_perf[0]*100:.1f}% | Volatility {sav_perf[1]*100:.1f}% | Sharpe {sav_perf[2]:.3f}
- Retirement pot: Return {ret_perf[0]*100:.1f}% | Volatility {ret_perf[1]*100:.1f}% | Sharpe {ret_perf[2]:.3f}
- Key constraint: Regulation 28 caps offshore exposure at 45%

MEMBER OUTCOME PROJECTOR ({member_age}yo, R{monthly_salary:,}/month, {years_to_retire} yrs to retirement):
- Median balance if PRESERVING: R{np.median(preserve)/1e6:.2f}m
- Median balance if WITHDRAWING annually: R{np.median(withdraw)/1e6:.2f}m
- Median cost of withdrawals: R{cost:.2f}m less at retirement
- P10 preserve: R{np.percentile(preserve,10)/1e6:.2f}m vs withdraw: R{np.percentile(withdraw,10)/1e6:.2f}m
"""

    system_prompt = """You are a senior actuarial consultant specialising in South African pension fund
asset-liability management (ALM) and investment strategy, with 20 years advising trustees on the
SA Two-Pot retirement system. Produce a structured, professional board-level risk advisory report.
Be specific, cite the numbers, and give actionable recommendations. Authoritative but accessible."""

    user_prompt = f"""Based on these ALM analysis outputs for our SA DC pension fund under the Two-Pot
system, produce a structured board-level risk advisory report with these sections:

1. EXECUTIVE SUMMARY (3-4 sentences)
2. LIQUIDITY RISK ASSESSMENT
3. INVESTMENT STRATEGY RECOMMENDATIONS
4. MEMBER OUTCOME IMPLICATIONS
5. KEY RISKS & MITIGANTS (top 3)
6. TRUSTEE ACTION ITEMS (3 concrete decisions)

QUANTITATIVE INPUTS:
{fund_summary}"""

    if run_ai:
        if not api_key_input:
            st.error("Please enter your Anthropic API key in the sidebar first.")
        else:
            try:
                import anthropic
                with st.spinner("Calling Claude API..."):
                    client = anthropic.Anthropic(api_key=api_key_input)
                    message = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=2000,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                    )
                ai_report = message.content[0].text
                st.markdown("### 🤖 Claude ALM Advisory Report")
                st.caption("SA Two-Pot Pension Fund · Board Risk Committee")
                st.divider()
                st.markdown(ai_report)
            except Exception as e:
                st.error(f"API Error: {e}")
    else:
        st.info("👈 Enter your Claude API key in the sidebar and click **Run AI Advisory Report** "
                "to generate the board-level ALM report.")
        with st.expander("Preview the quantitative inputs that will be sent to Claude"):
            st.code(fund_summary, language="text")


# ═════════════════════════════════════════════════════════════════════
#  FOOTER
# ═════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    """
    <div style='text-align:center; color:#484F58; font-size:0.85em;'>
    Built with Python · PyPortfolioOpt · Plotly · Streamlit · NumPy · SciPy · Claude API (Anthropic)<br>
    SA market assumptions based on ASISA · SARB · JSE · Regulation 28 of the Pension Funds Act
    </div>
    """,
    unsafe_allow_html=True,
)