# ============================================================
#  SA Two-Pot ALM Decision-Support Simulator 
# ============================================================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ── PyPortfolioOpt imports ────────────────────────────────────────────
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt import plotting as ppt
from pypfopt.efficient_frontier import EfficientFrontier

# ─────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SA Two-Pot ALM Dashboard",
    page_icon="🇿🇦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────
C_GOLD   = '#F0B429'
C_GREEN  = '#39D353'
C_RED    = '#FF6B6B'
C_BLUE   = '#58A6FF'
C_PURPLE = '#BC8CFF'
C_ORANGE = '#FF9A3C'
C_TEAL   = '#26C6DA'

# ── Dark chart theme ──────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0D1117', 'axes.facecolor': '#161B22',
    'axes.edgecolor': '#30363D',   'axes.labelcolor': '#C9D1D9',
    'axes.titlecolor': '#E6EDF3',  'axes.titlesize': 12,
    'axes.titleweight': 'bold',    'axes.labelsize': 10,
    'xtick.color': '#8B949E',      'ytick.color': '#8B949E',
    'text.color': '#C9D1D9',       'grid.color': '#21262D',
    'grid.linewidth': 0.7,         'legend.facecolor': '#161B22',
    'legend.edgecolor': '#30363D', 'font.family': 'monospace',
    'figure.dpi': 110,
})

# ─────────────────────────────────────────────────────────────────────
#  SIDEBAR — CONTROLS
# ─────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Dashboard Controls")

st.sidebar.header("Fund Parameters")
N_MEMBERS       = st.sidebar.slider("Number of Members",    500, 5000, 1000, step=100)
CONTRIBUTION_RATE = st.sidebar.slider("Contribution Rate (%)", 10, 25, 15) / 100
RISK_FREE       = st.sidebar.slider("Risk-Free Rate (Repo, %)", 6.0, 10.0, 8.5) / 100

st.sidebar.header("Member Profile (Module 4)")
member_age      = st.sidebar.slider("Current Age",          25, 55, 35)
retirement_age  = st.sidebar.slider("Retirement Age",       60, 70, 65)
monthly_salary  = st.sidebar.slider("Monthly Salary (R)",   10000, 150000, 35000, step=1000)
N_SIMS          = st.sidebar.slider("Monte Carlo Paths",    500, 5000, 2000, step=500)

st.sidebar.header("🤖 Claude AI Advisory")
api_key_input   = st.sidebar.text_input("Anthropic API Key (optional)", type="password",
                                         placeholder="sk-ant-...")
run_ai          = st.sidebar.button("Run AI Advisory Report", type="primary")

# ─────────────────────────────────────────────────────────────────────
#  TITLE
# ─────────────────────────────────────────────────────────────────────
st.title("🇿🇦 Two-Pot Pension Fund ALM Simulator")
st.subheader("Liquidity Risk, Investment Strategy & Member Outcome Analysis")
st.markdown("""
**Tools:** Python · PyPortfolioOpt · Streamlit · Actuarial Science · Stochastic Modelling  
**Context:** SA Defined Contribution (DC) pension fund — Two-Pot System (effective 1 September 2024)
""")
st.divider()

# ─────────────────────────────────────────────────────────────────────
# DASHBOARD TABS
# ─────────────────────────────────────────────────────────────────────
tab_intro, tab_fund, tab_liquidity, tab_investment, tab_member, tab_ai = st.tabs([
    "🏠 Intro & Two-Pot Context",
    "👥 Understanding the Fund",
    "🚨 Liquidity Risk",
    "📈 Investment Strategy",
    "👤 Member Outcomes",
    "🤖 AI Trustee Advisory"
])
with tab_intro:

    st.header("🇿🇦 South African Two-Pot Retirement System")

    st.markdown("""
The South African Two-Pot retirement system, effective from **1 September 2024**, fundamentally changes how retirement fund members access their pension savings.

Under the new system:

- **1/3 of future contributions** are allocated to a **Savings Pot**
  - Members may withdraw from this pot before retirement (subject to rules and taxes)

- **2/3 of future contributions** are allocated to a **Retirement Pot**
  - Preserved until retirement

- Existing accumulated savings remain in a **Vested Pot**
  - Subject to previous retirement rules

This reform improves short-term financial flexibility for members, but introduces significant new risks for pension funds and trustees.
""")

    st.divider()

    st.header("⚠️ The Core Problem")

    st.markdown("""
The Two-Pot system creates major challenges for South African pension funds:

### 🚨 Liquidity Risk
Large-scale member withdrawals may force funds to sell assets unexpectedly, creating liquidity pressure and market impact costs.

### 📉 Investment Strategy Challenges
Funds must now balance:
- Long-term growth objectives
- Liquidity requirements
- Regulation 28 constraints
- Member withdrawal behaviour

### 👤 Member Outcome Risk
Frequent withdrawals can materially reduce long-term retirement wealth and increase retirement insecurity.

### 🧠 Trustee Decision Complexity
Trustees need integrated tools to understand:
- withdrawal stress scenarios,
- portfolio resilience,
- and long-term member impacts.
""")

    st.divider()

    st.header("💡 How This Dashboard Helps")

    st.markdown("""
This dashboard is designed as an **Asset-Liability Management (ALM) decision-support tool** for:

- Pension fund trustees
- Investment committees
- Actuaries
- Consultants
- Asset managers

The platform combines:
- actuarial fund modelling,
- liquidity stress testing,
- portfolio optimisation,
- Monte Carlo simulation,
- and AI-generated trustee reporting

into a single integrated environment.
""")

    st.divider()

    st.header("🧭 Dashboard Roadmap")

    st.markdown("""
### 👥 Understanding the Fund
Simulates the demographic structure, salaries, balances, and contribution flows of a South African defined contribution pension fund.

### 🚨 Liquidity Risk
Stress tests mass withdrawal scenarios to assess whether the fund can withstand large-scale savings-pot withdrawals.

### 📈 Investment Strategy
Uses modern portfolio theory and PyPortfolioOpt to determine optimal asset allocations for the Savings Pot and Retirement Pot.

### 👤 Member Outcomes
Projects retirement outcomes under thousands of market scenarios to assess the long-term cost of withdrawals.

### 🤖 AI Trustee Advisory
Transforms quantitative results into a board-level strategic advisory report using Claude AI.
""")

    st.divider()

    st.info("""
🎯 Core Question Answered by the Dashboard:

'How should South African pension funds adapt their liquidity management and investment strategy under the Two-Pot retirement system?'
""")

# ─────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────
SAVINGS_SPLIT = 1/3
RETIRE_SPLIT  = 2/3
np.random.seed(42)

ASSET_CLASSES = ['SA Equities', 'SA Bonds', 'Global Equities',
                 'SA Listed Property', 'Inflation-Linked', 'Cash & MM']
n_assets = len(ASSET_CLASSES)

expected_returns_arr = np.array([0.1350, 0.1050, 0.1200, 0.1100, 0.0950, 0.0850])
volatilities_arr     = np.array([0.18,   0.08,   0.22,   0.16,   0.07,   0.01])

corr_matrix = np.array([
    [1.00, 0.05, 0.55, 0.55, 0.00, 0.00],
    [0.05, 1.00, 0.05, 0.20, 0.75, 0.35],
    [0.55, 0.05, 1.00, 0.40, 0.00, 0.00],
    [0.55, 0.20, 0.40, 1.00, 0.10, 0.00],
    [0.00, 0.75, 0.00, 0.10, 1.00, 0.20],
    [0.00, 0.35, 0.00, 0.00, 0.20, 1.00],
])
cov_matrix = np.outer(volatilities_arr, volatilities_arr) * corr_matrix

# ─────────────────────────────────────────────────────────────────────
#  MODULE 1 — FUND SETUP
# ─────────────────────────────────────────────────────────────────────
with tab_fund:
    st.divider()
    st.header("👥 Understanding the Pension Fund")
    st.markdown("""
> Before assessing risk and investment strategy, trustees first need to understand the structure of the pension fund itself.
> 
> This section simulates a realistic South African defined contribution pension fund, including member demographics, salaries, contributions, and accumulated balances across the Vested, Savings, and Retirement pots.
> 
> These insights help assess:
> - member demographics,
> - contribution concentration,
> - wealth distribution,
> - and the size of the liquid savings pot exposure.
""")
    @st.cache_data
    def generate_fund(n_members, contribution_rate, seed=42):
        np.random.seed(seed)
        ages             = np.random.uniform(22, 62, n_members)
        years_of_service = ages - 22
        monthly_salaries = np.clip(np.random.lognormal(np.log(35000), 0.65, n_members), 8000, 500000)
        annual_salaries  = monthly_salaries * 12
        annual_contribs  = annual_salaries * contribution_rate

        def fv_annuity(pmt, r, n):
            if r == 0 or n == 0: return pmt * n
            return pmt * ((1 + r)**n - 1) / r

        r = 0.07
        vested_balances  = np.array([fv_annuity(annual_contribs[i], r, max(0, years_of_service[i]-1))
                                    for i in range(n_members)])
        savings_balances = annual_contribs * SAVINGS_SPLIT * 0.5
        retire_balances  = annual_contribs * RETIRE_SPLIT  * 0.5

        df = pd.DataFrame({
            'member_id': range(1, n_members + 1),
            'age': ages, 'years_of_service': years_of_service,
            'monthly_salary': monthly_salaries, 'annual_salary': annual_salaries,
            'annual_contribution': annual_contribs,
            'vested_balance': vested_balances, 'savings_balance': savings_balances,
            'retire_balance': retire_balances,
        })
        df['total_balance'] = df['vested_balance'] + df['savings_balance'] + df['retire_balance']
        return df

    fund_df = generate_fund(N_MEMBERS, CONTRIBUTION_RATE)
    total_aum = fund_df['total_balance'].sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Active Members",       f"{N_MEMBERS:,}")
    col2.metric("Total AUM",            f"R{total_aum/1e9:.2f}bn")
    col3.metric("Total Savings Pot",    f"R{fund_df['savings_balance'].sum()/1e6:.1f}m")
    col4.metric("Average Age",          f"{fund_df['age'].mean():.1f} yrs")
    col5.metric("Median Monthly Salary",f"R{fund_df['monthly_salary'].median():,.0f}")

    fig1, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig1.patch.set_facecolor('#0D1117')
    fig1.suptitle('FUND MEMBER DEMOGRAPHICS & BALANCE STRUCTURE',
                fontsize=13, fontweight='bold', color='#E6EDF3')

    ax = axes[0, 0]
    ax.hist(fund_df['age'], bins=30, color=C_BLUE, alpha=0.85, edgecolor='#0D1117')
    ax.axvline(fund_df['age'].mean(), color=C_GOLD, linewidth=2, linestyle='--',
            label=f"Mean: {fund_df['age'].mean():.1f}")
    ax.set_title('Age Distribution'); ax.set_xlabel('Age'); ax.set_ylabel('Members')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(fund_df['monthly_salary']/1000, bins=40, color=C_GREEN, alpha=0.85, edgecolor='#0D1117')
    ax.axvline(fund_df['monthly_salary'].median()/1000, color=C_GOLD, linewidth=2, linestyle='--',
            label=f"Median: R{fund_df['monthly_salary'].median()/1000:.0f}k")
    ax.set_title('Monthly Salary Distribution'); ax.set_xlabel('Salary (Rk)'); ax.set_ylabel('Members')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    age_bins   = [22, 30, 40, 50, 62]
    age_labels = ['22–30', '31–40', '41–50', '51–62']
    fund_df['age_group'] = pd.cut(fund_df['age'], bins=age_bins, labels=age_labels)
    grouped = fund_df.groupby('age_group', observed=True)[['vested_balance','savings_balance','retire_balance']].mean()/1000
    ax.bar(age_labels, grouped['vested_balance'],  color=C_PURPLE, label='Vested Pot',     alpha=0.9)
    ax.bar(age_labels, grouped['retire_balance'],  bottom=grouped['vested_balance'], color=C_BLUE,   label='Retirement Pot', alpha=0.9)
    ax.bar(age_labels, grouped['savings_balance'], bottom=grouped['vested_balance']+grouped['retire_balance'], color=C_GOLD, label='Savings Pot', alpha=0.9)
    ax.set_title('Avg Balance by Age Group'); ax.set_xlabel('Age Group'); ax.set_ylabel('Avg Balance (Rk)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 0]
    pot_totals = [fund_df['vested_balance'].sum(), fund_df['retire_balance'].sum(), fund_df['savings_balance'].sum()]
    pot_labels = ['Vested Pot', 'Retirement Pot', 'Savings Pot']
    total_p = sum(pot_totals)
    pcts    = [x/total_p*100 for x in pot_totals]
    wedges, _ = ax.pie(pot_totals, labels=None, colors=[C_PURPLE, C_BLUE, C_GOLD], startangle=90)
    ax.legend(wedges, [f"{l} ({p:.1f}%)" for l, p in zip(pot_labels, pcts)],
            title='Pot Type', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.set_title('Fund Asset Composition')

    ax = axes[1, 1]
    sc = ax.scatter(fund_df['age'], fund_df['savings_balance']/1000,
                    c=fund_df['monthly_salary']/1000, cmap='YlOrRd', alpha=0.5, s=12)
    plt.colorbar(sc, ax=ax, label='Salary (Rk)')
    ax.set_title('Savings Pot Balance vs Age'); ax.set_xlabel('Age'); ax.set_ylabel('Balance (Rk)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    deciles     = np.percentile(fund_df['annual_salary'], np.arange(10, 110, 10))/1000
    sav_flow    = deciles * CONTRIBUTION_RATE * SAVINGS_SPLIT
    ret_flow    = deciles * CONTRIBUTION_RATE * RETIRE_SPLIT
    dlabels     = [f'D{i}' for i in range(1, 11)]
    ax.bar(dlabels, ret_flow,  color=C_BLUE, alpha=0.9, label='Retirement (2/3)')
    ax.bar(dlabels, sav_flow, bottom=ret_flow, color=C_GOLD, alpha=0.9, label='Savings (1/3)')
    ax.set_title('Annual Contribution Flow by Salary Decile'); ax.set_xlabel('Decile'); ax.set_ylabel('Annual Contrib (Rk)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

# ─────────────────────────────────────────────────────────────────────
#  MODULE 2 — CASHFLOW STRESS TEST
# ─────────────────────────────────────────────────────────────────────
with tab_liquidity:
    st.divider()
    st.header("🚨 Can the Fund Survive Mass Withdrawals?")
    st.markdown("""
> One of the biggest risks introduced by the Two-Pot system is the possibility of large-scale member withdrawals from the Savings Pot.
>
> Trustees therefore need to understand whether the fund has sufficient liquidity to meet withdrawal demands without disrupting long-term investment strategy or forcing distressed asset sales.
>
> This section stress tests the fund under multiple withdrawal scenarios, modelling:
> - member withdrawal behaviour,
> - monthly cashflow pressure,
> - liquidity buffer depletion,
> - forced asset sales,
> - and estimated market impact costs.
>
> The analysis helps assess the resilience of the fund under withdrawal stress and highlights where liquidity management strategies may need to change under the Two-Pot regime.
""")

    asset_allocation = {
        'SA Equities': 0.35, 'SA Bonds': 0.25, 'Global Equities': 0.20,
        'SA Listed Property': 0.095, 'Inflation-Linked': 0.10, 'Cash & MM': 0.005,
    }
    liquidity_scores = {
        'SA Equities': 0.70, 'SA Bonds': 0.80, 'Global Equities': 0.60,
        'SA Listed Property': 0.45, 'Inflation-Linked': 0.65, 'Cash & MM': 1.00,
    }
    MARKET_IMPACT_FACTOR = 0.05
    MONTHS = 12
    scenarios = {
        'Base Case (20%)':      0.20,
        'Moderate Stress (45%)': 0.45,
        'Severe Stress (70%)':  0.70,
    }
    scenario_colors = {
        'Base Case (20%)':      C_GREEN,
        'Moderate Stress (45%)': C_GOLD,
        'Severe Stress (70%)':  C_RED,
    }

    def run_stress(pct, fund_df, total_aum, months):
        eligible   = fund_df[fund_df['savings_balance'] >= 2000]
        withdrawers = eligible.sample(frac=pct, random_state=42)
        total_w    = withdrawers['savings_balance'].sum()
        monthly_w  = np.random.dirichlet(np.ones(months)) * total_w
        liquid_buf = total_aum * asset_allocation['Cash & MM']
        cumulative = np.cumsum(monthly_w)
        forced     = np.maximum(0, cumulative - liquid_buf)
        impact     = forced * MARKET_IMPACT_FACTOR
        return {
            'monthly_withdrawals':  monthly_w,
            'cumulative':           cumulative,
            'forced_equity_sales':  forced,
            'total_withdrawal':     total_w,
            'total_market_impact': impact[-1] / 1e6,
        }

    results = {name: run_stress(pct, fund_df, total_aum, MONTHS) for name, pct in scenarios.items()}

    fig2, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig2.patch.set_facecolor('#0D1117')
    fig2.suptitle('CASHFLOW STRESS TEST: MASS WITHDRAWAL SCENARIOS',
                fontsize=13, fontweight='bold', color='#E6EDF3')
    months_x = np.arange(1, MONTHS + 1)

    ax = axes[0, 0]
    for name, r in results.items():
        ax.plot(months_x, r['monthly_withdrawals']/1e6, color=scenario_colors[name], linewidth=2, label=name, marker='o', markersize=5)
    ax.set_title('Monthly Withdrawal Cashflows'); ax.set_xlabel('Month'); ax.set_ylabel('Withdrawals (Rm)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    liquid_line = total_aum * asset_allocation['Cash & MM'] / 1e6
    ax.axhline(liquid_line, color='white', linewidth=1.5, linestyle='--', alpha=0.6, label=f'Cash Buffer: R{liquid_line:.0f}m')
    for name, r in results.items():
        ax.plot(months_x, r['cumulative']/1e6, color=scenario_colors[name], linewidth=2, label=name)
    sev_cum = results['Severe Stress (70%)']['cumulative']/1e6
    ax.fill_between(months_x, sev_cum, liquid_line, where=sev_cum > liquid_line, alpha=0.15, color=C_RED, label='Forced Sale Zone')
    ax.set_title('Cumulative Withdrawals vs Liquidity Buffer'); ax.set_xlabel('Month'); ax.set_ylabel('Cumulative (Rm)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    anames = list(asset_allocation.keys())
    avals  = [total_aum * v / 1e6 for v in asset_allocation.values()]
    lcolors= [C_GREEN if liquidity_scores[a] >= 0.75 else C_GOLD if liquidity_scores[a] >= 0.60 else C_RED for a in anames]
    bars   = ax.barh(anames, avals, color=lcolors, alpha=0.85, edgecolor='#0D1117')
    ax.set_title('Asset Portfolio — Liquidity Map\n(Green=High | Amber=Med | Red=Low)')
    ax.set_xlabel('Market Value (Rm)'); ax.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, avals):
        ax.text(val+2, bar.get_y()+bar.get_height()/2, f'R{val:.0f}m', va='center', fontsize=7.5, color='#C9D1D9')

    ax = axes[1, 0]
    for name, r in results.items():
        ax.plot(months_x, r['forced_equity_sales']/1e6, color=scenario_colors[name], linewidth=2, label=name)
        ax.fill_between(months_x, r['forced_equity_sales']/1e6, alpha=0.08, color=scenario_colors[name])
    ax.set_title('Cumulative Forced Equity Sales'); ax.set_xlabel('Month'); ax.set_ylabel('Forced Sales (Rm)')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    names   = list(results.keys())
    impacts = [r['total_market_impact'] for r in results.values()]
    bcolors = [scenario_colors[n] for n in names]
    ax.bar([n.split('(')[0].strip() for n in names], impacts, color=bcolors, alpha=0.85, edgecolor='#0D1117')
    for i, (name, imp) in enumerate(zip(names, impacts)):
        if imp > 0:
            ax.text(i, imp, f'R{imp:.2f}m', ha='center', fontsize=8)
    ax.set_title('Estimated Market Impact Cost'); ax.set_ylabel('Impact Cost (Rm)'); ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 2]
    eligible = fund_df[fund_df['savings_balance'] >= 2000]
    ax.hist(eligible['savings_balance']/1000, bins=40, color=C_PURPLE, alpha=0.85, edgecolor='#0D1117')
    ax.axvline(eligible['savings_balance'].median()/1000, color=C_GOLD, linewidth=2, linestyle='--',
            label=f"Median: R{eligible['savings_balance'].median()/1000:.1f}k")
    ax.set_title('Distribution of Savings Pot Balances'); ax.set_xlabel('Savings Balance (Rk)'); ax.set_ylabel('Members')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig2.subplots_adjust(wspace=0.30, hspace=0.40)
    st.pyplot(fig2, use_container_width=True)
    plt.close()

# ─────────────────────────────────────────────────────────────────────
#  MODULE 3 — INVESTMENT OPTIMISER (PyPortfolioOpt)
# ─────────────────────────────────────────────────────────────────────
with tab_investment:
    st.divider()
    st.header("📈 How Should the Pots Be Invested?")
    st.markdown("""
> Once liquidity risks are understood, the next challenge is determining how the Savings Pot and Retirement Pot should be invested.
>
> The Two-Pot system fundamentally changes pension fund portfolio construction:
> - the Savings Pot requires higher liquidity and lower volatility,
> - while the Retirement Pot remains focused on long-term growth.
>
> This section applies modern portfolio theory using PyPortfolioOpt to identify optimal asset allocations under realistic South African market assumptions and Regulation 28 constraints.
>
> The efficient frontier analysis illustrates the trade-off between risk and return, while the optimal portfolios demonstrate how investment strategy differs between liquidity-focused and long-term retirement objectives.
""")

    # ── Build covariance matrix as a DataFrame (PyPortfolioOpt format) ───
    mu_series  = pd.Series(expected_returns_arr, index=ASSET_CLASSES)
    cov_df     = pd.DataFrame(cov_matrix, index=ASSET_CLASSES, columns=ASSET_CLASSES)

    # ── Savings Pot: Max Sharpe with conservative constraints ─────────────
    # PyPortfolioOpt EfficientFrontier accepts weight_bounds and sector_mapper constraints
    ef_savings = EfficientFrontier(mu_series, cov_df, weight_bounds=(0, 1))
    # Savings pot constraints: ≥15% cash, ≤20% SA Equities, ≤20% Global Equities
    ef_savings.add_constraint(lambda w: w[5] >= 0.15)          # Cash >= 15%
    ef_savings.add_constraint(lambda w: w[0] <= 0.20)          # SA Equities <= 20%
    ef_savings.add_constraint(lambda w: w[2] <= 0.20)          # Global Equities <= 20%
    ef_savings.max_sharpe(risk_free_rate=RISK_FREE)
    sav_weights_clean = ef_savings.clean_weights()
    sav_perf          = ef_savings.portfolio_performance(risk_free_rate=RISK_FREE)

    # ── Retirement Pot: Max Sharpe with Regulation 28 constraints ────────
    ef_retire = EfficientFrontier(mu_series, cov_df, weight_bounds=(0, 1))
    ef_retire.add_constraint(lambda w: w[2] <= 0.45)                        # Offshore <= 45%
    ef_retire.add_constraint(lambda w: w[0] + w[2] + w[3] <= 0.75)         # Total equity <= 75%
    ef_retire.max_sharpe(risk_free_rate=RISK_FREE)
    ret_weights_clean = ef_retire.clean_weights()
    ret_perf          = ef_retire.portfolio_performance(risk_free_rate=RISK_FREE)

    # ── Efficient frontier by scanning target returns ─────────────────────
    def scan_frontier(mu, cov, extra_constraints=None, n_points=120):
        rets_out, vols_out, weights_out = [], [], []
        r_min = mu.min() * 0.9
        r_max = mu.max() * 1.1
        for target in np.linspace(r_min, r_max, n_points):
            try:
                ef_tmp = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
                if extra_constraints:
                    for c in extra_constraints:
                        ef_tmp.add_constraint(c)
                ef_tmp.efficient_return(target_return=target)
                p = ef_tmp.portfolio_performance(risk_free_rate=RISK_FREE)
                rets_out.append(p[0]); vols_out.append(p[1])
                weights_out.append(list(ef_tmp.clean_weights().values()))
            except Exception:
                pass
        return np.array(rets_out), np.array(vols_out), weights_out

    sav_cons = [lambda w: w[5] >= 0.15, lambda w: w[0] <= 0.20, lambda w: w[2] <= 0.20]
    ret_cons = [lambda w: w[2] <= 0.45, lambda w: w[0]+w[2]+w[3] <= 0.75]

    with st.spinner("Computing efficient frontiers with PyPortfolioOpt..."):
        sav_fr, sav_fv, _  = scan_frontier(mu_series, cov_df, sav_cons)
        ret_fr, ret_fv, _  = scan_frontier(mu_series, cov_df, ret_cons)

    sav_sharpe_curve = (sav_fr - RISK_FREE) / sav_fv if len(sav_fv) > 0 else np.array([])
    ret_sharpe_curve = (ret_fr - RISK_FREE) / ret_fv if len(ret_fv) > 0 else np.array([])

    # ── Display optimal weights ───────────────────────────────────────────
    col_s, col_r = st.columns(2)
    with col_s:
        st.markdown("#### 🏦 Optimal Savings Pot Portfolio (Max Sharpe)")
        st.markdown(f"**Return: {sav_perf[0]*100:.2f}%** | **Vol: {sav_perf[1]*100:.2f}%** | **Sharpe: {sav_perf[2]:.3f}**")
        sav_df = pd.DataFrame(list(sav_weights_clean.items()), columns=['Asset Class', 'Weight'])
        sav_df['Weight (%)'] = (sav_df['Weight'] * 100).round(1)
        sav_df = sav_df[sav_df['Weight (%)'] > 0.5][['Asset Class', 'Weight (%)']]
        st.dataframe(sav_df, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("#### 🔒 Optimal Retirement Pot Portfolio (Max Sharpe)")
        st.markdown(f"**Return: {ret_perf[0]*100:.2f}%** | **Vol: {ret_perf[1]*100:.2f}%** | **Sharpe: {ret_perf[2]:.3f}**")
        ret_df = pd.DataFrame(list(ret_weights_clean.items()), columns=['Asset Class', 'Weight'])
        ret_df['Weight (%)'] = (ret_df['Weight'] * 100).round(1)
        ret_df = ret_df[ret_df['Weight (%)'] > 0.5][['Asset Class', 'Weight (%)']]
        st.dataframe(ret_df, use_container_width=True, hide_index=True)

    # ── Plot Module 3 ─────────────────────────────────────────────────────
    fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig3.patch.set_facecolor('#0D1117')
    fig3.suptitle('INVESTMENT STRATEGY OPTIMISER (PyPortfolioOpt)',
                fontsize=13, fontweight='bold', color='#E6EDF3')

    # Plot 1: Dual efficient frontier
    ax = axes[0, 0]
    ax.set_title('Mean-Variance Efficient Frontiers')
    if len(sav_fv) > 0:
        idx = np.argsort(sav_fv)
        sav_fv_sorted = sav_fv[idx]
        sav_fr_sorted = sav_fr[idx]
        sav_sharpe_sorted = sav_sharpe_curve[idx]

        ax.plot(sav_fv_sorted * 100, sav_fr_sorted * 100,
                color=C_GOLD, linewidth=2, label='Savings Pot')

        ax.scatter(sav_fv_sorted * 100, sav_fr_sorted * 100,
                c=sav_sharpe_sorted, cmap='YlOrRd', s=25, alpha=0.9)
    if len(ret_fv) > 0:
        idx = np.argsort(ret_fv)
        ret_fv_sorted = ret_fv[idx]
        ret_fr_sorted = ret_fr[idx]
        ret_sharpe_sorted = ret_sharpe_curve[idx]

        ax.plot(ret_fv_sorted * 100, ret_fr_sorted * 100,
                color=C_BLUE, linewidth=2, label='Retirement Pot')

        ax.scatter(ret_fv_sorted * 100, ret_fr_sorted * 100,
                c=ret_sharpe_sorted, cmap='Blues', s=25, alpha=0.9)
    ax.scatter(sav_perf[1]*100, sav_perf[0]*100, s=200, color=C_GOLD, marker='*', zorder=5,
            label=f'Sav Opt: {sav_perf[0]*100:.1f}%/{sav_perf[1]*100:.1f}%')
    ax.scatter(ret_perf[1]*100, ret_perf[0]*100, s=200, color=C_BLUE, marker='*', zorder=5,
            label=f'Ret Opt: {ret_perf[0]*100:.1f}%/{ret_perf[1]*100:.1f}%')
    asset_colors_list = [C_GREEN, C_BLUE, C_PURPLE, C_ORANGE, C_TEAL, 'white']
    for i, (asset, col) in enumerate(zip(ASSET_CLASSES, asset_colors_list)):
        ax.scatter(volatilities_arr[i]*100, expected_returns_arr[i]*100, s=70, color=col, marker='D', alpha=0.85, zorder=4)
        ax.annotate(asset, (volatilities_arr[i]*100, expected_returns_arr[i]*100),
                    textcoords='offset points', xytext=(4, 2), fontsize=6.5, color=col)
    ax.axhline(RISK_FREE*100, color='white', linewidth=0.8, linestyle='--', alpha=0.35)
    ax.set_xlabel('Volatility (%)'); ax.set_ylabel('Expected Return (%)'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Plot 2: Savings pot optimal allocation pie
    ax = axes[0, 1]
    sav_vals   = [v for v in sav_weights_clean.values() if v > 0.005]
    sav_labels = [k for k, v in sav_weights_clean.items() if v > 0.005]
    ax.pie(sav_vals, labels=sav_labels, autopct='%1.1f%%',
        colors=asset_colors_list[:len(sav_labels)], startangle=90,
        textprops={'fontsize': 9, 'color': '#C9D1D9'})
    ax.axis('equal')
    ax.set_title(f'🏦 Savings Pot Allocation\nRet: {sav_perf[0]*100:.1f}% | Vol: {sav_perf[1]*100:.1f}%')

    # Plot 3: Retirement pot optimal allocation pie
    ax = axes[1, 0]
    ret_vals   = [v for v in ret_weights_clean.values() if v > 0.005]
    ret_labels = [k for k, v in ret_weights_clean.items() if v > 0.005]
    ax.pie(ret_vals, labels=ret_labels, autopct='%1.1f%%',
        colors=asset_colors_list[:len(ret_labels)], startangle=90,
        textprops={'fontsize': 9, 'color': '#C9D1D9'})
    ax.axis('equal')
    ax.set_title(f'🔒 Retirement Pot Allocation\nRet: {ret_perf[0]*100:.1f}% | Vol: {ret_perf[1]*100:.1f}%')

    # Plot 4: Sharpe curve
    ax = axes[1, 1]
    if len(sav_sharpe_curve) > 0:
        ax.plot(sav_fr_sorted * 100, sav_sharpe_sorted,
                color=C_GOLD, linewidth=2, label='Savings Pot')
        ax.scatter(sav_perf[0]*100, sav_perf[2], s=100, color=C_GOLD, zorder=5)
    if len(ret_sharpe_curve) > 0:
        ax.plot(ret_fr_sorted * 100, ret_sharpe_sorted,
                color=C_BLUE, linewidth=2, label='Retirement Pot')
        ax.scatter(ret_perf[0]*100, ret_perf[2], s=100, color=C_BLUE, zorder=5)
    ax.set_title('Sharpe Ratio Along the Frontier')
    ax.set_xlabel('Expected Return (%)'); ax.set_ylabel('Sharpe Ratio')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout(pad=3)
    st.pyplot(fig3)
    plt.close()

# ─────────────────────────────────────────────────────────────────────
#  MODULE 4 — MEMBER OUTCOME PROJECTOR (Monte Carlo)
# ─────────────────────────────────────────────────────────────────────
with tab_member:
    st.divider()
    st.header("👤 How Do Withdrawals Affect Retirement Outcomes?")
    st.markdown("""
> While the Two-Pot system improves short-term financial flexibility, it may also reduce long-term retirement security if members repeatedly withdraw from their Savings Pot.
>
> Trustees and policymakers therefore need to understand the long-term behavioural consequences of early access to retirement savings.
>
> This section uses Monte Carlo simulation to project retirement outcomes under thousands of possible market scenarios, comparing members who preserve their Savings Pot against those who withdraw regularly.
>
> The results illustrate:
> - uncertainty in retirement outcomes,
> - the compounding impact of investment returns,
> - and the long-term cost of repeated withdrawals on retirement wealth.
""")

    years_to_retire = max(1, retirement_age - member_age)
    annual_salary_val = monthly_salary * 12
    vested_start      = 450_000
    savings_start     = 7_000
    retire_start      = 14_000
    annual_contrib_sav = annual_salary_val * CONTRIBUTION_RATE * SAVINGS_SPLIT
    annual_contrib_ret = annual_salary_val * CONTRIBUTION_RATE * RETIRE_SPLIT
    salary_growth     = 0.06

    # Use PyPortfolioOpt optimal portfolio performance numbers
    savings_mu_mc  = sav_perf[0]
    savings_sig_mc = sav_perf[1]
    retire_mu_mc   = ret_perf[0]
    retire_sig_mc  = ret_perf[1]

    @st.cache_data
    def simulate_member(withdraw_savings, n_sims, years, savings_mu, savings_sigma,
                        retire_mu, retire_sigma, vested_start, savings_start,
                        retire_start, annual_contrib_sav, annual_contrib_ret,
                        contribution_rate, savings_split, salary_growth, monthly_salary, seed=42):
        np.random.seed(seed)
        annual_salary = monthly_salary * 12
        final_balances = []
        for _ in range(n_sims):
            vested  = vested_start
            savings = savings_start
            retire  = retire_start
            salary  = annual_salary
            sav_rets  = np.random.lognormal(np.log(1+savings_mu)-0.5*savings_sigma**2, savings_sigma, years)
            ret_rets  = np.random.lognormal(np.log(1+retire_mu)-0.5*retire_sigma**2,  retire_sigma,  years)
            vest_rets = np.random.lognormal(np.log(1+retire_mu)-0.5*retire_sigma**2,  retire_sigma,  years)
            for y in range(years):
                ac_sav = salary * contribution_rate * savings_split
                ac_ret = salary * contribution_rate * (1 - savings_split)
                savings += ac_sav; retire += ac_ret
                savings *= sav_rets[y]; retire *= ret_rets[y]; vested *= vest_rets[y]
                if withdraw_savings and savings >= 2000:
                    savings = 0
                salary *= (1 + salary_growth)
            final_balances.append(vested + retire + (savings if not withdraw_savings else 0))
        return np.array(final_balances)

    with st.spinner(f"Running {N_SIMS:,} Monte Carlo simulations..."):
        preserve_balances = simulate_member(
            False, N_SIMS, max(1, years_to_retire), savings_mu_mc, savings_sig_mc,
            retire_mu_mc, retire_sig_mc, vested_start, savings_start, retire_start,
            annual_contrib_sav, annual_contrib_ret, CONTRIBUTION_RATE, SAVINGS_SPLIT,
            salary_growth, monthly_salary)
        withdraw_balances = simulate_member(
            True,  N_SIMS, max(1, years_to_retire), savings_mu_mc, savings_sig_mc,
            retire_mu_mc, retire_sig_mc, vested_start, savings_start, retire_start,
            annual_contrib_sav, annual_contrib_ret, CONTRIBUTION_RATE, SAVINGS_SPLIT,
            salary_growth, monthly_salary)

    cost = (np.median(preserve_balances) - np.median(withdraw_balances)) / 1e6

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Median Balance — Preserve",  f"R{np.median(preserve_balances)/1e6:.2f}m")
    col2.metric("Median Balance — Withdraw",  f"R{np.median(withdraw_balances)/1e6:.2f}m")
    col3.metric("Cost of Annual Withdrawals", f"R{cost:.2f}m less", delta_color="inverse")
    col4.metric("Simulation Paths",           f"{N_SIMS:,}")

    fig4, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig4.patch.set_facecolor('#0D1117')
    fig4.suptitle(f'MEMBER OUTCOME PROJECTOR: PRESERVE vs WITHDRAW ({N_SIMS:,}-Path Monte Carlo)',
                fontsize=12, fontweight='bold', color='#E6EDF3')

    ax = axes[0]
    bins = np.linspace(0, max(preserve_balances.max(), withdraw_balances.max())/1e6, 60)
    ax.hist(preserve_balances/1e6, bins=bins, color=C_GREEN, alpha=0.6, edgecolor='none', label='Preserve')
    ax.hist(withdraw_balances/1e6, bins=bins, color=C_RED,   alpha=0.6, edgecolor='none', label='Withdraw')
    ax.axvline(np.median(preserve_balances)/1e6, color=C_GREEN, linewidth=2.5, linestyle='--')
    ax.axvline(np.median(withdraw_balances)/1e6, color=C_RED,   linewidth=2.5, linestyle='--')
    ax.set_title('Distribution of Retirement Balances'); ax.set_xlabel('Total Retirement Wealth (Rm)'); ax.set_ylabel('Frequency')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    ax = axes[1]
    YEARS_MC = max(1, years_to_retire); N_FAN = min(300, N_SIMS)
    np.random.seed(42)
    annual_preserve = np.zeros((N_FAN, YEARS_MC + 1))
    annual_withdraw = np.zeros((N_FAN, YEARS_MC + 1))
    for mode_idx, (withdraw_flag, storage) in enumerate([(False, annual_preserve), (True, annual_withdraw)]):
        for sim in range(N_FAN):
            v = vested_start; s = savings_start; r = retire_start; sal = annual_salary_val
            storage[sim, 0] = v + s + r
            sr = np.random.lognormal(np.log(1+savings_mu_mc)-0.5*savings_sig_mc**2, savings_sig_mc, YEARS_MC)
            rr = np.random.lognormal(np.log(1+retire_mu_mc)-0.5*retire_sig_mc**2,   retire_sig_mc,  YEARS_MC)
            vr = np.random.lognormal(np.log(1+retire_mu_mc)-0.5*retire_sig_mc**2,   retire_sig_mc,  YEARS_MC)
            for y in range(YEARS_MC):
                s += sal * CONTRIBUTION_RATE * SAVINGS_SPLIT
                r += sal * CONTRIBUTION_RATE * RETIRE_SPLIT
                s *= sr[y]; r *= rr[y]; v *= vr[y]
                if withdraw_flag and s >= 2000: s = 0
                sal *= (1 + salary_growth)
                storage[sim, y+1] = v + r + (s if not withdraw_flag else 0)
    years_axis = np.arange(member_age, retirement_age + 1)[:YEARS_MC+1]
    for pct_lo, pct_hi, alpha in [(10, 90, 0.12), (25, 75, 0.22)]:
        ax.fill_between(years_axis, np.percentile(annual_preserve, pct_lo, axis=0)/1e6,
                        np.percentile(annual_preserve, pct_hi, axis=0)/1e6, alpha=alpha, color=C_GREEN)
        ax.fill_between(years_axis, np.percentile(annual_withdraw, pct_lo, axis=0)/1e6,
                        np.percentile(annual_withdraw, pct_hi, axis=0)/1e6, alpha=alpha, color=C_RED)
    ax.plot(years_axis, np.median(annual_preserve, axis=0)/1e6, color=C_GREEN, linewidth=2.5, label='Preserve (median)')
    ax.plot(years_axis, np.median(annual_withdraw, axis=0)/1e6, color=C_RED,   linewidth=2.5, label='Withdraw (median)')
    ax.set_title('Wealth Accumulation Fan Chart'); ax.set_xlabel('Age'); ax.set_ylabel('Total Retirement Wealth (Rm)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[2]
    pcts = [10, 25, 50, 75, 90]
    plabels = ['P10\n(Bad)', 'P25', 'P50\n(Median)', 'P75', 'P90\n(Good)']
    pp = [np.percentile(preserve_balances, p)/1e6 for p in pcts]
    wp = [np.percentile(withdraw_balances, p)/1e6 for p in pcts]
    x = np.arange(len(pcts)); w = 0.38
    b1 = ax.bar(x-w/2, pp, w, color=C_GREEN, alpha=0.85, edgecolor='#0D1117', label='Preserve')
    b2 = ax.bar(x+w/2, wp, w, color=C_RED,   alpha=0.85, edgecolor='#0D1117', label='Withdraw')
    for bar in b1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                            f'R{bar.get_height():.1f}m', ha='center', fontsize=6.5, color=C_GREEN)
    for bar in b2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                            f'R{bar.get_height():.1f}m', ha='center', fontsize=6.5, color=C_RED)
    ax.set_xticks(x); ax.set_xticklabels(plabels, fontsize=9)
    ax.set_title('Retirement Balance by Percentile'); ax.set_ylabel('Total Retirement Wealth (Rm)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

# ─────────────────────────────────────────────────────────────────────
#  MODULE 5 — CLAUDE AI ADVISORY
# ─────────────────────────────────────────────────────────────────────
with tab_ai:
    st.divider()
    st.header("🤖 Trustee AI Advisory Report")
    st.markdown("""
> Pension fund modelling often produces highly technical outputs that can be difficult for trustees and investment committees to interpret quickly.
>
> This section bridges the gap between quantitative modelling and strategic decision-making by using a large language model (Claude AI) to synthesise the dashboard outputs into a structured board-level advisory report.
>
> The AI-generated report translates the modelling results into:
> - liquidity risk insights,
> - investment strategy recommendations,
> - member outcome implications,
> - key risks and mitigants,
> - and practical trustee action items.
>
> This demonstrates how AI can support actuarial and investment professionals by transforming complex analytical outputs into accessible strategic recommendations.
""")

    st.markdown("""
    Paste your Anthropic API key in the sidebar and click **Run AI Advisory Report** to generate 
    a board-level ALM risk report powered by Claude.
    """)

    if run_ai:
        if not api_key_input or api_key_input == "":
            st.error("Please enter your Anthropic API key in the sidebar first.")
        else:
            fund_summary = f"""
    FUND PROFILE:
    - Active members: {N_MEMBERS:,}
    - Total AUM: R{total_aum/1e9:.2f}bn
    - Total savings pot: R{fund_df['savings_balance'].sum()/1e6:.1f}m ({fund_df['savings_balance'].sum()/total_aum*100:.1f}% of AUM)
    - Average member age: {fund_df['age'].mean():.1f} years
    - Median monthly salary: R{fund_df['monthly_salary'].median():,.0f}

    STRESS TEST RESULTS:
    - Base case (20% withdraw): R{results['Base Case (20%)']['total_withdrawal']/1e6:.1f}m withdrawn
    - Moderate stress (45% withdraw): R{results['Moderate Stress (45%)']['total_withdrawal']/1e6:.1f}m withdrawn
    - Severe stress (70% withdraw): R{results['Severe Stress (70%)']['total_withdrawal']/1e6:.1f}m withdrawn
    - Liquid buffer (Cash & MM): R{total_aum*asset_allocation['Cash & MM']/1e6:.1f}m
    - Severe scenario exceeds liquid buffer: {'YES' if results['Severe Stress (70%)']['total_withdrawal'] > total_aum*asset_allocation['Cash & MM'] else 'NO'}

    OPTIMAL INVESTMENT STRATEGY (PyPortfolioOpt — Max Sharpe):
    - Savings pot: Return {sav_perf[0]*100:.1f}% p.a. | Volatility {sav_perf[1]*100:.1f}% | Sharpe {sav_perf[2]:.3f}
    - Retirement pot: Return {ret_perf[0]*100:.1f}% p.a. | Volatility {ret_perf[1]*100:.1f}% | Sharpe {ret_perf[2]:.3f}
    - Key constraint: Regulation 28 caps offshore exposure at 45%

    MEMBER OUTCOME PROJECTOR ({member_age}-year-old, R{monthly_salary:,}/month, {years_to_retire} years to retirement):
    - Median balance if PRESERVING savings pot: R{np.median(preserve_balances)/1e6:.2f}m
    - Median balance if WITHDRAWING savings pot annually: R{np.median(withdraw_balances)/1e6:.2f}m
    - Median cost of annual withdrawals: R{cost:.2f}m less at retirement
    - P10 (bad outcome) preserve: R{np.percentile(preserve_balances,10)/1e6:.2f}m vs withdraw: R{np.percentile(withdraw_balances,10)/1e6:.2f}m
    """

            system_prompt = """You are a senior actuarial consultant specialising in South African pension fund
    asset-liability management (ALM) and investment strategy. You have 20 years of experience advising
    pension fund trustees and investment managers on the SA Two-Pot retirement system.
    Produce a structured, professional board-level risk advisory report. Be specific, cite the numbers,
    and give actionable recommendations. Write in a tone appropriate for a board pack — authoritative but accessible."""

            user_prompt = f"""Based on these ALM analysis outputs for our SA DC pension fund under the Two-Pot system,
    produce a structured board-level risk advisory report with these sections:

    1. EXECUTIVE SUMMARY (3-4 sentences)
    2. LIQUIDITY RISK ASSESSMENT
    3. INVESTMENT STRATEGY RECOMMENDATIONS  
    4. MEMBER OUTCOME IMPLICATIONS
    5. KEY RISKS & MITIGANTS (top 3)
    6. TRUSTEE ACTION ITEMS (3 concrete decisions)

    QUANTITATIVE INPUTS:
    {fund_summary}"""

            try:
                import anthropic
                with st.spinner("Calling Claude API..."):
                    client  = anthropic.Anthropic(api_key=api_key_input)
                    message = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=2000,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}]
                    )
                ai_report = message.content[0].text
                st.markdown("---")
                st.markdown("### 🤖 Claude ALM Advisory Report")
                st.markdown(f"*SA Two-Pot Pension Fund | Board Risk Committee*")
                st.divider()
                st.markdown(ai_report)
            except Exception as e:
                st.error(f"API Error: {e}")

    else:
        st.info("👈 Enter your Claude API key in the sidebar and click **Run AI Advisory Report** to generate the board-level ALM report.")

# ─────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align: center; color: #484F58; font-size: 0.85em;'>
Built with Python · PyPortfolioOpt · Streamlit · NumPy · SciPy · Matplotlib · Claude API (Anthropic)<br>
SA market assumptions based on ASISA · SARB · JSE · Regulation 28 of the Pension Funds Act
</div>
""", unsafe_allow_html=True)
