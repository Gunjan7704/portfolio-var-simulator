"""
Streamlit Web Application
==========================
Interactive web interface for the Portfolio VaR Simulator.

Run with: streamlit run web_app.py
Access at: http://localhost:8501
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
import yfinance as yf
from data_manager import DataManager
from monte_carlo_engine import MonteCarloEngine
from var_calculator import VaRCalculator
from visualizer import PortfolioVisualizer

# Suppress yfinance FutureWarnings only
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

st.set_page_config(
    page_title="Portfolio Risk VaR Simulator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def fetch_market_data():
    dm = DataManager()
    historical_data = dm.fetch_historical_data()
    return historical_data

def generate_sample_asset_data(asset_name, start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.dayofweek < 5]
    
    asset_params = {
        'Nifty': {'start_price': 18000, 'volatility': 0.015, 'drift': 0.0003},
        'Gold': {'start_price': 2000, 'volatility': 0.012, 'drift': 0.0002},
        'Crude': {'start_price': 75, 'volatility': 0.025, 'drift': 0.0001}
    }
    
    params = asset_params.get(asset_name, asset_params['Nifty'])
    n_days = len(dates)
    
    np.random.seed(hash(asset_name) % 2147483647)
    returns = np.random.normal(params['drift'], params['volatility'], n_days)
    
    prices = [params['start_price']]
    for i in range(1, n_days):
        prices.append(prices[-1] * (1 + returns[i]))
    
    prices = np.array(prices)
    
    ohlc_data = []
    for i, close_price in enumerate(prices):
        if i == 0:
            open_price = close_price
        else:
            open_price = prices[i-1] * (1 + np.random.normal(0, 0.002))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
        volume = np.random.randint(1000000, 5000000)
        ohlc_data.append([open_price, high_price, low_price, close_price, volume])
    
    if len(ohlc_data) != len(dates):
        st.error(f"Length mismatch: ohlc_data ({len(ohlc_data)}) vs dates ({len(dates)})")
        dates = dates[:len(ohlc_data)]
    
    df = pd.DataFrame(ohlc_data, 
                     columns=['Open', 'High', 'Low', 'Close', 'Volume'],
                     index=dates)
    df['Adj Close'] = df['Close']
    return df

def generate_sample_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    sample_data = {}
    assets = ['Nifty', 'Gold', 'Crude']
    for asset in assets:
        sample_data[asset] = generate_sample_asset_data(asset, start_date, end_date)
    return sample_data

@st.cache_data
def calculate_returns(historical_data):
    dm = DataManager()
    return dm.calculate_returns(historical_data)

def run_monte_carlo_simulation(returns_data, weights, initial_portfolio_value, n_simulations, time_horizon):
    n_assets = len(weights)
    n_days = time_horizon
    
    mean_returns = returns_data.mean().values
    cov_matrix = returns_data.cov().values
    
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    eigenvals = np.maximum(eigenvals, 1e-8)
    cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    try:
        chol_matrix = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        U, s, Vt = np.linalg.svd(cov_matrix)
        chol_matrix = U @ np.diag(np.sqrt(s))
    
    np.random.seed(42)
    random_numbers = np.random.normal(0, 1, (n_simulations, n_days, n_assets))
    
    correlated_returns = np.zeros_like(random_numbers)
    for i in range(n_simulations):
        for j in range(n_days):
            correlated_returns[i, j, :] = mean_returns + chol_matrix @ random_numbers[i, j, :]
    
    portfolio_returns = np.sum(correlated_returns * weights, axis=2)
    
    portfolio_values = np.zeros((n_simulations, n_days + 1))
    portfolio_values[:, 0] = initial_portfolio_value
    
    for day in range(n_days):
        portfolio_values[:, day + 1] = portfolio_values[:, day] * (1 + portfolio_returns[:, day])
    
    return portfolio_values, portfolio_returns

def create_price_paths_chart(portfolio_values, n_paths_to_show=100):
    n_simulations = portfolio_values.shape[0]
    n_days = portfolio_values.shape[1]
    fig = go.Figure()
    
    paths_to_show = min(n_paths_to_show, n_simulations)
    np.random.seed(42)
    selected_paths = np.random.choice(n_simulations, paths_to_show, replace=False)
    
    for i, path_idx in enumerate(selected_paths):
        fig.add_trace(go.Scatter(
            x=list(range(n_days)),
            y=portfolio_values[path_idx, :],
            mode='lines',
            line=dict(width=0.5, color='lightblue'),
            opacity=0.3,
            showlegend=False,
            hovertemplate="Day: %{x}<br>Portfolio Value: $%{y:,.0f}<extra></extra>"
        ))
    
    mean_path = np.mean(portfolio_values, axis=0)
    fig.add_trace(go.Scatter(
        x=list(range(n_days)),
        y=mean_path,
        mode='lines',
        line=dict(width=3, color='red'),
        name='Mean Path',
        hovertemplate="Day: %{x}<br>Mean Portfolio Value: $%{y:,.0f}<extra></extra>"
    ))
    
    p5 = np.percentile(portfolio_values, 5, axis=0)
    p95 = np.percentile(portfolio_values, 95, axis=0)
    
    fig.add_trace(go.Scatter(
        x=list(range(n_days)),
        y=p95,
        mode='lines',
        line=dict(width=1, color='green', dash='dash'),
        name='95th Percentile',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(n_days)),
        y=p5,
        mode='lines',
        line=dict(width=1, color='orange', dash='dash'),
        name='5th Percentile',
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.1)',
        showlegend=True
    ))
    
    fig.update_layout(
        title="Monte Carlo Portfolio Value Simulations",
        xaxis_title="Days",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        height=500
    )
    return fig

def create_return_distribution_chart(portfolio_returns, var_results):
    fig = go.Figure()
    one_day_returns = portfolio_returns[:, 0] if portfolio_returns.shape[1] > 0 else portfolio_returns.flatten()
    fig.add_trace(go.Histogram(
        x=one_day_returns * 100,
        nbinsx=50,
        name="1-Day Portfolio Returns",
        opacity=0.7,
        marker_color='lightblue'
    ))
    colors = ['red', 'darkred', 'crimson']
    color_idx = 0
    for key, var_data in var_results.items():
        if var_data['horizon_days'] == 1:
            var_value = var_data['var_percentage'] * 100
            confidence = var_data['confidence_level']
            fig.add_vline(
                x=var_value,
                line_dash="dash",
                line_color=colors[color_idx % len(colors)],
                annotation_text=f"VaR {confidence:.0%}: {abs(var_value):.2f}%",
                annotation_position="top"
            )
            color_idx += 1
    fig.update_layout(
        title="Portfolio Return Distribution with VaR Estimates",
        xaxis_title="Portfolio Returns (%)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=500
    )
    return fig

def create_var_comparison_chart(var_results):
    var_data = []
    for key, data in var_results.items():
        var_data.append({
            'Scenario': key,
            'Confidence': f"{data['confidence_level']:.0%}",
            'Horizon': f"{data['horizon_days']} Day{'s' if data['horizon_days'] > 1 else ''}",
            'VaR_Dollar': abs(data['var_dollar']),
            'VaR_Percentage': abs(data['var_percentage']) * 100,
            'ES_Dollar': abs(data['expected_shortfall']),
            'ES_Percentage': abs(data['expected_shortfall']) * 100
        })
    df = pd.DataFrame(var_data)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['VaR by Dollar Amount', 'VaR by Percentage', 
                       'Expected Shortfall ($)', 'Expected Shortfall (%)']
    )
    fig.add_trace(
        go.Bar(
            x=df['Scenario'],
            y=df['VaR_Dollar'],
            name='VaR ($)',
            marker_color='lightcoral',
            text=[f"${x:,.0f}" for x in df['VaR_Dollar']],
            textposition='auto'
        ), row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            x=df['Scenario'],
            y=df['VaR_Percentage'],
            name='VaR (%)',
            marker_color='lightblue',
            text=[f"{x:.1f}%" for x in df['VaR_Percentage']],
            textposition='auto'
        ), row=1, col=2
    )
    fig.add_trace(
        go.Bar(
            x=df['Scenario'],
            y=df['ES_Dollar'],
            name='ES ($)',
            marker_color='lightsalmon',
            text=[f"${x:,.0f}" for x in df['ES_Dollar']],
            textposition='auto'
        ), row=2, col=1
    )
    fig.add_trace(
        go.Bar(
            x=df['Scenario'],
            y=df['ES_Percentage'],
            name='ES (%)',
            marker_color='lightgreen',
            text=[f"{x:.1f}%" for x in df['ES_Percentage']],
            textposition='auto'
        ), row=2, col=2
    )
    fig.update_layout(
        title="Value-at-Risk and Expected Shortfall Comparison",
        template="plotly_white",
        height=700,
        showlegend=False
    )
    return fig

def create_correlation_heatmap(returns_data):
    correlation_matrix = returns_data.corr()
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix.values, 3),
        texttemplate="%{text}",
        textfont={"size": 14},
        hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>"
    ))
    fig.update_layout(
        title="Asset Correlation Matrix",
        template="plotly_white",
        height=400,
        width=500
    )
    return fig

def main():
    st.markdown('<h1 class="main-header">🎯 Portfolio Risk: Value-at-Risk (VaR) Simulation</h1>', unsafe_allow_html=True)
    st.markdown("### Monte Carlo simulation model to estimate 1-day and 5-day VaR for diversified futures portfolio")
    st.markdown("*This application simulates portfolio risk using Monte Carlo methods for Nifty 50, Gold, and Crude Oil futures.*")
    
    with st.sidebar:
        st.header("⚙️ Simulation Parameters")
        st.subheader("Portfolio Allocation")
        nifty_weight = st.slider("Nifty 50 Weight (%)", 0, 100, 40) / 100
        gold_weight = st.slider("Gold Weight (%)", 0, 100, 30) / 100
        crude_weight = st.slider("Crude Oil Weight (%)", 0, 100, 30) / 100
        
        total_weight = nifty_weight + gold_weight + crude_weight
        if total_weight > 0:
            nifty_weight /= total_weight
            gold_weight /= total_weight
            crude_weight /= total_weight
        else:
            nifty_weight = gold_weight = crude_weight = 1/3
        
        weights = np.array([nifty_weight, gold_weight, crude_weight])
        st.info(f"**Normalized Weights:**\n\n"
                f"Nifty: {nifty_weight:.1%}\n\n"
                f"Gold: {gold_weight:.1%}\n\n"
                f"Crude: {crude_weight:.1%}")
        
        st.subheader("Simulation Settings")
        initial_portfolio_value = st.number_input(
            "Initial Portfolio Value ($)", 
            min_value=10000, 
            max_value=10000000, 
            value=1000000,
            step=10000
        )
        n_simulations = st.selectbox(
            "Number of Simulations",
            [1000, 5000, 10000, 25000],
            index=1
        )
        time_horizon = st.selectbox(
            "Maximum Time Horizon (Days)",
            [5, 10, 21, 63],
            index=0
        )
        confidence_levels = st.multiselect(
            "Confidence Levels",
            [0.90, 0.95, 0.99],
            default=[0.95, 0.99]
        )
        run_simulation = st.button("🚀 Run Monte Carlo Simulation", type="primary")
    
    try:
        with st.spinner("Loading market data..."):
            historical_data = fetch_market_data()
            if not historical_data:
                raise ValueError("Failed to load market data.")
            st.success("✅ Market data loaded successfully!")
        
        for asset_name, df in historical_data.items():
            if df.empty:
                st.warning(f"Data for {asset_name} is empty.")
            elif df.index.name is None:
                st.warning(f"Data for {asset_name} has no index.")
        
        returns_data = calculate_returns(historical_data)
        if returns_data.empty:
            raise ValueError("Returns data is empty after calculation.")
        
        with st.expander("📊 Data Overview", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Assets", len(historical_data))
                st.metric("Data Points", len(returns_data))
            with col2:
                st.metric("Date Range", f"{returns_data.index.min().strftime('%Y-%m-%d')} to {returns_data.index.max().strftime('%Y-%m-%d')}")
            with col3:
                st.metric("Portfolio Value", f"${initial_portfolio_value:,.0f}")
            st.plotly_chart(create_correlation_heatmap(returns_data), use_container_width=True)
            st.subheader("Return Statistics")
            stats_df = returns_data.describe()
            st.dataframe(stats_df.style.format("{:.4f}"))
        
        if run_simulation and len(confidence_levels) > 0:
            with st.spinner(f"Running {n_simulations:,} Monte Carlo simulations..."):
                portfolio_values, portfolio_returns = run_monte_carlo_simulation(
                    returns_data, weights, initial_portfolio_value, n_simulations, time_horizon
                )
                # Create an instance of VaRCalculator
                var_calc = VaRCalculator(portfolio_returns, portfolio_values, initial_portfolio_value)
                # Calculate VaR using VaRCalculator
                var_results = var_calc.calculate_var(confidence_levels, [1, 5])
                
                # Calculate risk metrics for the risk report
                if portfolio_returns.shape[1] > 0:
                    risk_metrics = var_calc.calculate_risk_metrics(portfolio_returns[:, 0])
                else:
                    risk_metrics = {
                        'annual_return': 0,
                        'annual_volatility': 0,
                        'sharpe_ratio': 0,
                        'sortino_ratio': 0,
                        'skewness': 0,
                        'kurtosis': 0,
                        'max_drawdown': 0,
                        'calmar_ratio': 0
                    }
                
                # Generate the risk report, which will also export the VaR results to a CSV
                risk_report = var_calc.generate_risk_report(var_results, risk_metrics)
            
            st.success("✅ Simulation completed successfully!")
            st.header("📈 Simulation Results")
            st.subheader("🎯 Value-at-Risk Summary")
            
            var_summary = []
            for key, data in var_results.items():
                var_summary.append({
                    'Scenario': key.replace('_', ' ').replace('d ', '-Day '),
                    'Confidence Level': f"{data['confidence_level']:.0%}",
                    'Time Horizon': f"{data['horizon_days']} day{'s' if data['horizon_days'] > 1 else ''}",
                    'VaR ($)': f"${abs(data['var_dollar']):,.0f}",
                    'VaR (%)': f"{abs(data['var_percentage']*100):.2f}%",
                    'Expected Shortfall ($)': f"${abs(data['expected_shortfall']):,.0f}",
                    'Expected Shortfall (%)': f"{abs(data['var_percentage']*100):.2f}%"
                })
            
            var_df = pd.DataFrame(var_summary)
            st.dataframe(var_df, use_container_width=True)
            
            st.subheader("📊 Key Risk Metrics")
            col1, col2, col3, col4 = st.columns(4)
            worst_var = min([data['var_dollar'] for data in var_results.values()])
            worst_es = min([data['expected_shortfall'] for data in var_results.values()])
            
            with col1:
                st.metric("Worst 1-Day VaR (95%)", 
                         f"${abs(var_results.get('1d_95%', {}).get('var_dollar', 0)):,.0f}" if '1d_95%' in var_results else "N/A")
            with col2:
                st.metric("Worst 5-Day VaR (95%)", 
                         f"${abs(var_results.get('5d_95%', {}).get('var_dollar', 0)):,.0f}" if '5d_95%' in var_results else "N/A")
            with col3:
                st.metric("Maximum Potential Loss", f"${abs(worst_var):,.0f}")
            with col4:
                st.metric("Expected Shortfall", f"${abs(worst_es):,.0f}")
            
            # Display the risk report
            st.subheader("📜 Risk Report")
            st.text(risk_report)
            
            st.subheader("📊 Visualization")
            tab1, tab2, tab3 = st.tabs(["Portfolio Paths", "Return Distribution", "Risk Comparison"])
            with tab1:
                st.plotly_chart(create_price_paths_chart(portfolio_values), use_container_width=True)
            with tab2:
                st.plotly_chart(create_return_distribution_chart(portfolio_returns, var_results), use_container_width=True)
            with tab3:
                st.plotly_chart(create_var_comparison_chart(var_results), use_container_width=True)
        
        elif run_simulation and len(confidence_levels) == 0:
            st.error("Please select at least one confidence level.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try refreshing the page or adjusting the parameters.")

if __name__ == "__main__":
    main()