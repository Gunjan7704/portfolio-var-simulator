# Portfolio VaR Simulator

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-51%20passed-brightgreen.svg)](#testing)

Monte Carlo simulation engine for estimating Value-at-Risk (VaR) on a multi-asset futures portfolio — Nifty 50, Gold, and Crude Oil. Built with Python, NumPy, SciPy, and Streamlit.

## What It Does

- **Simulates 10,000+ correlated price paths** using Geometric Brownian Motion (GBM) with Cholesky-decomposed covariance
- **Computes VaR & Expected Shortfall (CVaR)** at 95%/99% confidence over 1-day and 5-day horizons
- **Backtests the VaR model** using Kupiec's Proportion of Failures (POF) likelihood ratio test
- **Generates risk reports** with Sharpe ratio, Sortino ratio, max drawdown, skewness, and kurtosis
- **Interactive Streamlit dashboard** for portfolio configuration and real-time analysis

## Architecture

```
portfolio-var-simulator/
├── main.py                   # CLI entry point — run full pipeline
├── web_app.py                # Streamlit interactive dashboard
├── data_manager.py           # Yahoo Finance data fetching + GBM sample data fallback
├── monte_carlo_engine.py     # Vectorized Monte Carlo simulation (NumPy einsum + cumsum)
├── var_calculator.py         # VaR, CVaR, component VaR, backtesting, risk metrics
├── visualizer.py             # Plotly + Matplotlib visualization engine
├── tests/
│   ├── test_data_manager.py  # 18 tests: data generation, cleaning, returns
│   ├── test_monte_carlo.py   # 16 tests: simulation shapes, GBM properties, Cholesky
│   └── test_var_calculator.py # 17 tests: VaR ordering, risk metrics, report generation
├── requirements.txt
├── setup.py
└── LICENSE
```

## Quick Start

### Installation

```bash
git clone https://github.com/gunjanmoradiya/portfolio-var-simulator.git
cd portfolio-var-simulator
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run from CLI

```bash
# Using sample data (no API needed — great for demo)
python main.py --sample-data

# Using live Yahoo Finance data
python main.py

# Custom parameters
python main.py --simulations 25000 --days 63 --portfolio-value 500000
```

### Run the Web Dashboard

```bash
streamlit run web_app.py
# Open http://localhost:8501
```

### Run Tests

```bash
python -m pytest tests/ -v
```

## How It Works

### 1. Data Pipeline (`data_manager.py`)

Fetches 2 years of OHLCV data from Yahoo Finance with fallback symbols per asset. Cleans outliers (>20% daily moves), handles missing data, and generates synthetic GBM data as a last-resort fallback.

### 2. Monte Carlo Engine (`monte_carlo_engine.py`)

Simulates correlated asset price paths using the GBM discretization:

```
S(t+1) = S(t) · exp((μ - σ²/2) + L · Z)
```

Where `L` is the Cholesky factor of the covariance matrix (`Σ = LLᵀ`) and `Z ~ N(0,1)`. The implementation is **fully vectorized** using `np.einsum` for the Cholesky correlation and `np.cumsum` for the cumulative price path — no Python loops in the hot path.

### 3. VaR Calculator (`var_calculator.py`)

- **Historical Simulation VaR**: Percentile of simulated P&L distribution
- **Expected Shortfall**: `E[Loss | Loss > VaR]` — average tail loss
- **Component VaR**: Marginal risk contribution via finite-difference perturbation
- **Kupiec Backtesting**: Likelihood ratio test comparing expected vs. actual VaR violations

### 4. Risk Metrics

| Metric | Formula |
|--------|---------|
| Sharpe Ratio | `E[R] / σ(R)` annualized |
| Sortino Ratio | `E[R] / σ_downside(R)` annualized |
| Max Drawdown | `max(peak - trough) / peak` |
| Calmar Ratio | `Annual Return / |Max Drawdown|` |
| Skewness | Third standardized moment |
| Kurtosis | Excess kurtosis (normal = 0) |

### 5. Visualizations (`visualizer.py`)

- Monte Carlo price path simulations with mean + percentile bands
- Return distribution histogram with VaR threshold markers
- VaR comparison bar charts (dollar and percentage)
- Asset correlation heatmap
- Efficient frontier via random portfolio sampling

## Tech Stack

- **Core**: Python, NumPy, SciPy, Pandas
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Web**: Streamlit
- **Data**: Yahoo Finance (yfinance)
- **Testing**: pytest (51 tests)

## Example Output

```
============================================================
MONTE CARLO SIMULATION
============================================================
Assets: ['Nifty', 'Gold', 'Crude']
Portfolio weights: {'Nifty': 0.4, 'Gold': 0.3, 'Crude': 0.3}
Initial portfolio value: $100,000.00
Simulation parameters: 10,000 paths, 252 days

1-Day VaR at 95% confidence:
  VaR (Dollar): $1,698.42
  VaR (Percentage): 1.70%
  Expected Shortfall: $2,411.15

Overall Risk Level: LOW
1-Day 95% VaR represents 1.7% of portfolio value
```

## License

MIT — see [LICENSE](LICENSE).

## Author

**Gunjan Moradiya**
B.Tech CSE (Data Science) — DJSCE, Mumbai