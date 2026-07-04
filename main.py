"""
Portfolio VaR Simulator — CLI Entry Point
==========================================
Runs the full Monte Carlo VaR analysis pipeline from the command line.

Usage:
    python main.py                    # Fetch live data from Yahoo Finance
    python main.py --sample-data      # Use synthetic sample data (no API needed)
    streamlit run web_app.py          # Launch interactive web dashboard
"""

import argparse
import numpy as np
from data_manager import DataManager
from monte_carlo_engine import MonteCarloEngine
from var_calculator import VaRCalculator
from visualizer import PortfolioVisualizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo VaR Simulation for Futures Portfolio"
    )
    parser.add_argument(
        "--sample-data", action="store_true",
        help="Use synthetic GBM sample data instead of live Yahoo Finance data"
    )
    parser.add_argument(
        "--simulations", type=int, default=10000,
        help="Number of Monte Carlo simulation paths (default: 10000)"
    )
    parser.add_argument(
        "--days", type=int, default=252,
        help="Number of trading days to simulate (default: 252)"
    )
    parser.add_argument(
        "--portfolio-value", type=float, default=100000,
        help="Initial portfolio value in USD (default: 100000)"
    )
    args = parser.parse_args()

    # ── Step 1: Fetch Data ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PORTFOLIO VaR SIMULATOR")
    print("=" * 60)

    dm = DataManager()
    historical_data = dm.fetch_historical_data(use_sample_data=args.sample_data)

    # Validate data quality
    validation = dm.validate_data_quality(historical_data)
    for asset, report in validation.items():
        status_icon = "✓" if report['status'] == 'SUCCESS' else "⚠"
        print(f"  {status_icon} {asset}: {report['message']}")

    # ── Step 2: Calculate Returns ───────────────────────────────────────
    returns_df = dm.calculate_returns(historical_data)
    latest_prices = dm.get_latest_prices(historical_data)
    initial_prices = np.array([latest_prices[asset] for asset in returns_df.columns])

    print(f"\nAssets: {list(returns_df.columns)}")
    print(f"Latest prices: {latest_prices}")
    print(f"Data points: {len(returns_df)}")

    # ── Step 3: Run Monte Carlo Simulation ──────────────────────────────
    portfolio_weights = {'Nifty': 0.4, 'Gold': 0.3, 'Crude': 0.3}
    mc_engine = MonteCarloEngine(returns_df, portfolio_weights)

    results = mc_engine.run_simulation(
        initial_prices=initial_prices,
        n_simulations=args.simulations,
        n_days=args.days,
        initial_portfolio_value=args.portfolio_value
    )

    # ── Step 4: Calculate VaR ───────────────────────────────────────────
    var_calc = VaRCalculator(
        results['portfolio_returns'],
        results['portfolio_values'],
        args.portfolio_value
    )
    var_results = var_calc.calculate_var(
        confidence_levels=[0.95, 0.99],
        time_horizons=[1, 5]
    )

    # ── Step 5: Risk Metrics & Report ───────────────────────────────────
    risk_metrics = var_calc.calculate_risk_metrics(results['portfolio_returns'][:, 0])

    # Backtest against historical data
    historical_portfolio_returns = np.dot(returns_df.values, mc_engine.weights)
    backtest_results = var_calc.backtest_var(historical_portfolio_returns, var_results)

    # Generate full risk report
    report = var_calc.generate_risk_report(var_results, risk_metrics, backtest_results)
    print(report)

    # ── Step 6: Correlation Analysis ────────────────────────────────────
    print("\n" + "=" * 50)
    print("CORRELATION MATRIX")
    print("=" * 50)
    corr_matrix = returns_df.corr()
    print(corr_matrix.to_string())

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("For interactive dashboard, run: streamlit run web_app.py")


if __name__ == "__main__":
    main()
