"""
Value-at-Risk (VaR) Calculator
===============================
Computes portfolio risk metrics including parametric and historical VaR,
Expected Shortfall (CVaR), component VaR, and performs Kupiec backtesting.

Risk Metrics:
    - VaR: Maximum expected loss at a given confidence level over a time horizon.
    - Expected Shortfall (ES/CVaR): Average loss in the tail beyond VaR.
    - Component VaR: Marginal risk contribution of each asset.
    - Sharpe/Sortino ratios, Maximum Drawdown, Skewness, Kurtosis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import os
from pathlib import Path
from typing import Optional


class VaRCalculator:
    """Calculates Value-at-Risk and comprehensive risk metrics for a portfolio.

    Uses Monte Carlo simulation outputs to estimate VaR at multiple confidence
    levels and time horizons. Includes Kupiec likelihood ratio backtesting
    to validate model accuracy.

    Args:
        portfolio_returns: 2D array (n_simulations, n_days) of daily log returns.
        portfolio_values: 2D array (n_simulations, n_days+1) of portfolio values.
        initial_value: Starting portfolio value in dollars.

    Attributes:
        n_simulations: Number of Monte Carlo paths used.
    """

    def __init__(
        self,
        portfolio_returns: np.ndarray,
        portfolio_values: np.ndarray,
        initial_value: float = 100000
    ) -> None:
        self.portfolio_returns = portfolio_returns
        self.portfolio_values = portfolio_values
        self.initial_value = initial_value
        self.n_simulations = len(portfolio_returns)

    def calculate_var(
        self,
        confidence_levels: list[float] = [0.95, 0.99],
        time_horizons: list[int] = [1, 5]
    ) -> dict[str, dict]:
        """Calculate VaR and Expected Shortfall at multiple confidence/horizon levels.

        VaR is computed as the α-percentile of the simulated P&L distribution.
        Expected Shortfall is the conditional mean of losses exceeding VaR.

        Args:
            confidence_levels: List of confidence levels (e.g., [0.95, 0.99]).
            time_horizons: List of holding periods in days (e.g., [1, 5]).

        Returns:
            Dictionary keyed by '{horizon}d_{confidence}%' with VaR metrics.
        """
        var_results: dict[str, dict] = {}
        print("\n" + "=" * 50)
        print("VALUE-AT-RISK CALCULATION")
        print("=" * 50)

        for horizon in time_horizons:
            for confidence in confidence_levels:
                if horizon == 1:
                    returns = self.portfolio_returns[:, 0]
                else:
                    returns = np.sum(self.portfolio_returns[:, :horizon], axis=1)

                dollar_returns = returns * self.initial_value
                var_percentile = (1 - confidence) * 100
                var_value = np.percentile(dollar_returns, var_percentile)
                var_percentage = np.percentile(returns, var_percentile)

                # Expected Shortfall: E[L | L > VaR]
                tail_losses = dollar_returns[dollar_returns <= var_value]
                expected_shortfall = np.mean(tail_losses) if len(tail_losses) > 0 else var_value

                key = f"{horizon}d_{int(confidence*100)}%"
                var_results[key] = {
                    'horizon_days': horizon,
                    'confidence_level': confidence,
                    'var_dollar': var_value,
                    'var_percentage': var_percentage,
                    'expected_shortfall': expected_shortfall,
                    'sample_returns': dollar_returns,
                    'tail_observations': len(tail_losses)
                }

                print(f"\n{horizon}-Day VaR at {confidence:.0%} confidence:")
                print(f"  VaR (Dollar): ${abs(var_value):,.2f}")
                print(f"  VaR (Percentage): {abs(var_percentage):.2%}")
                print(f"  Expected Shortfall: ${abs(expected_shortfall):,.2f}")
                print(f"  Tail Observations: {len(tail_losses):,}")

        return var_results

    def calculate_component_var(
        self,
        asset_returns: pd.DataFrame,
        asset_weights: np.ndarray,
        confidence_level: float = 0.95,
        time_horizon: int = 1
    ) -> tuple[dict[str, dict], float]:
        """Calculate marginal and component VaR for each asset.

        Uses finite-difference approximation to estimate marginal VaR:
            Marginal VaR_i ≈ (VaR(w + ε·e_i) - VaR(w)) / ε

        Component VaR_i = Marginal VaR_i × w_i × Portfolio Value

        Args:
            asset_returns: DataFrame of individual asset returns.
            asset_weights: Array of portfolio weights.
            confidence_level: Confidence level for VaR.
            time_horizon: Holding period in days.

        Returns:
            Tuple of (component_vars dict, total_portfolio_var).
        """
        component_vars: dict[str, dict] = {}

        portfolio_var = self.calculate_var([confidence_level], [time_horizon])
        portfolio_var_value = abs(portfolio_var[f"{time_horizon}d_{int(confidence_level*100)}%"]['var_dollar'])

        for i, asset in enumerate(asset_returns.columns):
            epsilon = 0.01
            perturbed_weights = asset_weights.copy()
            perturbed_weights[i] += epsilon
            perturbed_weights = perturbed_weights / perturbed_weights.sum()

            perturbed_returns = np.dot(asset_returns.values, perturbed_weights)
            perturbed_dollar_returns = perturbed_returns * self.initial_value
            perturbed_var = abs(np.percentile(perturbed_dollar_returns, (1 - confidence_level) * 100))

            marginal_var = (perturbed_var - portfolio_var_value) / epsilon
            component_var = marginal_var * asset_weights[i] * self.initial_value

            component_vars[asset] = {
                'weight': asset_weights[i],
                'marginal_var': marginal_var,
                'component_var': component_var,
                'contribution_pct': component_var / portfolio_var_value * 100 if portfolio_var_value > 0 else 0
            }

        return component_vars, portfolio_var_value

    def backtest_var(
        self,
        historical_returns: np.ndarray,
        var_results: dict[str, dict],
        test_period_days: int = 252
    ) -> dict[str, dict]:
        """Backtest VaR model using Kupiec's Proportion of Failures (POF) test.

        Compares expected vs. actual VaR violations over a historical period.
        Uses likelihood ratio test to assess whether the violation rate is
        statistically consistent with the model's confidence level.

        Kupiec LR = 2 × [n_v × ln(p̂/p) + (T - n_v) × ln((1-p̂)/(1-p))]
        where p̂ = actual violation rate, p = expected rate.

        Args:
            historical_returns: Array of historical portfolio returns.
            var_results: VaR results dictionary from calculate_var().
            test_period_days: Number of days to use for backtesting.

        Returns:
            Dictionary with backtest statistics for each 1-day VaR scenario.
        """
        backtest_results: dict[str, dict] = {}
        print("\n" + "=" * 50)
        print("VaR BACKTESTING")
        print("=" * 50)

        test_returns = historical_returns[-test_period_days:] * self.initial_value
        for key, var_data in var_results.items():
            if var_data['horizon_days'] == 1:
                var_threshold = var_data['var_dollar']
                confidence = var_data['confidence_level']
                expected_violations = test_period_days * (1 - confidence)

                violations = np.sum(test_returns <= var_threshold)
                violation_rate = violations / test_period_days

                # Guard against log(0) in edge cases
                if violations == 0 or violations == test_period_days:
                    lr_stat = 0.0
                    p_value = 1.0
                else:
                    lr_stat = 2 * (violations * np.log(violation_rate / (1 - confidence)) +
                                  (test_period_days - violations) * np.log((1 - violation_rate) / confidence))
                    p_value = 1 - stats.chi2.cdf(abs(lr_stat), df=1)

                backtest_results[key] = {
                    'expected_violations': expected_violations,
                    'actual_violations': violations,
                    'violation_rate': violation_rate,
                    'expected_rate': 1 - confidence,
                    'lr_statistic': lr_stat,
                    'p_value': p_value,
                    'test_passed': p_value > 0.05,
                    'violation_dates': np.where(test_returns <= var_threshold)[0]
                }

                print(f"\n{key} Backtest Results:")
                print(f"  Expected violations: {expected_violations:.1f}")
                print(f"  Actual violations: {violations}")
                print(f"  Violation rate: {violation_rate:.2%}")
                print(f"  LR statistic: {lr_stat:.2f}")
                print(f"  P-value: {p_value:.4f}")
                print(f"  Test result: {'PASS' if p_value > 0.05 else 'FAIL'}")

        return backtest_results

    def calculate_risk_metrics(self, returns_data: np.ndarray) -> dict[str, float]:
        """Compute a comprehensive suite of portfolio risk metrics.

        Metrics:
            - Annualized return and volatility (assuming 252 trading days)
            - Sharpe ratio: excess return per unit of total risk
            - Sortino ratio: excess return per unit of downside risk
            - Skewness: asymmetry of the return distribution
            - Kurtosis: tail heaviness (excess kurtosis, normal = 0)
            - Maximum drawdown: largest peak-to-trough decline
            - Calmar ratio: annualized return / max drawdown

        Args:
            returns_data: Array of portfolio returns.

        Returns:
            Dictionary of risk metric name-value pairs.
        """
        if np.abs(returns_data).max() > 1:
            returns_pct = returns_data / self.initial_value
        else:
            returns_pct = returns_data

        annual_return = np.mean(returns_pct) * 252
        annual_volatility = np.std(returns_pct) * np.sqrt(252)
        downside_returns = returns_pct[returns_pct < 0]
        max_dd = self._calculate_max_drawdown(returns_pct)

        metrics = {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': annual_return / annual_volatility if annual_volatility > 0 else 0,
            'sortino_ratio': (
                annual_return / (np.std(downside_returns) * np.sqrt(252))
                if len(downside_returns) > 0 else 0
            ),
            'skewness': float(stats.skew(returns_pct)),
            'kurtosis': float(stats.kurtosis(returns_pct)),
            'max_drawdown': max_dd,
            'calmar_ratio': annual_return / abs(max_dd) if max_dd != 0 else 0
        }
        return metrics

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from a return series.

        Max drawdown = max(peak - trough) / peak, representing the worst
        cumulative loss from a historical high.

        Args:
            returns: Array of periodic returns.

        Returns:
            Maximum drawdown as a negative decimal (e.g., -0.15 = 15% drawdown).
        """
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return float(np.min(drawdown))

    def generate_risk_report(
        self,
        var_results: dict[str, dict],
        risk_metrics: dict[str, float],
        backtest_results: Optional[dict[str, dict]] = None
    ) -> str:
        """Generate a formatted text risk report.

        Includes portfolio metrics, VaR analysis, backtesting results (if
        available), risk level classification, and actionable recommendations.

        Args:
            var_results: VaR calculation results.
            risk_metrics: Portfolio risk metrics.
            backtest_results: Optional backtesting results.

        Returns:
            Formatted multi-line string report.
        """
        report = []
        report.append("=" * 70)
        report.append("PORTFOLIO RISK ANALYSIS REPORT")
        report.append("=" * 70)
        report.append(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Initial Portfolio Value: ${self.initial_value:,.2f}")
        report.append(f"Number of Simulations: {self.n_simulations:,}")

        report.append("\n" + "-" * 50)
        report.append("PORTFOLIO RISK METRICS")
        report.append("-" * 50)
        report.append(f"Expected Annual Return: {risk_metrics['annual_return']:.2%}")
        report.append(f"Annual Volatility: {risk_metrics['annual_volatility']:.2%}")
        report.append(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.3f}")
        report.append(f"Sortino Ratio: {risk_metrics['sortino_ratio']:.3f}")
        report.append(f"Skewness: {risk_metrics['skewness']:.3f}")
        report.append(f"Kurtosis: {risk_metrics['kurtosis']:.3f}")
        report.append(f"Maximum Drawdown: {risk_metrics['max_drawdown']:.2%}")
        report.append(f"Calmar Ratio: {risk_metrics['calmar_ratio']:.3f}")

        report.append("\n" + "-" * 50)
        report.append("VALUE-AT-RISK ANALYSIS")
        report.append("-" * 50)
        for key, var_data in var_results.items():
            report.append(f"\n{key}:")
            report.append(f"  Potential Loss (Dollar): ${abs(var_data['var_dollar']):,.2f}")
            report.append(f"  Potential Loss (Percentage): {abs(var_data['var_percentage']):.2%}")
            report.append(f"  Expected Shortfall: ${abs(var_data['expected_shortfall']):,.2f}")
            report.append(f"  Confidence Level: {var_data['confidence_level']:.0%}")

        if backtest_results:
            report.append("\n" + "-" * 50)
            report.append("BACKTESTING RESULTS")
            report.append("-" * 50)
            for key, backtest_data in backtest_results.items():
                report.append(f"\n{key}:")
                report.append(f"  Expected Violations: {backtest_data['expected_violations']:.1f}")
                report.append(f"  Actual Violations: {backtest_data['actual_violations']}")
                report.append(f"  Violation Rate: {backtest_data['violation_rate']:.2%}")
                report.append(f"  Model Accuracy: {'ACCEPTABLE' if backtest_data['test_passed'] else 'NEEDS REVIEW'}")

        report.append("\n" + "-" * 50)
        report.append("RISK INTERPRETATION")
        report.append("-" * 50)

        one_day_95_var = None
        for key, var_data in var_results.items():
            if "1d_95%" in key:
                one_day_95_var = abs(var_data['var_dollar'])
                break

        if one_day_95_var:
            var_as_pct_of_portfolio = (one_day_95_var / self.initial_value) * 100
            if var_as_pct_of_portfolio < 2:
                risk_level = "LOW"
            elif var_as_pct_of_portfolio < 5:
                risk_level = "MODERATE"
            elif var_as_pct_of_portfolio < 10:
                risk_level = "HIGH"
            else:
                risk_level = "VERY HIGH"
            report.append(f"Overall Risk Level: {risk_level}")
            report.append(f"1-Day 95% VaR represents {var_as_pct_of_portfolio:.1f}% of portfolio value")

        report.append("\n" + "-" * 50)
        report.append("RECOMMENDATIONS")
        report.append("-" * 50)

        if risk_metrics['sharpe_ratio'] < 0.5:
            report.append("• Consider reviewing asset allocation - low risk-adjusted returns")
        if abs(risk_metrics['skewness']) > 1:
            report.append("• Portfolio returns show significant skewness - consider tail risk hedging")
        if risk_metrics['kurtosis'] > 3:
            report.append("• High kurtosis indicates fat tails - increase VaR confidence levels")
        if backtest_results:
            failed_tests = [k for k, v in backtest_results.items() if not v['test_passed']]
            if failed_tests:
                report.append(f"• VaR model failed backtesting for: {', '.join(failed_tests)}")
                report.append("• Consider recalibrating model parameters")

        report.append("\n" + "=" * 70)

        # Export VaR results to CSV
        self.export_results(var_results)

        return "\n".join(report)

    def export_results(
        self, var_results: dict[str, dict], filename_prefix: str = "var_analysis"
    ) -> Path:
        """Export VaR summary to CSV file.

        Args:
            var_results: VaR calculation results.
            filename_prefix: Prefix for the output filename.

        Returns:
            Path to the exported CSV file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        var_summary = []
        for key, data in var_results.items():
            var_summary.append({
                'Time_Horizon': f"{data['horizon_days']} days",
                'Confidence_Level': f"{data['confidence_level']:.0%}",
                'VaR_Dollar': abs(data['var_dollar']),
                'VaR_Percentage': abs(data['var_percentage']),
                'Expected_Shortfall': abs(data['expected_shortfall'])
            })
        var_df = pd.DataFrame(var_summary)

        # Use project-relative output directory
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        var_filename = output_dir / f"{filename_prefix}_summary_{timestamp}.csv"
        var_df.to_csv(var_filename, index=False)
        print(f"\nVaR results exported to: {var_filename}")
        return var_filename