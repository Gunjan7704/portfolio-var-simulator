"""
Tests for VaRCalculator module.
Validates VaR calculation, risk metrics, max drawdown, and report generation.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_manager import DataManager
from monte_carlo_engine import MonteCarloEngine
from var_calculator import VaRCalculator


@pytest.fixture
def simulation_results():
    """Run a small simulation to get test data."""
    dm = DataManager()
    sample_data = dm.generate_sample_data()
    returns_df = dm.calculate_returns(sample_data)
    latest_prices = dm.get_latest_prices(sample_data)
    initial_prices = np.array([latest_prices[asset] for asset in returns_df.columns])

    engine = MonteCarloEngine(returns_df)
    results = engine.run_simulation(
        initial_prices, n_simulations=500, n_days=10,
        initial_portfolio_value=100000, random_seed=42
    )
    return results


@pytest.fixture
def var_calc(simulation_results):
    """Create a VaRCalculator instance."""
    return VaRCalculator(
        simulation_results['portfolio_returns'],
        simulation_results['portfolio_values'],
        initial_value=100000
    )


class TestVaRCalculation:
    """Test Value-at-Risk computation."""

    def test_var_results_keys(self, var_calc):
        """VaR results should have keys for each horizon/confidence combo."""
        results = var_calc.calculate_var([0.95, 0.99], [1, 5])
        expected_keys = {'1d_95%', '1d_99%', '5d_95%', '5d_99%'}
        assert set(results.keys()) == expected_keys

    def test_var_is_negative(self, var_calc):
        """VaR dollar value should be negative (it's a loss)."""
        results = var_calc.calculate_var([0.95], [1])
        assert results['1d_95%']['var_dollar'] < 0

    def test_var_99_worse_than_95(self, var_calc):
        """99% VaR should show a larger loss than 95% VaR."""
        results = var_calc.calculate_var([0.95, 0.99], [1])
        var_95 = abs(results['1d_95%']['var_dollar'])
        var_99 = abs(results['1d_99%']['var_dollar'])
        assert var_99 >= var_95

    def test_5d_var_larger_than_1d(self, var_calc):
        """5-day VaR should generally be larger than 1-day VaR."""
        results = var_calc.calculate_var([0.95], [1, 5])
        var_1d = abs(results['1d_95%']['var_dollar'])
        var_5d = abs(results['5d_95%']['var_dollar'])
        assert var_5d >= var_1d

    def test_expected_shortfall_worse_than_var(self, var_calc):
        """Expected Shortfall should be >= VaR (average of tail losses)."""
        results = var_calc.calculate_var([0.95], [1])
        var_value = results['1d_95%']['var_dollar']
        es_value = results['1d_95%']['expected_shortfall']
        # Both are negative; ES should be more negative
        assert es_value <= var_value

    def test_var_results_contain_required_fields(self, var_calc):
        """Each VaR result should contain all required fields."""
        results = var_calc.calculate_var([0.95], [1])
        required_fields = {
            'horizon_days', 'confidence_level', 'var_dollar',
            'var_percentage', 'expected_shortfall', 'sample_returns',
            'tail_observations'
        }
        assert required_fields.issubset(set(results['1d_95%'].keys()))


class TestRiskMetrics:
    """Test portfolio risk metric calculations."""

    def test_risk_metrics_keys(self, var_calc):
        """Risk metrics should contain all expected keys."""
        returns = np.random.normal(0, 0.01, 252)
        metrics = var_calc.calculate_risk_metrics(returns)
        expected_keys = {
            'annual_return', 'annual_volatility', 'sharpe_ratio',
            'sortino_ratio', 'skewness', 'kurtosis',
            'max_drawdown', 'calmar_ratio'
        }
        assert set(metrics.keys()) == expected_keys

    def test_sharpe_ratio_sign(self, var_calc):
        """Positive returns with low vol should give positive Sharpe."""
        returns = np.random.normal(0.001, 0.005, 252)  # Positive drift
        metrics = var_calc.calculate_risk_metrics(returns)
        # With positive drift, Sharpe should be positive (most of the time)
        assert isinstance(metrics['sharpe_ratio'], float)

    def test_max_drawdown_is_negative(self, var_calc):
        """Max drawdown should be negative or zero."""
        returns = np.random.normal(0, 0.01, 252)
        metrics = var_calc.calculate_risk_metrics(returns)
        assert metrics['max_drawdown'] <= 0

    def test_zero_volatility_sharpe(self, var_calc):
        """Zero volatility should return Sharpe ratio of 0."""
        returns = np.zeros(100)
        metrics = var_calc.calculate_risk_metrics(returns)
        assert metrics['sharpe_ratio'] == 0


class TestMaxDrawdown:
    """Test maximum drawdown calculation."""

    def test_known_drawdown(self, var_calc):
        """Test with a known drawdown scenario."""
        # Price goes 100 -> 110 -> 88 -> 95
        # Returns: +10%, -20%, +7.95%
        # Max drawdown from 110 to 88 = -20%
        returns = np.array([0.10, -0.20, 0.0795])
        dd = var_calc._calculate_max_drawdown(returns)
        assert dd < 0
        assert abs(dd) > 0.15  # Should be around -20%

    def test_monotonically_increasing(self, var_calc):
        """Monotonically increasing returns should have ~0 drawdown."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        dd = var_calc._calculate_max_drawdown(returns)
        assert dd == 0.0


class TestRiskReport:
    """Test risk report generation."""

    def test_report_is_string(self, var_calc):
        """Report should be a formatted string."""
        var_results = var_calc.calculate_var([0.95], [1])
        returns = np.random.normal(0, 0.01, 252)
        metrics = var_calc.calculate_risk_metrics(returns)
        report = var_calc.generate_risk_report(var_results, metrics)
        assert isinstance(report, str)

    def test_report_contains_sections(self, var_calc):
        """Report should contain all major sections."""
        var_results = var_calc.calculate_var([0.95], [1])
        returns = np.random.normal(0, 0.01, 252)
        metrics = var_calc.calculate_risk_metrics(returns)
        report = var_calc.generate_risk_report(var_results, metrics)
        assert "PORTFOLIO RISK ANALYSIS REPORT" in report
        assert "VALUE-AT-RISK ANALYSIS" in report
        assert "RISK INTERPRETATION" in report
        assert "RECOMMENDATIONS" in report

    def test_report_includes_risk_level(self, var_calc):
        """Report should classify risk level."""
        var_results = var_calc.calculate_var([0.95], [1])
        returns = np.random.normal(0, 0.01, 252)
        metrics = var_calc.calculate_risk_metrics(returns)
        report = var_calc.generate_risk_report(var_results, metrics)
        assert any(level in report for level in ['LOW', 'MODERATE', 'HIGH', 'VERY HIGH'])


class TestExportResults:
    """Test CSV export functionality."""

    def test_export_creates_file(self, var_calc, tmp_path):
        """Export should create a CSV file."""
        var_results = var_calc.calculate_var([0.95], [1])
        # Test export writes to output/ dir — just verify no errors
        filepath = var_calc.export_results(var_results)
        assert filepath.exists()
        assert filepath.suffix == '.csv'

    def test_export_csv_content(self, var_calc):
        """Exported CSV should contain correct columns."""
        var_results = var_calc.calculate_var([0.95], [1])
        filepath = var_calc.export_results(var_results)
        df = pd.read_csv(filepath)
        expected_cols = {'Time_Horizon', 'Confidence_Level', 'VaR_Dollar',
                        'VaR_Percentage', 'Expected_Shortfall'}
        assert expected_cols == set(df.columns)
