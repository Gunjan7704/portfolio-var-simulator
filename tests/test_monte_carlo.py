"""
Tests for MonteCarloEngine module.
Validates simulation output shapes, statistical properties, and reproducibility.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_manager import DataManager
from monte_carlo_engine import MonteCarloEngine


@pytest.fixture
def returns_data():
    """Generate returns data from sample data."""
    dm = DataManager()
    sample_data = dm.generate_sample_data()
    return dm.calculate_returns(sample_data)


@pytest.fixture
def engine(returns_data):
    """Create a MonteCarloEngine instance."""
    return MonteCarloEngine(returns_data)


@pytest.fixture
def initial_prices(returns_data):
    """Get initial prices from sample data."""
    dm = DataManager()
    sample_data = dm.generate_sample_data()
    latest = dm.get_latest_prices(sample_data)
    return np.array([latest[asset] for asset in returns_data.columns])


class TestEngineInitialization:
    """Test MonteCarloEngine constructor and parameter estimation."""

    def test_equal_weights_default(self, returns_data):
        """Default weights should be equal (1/n)."""
        engine = MonteCarloEngine(returns_data)
        expected = np.array([1/3, 1/3, 1/3])
        np.testing.assert_array_almost_equal(engine.weights, expected)

    def test_custom_weights_normalized(self, returns_data):
        """Custom weights should be normalized to sum to 1."""
        weights = {'Nifty': 0.5, 'Gold': 0.3, 'Crude': 0.2}
        engine = MonteCarloEngine(returns_data, weights)
        assert np.isclose(engine.weights.sum(), 1.0)

    def test_covariance_matrix_shape(self, engine):
        """Covariance matrix should be (n_assets × n_assets)."""
        n = engine.n_assets
        assert engine.cov_matrix.shape == (n, n)

    def test_covariance_matrix_symmetric(self, engine):
        """Covariance matrix should be symmetric."""
        np.testing.assert_array_almost_equal(
            engine.cov_matrix, engine.cov_matrix.T
        )

    def test_cholesky_decomposition_valid(self, engine):
        """L @ L^T should reconstruct the covariance matrix."""
        L = engine.cholesky_matrix
        reconstructed = L @ L.T
        np.testing.assert_array_almost_equal(reconstructed, engine.cov_matrix, decimal=10)


class TestSimulation:
    """Test Monte Carlo simulation output."""

    def test_price_paths_shape(self, engine, initial_prices):
        """Output shape should be (n_simulations, n_days + 1, n_assets)."""
        n_sims, n_days = 100, 10
        paths = engine.simulate_price_paths(initial_prices, n_sims, n_days, random_seed=42)
        assert paths.shape == (n_sims, n_days + 1, engine.n_assets)

    def test_initial_prices_set_correctly(self, engine, initial_prices):
        """First row of price paths should equal initial prices for all sims."""
        paths = engine.simulate_price_paths(initial_prices, 100, 10, random_seed=42)
        # Each simulation's t=0 should match initial prices
        for sim_idx in range(paths.shape[0]):
            np.testing.assert_array_almost_equal(paths[sim_idx, 0, :], initial_prices)

    def test_prices_are_positive(self, engine, initial_prices):
        """GBM guarantees positive prices (exp never goes negative)."""
        paths = engine.simulate_price_paths(initial_prices, 1000, 50, random_seed=42)
        assert (paths > 0).all()

    def test_reproducibility_with_seed(self, engine, initial_prices):
        """Same seed should produce identical simulation results."""
        paths1 = engine.simulate_price_paths(initial_prices, 100, 10, random_seed=42)
        paths2 = engine.simulate_price_paths(initial_prices, 100, 10, random_seed=42)
        np.testing.assert_array_equal(paths1, paths2)

    def test_different_seeds_differ(self, engine, initial_prices):
        """Different seeds should produce different results."""
        paths1 = engine.simulate_price_paths(initial_prices, 100, 10, random_seed=42)
        paths2 = engine.simulate_price_paths(initial_prices, 100, 10, random_seed=99)
        assert not np.array_equal(paths1, paths2)


class TestPortfolioValues:
    """Test portfolio value and return calculations."""

    def test_portfolio_values_shape(self, engine, initial_prices):
        """Portfolio values should be (n_simulations, n_days + 1)."""
        n_sims, n_days = 100, 10
        paths = engine.simulate_price_paths(initial_prices, n_sims, n_days, random_seed=42)
        values = engine.calculate_portfolio_values(paths, 100000)
        assert values.shape == (n_sims, n_days + 1)

    def test_initial_portfolio_value(self, engine, initial_prices):
        """First day portfolio values should equal initial value."""
        paths = engine.simulate_price_paths(initial_prices, 100, 10, random_seed=42)
        values = engine.calculate_portfolio_values(paths, 100000)
        np.testing.assert_array_almost_equal(values[:, 0], 100000, decimal=0)

    def test_portfolio_returns_shape(self, engine, initial_prices):
        """Portfolio returns should have one fewer column than values."""
        n_sims, n_days = 100, 10
        paths = engine.simulate_price_paths(initial_prices, n_sims, n_days, random_seed=42)
        values = engine.calculate_portfolio_values(paths, 100000)
        returns = engine.calculate_portfolio_returns(values)
        assert returns.shape == (n_sims, n_days)


class TestFullSimulation:
    """Test the full run_simulation pipeline."""

    def test_run_simulation_returns_all_keys(self, engine, initial_prices):
        """Results dict should contain all expected keys."""
        results = engine.run_simulation(initial_prices, n_simulations=100, n_days=10)
        expected_keys = {
            'price_paths', 'portfolio_values', 'portfolio_returns',
            'final_values', 'total_returns', 'mean_return', 'std_return',
            'min_return', 'max_return', 'simulation_params'
        }
        assert set(results.keys()) == expected_keys

    def test_mean_return_is_scalar(self, engine, initial_prices):
        """Summary statistics should be scalar values."""
        results = engine.run_simulation(initial_prices, n_simulations=100, n_days=10)
        assert np.isscalar(results['mean_return'])
        assert np.isscalar(results['std_return'])

    def test_min_max_return_order(self, engine, initial_prices):
        """Min return should be <= mean <= max return."""
        results = engine.run_simulation(initial_prices, n_simulations=1000, n_days=10)
        assert results['min_return'] <= results['mean_return'] <= results['max_return']
