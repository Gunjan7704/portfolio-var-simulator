"""
Monte Carlo Simulation Engine
===============================
Generates correlated multi-asset price paths using Geometric Brownian Motion (GBM)
and Cholesky decomposition for portfolio risk analysis.

Mathematical Framework:
    - Price evolution: dS = μS·dt + σS·dW  (GBM)
    - Discrete form: S(t+1) = S(t) · exp((μ - σ²/2)·dt + σ·√dt·Z)
    - Correlation: Z_correlated = L · Z_independent  (Cholesky: Σ = L·Lᵀ)

The engine supports both sequential and parallel (multi-threaded) simulation
modes for scalability.
"""

import numpy as np
import pandas as pd
from scipy.linalg import cholesky
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional


class MonteCarloEngine:
    """Monte Carlo simulation engine for multi-asset portfolio risk analysis.

    Simulates correlated price paths using GBM with drift and diffusion
    parameters estimated from historical return data. Uses Cholesky
    decomposition of the covariance matrix to preserve cross-asset correlations.

    Args:
        returns_data: DataFrame of historical log returns (assets as columns).
        portfolio_weights: Optional dict mapping asset names to weights.
            If None, uses equal weighting (1/n).

    Attributes:
        mean_returns: Annualized mean return vector.
        cov_matrix: Covariance matrix of asset returns.
        cholesky_matrix: Lower-triangular Cholesky factor of covariance matrix.
    """

    def __init__(
        self,
        returns_data: pd.DataFrame,
        portfolio_weights: Optional[dict[str, float]] = None
    ) -> None:
        self.returns_data = returns_data
        self.assets = list(returns_data.columns)
        self.n_assets = len(self.assets)

        if portfolio_weights is None:
            self.weights = np.array([1 / self.n_assets] * self.n_assets)
        else:
            self.weights = np.array([portfolio_weights.get(asset, 0) for asset in self.assets])
        self.weights = self.weights / self.weights.sum()

        self._calculate_parameters()

    def _calculate_parameters(self) -> None:
        """Estimate statistical parameters from historical returns.

        Computes mean return vector, covariance matrix, and performs Cholesky
        decomposition. Falls back to correlation matrix if covariance matrix
        is not positive-definite.
        """
        self.mean_returns = self.returns_data.mean().values
        self.cov_matrix = self.returns_data.cov().values
        self.corr_matrix = self.returns_data.corr().values
        self.std_returns = self.returns_data.std().values

        try:
            self.cholesky_matrix = cholesky(self.cov_matrix, lower=True)
        except np.linalg.LinAlgError:
            print("Warning: Covariance matrix not positive-definite, using correlation matrix")
            self.cholesky_matrix = cholesky(self.corr_matrix, lower=True)

    def simulate_price_paths(
        self,
        initial_prices: np.ndarray,
        n_simulations: int = 10000,
        n_days: int = 252,
        use_correlation: bool = True,
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        """Simulate correlated asset price paths via GBM.

        Uses vectorized NumPy operations for performance. The Cholesky matrix
        transforms independent standard normal draws into correlated shocks
        that preserve the historical covariance structure.

        Discrete GBM step:
            S(t+1) = S(t) · exp((μ - σ²/2) + σ · ε_correlated)

        Args:
            initial_prices: Array of starting prices for each asset.
            n_simulations: Number of Monte Carlo paths to generate.
            n_days: Number of trading days to simulate forward.
            use_correlation: If True, apply Cholesky correlation to shocks.
            random_seed: Seed for reproducibility.

        Returns:
            3D array of shape (n_simulations, n_days + 1, n_assets) containing
            simulated price paths.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        print(f"Running {n_simulations:,} simulations for {n_days} days...")
        start_time = time.time()

        # Pre-compute drift term: μ - σ²/2 (Itô correction)
        drift = self.mean_returns - 0.5 * self.std_returns ** 2

        # Generate all random shocks at once: (n_simulations, n_days, n_assets)
        random_shocks = np.random.standard_normal((n_simulations, n_days, self.n_assets))

        # Apply Cholesky correlation: vectorized matrix multiply across all sims and days
        if use_correlation:
            # L @ Z for each (sim, day) pair — use einsum for fully vectorized operation
            # cholesky_matrix is (n_assets, n_assets), random_shocks is (n_sims, n_days, n_assets)
            random_shocks = np.einsum('ij,mkj->mki', self.cholesky_matrix, random_shocks)

        # Compute log returns: drift + diffusion (vectorized, no Python loops)
        log_returns = drift[np.newaxis, np.newaxis, :] + random_shocks

        # Build price paths via cumulative product of exp(log_returns)
        price_paths = np.zeros((n_simulations, n_days + 1, self.n_assets))
        price_paths[:, 0, :] = initial_prices
        # Cumulative product along the time axis
        cumulative_returns = np.exp(np.cumsum(log_returns, axis=1))
        price_paths[:, 1:, :] = initial_prices[np.newaxis, np.newaxis, :] * cumulative_returns

        elapsed_time = time.time() - start_time
        print(f"Simulation completed in {elapsed_time:.2f} seconds")

        return price_paths

    def calculate_portfolio_values(
        self,
        price_paths: np.ndarray,
        initial_portfolio_value: float = 100000
    ) -> np.ndarray:
        """Convert asset price paths to portfolio value paths.

        Computes the number of units held in each asset based on initial
        allocation weights, then tracks total portfolio value over time.

        Args:
            price_paths: 3D array (n_simulations, n_days+1, n_assets).
            initial_portfolio_value: Starting portfolio value in dollars.

        Returns:
            2D array of shape (n_simulations, n_days+1) with portfolio values.
        """
        initial_prices = price_paths[0, 0, :]  # Same initial prices across all sims
        initial_asset_values = initial_portfolio_value * self.weights
        asset_quantities = initial_asset_values / initial_prices

        # Vectorized: portfolio_value = sum(quantities * prices) for each sim and day
        portfolio_values = np.sum(asset_quantities[np.newaxis, np.newaxis, :] * price_paths, axis=2)

        return portfolio_values

    def calculate_portfolio_returns(self, portfolio_values: np.ndarray) -> np.ndarray:
        """Compute log returns of portfolio values across all simulations.

        Args:
            portfolio_values: 2D array (n_simulations, n_days+1).

        Returns:
            2D array of shape (n_simulations, n_days) with daily log returns.
        """
        # Vectorized log return calculation
        portfolio_returns = np.diff(np.log(portfolio_values), axis=1)
        return portfolio_returns

    def run_simulation(
        self,
        initial_prices: np.ndarray,
        n_simulations: int = 10000,
        n_days: int = 252,
        initial_portfolio_value: float = 100000,
        use_correlation: bool = True,
        random_seed: int = 42
    ) -> dict:
        """Execute full Monte Carlo simulation pipeline.

        Runs price path simulation → portfolio valuation → return calculation
        and prints a summary of results.

        Args:
            initial_prices: Starting price for each asset.
            n_simulations: Number of simulation paths.
            n_days: Trading days to simulate.
            initial_portfolio_value: Starting portfolio value.
            use_correlation: Whether to apply correlation structure.
            random_seed: Random seed for reproducibility.

        Returns:
            Dictionary containing price_paths, portfolio_values,
            portfolio_returns, final_values, total_returns, and summary stats.
        """
        print("\n" + "=" * 60)
        print("MONTE CARLO SIMULATION")
        print("=" * 60)
        print(f"Assets: {self.assets}")
        print(f"Portfolio weights: {dict(zip(self.assets, self.weights.round(3)))}")
        print(f"Initial portfolio value: ${initial_portfolio_value:,.2f}")
        print(f"Simulation parameters: {n_simulations:,} paths, {n_days} days")

        price_paths = self.simulate_price_paths(
            initial_prices, n_simulations, n_days, use_correlation, random_seed
        )
        portfolio_values = self.calculate_portfolio_values(price_paths, initial_portfolio_value)
        portfolio_returns = self.calculate_portfolio_returns(portfolio_values)

        final_values = portfolio_values[:, -1]
        total_returns = (final_values - initial_portfolio_value) / initial_portfolio_value

        results = {
            'price_paths': price_paths,
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns,
            'final_values': final_values,
            'total_returns': total_returns,
            'mean_return': np.mean(total_returns),
            'std_return': np.std(total_returns),
            'min_return': np.min(total_returns),
            'max_return': np.max(total_returns),
            'simulation_params': {
                'n_simulations': n_simulations,
                'n_days': n_days,
                'initial_value': initial_portfolio_value,
                'use_correlation': use_correlation,
                'random_seed': random_seed
            }
        }

        print(f"\nSimulation Results:")
        print(f"Mean Portfolio Return: {results['mean_return']:.2%}")
        print(f"Portfolio Volatility: {results['std_return']:.2%}")
        print(f"Best Case Return: {results['max_return']:.2%}")
        print(f"Worst Case Return: {results['min_return']:.2%}")

        return results

    def parallel_simulation(
        self,
        initial_prices: np.ndarray,
        n_simulations: int = 10000,
        n_days: int = 252,
        initial_portfolio_value: float = 100000,
        n_threads: int = 4
    ) -> dict:
        """Run Monte Carlo simulation in parallel using thread pool.

        Splits the total simulations across multiple threads and combines
        results. Each thread uses a different random seed for independence.

        Args:
            initial_prices: Starting price for each asset.
            n_simulations: Total number of simulation paths.
            n_days: Trading days to simulate.
            initial_portfolio_value: Starting portfolio value.
            n_threads: Number of parallel threads.

        Returns:
            Combined results dictionary from all threads.
        """
        print(f"Running parallel simulation with {n_threads} threads...")
        sims_per_thread = n_simulations // n_threads
        remaining_sims = n_simulations % n_threads

        def run_batch(batch_size: int, seed_offset: int) -> dict:
            return self.run_simulation(
                initial_prices, batch_size, n_days,
                initial_portfolio_value, True, 42 + seed_offset
            )

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = []
            for i in range(n_threads):
                batch_size = sims_per_thread + (1 if i < remaining_sims else 0)
                futures.append(executor.submit(run_batch, batch_size, i))
            all_results = [future.result() for future in futures]

        combined_results = self._combine_simulation_results(all_results)
        return combined_results

    def _combine_simulation_results(self, results_list: list[dict]) -> dict:
        """Merge results from parallel simulation batches.

        Concatenates portfolio values, returns, and final values from
        multiple simulation runs into a single results dictionary.

        Args:
            results_list: List of result dictionaries from each batch.

        Returns:
            Single combined results dictionary.
        """
        combined_portfolio_values = np.vstack([r['portfolio_values'] for r in results_list])
        combined_portfolio_returns = np.vstack([r['portfolio_returns'] for r in results_list])
        combined_final_values = np.concatenate([r['final_values'] for r in results_list])
        combined_total_returns = np.concatenate([r['total_returns'] for r in results_list])

        combined_results = {
            'portfolio_values': combined_portfolio_values,
            'portfolio_returns': combined_portfolio_returns,
            'final_values': combined_final_values,
            'total_returns': combined_total_returns,
            'mean_return': np.mean(combined_total_returns),
            'std_return': np.std(combined_total_returns),
            'min_return': np.min(combined_total_returns),
            'max_return': np.max(combined_total_returns),
            'simulation_params': results_list[0]['simulation_params']
        }
        combined_results['simulation_params']['n_simulations'] = len(combined_total_returns)
        return combined_results