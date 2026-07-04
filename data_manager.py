"""
Data Manager Module
====================
Fetches, validates, and preprocesses historical market data for portfolio risk analysis.

Supports real-time data from Yahoo Finance with multi-level fallback mechanisms,
and generates synthetic GBM-based sample data when APIs are unavailable.

Assets tracked: Nifty 50 (Indian equity index), Gold futures, Crude Oil futures.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import time
from datetime import datetime, timedelta
from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class DataManager:
    """Manages historical market data acquisition, cleaning, and preprocessing.

    Implements a resilient data pipeline with primary/fallback symbol resolution,
    exponential backoff retry logic, and automatic sample data generation as a
    last-resort fallback.

    Attributes:
        asset_symbols: Primary Yahoo Finance ticker symbols for each asset.
        fallback_symbols: Alternative tickers tried when primary symbols fail.
        session: Requests session with retry strategy for robust HTTP calls.
    """

    def __init__(self) -> None:
        self.asset_symbols: dict[str, str] = {
            'Nifty': '^NSEI',
            'Gold': 'GC=F',
            'Crude': 'CL=F'
        }
        self.fallback_symbols: dict[str, list[str]] = {
            'Nifty': ['NIFTYBEES.NS', '^NSEI', 'INFY'],
            'Gold': ['GLD', 'GOLD', 'GC=F'],
            'Crude': ['USO', 'OIL', 'CL=F']
        }
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def fetch_data_with_fallback(
        self, symbol: str, period: str = '2y', interval: str = '1d', max_retries: int = 3
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single symbol with exponential backoff retry.

        Args:
            symbol: Yahoo Finance ticker symbol (e.g., '^NSEI', 'GC=F').
            period: Lookback period ('1y', '2y', '5y', etc.).
            interval: Bar interval ('1d', '1h', etc.).
            max_retries: Maximum number of retry attempts.

        Returns:
            DataFrame with OHLCV columns, or empty DataFrame on failure.
        """
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(2 ** attempt)
                ticker = yf.Ticker(symbol, session=self.session)
                data = ticker.history(period=period, interval=interval, timeout=30)
                if not data.empty and len(data) > 50:
                    return data
                else:
                    print(f"Insufficient data for {symbol}, attempt {attempt + 1}")
            except Exception as e:
                print(f"Error fetching {symbol} (attempt {attempt + 1}): {str(e)}")
                continue
        return pd.DataFrame()

    def fetch_historical_data(
        self, period: str = '2y', use_sample_data: bool = False
    ) -> dict[str, pd.DataFrame]:
        """Fetch historical data for all portfolio assets.

        Tries primary symbols first, then iterates through fallback symbols.
        If all API calls fail, generates synthetic sample data.

        Args:
            period: Lookback period for historical data.
            use_sample_data: If True, skip API calls and use synthetic data.

        Returns:
            Dictionary mapping asset names to DataFrames with OHLCV data.
        """
        if use_sample_data:
            return self.generate_sample_data()

        historical_data: dict[str, pd.DataFrame] = {}
        successful_fetches = 0

        print("Fetching market data...")
        for asset_name, primary_symbol in self.asset_symbols.items():
            print(f"Fetching data for {asset_name}...")
            data = self.fetch_data_with_fallback(primary_symbol, period)

            if data.empty and asset_name in self.fallback_symbols:
                print(f"Primary symbol failed for {asset_name}, trying fallbacks...")
                for fallback_symbol in self.fallback_symbols[asset_name]:
                    data = self.fetch_data_with_fallback(fallback_symbol, period)
                    if not data.empty:
                        print(f"Successfully fetched {asset_name} using {fallback_symbol}")
                        break

            if not data.empty:
                data = self.clean_data(data)
                if data.index.name is None and not data.empty:
                    data.index.name = 'Date'
                historical_data[asset_name] = data
                successful_fetches += 1
                print(f"✓ Successfully fetched {asset_name} data ({len(data)} records)")
            else:
                print(f"✗ Failed to fetch data for {asset_name}")

        if successful_fetches == 0:
            print("All API calls failed. Using sample data for demonstration...")
            return self.generate_sample_data()

        if successful_fetches < len(self.asset_symbols):
            print("Some assets missing. Filling with sample data...")
            sample_data = self.generate_sample_data()
            for asset_name in self.asset_symbols.keys():
                if asset_name not in historical_data:
                    historical_data[asset_name] = sample_data[asset_name]
                    print(f"Using sample data for {asset_name}")

        return historical_data

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate OHLCV data.

        Removes NaN rows, fills missing columns, and filters out daily returns
        exceeding a 20% absolute threshold as outliers.

        Args:
            data: Raw OHLCV DataFrame from Yahoo Finance.

        Returns:
            Cleaned DataFrame with outliers removed.
        """
        data = data.dropna()
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                if col == 'Volume':
                    data[col] = 1000000
                else:
                    print(f"Warning: Missing {col} column")

        if len(data) > 1:
            daily_change = data['Close'].pct_change().abs()
            outlier_threshold = 0.20
            outliers = daily_change > outlier_threshold
            if outliers.sum() > 0:
                print(f"Removing {outliers.sum()} outlier(s)")
                data = data[~outliers]

        return data

    def generate_sample_data(self) -> dict[str, pd.DataFrame]:
        """Generate synthetic OHLCV data using Geometric Brownian Motion (GBM).

        Creates realistic market data with configurable drift and volatility
        parameters for each asset class. Useful for demo/testing when
        Yahoo Finance API is unavailable.

        Returns:
            Dictionary mapping asset names to synthetic OHLCV DataFrames.
        """
        print("Generating sample market data...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates[dates.dayofweek < 5]  # Business days only

        sample_data: dict[str, pd.DataFrame] = {}
        asset_params = {
            'Nifty': {'start_price': 15000, 'volatility': 0.015, 'drift': 0.0003},
            'Gold': {'start_price': 1800, 'volatility': 0.012, 'drift': 0.0002},
            'Crude': {'start_price': 70, 'volatility': 0.025, 'drift': 0.0001}
        }

        np.random.seed(42)
        for asset_name, params in asset_params.items():
            n_days = len(dates)
            returns = np.random.normal(params['drift'], params['volatility'], n_days)

            # Vectorized cumulative price path via GBM
            price_path = [params['start_price']]
            for i in range(1, n_days):
                price_path.append(price_path[-1] * (1 + returns[i]))

            prices = np.array(price_path)
            high_low_spread = 0.01
            open_close_spread = 0.005

            ohlc_data = []
            for i, close_price in enumerate(prices):
                if i == 0:
                    open_price = close_price
                else:
                    open_price = prices[i-1] * (1 + np.random.normal(0, open_close_spread))
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, high_low_spread)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, high_low_spread)))
                volume = np.random.randint(1000000, 5000000)
                ohlc_data.append({
                    'Open': open_price,
                    'High': high_price,
                    'Low': low_price,
                    'Close': close_price,
                    'Volume': volume
                })

            df = pd.DataFrame(ohlc_data, index=dates[:len(prices)])
            sample_data[asset_name] = df

        return sample_data

    def calculate_returns(self, historical_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate log returns for each asset in the portfolio.

        Uses log returns: r_t = ln(P_t / P_{t-1}), which are preferred in
        quantitative finance for their additive property across time.

        Args:
            historical_data: Dictionary mapping asset names to OHLCV DataFrames.

        Returns:
            DataFrame with log returns for each asset, indexed by date.
        """
        returns_data: dict[str, pd.Series] = {}
        for asset_name, data in historical_data.items():
            if 'Close' in data.columns and not data.empty:
                returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
                if returns.index.name is None:
                    returns.index.name = 'Date'
                returns_data[asset_name] = returns
            else:
                returns_data[asset_name] = pd.Series(dtype=float)
        returns_df = pd.DataFrame(returns_data)
        if returns_df.index.name is None and not returns_df.empty:
            returns_df.index.name = 'Date'
        return returns_df

    def get_latest_prices(self, historical_data: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Extract the most recent closing price for each asset.

        Args:
            historical_data: Dictionary mapping asset names to OHLCV DataFrames.

        Returns:
            Dictionary mapping asset names to their latest closing prices.
        """
        latest_prices: dict[str, float] = {}
        for asset_name, data in historical_data.items():
            if not data.empty and 'Close' in data.columns:
                latest_prices[asset_name] = data['Close'].iloc[-1]
            else:
                sample_prices = {'Nifty': 18000, 'Gold': 1900, 'Crude': 75}
                latest_prices[asset_name] = sample_prices.get(asset_name, 100)
        return latest_prices

    def validate_data_quality(self, historical_data: dict[str, pd.DataFrame]) -> dict[str, dict]:
        """Run data quality checks on historical data.

        Validates minimum data length (250 trading days recommended) and
        checks for large gaps (>7 calendar days) in the time series.

        Args:
            historical_data: Dictionary mapping asset names to OHLCV DataFrames.

        Returns:
            Dictionary with validation status ('SUCCESS', 'WARNING', 'FAILED')
            and diagnostic messages for each asset.
        """
        validation_report: dict[str, dict] = {}
        for asset_name, data in historical_data.items():
            if data.empty:
                validation_report[asset_name] = {
                    'status': 'FAILED',
                    'message': 'No data available'
                }
                continue
            min_required_days = 250
            if len(data) < min_required_days:
                validation_report[asset_name] = {
                    'status': 'WARNING',
                    'message': f'Only {len(data)} days of data (recommended: {min_required_days}+)'
                }
            else:
                validation_report[asset_name] = {
                    'status': 'SUCCESS',
                    'message': f'{len(data)} days of data available'
                }
            if len(data) > 1:
                date_gaps = pd.Series(data.index).diff().dt.days
                large_gaps = date_gaps[date_gaps > 7].count()
                if large_gaps > 0:
                    validation_report[asset_name]['message'] += f' (Warning: {large_gaps} large gaps)'
        return validation_report