"""
Tests for DataManager module.
Validates sample data generation, data cleaning, and returns calculation.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_manager import DataManager


@pytest.fixture
def data_manager():
    """Create a DataManager instance for testing."""
    return DataManager()


@pytest.fixture
def sample_data(data_manager):
    """Generate sample data for testing."""
    return data_manager.generate_sample_data()


class TestSampleDataGeneration:
    """Test synthetic data generation via GBM."""

    def test_generates_all_assets(self, sample_data):
        """Sample data should contain all three asset classes."""
        assert set(sample_data.keys()) == {'Nifty', 'Gold', 'Crude'}

    def test_ohlcv_columns_present(self, sample_data):
        """Each asset DataFrame should have OHLCV columns."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for asset_name, df in sample_data.items():
            for col in required_columns:
                assert col in df.columns, f"Missing {col} in {asset_name}"

    def test_no_nan_values(self, sample_data):
        """Generated data should contain no NaN values."""
        for asset_name, df in sample_data.items():
            assert not df.isnull().any().any(), f"NaN values found in {asset_name}"

    def test_positive_prices(self, sample_data):
        """All prices should be strictly positive (GBM property)."""
        for asset_name, df in sample_data.items():
            assert (df[['Open', 'High', 'Low', 'Close']] > 0).all().all(), \
                f"Non-positive prices in {asset_name}"

    def test_high_low_relationship(self, sample_data):
        """High should be >= Low for all rows."""
        for asset_name, df in sample_data.items():
            assert (df['High'] >= df['Low']).all(), \
                f"High < Low found in {asset_name}"

    def test_sufficient_data_points(self, sample_data):
        """Should generate at least 250 trading days (roughly 1 year)."""
        for asset_name, df in sample_data.items():
            assert len(df) >= 250, \
                f"Insufficient data for {asset_name}: {len(df)} rows"

    def test_reproducibility_with_seed(self, data_manager):
        """Same seed should produce identical data."""
        data1 = data_manager.generate_sample_data()
        data2 = data_manager.generate_sample_data()
        for asset in data1.keys():
            # Compare values only — index uses datetime.now() so timestamps differ slightly
            np.testing.assert_array_almost_equal(
                data1[asset].values, data2[asset].values
            )

    def test_business_days_only(self, sample_data):
        """Index should only contain business days (Mon-Fri)."""
        for asset_name, df in sample_data.items():
            weekdays = df.index.dayofweek
            assert (weekdays < 5).all(), \
                f"Weekend dates found in {asset_name}"


class TestDataCleaning:
    """Test data cleaning and outlier removal."""

    def test_clean_removes_nan(self, data_manager):
        """Cleaning should remove rows with NaN values."""
        df = pd.DataFrame({
            'Open': [100, np.nan, 102],
            'High': [101, 101, 103],
            'Low': [99, 99, 101],
            'Close': [100.5, 100.5, 102.5],
            'Volume': [1000, 1000, 1000]
        })
        cleaned = data_manager.clean_data(df)
        assert not cleaned.isnull().any().any()

    def test_clean_removes_outliers(self, data_manager):
        """Daily changes > 20% should be removed as outliers."""
        prices = [100, 101, 130, 102, 103]  # 130 is a 28.7% jump
        df = pd.DataFrame({
            'Open': prices,
            'High': [p + 1 for p in prices],
            'Low': [p - 1 for p in prices],
            'Close': prices,
            'Volume': [1000] * 5
        }, index=pd.date_range('2024-01-01', periods=5))
        cleaned = data_manager.clean_data(df)
        assert len(cleaned) < len(df)

    def test_clean_adds_missing_volume(self, data_manager):
        """Should add default Volume column if missing."""
        df = pd.DataFrame({
            'Open': [100, 101],
            'High': [101, 102],
            'Low': [99, 100],
            'Close': [100.5, 101.5],
        })
        cleaned = data_manager.clean_data(df)
        assert 'Volume' in cleaned.columns


class TestReturnsCalculation:
    """Test log returns calculation."""

    def test_returns_shape(self, data_manager, sample_data):
        """Returns DataFrame should have n-1 rows (one less than prices)."""
        returns_df = data_manager.calculate_returns(sample_data)
        for asset_name in sample_data:
            assert len(returns_df[asset_name].dropna()) == len(sample_data[asset_name]) - 1

    def test_returns_are_log_returns(self, data_manager):
        """Verify log return formula: r_t = ln(P_t / P_{t-1})."""
        data = {
            'TestAsset': pd.DataFrame({
                'Close': [100.0, 110.0, 105.0]
            }, index=pd.date_range('2024-01-01', periods=3))
        }
        returns = data_manager.calculate_returns(data)
        expected = np.log(110.0 / 100.0)
        np.testing.assert_almost_equal(returns['TestAsset'].iloc[0], expected, decimal=10)

    def test_returns_columns_match_assets(self, data_manager, sample_data):
        """Returns DataFrame columns should match input asset names."""
        returns_df = data_manager.calculate_returns(sample_data)
        assert set(returns_df.columns) == set(sample_data.keys())


class TestLatestPrices:
    """Test latest price extraction."""

    def test_latest_prices_correct(self, data_manager, sample_data):
        """Latest prices should match the last Close value in each DataFrame."""
        latest = data_manager.get_latest_prices(sample_data)
        for asset_name, price in latest.items():
            expected = sample_data[asset_name]['Close'].iloc[-1]
            assert price == expected


class TestDataValidation:
    """Test data quality validation."""

    def test_validation_returns_all_assets(self, data_manager, sample_data):
        """Validation report should cover all assets."""
        report = data_manager.validate_data_quality(sample_data)
        assert set(report.keys()) == set(sample_data.keys())

    def test_sample_data_passes_validation(self, data_manager, sample_data):
        """Sample data with 2 years should pass validation."""
        report = data_manager.validate_data_quality(sample_data)
        for asset_name, result in report.items():
            assert result['status'] in ('SUCCESS', 'WARNING')

    def test_empty_data_fails_validation(self, data_manager):
        """Empty DataFrames should fail validation."""
        empty_data = {'TestAsset': pd.DataFrame()}
        report = data_manager.validate_data_quality(empty_data)
        assert report['TestAsset']['status'] == 'FAILED'
