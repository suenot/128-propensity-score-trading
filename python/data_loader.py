"""
Data Loader for Propensity Score Trading

This module provides data loading utilities for both traditional
stock market data and cryptocurrency data from Bybit.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Container for market data with computed features."""
    prices: pd.DataFrame  # OHLCV data
    features: pd.DataFrame  # Computed features for propensity modeling
    returns: pd.Series  # Forward returns (outcome variable)
    signal: pd.Series  # Trading signal (treatment variable)
    metadata: Dict


def generate_synthetic_data(
    n_samples: int = 1000,
    n_assets: int = 1,
    start_date: str = "2020-01-01",
    true_treatment_effect: float = 0.005,
    confounding_strength: float = 0.5,
    signal_threshold: float = 0.02,
    random_state: int = 42,
) -> MarketData:
    """
    Generate synthetic trading data with known causal structure.

    This is useful for testing and validating propensity score methods
    because we know the true treatment effect.

    Args:
        n_samples: Number of trading days
        n_assets: Number of assets (currently supports 1)
        start_date: Start date for the data
        true_treatment_effect: True causal effect of the signal on returns
        confounding_strength: How much confounders affect both signal and outcome
        signal_threshold: Threshold for signal activation
        random_state: Random seed

    Returns:
        MarketData with synthetic prices, features, and outcomes
    """
    np.random.seed(random_state)

    dates = pd.date_range(start=start_date, periods=n_samples, freq='D')

    # Generate confounders (market conditions)
    # These affect both signal activation and future returns
    volatility = np.abs(np.random.normal(0.02, 0.01, n_samples))
    volume = np.random.lognormal(10, 0.5, n_samples)
    market_regime = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])  # 0=bear, 1=bull
    momentum = np.cumsum(np.random.normal(0, 0.01, n_samples))

    # Generate price series
    initial_price = 100
    returns_noise = np.random.normal(0, volatility)

    # Confounder effect on returns
    base_returns = (
        0.0005 * market_regime +  # Bull markets have higher returns
        -0.5 * volatility +  # High vol → lower returns
        0.01 * np.sign(momentum) +  # Trend following
        returns_noise
    )

    prices = initial_price * np.exp(np.cumsum(base_returns))

    # Create OHLCV data
    high = prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples)))
    open_price = prices * (1 + np.random.normal(0, 0.005, n_samples))

    ohlcv = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume,
    }, index=dates)

    # Generate signal based on confounders (creates confounding)
    # Signal is more likely to fire in certain market conditions
    signal_propensity = 1 / (1 + np.exp(-(
        -2 +
        confounding_strength * 10 * (volatility - 0.02) +  # More likely in low vol
        confounding_strength * 2 * market_regime +  # More likely in bull markets
        confounding_strength * 5 * np.clip(momentum, -0.5, 0.5)  # More likely with positive momentum
    )))

    # Compute 5-day momentum as observable signal criterion
    momentum_5d = pd.Series(prices).pct_change(5).fillna(0).values
    signal_fires = (momentum_5d > signal_threshold) | (np.random.random(n_samples) < signal_propensity)
    signal = signal_fires.astype(int)

    # Forward returns (1-day ahead) - this is our outcome
    # TRUE DATA GENERATING PROCESS:
    # return = base_effect_from_confounders + true_treatment_effect * signal + noise
    forward_returns = np.roll(base_returns, -1) + true_treatment_effect * signal
    forward_returns[-1] = 0  # Last observation has no forward return

    # Feature matrix for propensity estimation
    price_sma = pd.Series(prices).rolling(20).mean()
    price_sma = price_sma.fillna(pd.Series(prices))
    volume_sma = pd.Series(volume).rolling(20).mean()
    volume_sma = volume_sma.fillna(pd.Series(volume))

    features = pd.DataFrame({
        'volatility': volatility,
        'volume_log': np.log(volume),
        'market_regime': market_regime,
        'momentum': momentum,
        'momentum_5d': momentum_5d,
        'price_sma_ratio': prices / price_sma.values,
        'volume_sma_ratio': volume / volume_sma.values,
    }, index=dates)

    metadata = {
        'true_treatment_effect': true_treatment_effect,
        'confounding_strength': confounding_strength,
        'signal_threshold': signal_threshold,
        'n_samples': n_samples,
        'n_treated': signal.sum(),
        'treatment_rate': signal.mean(),
        'data_type': 'synthetic',
    }

    return MarketData(
        prices=ohlcv,
        features=features,
        returns=pd.Series(forward_returns, index=dates, name='forward_return'),
        signal=pd.Series(signal, index=dates, name='signal'),
        metadata=metadata
    )


class DataLoader:
    """Base class for data loading."""

    def load(self, *args, **kwargs) -> MarketData:
        raise NotImplementedError


class BybitDataLoader(DataLoader):
    """
    Load cryptocurrency data from Bybit exchange.

    Uses Bybit's public API to fetch historical OHLCV data.
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self, rate_limit_pause: float = 0.1):
        """
        Args:
            rate_limit_pause: Seconds to wait between API calls
        """
        self.rate_limit_pause = rate_limit_pause

    def load(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "D",  # D=daily, 60=hourly, etc.
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000,
        signal_function: Optional[callable] = None,
    ) -> MarketData:
        """
        Load data from Bybit.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            interval: Candle interval ('1', '5', '15', '60', '240', 'D', 'W')
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            limit: Maximum number of candles
            signal_function: Function to generate trading signal from features

        Returns:
            MarketData object
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests library required: pip install requests")

        # Convert dates to timestamps
        if start_date:
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        else:
            start_ts = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)

        if end_date:
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        else:
            end_ts = int(datetime.now().timestamp() * 1000)

        # Fetch data from Bybit API (v5)
        endpoint = f"{self.BASE_URL}/v5/market/kline"
        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "start": start_ts,
            "end": end_ts,
            "limit": min(limit, 1000),
        }

        logger.info(f"Fetching {symbol} data from Bybit...")

        all_data = []

        while True:
            response = requests.get(endpoint, params=params)

            if response.status_code != 200:
                raise RuntimeError(f"API error: {response.status_code} - {response.text}")

            result = response.json()

            if result['retCode'] != 0:
                raise RuntimeError(f"Bybit API error: {result['retMsg']}")

            data = result['result']['list']
            if not data:
                break

            all_data.extend(data)

            # Check if we have more data to fetch
            if len(data) < params['limit']:
                break

            # Update start time for next batch
            oldest_ts = int(data[-1][0])
            if oldest_ts <= start_ts:
                break
            params['end'] = oldest_ts

            time.sleep(self.rate_limit_pause)

        if not all_data:
            raise ValueError(f"No data returned for {symbol}")

        # Parse data into DataFrame
        # Bybit kline format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df = df.sort_values('timestamp').set_index('timestamp')

        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = pd.to_numeric(df[col])

        # Compute features
        features = self._compute_features(df)

        # Compute forward returns (outcome)
        forward_returns = df['close'].pct_change().shift(-1).fillna(0)

        # Generate signal (treatment)
        if signal_function is None:
            # Default: momentum signal
            momentum_5d = df['close'].pct_change(5).fillna(0)
            signal = (momentum_5d > 0.02).astype(int)
        else:
            signal = signal_function(df, features)

        metadata = {
            'symbol': symbol,
            'interval': interval,
            'start_date': df.index.min().strftime('%Y-%m-%d'),
            'end_date': df.index.max().strftime('%Y-%m-%d'),
            'n_samples': len(df),
            'n_treated': signal.sum(),
            'treatment_rate': signal.mean(),
            'data_type': 'bybit_crypto',
        }

        return MarketData(
            prices=df[['open', 'high', 'low', 'close', 'volume']],
            features=features,
            returns=pd.Series(forward_returns, name='forward_return'),
            signal=pd.Series(signal, name='signal'),
            metadata=metadata
        )

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features for propensity score estimation."""
        features = pd.DataFrame(index=df.index)

        # Volatility features
        features['volatility_20d'] = df['close'].pct_change().rolling(20).std().fillna(0.01)
        features['volatility_5d'] = df['close'].pct_change().rolling(5).std().fillna(0.01)

        # Volume features
        features['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean().fillna(df['volume'])

        # Price features
        features['price_sma20_ratio'] = df['close'] / df['close'].rolling(20).mean().fillna(df['close'])
        features['price_sma50_ratio'] = df['close'] / df['close'].rolling(50).mean().fillna(df['close'])

        # Momentum features
        features['momentum_5d'] = df['close'].pct_change(5).fillna(0)
        features['momentum_10d'] = df['close'].pct_change(10).fillna(0)
        features['momentum_20d'] = df['close'].pct_change(20).fillna(0)

        # Range features
        features['range_ratio'] = (df['high'] - df['low']) / df['close']

        # RSI-like feature
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.inf)
        features['rsi'] = (100 - (100 / (1 + rs))).fillna(50) / 100  # Normalize to 0-1

        return features.fillna(0)


class StockDataLoader(DataLoader):
    """
    Load stock market data.

    Uses Yahoo Finance (via yfinance) or local CSV files.
    """

    def load(
        self,
        symbol: str = "SPY",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        source: str = "yfinance",
        csv_path: Optional[str] = None,
        signal_function: Optional[callable] = None,
    ) -> MarketData:
        """
        Load stock data.

        Args:
            symbol: Stock ticker
            start_date: Start date
            end_date: End date
            source: Data source ('yfinance' or 'csv')
            csv_path: Path to CSV file if source='csv'
            signal_function: Function to generate trading signal

        Returns:
            MarketData object
        """
        if source == "yfinance":
            df = self._load_from_yfinance(symbol, start_date, end_date)
        elif source == "csv":
            df = self._load_from_csv(csv_path)
        else:
            raise ValueError(f"Unknown source: {source}")

        # Compute features
        features = self._compute_features(df)

        # Forward returns
        forward_returns = df['Close'].pct_change().shift(-1).fillna(0)

        # Generate signal
        if signal_function is None:
            momentum_5d = df['Close'].pct_change(5).fillna(0)
            signal = (momentum_5d > 0.02).astype(int)
        else:
            signal = signal_function(df, features)

        metadata = {
            'symbol': symbol,
            'start_date': df.index.min().strftime('%Y-%m-%d'),
            'end_date': df.index.max().strftime('%Y-%m-%d'),
            'n_samples': len(df),
            'n_treated': signal.sum(),
            'treatment_rate': signal.mean(),
            'data_type': 'stock',
        }

        return MarketData(
            prices=df,
            features=features,
            returns=pd.Series(forward_returns, name='forward_return'),
            signal=pd.Series(signal, name='signal'),
            metadata=metadata
        )

    def _load_from_yfinance(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """Load data from Yahoo Finance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance required: pip install yfinance")

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        return df

    def _load_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
        return df

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features for stock data."""
        # Handle both yfinance column names (Capital) and standard (lowercase)
        close_col = 'Close' if 'Close' in df.columns else 'close'
        volume_col = 'Volume' if 'Volume' in df.columns else 'volume'
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'

        features = pd.DataFrame(index=df.index)

        # Volatility
        features['volatility_20d'] = df[close_col].pct_change().rolling(20).std().fillna(0.01)
        features['volatility_5d'] = df[close_col].pct_change().rolling(5).std().fillna(0.01)

        # Volume
        features['volume_sma_ratio'] = df[volume_col] / df[volume_col].rolling(20).mean().fillna(df[volume_col])

        # Price ratios
        features['price_sma20_ratio'] = df[close_col] / df[close_col].rolling(20).mean().fillna(df[close_col])
        features['price_sma50_ratio'] = df[close_col] / df[close_col].rolling(50).mean().fillna(df[close_col])

        # Momentum
        features['momentum_5d'] = df[close_col].pct_change(5).fillna(0)
        features['momentum_10d'] = df[close_col].pct_change(10).fillna(0)
        features['momentum_20d'] = df[close_col].pct_change(20).fillna(0)

        # Range
        features['range_ratio'] = (df[high_col] - df[low_col]) / df[close_col]

        # RSI
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.inf)
        features['rsi'] = (100 - (100 / (1 + rs))).fillna(50) / 100

        return features.fillna(0)


def create_trading_signal(
    df: pd.DataFrame,
    features: pd.DataFrame,
    signal_type: str = "momentum",
    **kwargs
) -> pd.Series:
    """
    Factory function to create different trading signals.

    Args:
        df: Price data
        features: Computed features
        signal_type: Type of signal to generate
        **kwargs: Signal-specific parameters

    Returns:
        Binary signal series
    """
    close = df['close'] if 'close' in df.columns else df['Close']

    if signal_type == "momentum":
        lookback = kwargs.get('lookback', 5)
        threshold = kwargs.get('threshold', 0.02)
        momentum = close.pct_change(lookback).fillna(0)
        return (momentum > threshold).astype(int)

    elif signal_type == "mean_reversion":
        lookback = kwargs.get('lookback', 20)
        threshold = kwargs.get('threshold', -1.5)
        zscore = (close - close.rolling(lookback).mean()) / close.rolling(lookback).std()
        return (zscore.fillna(0) < threshold).astype(int)

    elif signal_type == "breakout":
        lookback = kwargs.get('lookback', 20)
        high_col = 'high' if 'high' in df.columns else 'High'
        breakout_level = df[high_col].rolling(lookback).max()
        return (close > breakout_level.shift(1)).astype(int)

    elif signal_type == "rsi":
        threshold = kwargs.get('threshold', 0.3)
        return (features['rsi'] < threshold).astype(int)

    else:
        raise ValueError(f"Unknown signal type: {signal_type}")


if __name__ == "__main__":
    # Example: Generate synthetic data
    print("=== Generating Synthetic Data ===")
    synthetic_data = generate_synthetic_data(
        n_samples=500,
        true_treatment_effect=0.005,
        confounding_strength=0.3,
    )

    print(f"Samples: {synthetic_data.metadata['n_samples']}")
    print(f"Treatment rate: {synthetic_data.metadata['treatment_rate']:.2%}")
    print(f"True ATE: {synthetic_data.metadata['true_treatment_effect']}")
    print(f"\nFeatures:\n{synthetic_data.features.head()}")
    print(f"\nPrices:\n{synthetic_data.prices.head()}")

    # Example: Try to load Bybit data (will fail without network)
    print("\n=== Bybit Data Loader Example ===")
    try:
        bybit_loader = BybitDataLoader()
        # This would work with network access:
        # bybit_data = bybit_loader.load(symbol="BTCUSDT", interval="D", limit=100)
        print("BybitDataLoader initialized successfully")
        print("To load data, use: bybit_loader.load(symbol='BTCUSDT', interval='D')")
    except Exception as e:
        print(f"Note: Bybit loading requires network access: {e}")

    # Example: Stock data loader
    print("\n=== Stock Data Loader Example ===")
    try:
        stock_loader = StockDataLoader()
        print("StockDataLoader initialized successfully")
        print("To load data, use: stock_loader.load(symbol='SPY', source='yfinance')")
    except Exception as e:
        print(f"Note: Stock loading requires yfinance: {e}")
