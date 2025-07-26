"""
Chapter 107: Propensity Score Methods for Trading

This module provides tools for applying propensity score methods
to causal inference in trading strategies.

Main components:
- PropensityScoreModel: Estimate propensity scores using various methods
- PropensityMatcher: Match treated and control observations
- CausalEffectEstimator: Estimate average treatment effects (ATE/ATT/ATC)
- DataLoader: Load stock and crypto data from various sources
- Backtester: Backtest propensity-adjusted trading strategies
"""

from .model import (
    PropensityScoreModel,
    PropensityMatcher,
    CausalEffectEstimator,
    IPWEstimator,
    DoublyRobustEstimator,
)
from .data_loader import (
    DataLoader,
    BybitDataLoader,
    StockDataLoader,
    generate_synthetic_data,
)
from .backtest import (
    PropensityBacktester,
    BacktestResult,
    PerformanceMetrics,
)

__version__ = "0.1.0"
__all__ = [
    "PropensityScoreModel",
    "PropensityMatcher",
    "CausalEffectEstimator",
    "IPWEstimator",
    "DoublyRobustEstimator",
    "DataLoader",
    "BybitDataLoader",
    "StockDataLoader",
    "generate_synthetic_data",
    "PropensityBacktester",
    "BacktestResult",
    "PerformanceMetrics",
]
