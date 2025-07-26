"""
Backtesting Framework for Propensity Score Trading Strategies

This module provides tools to backtest trading strategies while
accounting for causal effects using propensity score methods.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

from .model import (
    PropensityScoreModel,
    PropensityMatcher,
    IPWEstimator,
    DoublyRobustEstimator,
    CausalEstimate,
)
from .data_loader import MarketData

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a trading strategy."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    n_trades: int
    avg_return_per_trade: float

    # Causal metrics (if available)
    causal_ate: Optional[float] = None
    ate_confidence_interval: Optional[Tuple[float, float]] = None
    ate_p_value: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'n_trades': self.n_trades,
            'avg_return_per_trade': self.avg_return_per_trade,
            'causal_ate': self.causal_ate,
            'ate_ci': self.ate_confidence_interval,
            'ate_p_value': self.ate_p_value,
        }

    def __str__(self) -> str:
        lines = [
            "=== Performance Metrics ===",
            f"Total Return:      {self.total_return:>10.2%}",
            f"Annual Return:     {self.annualized_return:>10.2%}",
            f"Volatility:        {self.volatility:>10.2%}",
            f"Sharpe Ratio:      {self.sharpe_ratio:>10.2f}",
            f"Max Drawdown:      {self.max_drawdown:>10.2%}",
            f"Win Rate:          {self.win_rate:>10.2%}",
            f"Profit Factor:     {self.profit_factor:>10.2f}",
            f"Number of Trades:  {self.n_trades:>10d}",
            f"Avg Return/Trade:  {self.avg_return_per_trade:>10.4%}",
        ]

        if self.causal_ate is not None:
            lines.extend([
                "",
                "=== Causal Analysis ===",
                f"Causal ATE:        {self.causal_ate:>10.4%}",
            ])
            if self.ate_confidence_interval:
                lines.append(
                    f"95% CI:            [{self.ate_confidence_interval[0]:.4%}, {self.ate_confidence_interval[1]:.4%}]"
                )
            if self.ate_p_value is not None:
                lines.append(f"P-value:           {self.ate_p_value:>10.4f}")

        return "\n".join(lines)


@dataclass
class BacktestResult:
    """Complete result from a backtest run."""
    metrics: PerformanceMetrics
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    causal_estimate: Optional[CausalEstimate] = None
    propensity_scores: Optional[pd.Series] = None

    def plot(self, save_path: Optional[str] = None):
        """Plot the backtest results."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Equity curve
        axes[0].plot(self.equity_curve.index, self.equity_curve.values)
        axes[0].set_title('Equity Curve')
        axes[0].set_ylabel('Portfolio Value')
        axes[0].grid(True, alpha=0.3)

        # Drawdown
        rolling_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - rolling_max) / rolling_max
        axes[1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        axes[1].plot(drawdown.index, drawdown.values, color='red', linewidth=0.5)
        axes[1].set_title('Drawdown')
        axes[1].set_ylabel('Drawdown %')
        axes[1].grid(True, alpha=0.3)

        # Propensity scores distribution (if available)
        if self.propensity_scores is not None:
            treated_ps = self.propensity_scores[self.positions == 1]
            untreated_ps = self.propensity_scores[self.positions == 0]

            axes[2].hist(treated_ps, bins=30, alpha=0.5, label='Signal=1 (Treated)', density=True)
            axes[2].hist(untreated_ps, bins=30, alpha=0.5, label='Signal=0 (Control)', density=True)
            axes[2].set_title('Propensity Score Distribution')
            axes[2].set_xlabel('Propensity Score')
            axes[2].legend()
        else:
            # Plot returns distribution
            axes[2].hist(self.returns, bins=50, alpha=0.7, edgecolor='black')
            axes[2].axvline(0, color='red', linestyle='--')
            axes[2].set_title('Returns Distribution')
            axes[2].set_xlabel('Return')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return fig


class PropensityBacktester:
    """
    Backtester that incorporates propensity score analysis.

    This backtester not only evaluates strategy performance but also
    estimates the causal effect of the trading signal using propensity
    score methods.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        position_sizing: str = "equal",
        use_propensity_weighting: bool = True,
        propensity_method: str = "gradient_boosting",
        causal_estimator: str = "doubly_robust",
    ):
        """
        Initialize the backtester.

        Args:
            initial_capital: Starting capital
            transaction_cost: Cost per trade (as fraction)
            position_sizing: Position sizing method ('equal', 'propensity_weighted')
            use_propensity_weighting: Whether to use propensity-weighted positions
            propensity_method: Method for propensity score estimation
            causal_estimator: Method for causal effect estimation
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.position_sizing = position_sizing
        self.use_propensity_weighting = use_propensity_weighting
        self.propensity_method = propensity_method
        self.causal_estimator = causal_estimator

    def run(
        self,
        data: MarketData,
        warmup_period: int = 50,
        refit_frequency: int = 20,
    ) -> BacktestResult:
        """
        Run the backtest.

        Args:
            data: MarketData object with prices, features, returns, and signal
            warmup_period: Number of periods for warmup (propensity estimation)
            refit_frequency: How often to refit the propensity model

        Returns:
            BacktestResult with performance metrics and equity curve
        """
        n = len(data.prices)

        if n < warmup_period + 10:
            raise ValueError(f"Not enough data. Need at least {warmup_period + 10} periods.")

        # Initialize arrays
        positions = np.zeros(n)
        returns = np.zeros(n)
        equity = np.zeros(n)
        equity[0] = self.initial_capital
        propensity_scores = np.zeros(n)

        # Prepare data
        X = data.features.values
        treatment = data.signal.values
        outcomes = data.returns.values

        # Initialize propensity model
        ps_model = PropensityScoreModel(method=self.propensity_method)

        # Track trades
        trades_list = []
        current_position = 0

        for t in range(warmup_period, n):
            # Refit propensity model periodically
            if t == warmup_period or (t - warmup_period) % refit_frequency == 0:
                # Use data up to current point (no lookahead)
                X_train = X[:t]
                T_train = treatment[:t]

                try:
                    ps_model.fit(X_train, T_train)
                except Exception as e:
                    logger.warning(f"Could not fit propensity model at t={t}: {e}")
                    continue

            # Predict propensity score for current period
            try:
                ps_current = ps_model.predict_proba(X[t:t+1])[0]
            except Exception:
                ps_current = 0.5

            propensity_scores[t] = ps_current

            # Get signal for current period
            signal = treatment[t]

            # Determine position
            if self.position_sizing == "propensity_weighted":
                # Weight position by inverse propensity (de-emphasize likely signals)
                if signal == 1:
                    position_weight = 1.0 / max(ps_current, 0.1)  # Prevent extreme weights
                else:
                    position_weight = 0
                position = signal * min(position_weight, 3.0) / 3.0  # Normalize
            else:
                # Equal sizing
                position = signal

            positions[t] = position

            # Calculate return
            if t > warmup_period:
                position_return = positions[t-1] * outcomes[t-1]

                # Transaction costs
                position_change = abs(positions[t] - positions[t-1])
                cost = position_change * self.transaction_cost

                returns[t] = position_return - cost
                equity[t] = equity[t-1] * (1 + returns[t])

                # Track trades
                if position != current_position:
                    trades_list.append({
                        'date': data.prices.index[t],
                        'signal': signal,
                        'position_before': current_position,
                        'position_after': position,
                        'propensity_score': ps_current,
                        'return': returns[t],
                    })
                    current_position = position
            else:
                equity[t] = self.initial_capital

        # Create Series with proper index
        dates = data.prices.index
        equity_series = pd.Series(equity, index=dates, name='equity')
        returns_series = pd.Series(returns, index=dates, name='returns')
        positions_series = pd.Series(positions, index=dates, name='positions')
        ps_series = pd.Series(propensity_scores, index=dates, name='propensity_score')

        trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()

        # Calculate performance metrics
        metrics = self._calculate_metrics(
            equity_series[warmup_period:],
            returns_series[warmup_period:],
            positions_series[warmup_period:],
        )

        # Estimate causal effect
        causal_estimate = None
        if self.use_propensity_weighting:
            causal_estimate = self._estimate_causal_effect(
                outcomes[warmup_period:],
                treatment[warmup_period:],
                propensity_scores[warmup_period:],
                X[warmup_period:],
            )

            if causal_estimate is not None:
                metrics.causal_ate = causal_estimate.estimate
                metrics.ate_confidence_interval = causal_estimate.confidence_interval
                metrics.ate_p_value = causal_estimate.p_value

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_series,
            returns=returns_series,
            positions=positions_series,
            trades=trades_df,
            causal_estimate=causal_estimate,
            propensity_scores=ps_series,
        )

    def _calculate_metrics(
        self,
        equity: pd.Series,
        returns: pd.Series,
        positions: pd.Series,
    ) -> PerformanceMetrics:
        """Calculate performance metrics."""
        # Filter valid returns
        valid_returns = returns[returns != 0]

        # Basic metrics
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        n_periods = len(equity)
        annualized_return = (1 + total_return) ** (252 / max(n_periods, 1)) - 1

        volatility = returns.std() * np.sqrt(252)
        sharpe = (annualized_return - 0.02) / volatility if volatility > 0 else 0  # Assume 2% risk-free

        # Drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())

        # Win rate
        winning_trades = (valid_returns > 0).sum()
        total_trades = len(valid_returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Profit factor
        gross_profit = valid_returns[valid_returns > 0].sum()
        gross_loss = abs(valid_returns[valid_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # Average return per trade
        n_trades = (positions.diff().abs() > 0).sum()
        avg_return = valid_returns.mean() if len(valid_returns) > 0 else 0

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor if profit_factor != np.inf else 999.99,
            n_trades=int(n_trades),
            avg_return_per_trade=avg_return,
        )

    def _estimate_causal_effect(
        self,
        outcomes: np.ndarray,
        treatment: np.ndarray,
        propensity_scores: np.ndarray,
        X: np.ndarray,
    ) -> Optional[CausalEstimate]:
        """Estimate causal effect using specified method."""
        try:
            if self.causal_estimator == "ipw":
                estimator = IPWEstimator()
                return estimator.estimate_ate(outcomes, treatment, propensity_scores)
            elif self.causal_estimator == "doubly_robust":
                estimator = DoublyRobustEstimator()
                return estimator.estimate_ate(outcomes, treatment, propensity_scores, X)
            else:
                logger.warning(f"Unknown causal estimator: {self.causal_estimator}")
                return None
        except Exception as e:
            logger.warning(f"Could not estimate causal effect: {e}")
            return None


def compare_strategies(
    data: MarketData,
    strategies: Dict[str, callable],
    warmup_period: int = 50,
) -> pd.DataFrame:
    """
    Compare multiple trading strategies.

    Args:
        data: MarketData object
        strategies: Dictionary mapping strategy names to signal functions
        warmup_period: Warmup period for propensity estimation

    Returns:
        DataFrame with comparison metrics
    """
    results = {}

    for name, signal_func in strategies.items():
        logger.info(f"Running backtest for: {name}")

        # Generate signal for this strategy
        modified_data = MarketData(
            prices=data.prices,
            features=data.features,
            returns=data.returns,
            signal=signal_func(data.prices, data.features),
            metadata={**data.metadata, 'strategy': name}
        )

        # Run backtest
        backtester = PropensityBacktester(
            use_propensity_weighting=True,
            causal_estimator="doubly_robust",
        )

        result = backtester.run(modified_data, warmup_period=warmup_period)
        results[name] = result.metrics.to_dict()

    return pd.DataFrame(results).T


def rolling_causal_estimate(
    data: MarketData,
    window_size: int = 100,
    step_size: int = 20,
) -> pd.DataFrame:
    """
    Compute rolling causal estimates over time.

    Useful for detecting regime changes in the causal effect.

    Args:
        data: MarketData object
        window_size: Size of rolling window
        step_size: Steps between estimates

    Returns:
        DataFrame with rolling estimates
    """
    n = len(data.prices)
    results = []

    ps_model = PropensityScoreModel(method='gradient_boosting')
    estimator = IPWEstimator()

    X = data.features.values
    treatment = data.signal.values
    outcomes = data.returns.values

    for start in range(0, n - window_size, step_size):
        end = start + window_size

        # Fit propensity model on window
        ps_model.fit(X[start:end], treatment[start:end])
        ps = ps_model.predict_proba(X[start:end])

        # Estimate causal effect
        try:
            estimate = estimator.estimate_ate(
                outcomes[start:end],
                treatment[start:end],
                ps
            )

            results.append({
                'start_date': data.prices.index[start],
                'end_date': data.prices.index[end-1],
                'ate': estimate.estimate,
                'std_error': estimate.std_error,
                'ci_lower': estimate.confidence_interval[0],
                'ci_upper': estimate.confidence_interval[1],
                'p_value': estimate.p_value,
            })
        except Exception as e:
            logger.warning(f"Could not estimate for window {start}-{end}: {e}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    from .data_loader import generate_synthetic_data

    # Generate test data
    print("Generating synthetic data...")
    data = generate_synthetic_data(
        n_samples=500,
        true_treatment_effect=0.005,
        confounding_strength=0.3,
    )

    print(f"\nData summary:")
    print(f"  Samples: {data.metadata['n_samples']}")
    print(f"  Treatment rate: {data.metadata['treatment_rate']:.2%}")
    print(f"  True ATE: {data.metadata['true_treatment_effect']}")

    # Run backtest
    print("\nRunning backtest...")
    backtester = PropensityBacktester(
        initial_capital=100000,
        transaction_cost=0.001,
        use_propensity_weighting=True,
        causal_estimator="doubly_robust",
    )

    result = backtester.run(data, warmup_period=50)

    print("\n" + str(result.metrics))

    # Compare with naive estimate
    print("\n=== Naive vs Causal Analysis ===")
    treated_returns = data.returns[data.signal == 1].mean()
    control_returns = data.returns[data.signal == 0].mean()
    naive_effect = treated_returns - control_returns

    print(f"Naive effect estimate: {naive_effect:.4%}")
    print(f"True treatment effect: {data.metadata['true_treatment_effect']:.4%}")
    if result.causal_estimate:
        print(f"Causal ATE estimate:   {result.causal_estimate.estimate:.4%}")
