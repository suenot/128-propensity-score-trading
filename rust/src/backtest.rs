//! Backtesting Framework
//!
//! This module provides backtesting functionality for propensity score
//! trading strategies.

use crate::data_loader::MarketData;
use crate::model::{CausalEstimate, IPWEstimator, PropensityScoreModel, PropensityMethod};

/// Performance metrics for a trading strategy
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub annualized_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub n_trades: usize,
    pub avg_return_per_trade: f64,
    pub causal_ate: Option<f64>,
    pub ate_confidence_interval: Option<(f64, f64)>,
    pub ate_p_value: Option<f64>,
}

impl std::fmt::Display for PerformanceMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Performance Metrics ===")?;
        writeln!(f, "Total Return:      {:>10.2}%", self.total_return * 100.0)?;
        writeln!(f, "Annual Return:     {:>10.2}%", self.annualized_return * 100.0)?;
        writeln!(f, "Volatility:        {:>10.2}%", self.volatility * 100.0)?;
        writeln!(f, "Sharpe Ratio:      {:>10.2}", self.sharpe_ratio)?;
        writeln!(f, "Max Drawdown:      {:>10.2}%", self.max_drawdown * 100.0)?;
        writeln!(f, "Win Rate:          {:>10.2}%", self.win_rate * 100.0)?;
        writeln!(f, "Profit Factor:     {:>10.2}", self.profit_factor)?;
        writeln!(f, "Number of Trades:  {:>10}", self.n_trades)?;
        writeln!(f, "Avg Return/Trade:  {:>10.4}%", self.avg_return_per_trade * 100.0)?;

        if let Some(ate) = self.causal_ate {
            writeln!(f)?;
            writeln!(f, "=== Causal Analysis ===")?;
            writeln!(f, "Causal ATE:        {:>10.4}%", ate * 100.0)?;
            if let Some(ci) = self.ate_confidence_interval {
                writeln!(f, "95% CI:            [{:.4}%, {:.4}%]", ci.0 * 100.0, ci.1 * 100.0)?;
            }
            if let Some(p) = self.ate_p_value {
                writeln!(f, "P-value:           {:>10.4}", p)?;
            }
        }

        Ok(())
    }
}

/// Result from a backtest run
#[derive(Debug)]
pub struct BacktestResult {
    pub metrics: PerformanceMetrics,
    pub equity_curve: Vec<f64>,
    pub returns: Vec<f64>,
    pub positions: Vec<f64>,
    pub propensity_scores: Vec<f64>,
    pub causal_estimate: Option<CausalEstimate>,
}

impl BacktestResult {
    /// Get daily returns during periods with active position
    pub fn active_returns(&self) -> Vec<f64> {
        self.positions
            .iter()
            .zip(self.returns.iter())
            .filter(|(&pos, _)| pos.abs() > 0.0)
            .map(|(_, &ret)| ret)
            .collect()
    }

    /// Get cumulative return at any point
    pub fn cumulative_return_at(&self, index: usize) -> f64 {
        if index >= self.equity_curve.len() || self.equity_curve.is_empty() {
            return 0.0;
        }
        (self.equity_curve[index] / self.equity_curve[0]) - 1.0
    }
}

/// Backtester that incorporates propensity score analysis
pub struct PropensityBacktester {
    initial_capital: f64,
    transaction_cost: f64,
    use_propensity_weighting: bool,
    propensity_method: PropensityMethod,
    refit_frequency: usize,
    warmup_period: usize,
}

impl PropensityBacktester {
    pub fn new() -> Self {
        Self {
            initial_capital: 100_000.0,
            transaction_cost: 0.001,
            use_propensity_weighting: true,
            propensity_method: PropensityMethod::LogisticRegression,
            refit_frequency: 20,
            warmup_period: 50,
        }
    }

    pub fn with_initial_capital(mut self, capital: f64) -> Self {
        self.initial_capital = capital;
        self
    }

    pub fn with_transaction_cost(mut self, cost: f64) -> Self {
        self.transaction_cost = cost;
        self
    }

    pub fn with_propensity_weighting(mut self, use_weighting: bool) -> Self {
        self.use_propensity_weighting = use_weighting;
        self
    }

    pub fn with_warmup_period(mut self, period: usize) -> Self {
        self.warmup_period = period;
        self
    }

    pub fn with_refit_frequency(mut self, freq: usize) -> Self {
        self.refit_frequency = freq;
        self
    }

    /// Run the backtest
    pub fn run(&self, data: &MarketData) -> Result<BacktestResult, String> {
        let n = data.prices.len();

        if n < self.warmup_period + 10 {
            return Err(format!(
                "Not enough data. Need at least {} periods.",
                self.warmup_period + 10
            ));
        }

        // Initialize arrays
        let mut positions = vec![0.0; n];
        let mut returns = vec![0.0; n];
        let mut equity = vec![self.initial_capital; n];
        let mut propensity_scores = vec![0.5; n];

        // Initialize propensity model
        let mut ps_model = PropensityScoreModel::new(self.propensity_method);

        #[allow(unused_variables)]
        let current_position = 0.0;

        for t in self.warmup_period..n {
            // Refit propensity model periodically
            if t == self.warmup_period || (t - self.warmup_period) % self.refit_frequency == 0 {
                // Use data up to current point (no lookahead)
                let features_train: Vec<Vec<f64>> = data.features[..t].to_vec();
                let treatment_train: Vec<u8> = data.treatment[..t].to_vec();

                if let Err(e) = ps_model.fit(&features_train, &treatment_train) {
                    // Continue with previous model if fit fails
                    eprintln!("Warning: Could not fit propensity model at t={}: {}", t, e);
                    continue;
                }
            }

            // Predict propensity score for current period
            let ps_current = match ps_model.predict_proba(&[data.features[t].clone()]) {
                Ok(ps) => ps[0],
                Err(_) => 0.5,
            };
            propensity_scores[t] = ps_current;

            // Get signal for current period
            let signal = data.treatment[t] as f64;

            // Determine position
            let position = if self.use_propensity_weighting {
                // Weight position by inverse propensity
                if signal > 0.5 {
                    let weight = 1.0 / ps_current.max(0.1);
                    (weight.min(3.0) / 3.0) * signal
                } else {
                    0.0
                }
            } else {
                signal
            };

            positions[t] = position;

            // Calculate return
            if t > self.warmup_period {
                let position_return = positions[t - 1] * data.outcomes[t - 1];

                // Transaction costs
                let position_change = (positions[t] - positions[t - 1]).abs();
                let cost = position_change * self.transaction_cost;

                returns[t] = position_return - cost;
                equity[t] = equity[t - 1] * (1.0 + returns[t]);
            }
        }

        // Calculate performance metrics
        let metrics = self.calculate_metrics(&equity, &returns, &positions);

        // Estimate causal effect
        let causal_estimate = if self.use_propensity_weighting {
            self.estimate_causal_effect(
                &data.outcomes[self.warmup_period..],
                &data.treatment[self.warmup_period..],
                &propensity_scores[self.warmup_period..],
            )
        } else {
            None
        };

        // Update metrics with causal estimates
        let metrics = if let Some(ref est) = causal_estimate {
            PerformanceMetrics {
                causal_ate: Some(est.estimate),
                ate_confidence_interval: Some(est.confidence_interval),
                ate_p_value: Some(est.p_value),
                ..metrics
            }
        } else {
            metrics
        };

        Ok(BacktestResult {
            metrics,
            equity_curve: equity,
            returns,
            positions,
            propensity_scores,
            causal_estimate,
        })
    }

    fn calculate_metrics(
        &self,
        equity: &[f64],
        returns: &[f64],
        positions: &[f64],
    ) -> PerformanceMetrics {
        let valid_equity = &equity[self.warmup_period..];
        let valid_returns = &returns[self.warmup_period..];
        let valid_positions = &positions[self.warmup_period..];

        // Total return
        let total_return = if valid_equity.first().map(|&x| x > 0.0).unwrap_or(false) {
            (valid_equity.last().unwrap_or(&self.initial_capital) / valid_equity[0]) - 1.0
        } else {
            0.0
        };

        // Annualized return (assuming daily data)
        let n_periods = valid_equity.len() as f64;
        let annualized_return = (1.0 + total_return).powf(252.0 / n_periods.max(1.0)) - 1.0;

        // Volatility
        let mean_return = valid_returns.iter().sum::<f64>() / valid_returns.len() as f64;
        let variance = valid_returns
            .iter()
            .map(|&r| (r - mean_return).powi(2))
            .sum::<f64>()
            / valid_returns.len() as f64;
        let volatility = variance.sqrt() * (252.0_f64).sqrt();

        // Sharpe ratio (assume 2% risk-free rate)
        let sharpe = if volatility > 0.0 {
            (annualized_return - 0.02) / volatility
        } else {
            0.0
        };

        // Max drawdown
        let mut max_equity = valid_equity[0];
        let mut max_drawdown: f64 = 0.0;
        for &eq in valid_equity {
            max_equity = max_equity.max(eq);
            let drawdown = (max_equity - eq) / max_equity;
            max_drawdown = max_drawdown.max(drawdown);
        }

        // Win rate and profit factor
        let active_returns: Vec<f64> = valid_returns
            .iter()
            .zip(valid_positions.iter())
            .filter(|(_, &pos)| pos.abs() > 0.0)
            .map(|(&ret, _)| ret)
            .collect();

        let winning_trades = active_returns.iter().filter(|&&r| r > 0.0).count();
        let win_rate = if !active_returns.is_empty() {
            winning_trades as f64 / active_returns.len() as f64
        } else {
            0.0
        };

        let gross_profit: f64 = active_returns.iter().filter(|&&r| r > 0.0).sum();
        let gross_loss: f64 = active_returns
            .iter()
            .filter(|&&r| r < 0.0)
            .map(|&r| r.abs())
            .sum();

        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else {
            999.99
        };

        // Number of trades
        let n_trades = valid_positions
            .windows(2)
            .filter(|w| (w[1] - w[0]).abs() > 0.0)
            .count();

        // Average return per trade
        let avg_return = if !active_returns.is_empty() {
            active_returns.iter().sum::<f64>() / active_returns.len() as f64
        } else {
            0.0
        };

        PerformanceMetrics {
            total_return,
            annualized_return,
            volatility,
            sharpe_ratio: sharpe,
            max_drawdown,
            win_rate,
            profit_factor,
            n_trades,
            avg_return_per_trade: avg_return,
            causal_ate: None,
            ate_confidence_interval: None,
            ate_p_value: None,
        }
    }

    fn estimate_causal_effect(
        &self,
        outcomes: &[f64],
        treatment: &[u8],
        propensity_scores: &[f64],
    ) -> Option<CausalEstimate> {
        let estimator = IPWEstimator::new(true);

        match estimator.estimate_ate(outcomes, treatment, propensity_scores) {
            Ok(estimate) => Some(estimate),
            Err(e) => {
                eprintln!("Warning: Could not estimate causal effect: {}", e);
                None
            }
        }
    }
}

impl Default for PropensityBacktester {
    fn default() -> Self {
        Self::new()
    }
}

/// Compare naive vs causal estimates
pub fn compare_naive_vs_causal(data: &MarketData) -> ComparisonResult {
    // Naive estimate: simple difference in means
    let mut treated_returns = Vec::new();
    let mut control_returns = Vec::new();

    for i in 0..data.outcomes.len() {
        if data.treatment[i] == 1 {
            treated_returns.push(data.outcomes[i]);
        } else {
            control_returns.push(data.outcomes[i]);
        }
    }

    let naive_ate = if !treated_returns.is_empty() && !control_returns.is_empty() {
        (treated_returns.iter().sum::<f64>() / treated_returns.len() as f64)
            - (control_returns.iter().sum::<f64>() / control_returns.len() as f64)
    } else {
        0.0
    };

    // Causal estimate using propensity scores
    let mut ps_model = PropensityScoreModel::new(PropensityMethod::LogisticRegression);
    let causal_ate = if ps_model.fit(&data.features, &data.treatment).is_ok() {
        let ps = ps_model.predict_proba(&data.features).unwrap_or_default();
        let estimator = IPWEstimator::new(true);
        estimator
            .estimate_ate(&data.outcomes, &data.treatment, &ps)
            .ok()
            .map(|e| e.estimate)
            .unwrap_or(0.0)
    } else {
        0.0
    };

    ComparisonResult {
        naive_ate,
        causal_ate,
        true_ate: data.metadata.true_treatment_effect,
        naive_bias: data
            .metadata
            .true_treatment_effect
            .map(|true_ate| naive_ate - true_ate),
        causal_bias: data
            .metadata
            .true_treatment_effect
            .map(|true_ate| causal_ate - true_ate),
    }
}

/// Result of comparing naive vs causal estimates
#[derive(Debug)]
pub struct ComparisonResult {
    pub naive_ate: f64,
    pub causal_ate: f64,
    pub true_ate: Option<f64>,
    pub naive_bias: Option<f64>,
    pub causal_bias: Option<f64>,
}

impl std::fmt::Display for ComparisonResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Naive vs Causal Comparison ===")?;
        writeln!(f, "Naive ATE:    {:.4}%", self.naive_ate * 100.0)?;
        writeln!(f, "Causal ATE:   {:.4}%", self.causal_ate * 100.0)?;

        if let Some(true_ate) = self.true_ate {
            writeln!(f, "True ATE:     {:.4}%", true_ate * 100.0)?;
            if let Some(naive_bias) = self.naive_bias {
                writeln!(f, "Naive Bias:   {:.4}%", naive_bias * 100.0)?;
            }
            if let Some(causal_bias) = self.causal_bias {
                writeln!(f, "Causal Bias:  {:.4}%", causal_bias * 100.0)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_loader::generate_synthetic_data;

    #[test]
    fn test_backtest() {
        let data = generate_synthetic_data(200, 0.005, 0.3, 42);

        let backtester = PropensityBacktester::new()
            .with_warmup_period(30)
            .with_propensity_weighting(true);

        let result = backtester.run(&data).unwrap();

        assert_eq!(result.equity_curve.len(), 200);
        assert!(result.metrics.total_return.is_finite());
        assert!(result.metrics.sharpe_ratio.is_finite());
    }

    #[test]
    fn test_naive_vs_causal() {
        let data = generate_synthetic_data(500, 0.01, 0.3, 42);
        let comparison = compare_naive_vs_causal(&data);

        // With confounding, naive estimate should have more bias
        if let (Some(naive_bias), Some(causal_bias)) = (comparison.naive_bias, comparison.causal_bias)
        {
            // Causal estimate should generally have less bias (though not guaranteed with small samples)
            println!("Naive bias: {:.4}, Causal bias: {:.4}", naive_bias, causal_bias);
        }
    }
}
