//! Data Loading Utilities
//!
//! This module provides data loading and synthetic data generation
//! for testing propensity score methods.

use chrono::{DateTime, NaiveDate, Utc};
use rand::prelude::*;
use rand_distr::{Exp, LogNormal, Normal};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Container for market data
#[derive(Debug, Clone)]
pub struct MarketData {
    /// OHLCV price data
    pub prices: Vec<PriceBar>,
    /// Feature vectors for propensity estimation
    pub features: Vec<Vec<f64>>,
    /// Forward returns (outcome variable)
    pub outcomes: Vec<f64>,
    /// Trading signal (treatment variable)
    pub treatment: Vec<u8>,
    /// Metadata about the dataset
    pub metadata: DatasetMetadata,
}

/// Single price bar (OHLCV)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceBar {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Dataset metadata
#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    pub n_samples: usize,
    pub n_treated: usize,
    pub treatment_rate: f64,
    pub true_treatment_effect: Option<f64>,
    pub confounding_strength: Option<f64>,
    pub data_source: String,
}

/// Generate synthetic trading data with known causal structure
///
/// This is useful for testing and validating propensity score methods
/// because we know the true treatment effect.
///
/// # Arguments
/// * `n_samples` - Number of trading days
/// * `true_treatment_effect` - True causal effect of the signal on returns
/// * `confounding_strength` - How much confounders affect both signal and outcome
/// * `seed` - Random seed for reproducibility
///
/// # Example
/// ```rust
/// use propensity_score_trading::generate_synthetic_data;
///
/// let data = generate_synthetic_data(500, 0.005, 0.3, 42);
/// assert_eq!(data.features.len(), 500);
/// ```
pub fn generate_synthetic_data(
    n_samples: usize,
    true_treatment_effect: f64,
    confounding_strength: f64,
    seed: u64,
) -> MarketData {
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate confounders (market conditions)
    let normal = Normal::new(0.0, 1.0).unwrap();
    let exp_dist = Exp::new(50.0).unwrap(); // For volatility
    let lognormal = LogNormal::new(10.0, 0.5).unwrap(); // For volume

    let mut volatility = Vec::with_capacity(n_samples);
    let mut volume = Vec::with_capacity(n_samples);
    let mut market_regime = Vec::with_capacity(n_samples);
    let mut momentum = Vec::with_capacity(n_samples);

    let mut cumulative_momentum = 0.0;

    for _ in 0..n_samples {
        // Volatility: exponential distribution, scaled
        let vol: f64 = rng.sample(exp_dist);
        let vol = vol.max(0.01).min(0.1);
        volatility.push(vol);

        // Volume: log-normal
        let vol_sample = rng.sample(lognormal);
        volume.push(vol_sample);

        // Market regime: binary (0=bear, 1=bull)
        let regime = if rng.gen::<f64>() < 0.4 { 1u8 } else { 0u8 };
        market_regime.push(regime);

        // Momentum: cumulative random walk
        cumulative_momentum += rng.sample(normal) * 0.01;
        momentum.push(cumulative_momentum);
    }

    // Generate price series
    let initial_price = 100.0;
    let mut prices = Vec::with_capacity(n_samples);
    let mut base_returns = Vec::with_capacity(n_samples);
    let mut cumulative_return = 0.0;

    for i in 0..n_samples {
        // Base return depends on confounders
        let noise = rng.sample(normal) * volatility[i];
        let base_ret = 0.0005 * market_regime[i] as f64 // Bull markets have higher returns
            - 0.5 * volatility[i]  // High vol → lower returns
            + 0.01 * (momentum[i] as f64).signum()  // Trend following
            + noise;

        base_returns.push(base_ret);
        cumulative_return += base_ret;

        let close = initial_price * (1.0 + cumulative_return);
        let high = close * (1.0 + rng.gen::<f64>().abs() * 0.01);
        let low = close * (1.0 - rng.gen::<f64>().abs() * 0.01);
        let open = close * (1.0 + rng.sample(normal) * 0.005);

        prices.push(PriceBar {
            timestamp: i as i64 * 86400, // Daily timestamps
            open,
            high,
            low,
            close,
            volume: volume[i],
        });
    }

    // Generate trading signal based on confounders (creates confounding)
    let mut treatment = Vec::with_capacity(n_samples);
    let mut signal_propensity = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        // Signal is more likely to fire in certain market conditions
        let z = -2.0
            + confounding_strength * 10.0 * (volatility[i] - 0.02)  // Low vol
            + confounding_strength * 2.0 * market_regime[i] as f64  // Bull markets
            + confounding_strength * 5.0 * momentum[i].clamp(-0.5, 0.5);  // Positive momentum

        let propensity = sigmoid(z);
        signal_propensity.push(propensity);

        // Also consider 5-day momentum for observable signal criterion
        let momentum_5d = if i >= 5 {
            (prices[i].close - prices[i - 5].close) / prices[i - 5].close
        } else {
            0.0
        };

        let signal_fires = momentum_5d > 0.02 || rng.gen::<f64>() < propensity;
        treatment.push(if signal_fires { 1 } else { 0 });
    }

    // Forward returns (outcome)
    // TRUE DGP: return = base_effect + treatment_effect * signal + noise
    let mut outcomes = Vec::with_capacity(n_samples);
    for i in 0..n_samples - 1 {
        let forward_return =
            base_returns[i + 1] + true_treatment_effect * treatment[i] as f64;
        outcomes.push(forward_return);
    }
    outcomes.push(0.0); // Last observation has no forward return

    // Build feature matrix
    let mut features = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut feature_vec = Vec::with_capacity(7);

        // Volatility features
        feature_vec.push(volatility[i]);

        // Volume features (log)
        feature_vec.push(volume[i].ln());

        // Regime
        feature_vec.push(market_regime[i] as f64);

        // Momentum
        feature_vec.push(momentum[i]);

        // 5-day momentum
        let momentum_5d = if i >= 5 {
            (prices[i].close - prices[i - 5].close) / prices[i - 5].close
        } else {
            0.0
        };
        feature_vec.push(momentum_5d);

        // Price/SMA ratio (use rolling 20-day mean)
        let sma_20 = if i >= 20 {
            prices[i - 19..=i]
                .iter()
                .map(|p| p.close)
                .sum::<f64>()
                / 20.0
        } else {
            prices[i].close
        };
        feature_vec.push(prices[i].close / sma_20);

        // Volume/SMA ratio
        let vol_sma_20 = if i >= 20 {
            prices[i - 19..=i]
                .iter()
                .map(|p| p.volume)
                .sum::<f64>()
                / 20.0
        } else {
            volume[i]
        };
        feature_vec.push(volume[i] / vol_sma_20);

        features.push(feature_vec);
    }

    let n_treated = treatment.iter().filter(|&&t| t == 1).count();

    MarketData {
        prices,
        features,
        outcomes,
        treatment,
        metadata: DatasetMetadata {
            n_samples,
            n_treated,
            treatment_rate: n_treated as f64 / n_samples as f64,
            true_treatment_effect: Some(true_treatment_effect),
            confounding_strength: Some(confounding_strength),
            data_source: "synthetic".to_string(),
        },
    }
}

/// Trait for data loaders
pub trait DataLoader {
    fn load(&self) -> Result<MarketData, DataLoaderError>;
}

/// Errors that can occur during data loading
#[derive(Debug, thiserror::Error)]
pub enum DataLoaderError {
    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Invalid data: {0}")]
    InvalidData(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("CSV error: {0}")]
    CsvError(#[from] csv::Error),
}

/// CSV Data Loader
///
/// Loads market data from a CSV file.
pub struct CsvDataLoader {
    file_path: String,
    signal_column: Option<String>,
}

impl CsvDataLoader {
    pub fn new(file_path: &str) -> Self {
        Self {
            file_path: file_path.to_string(),
            signal_column: None,
        }
    }

    pub fn with_signal_column(mut self, column: &str) -> Self {
        self.signal_column = Some(column.to_string());
        self
    }
}

impl DataLoader for CsvDataLoader {
    fn load(&self) -> Result<MarketData, DataLoaderError> {
        let mut reader = csv::Reader::from_path(&self.file_path)?;

        let mut prices = Vec::new();
        let mut features = Vec::new();
        let mut outcomes = Vec::new();
        let mut treatment = Vec::new();

        let headers = reader.headers()?.clone();

        for result in reader.records() {
            let record = result.map_err(|e| DataLoaderError::ParseError(e.to_string()))?;

            // Parse OHLCV
            let open: f64 = record
                .get(headers.iter().position(|h| h == "open").unwrap_or(1))
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0);
            let high: f64 = record
                .get(headers.iter().position(|h| h == "high").unwrap_or(2))
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0);
            let low: f64 = record
                .get(headers.iter().position(|h| h == "low").unwrap_or(3))
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0);
            let close: f64 = record
                .get(headers.iter().position(|h| h == "close").unwrap_or(4))
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0);
            let volume: f64 = record
                .get(headers.iter().position(|h| h == "volume").unwrap_or(5))
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0);

            prices.push(PriceBar {
                timestamp: prices.len() as i64 * 86400,
                open,
                high,
                low,
                close,
                volume,
            });

            // Basic features
            features.push(vec![
                (high - low) / close,  // Range
                volume.ln(),           // Log volume
                0.0,                   // Placeholder for momentum
            ]);

            // Treatment (if column specified)
            if let Some(ref col) = self.signal_column {
                if let Some(pos) = headers.iter().position(|h| h == col) {
                    let sig: u8 = record
                        .get(pos)
                        .unwrap_or("0")
                        .parse()
                        .unwrap_or(0);
                    treatment.push(sig);
                }
            }
        }

        // Calculate returns and momentum features
        for i in 1..prices.len() {
            let ret = (prices[i].close - prices[i - 1].close) / prices[i - 1].close;
            outcomes.push(ret);

            // Update momentum feature
            if i >= 5 {
                let mom_5d = (prices[i].close - prices[i - 5].close) / prices[i - 5].close;
                features[i][2] = mom_5d;
            }
        }
        outcomes.push(0.0);

        // Generate default treatment if not in file
        if treatment.is_empty() {
            for i in 0..prices.len() {
                let momentum = features[i][2];
                treatment.push(if momentum > 0.02 { 1 } else { 0 });
            }
        }

        let n_treated = treatment.iter().filter(|&&t| t == 1).count();
        let n_samples = outcomes.len();

        Ok(MarketData {
            prices,
            features,
            outcomes,
            treatment,
            metadata: DatasetMetadata {
                n_samples,
                n_treated,
                treatment_rate: n_treated as f64 / n_samples as f64,
                true_treatment_effect: None,
                confounding_strength: None,
                data_source: format!("csv:{}", self.file_path),
            },
        })
    }
}

/// Compute trading features from price data
pub fn compute_features(prices: &[PriceBar]) -> Vec<Vec<f64>> {
    let n = prices.len();
    let mut features = Vec::with_capacity(n);

    for i in 0..n {
        let mut feature_vec = Vec::new();

        // Volatility (using ATR proxy)
        let range_pct = (prices[i].high - prices[i].low) / prices[i].close;
        feature_vec.push(range_pct);

        // Log volume
        feature_vec.push(prices[i].volume.ln());

        // Price relative to 20-day SMA
        let sma_20 = if i >= 20 {
            prices[i - 19..=i].iter().map(|p| p.close).sum::<f64>() / 20.0
        } else if i > 0 {
            prices[0..=i].iter().map(|p| p.close).sum::<f64>() / (i + 1) as f64
        } else {
            prices[i].close
        };
        feature_vec.push(prices[i].close / sma_20);

        // 5-day momentum
        let momentum_5d = if i >= 5 {
            (prices[i].close - prices[i - 5].close) / prices[i - 5].close
        } else {
            0.0
        };
        feature_vec.push(momentum_5d);

        // 10-day momentum
        let momentum_10d = if i >= 10 {
            (prices[i].close - prices[i - 10].close) / prices[i - 10].close
        } else {
            0.0
        };
        feature_vec.push(momentum_10d);

        // 20-day volatility (standard deviation of returns)
        let vol_20d = if i >= 20 {
            let returns: Vec<f64> = (1..=20)
                .map(|j| {
                    (prices[i - j + 1].close - prices[i - j].close) / prices[i - j].close
                })
                .collect();
            let mean = returns.iter().sum::<f64>() / 20.0;
            let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / 20.0;
            variance.sqrt()
        } else {
            0.02 // Default volatility
        };
        feature_vec.push(vol_20d);

        features.push(feature_vec);
    }

    features
}

/// Create a momentum-based trading signal
pub fn create_momentum_signal(prices: &[PriceBar], lookback: usize, threshold: f64) -> Vec<u8> {
    let n = prices.len();
    let mut signal = vec![0u8; n];

    for i in lookback..n {
        let momentum = (prices[i].close - prices[i - lookback].close) / prices[i - lookback].close;
        if momentum > threshold {
            signal[i] = 1;
        }
    }

    signal
}

/// Create a mean-reversion trading signal
pub fn create_mean_reversion_signal(
    prices: &[PriceBar],
    lookback: usize,
    threshold: f64,
) -> Vec<u8> {
    let n = prices.len();
    let mut signal = vec![0u8; n];

    for i in lookback..n {
        let sma = prices[i - lookback + 1..=i]
            .iter()
            .map(|p| p.close)
            .sum::<f64>()
            / lookback as f64;

        let returns: Vec<f64> = (1..lookback)
            .map(|j| {
                (prices[i - j + 1].close - prices[i - j].close) / prices[i - j].close
            })
            .collect();
        let std = {
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                / returns.len() as f64;
            variance.sqrt()
        };

        let zscore = if std > 0.0 {
            (prices[i].close - sma) / (sma * std)
        } else {
            0.0
        };

        // Buy when price is significantly below mean
        if zscore < threshold {
            signal[i] = 1;
        }
    }

    signal
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_data() {
        let data = generate_synthetic_data(100, 0.01, 0.3, 42);

        assert_eq!(data.prices.len(), 100);
        assert_eq!(data.features.len(), 100);
        assert_eq!(data.outcomes.len(), 100);
        assert_eq!(data.treatment.len(), 100);

        // Treatment should be binary
        assert!(data.treatment.iter().all(|&t| t == 0 || t == 1));

        // Metadata should be correct
        assert_eq!(data.metadata.n_samples, 100);
        assert!(data.metadata.true_treatment_effect == Some(0.01));
    }

    #[test]
    fn test_compute_features() {
        let prices = vec![
            PriceBar {
                timestamp: 0,
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.5,
                volume: 1000.0,
            },
            PriceBar {
                timestamp: 86400,
                open: 100.5,
                high: 102.0,
                low: 100.0,
                close: 101.5,
                volume: 1100.0,
            },
        ];

        let features = compute_features(&prices);
        assert_eq!(features.len(), 2);
        assert!(!features[0].is_empty());
    }

    #[test]
    fn test_momentum_signal() {
        let data = generate_synthetic_data(100, 0.01, 0.3, 42);
        let signal = create_momentum_signal(&data.prices, 5, 0.02);

        assert_eq!(signal.len(), 100);
        assert!(signal.iter().all(|&s| s == 0 || s == 1));
    }
}
