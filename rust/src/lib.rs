//! # Propensity Score Methods for Trading
//!
//! This crate provides tools for applying propensity score methods
//! to causal inference in trading strategies.
//!
//! ## Overview
//!
//! Propensity scores help estimate the causal effect of trading signals
//! by controlling for confounding market conditions. This is crucial for
//! understanding whether a trading signal truly causes better returns or
//! is merely correlated due to market conditions.
//!
//! ## Example
//!
//! ```rust
//! use propensity_score_trading::{
//!     PropensityScoreModel, PropensityMethod, IPWEstimator,
//!     generate_synthetic_data,
//! };
//!
//! // Generate synthetic data
//! let data = generate_synthetic_data(500, 0.005, 0.3, 42);
//!
//! // Fit propensity model
//! let mut model = PropensityScoreModel::new(PropensityMethod::LogisticRegression);
//! model.fit(&data.features, &data.treatment).unwrap();
//!
//! // Get propensity scores
//! let ps = model.predict_proba(&data.features).unwrap();
//!
//! // Estimate causal effect
//! let estimator = IPWEstimator::new(true);
//! let estimate = estimator.estimate_ate(&data.outcomes, &data.treatment, &ps).unwrap();
//!
//! println!("Estimated ATE: {:.4}", estimate.estimate);
//! ```

pub mod model;
pub mod data_loader;
pub mod backtest;
pub mod utils;

pub use model::{
    PropensityScoreModel,
    PropensityMethod,
    PropensityMatcher,
    MatchingMethod,
    IPWEstimator,
    DoublyRobustEstimator,
    CausalEstimate,
    MatchResult,
};

pub use data_loader::{
    MarketData,
    generate_synthetic_data,
    DataLoader,
};

pub use backtest::{
    PropensityBacktester,
    BacktestResult,
    PerformanceMetrics,
    compare_naive_vs_causal,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_data_generation() {
        let data = generate_synthetic_data(100, 0.01, 0.3, 42);
        assert_eq!(data.features.len(), 100);
        assert_eq!(data.treatment.len(), 100);
        assert_eq!(data.outcomes.len(), 100);
    }

    #[test]
    fn test_propensity_model() {
        let data = generate_synthetic_data(200, 0.01, 0.3, 42);

        let mut model = PropensityScoreModel::new(PropensityMethod::LogisticRegression);
        model.fit(&data.features, &data.treatment).unwrap();

        let ps = model.predict_proba(&data.features).unwrap();

        // Propensity scores should be between 0 and 1
        assert!(ps.iter().all(|&p| p > 0.0 && p < 1.0));
    }

    #[test]
    fn test_ipw_estimator() {
        let data = generate_synthetic_data(500, 0.02, 0.2, 42);

        let mut model = PropensityScoreModel::new(PropensityMethod::LogisticRegression);
        model.fit(&data.features, &data.treatment).unwrap();
        let ps = model.predict_proba(&data.features).unwrap();

        let estimator = IPWEstimator::new(true);
        let estimate = estimator.estimate_ate(&data.outcomes, &data.treatment, &ps).unwrap();

        // The estimate should be reasonably close to the true effect (0.02)
        // Allow for statistical variation
        assert!(estimate.estimate > -0.05 && estimate.estimate < 0.10);
    }
}
