//! Propensity Score Models
//!
//! This module provides implementations of propensity score estimation
//! and causal effect estimation methods.

use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during propensity score estimation
#[derive(Error, Debug)]
pub enum PropensityError {
    #[error("Model has not been fitted yet")]
    NotFitted,

    #[error("Invalid input dimensions: {0}")]
    InvalidDimensions(String),

    #[error("Treatment must be binary (0 or 1)")]
    NonBinaryTreatment,

    #[error("Numerical error: {0}")]
    NumericalError(String),

    #[error("Insufficient overlap between treatment groups")]
    InsufficientOverlap,
}

/// Methods for propensity score estimation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PropensityMethod {
    /// Logistic regression (default)
    LogisticRegression,
    /// Simple linear probability model (for comparison)
    LinearProbability,
}

/// Methods for propensity score matching
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatchingMethod {
    /// Nearest neighbor matching
    NearestNeighbor,
    /// Caliper matching (with distance threshold)
    Caliper,
    /// Stratification into bins
    Stratification,
}

/// Result of propensity score matching
#[derive(Debug, Clone)]
pub struct MatchResult {
    pub treated_indices: Vec<usize>,
    pub control_indices: Vec<usize>,
    pub propensity_scores: Vec<f64>,
    pub standardized_mean_differences: HashMap<String, f64>,
    pub balance_quality: f64,
}

/// Result of causal effect estimation
#[derive(Debug, Clone)]
pub struct CausalEstimate {
    pub estimate: f64,
    pub std_error: f64,
    pub confidence_interval: (f64, f64),
    pub p_value: f64,
    pub method: String,
    pub n_treated: usize,
    pub n_control: usize,
}

impl CausalEstimate {
    /// Check if the estimate is statistically significant at the given level
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }
}

impl std::fmt::Display for CausalEstimate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CausalEstimate(ATE={:.4}, SE={:.4}, 95% CI=[{:.4}, {:.4}], p={:.4})",
            self.estimate,
            self.std_error,
            self.confidence_interval.0,
            self.confidence_interval.1,
            self.p_value
        )
    }
}

/// Propensity Score Model
///
/// Estimates the probability of treatment given observed covariates.
pub struct PropensityScoreModel {
    method: PropensityMethod,
    coefficients: Option<Vec<f64>>,
    intercept: Option<f64>,
    feature_means: Option<Vec<f64>>,
    feature_stds: Option<Vec<f64>>,
    clip_bounds: (f64, f64),
    is_fitted: bool,
}

impl PropensityScoreModel {
    /// Create a new propensity score model
    pub fn new(method: PropensityMethod) -> Self {
        Self {
            method,
            coefficients: None,
            intercept: None,
            feature_means: None,
            feature_stds: None,
            clip_bounds: (0.01, 0.99),
            is_fitted: false,
        }
    }

    /// Set the clip bounds for propensity scores
    pub fn with_clip_bounds(mut self, lower: f64, upper: f64) -> Self {
        self.clip_bounds = (lower, upper);
        self
    }

    /// Fit the model to training data
    ///
    /// # Arguments
    /// * `features` - Feature vectors for each observation (n_samples x n_features)
    /// * `treatment` - Binary treatment indicator (0 or 1)
    pub fn fit(
        &mut self,
        features: &[Vec<f64>],
        treatment: &[u8],
    ) -> Result<(), PropensityError> {
        if features.is_empty() || treatment.is_empty() {
            return Err(PropensityError::InvalidDimensions(
                "Empty input".to_string(),
            ));
        }

        if features.len() != treatment.len() {
            return Err(PropensityError::InvalidDimensions(format!(
                "Features ({}) and treatment ({}) must have same length",
                features.len(),
                treatment.len()
            )));
        }

        // Check binary treatment
        if !treatment.iter().all(|&t| t == 0 || t == 1) {
            return Err(PropensityError::NonBinaryTreatment);
        }

        let n_features = features[0].len();
        let n_samples = features.len();

        // Standardize features
        let (scaled_features, means, stds) = standardize_features(features);
        self.feature_means = Some(means);
        self.feature_stds = Some(stds);

        // Fit model based on method
        match self.method {
            PropensityMethod::LogisticRegression => {
                self.fit_logistic(&scaled_features, treatment, n_features, n_samples)?;
            }
            PropensityMethod::LinearProbability => {
                self.fit_linear(&scaled_features, treatment, n_features)?;
            }
        }

        self.is_fitted = true;
        Ok(())
    }

    /// Fit logistic regression using gradient descent
    fn fit_logistic(
        &mut self,
        features: &[Vec<f64>],
        treatment: &[u8],
        n_features: usize,
        n_samples: usize,
    ) -> Result<(), PropensityError> {
        // Initialize coefficients
        let mut coef = vec![0.0; n_features];
        let mut intercept = 0.0;

        // Gradient descent parameters
        let learning_rate = 0.1;
        let max_iter = 1000;
        let tol = 1e-6;

        for _ in 0..max_iter {
            let mut grad_coef = vec![0.0; n_features];
            let mut grad_intercept = 0.0;

            for i in 0..n_samples {
                // Compute prediction
                let z: f64 = intercept
                    + features[i]
                        .iter()
                        .zip(coef.iter())
                        .map(|(x, c)| x * c)
                        .sum::<f64>();

                let p = sigmoid(z);
                let error = p - treatment[i] as f64;

                // Accumulate gradients
                for (j, &x) in features[i].iter().enumerate() {
                    grad_coef[j] += error * x;
                }
                grad_intercept += error;
            }

            // Update parameters
            let mut max_update = 0.0f64;
            for j in 0..n_features {
                let update = learning_rate * grad_coef[j] / n_samples as f64;
                coef[j] -= update;
                max_update = max_update.max(update.abs());
            }
            let intercept_update = learning_rate * grad_intercept / n_samples as f64;
            intercept -= intercept_update;
            max_update = max_update.max(intercept_update.abs());

            // Check convergence
            if max_update < tol {
                break;
            }
        }

        self.coefficients = Some(coef);
        self.intercept = Some(intercept);

        Ok(())
    }

    /// Fit linear probability model (OLS)
    fn fit_linear(
        &mut self,
        features: &[Vec<f64>],
        treatment: &[u8],
        n_features: usize,
    ) -> Result<(), PropensityError> {
        // Simple OLS: (X'X)^-1 X'y
        // Using gradient descent for simplicity
        let mut coef = vec![0.0; n_features];
        let mut intercept = 0.0;

        let learning_rate = 0.1;
        let max_iter = 1000;
        let n_samples = features.len();

        for _ in 0..max_iter {
            let mut grad_coef = vec![0.0; n_features];
            let mut grad_intercept = 0.0;

            for i in 0..n_samples {
                let pred: f64 = intercept
                    + features[i]
                        .iter()
                        .zip(coef.iter())
                        .map(|(x, c)| x * c)
                        .sum::<f64>();

                let error = pred - treatment[i] as f64;

                for (j, &x) in features[i].iter().enumerate() {
                    grad_coef[j] += 2.0 * error * x;
                }
                grad_intercept += 2.0 * error;
            }

            for j in 0..n_features {
                coef[j] -= learning_rate * grad_coef[j] / n_samples as f64;
            }
            intercept -= learning_rate * grad_intercept / n_samples as f64;
        }

        self.coefficients = Some(coef);
        self.intercept = Some(intercept);

        Ok(())
    }

    /// Predict propensity scores
    pub fn predict_proba(&self, features: &[Vec<f64>]) -> Result<Vec<f64>, PropensityError> {
        if !self.is_fitted {
            return Err(PropensityError::NotFitted);
        }

        let coef = self.coefficients.as_ref().unwrap();
        let intercept = self.intercept.unwrap();
        let means = self.feature_means.as_ref().unwrap();
        let stds = self.feature_stds.as_ref().unwrap();

        let mut proba = Vec::with_capacity(features.len());

        for feature_vec in features {
            // Standardize
            let scaled: Vec<f64> = feature_vec
                .iter()
                .zip(means.iter().zip(stds.iter()))
                .map(|(&x, (&m, &s))| if s > 1e-10 { (x - m) / s } else { 0.0 })
                .collect();

            // Compute linear combination
            let z: f64 = intercept
                + scaled
                    .iter()
                    .zip(coef.iter())
                    .map(|(x, c)| x * c)
                    .sum::<f64>();

            // Apply link function
            let p = match self.method {
                PropensityMethod::LogisticRegression => sigmoid(z),
                PropensityMethod::LinearProbability => z.clamp(0.0, 1.0),
            };

            // Clip to bounds
            proba.push(p.clamp(self.clip_bounds.0, self.clip_bounds.1));
        }

        Ok(proba)
    }
}

/// Propensity Score Matcher
///
/// Matches treated observations with control observations based on propensity scores.
pub struct PropensityMatcher {
    method: MatchingMethod,
    n_neighbors: usize,
    caliper: f64,
    n_strata: usize,
    with_replacement: bool,
}

impl PropensityMatcher {
    pub fn new(method: MatchingMethod) -> Self {
        Self {
            method,
            n_neighbors: 1,
            caliper: 0.2,
            n_strata: 5,
            with_replacement: false,
        }
    }

    pub fn with_n_neighbors(mut self, n: usize) -> Self {
        self.n_neighbors = n;
        self
    }

    pub fn with_caliper(mut self, caliper: f64) -> Self {
        self.caliper = caliper;
        self
    }

    pub fn with_replacement(mut self, replacement: bool) -> Self {
        self.with_replacement = replacement;
        self
    }

    /// Perform matching
    pub fn match_samples(
        &self,
        propensity_scores: &[f64],
        treatment: &[u8],
        features: Option<&[Vec<f64>]>,
    ) -> Result<MatchResult, PropensityError> {
        // Separate treated and control indices
        let mut treated_idx: Vec<usize> = Vec::new();
        let mut control_idx: Vec<usize> = Vec::new();

        for (i, &t) in treatment.iter().enumerate() {
            if t == 1 {
                treated_idx.push(i);
            } else {
                control_idx.push(i);
            }
        }

        if treated_idx.is_empty() || control_idx.is_empty() {
            return Err(PropensityError::InsufficientOverlap);
        }

        let (matched_treated, matched_control) = match self.method {
            MatchingMethod::NearestNeighbor | MatchingMethod::Caliper => {
                self.nearest_neighbor_match(propensity_scores, &treated_idx, &control_idx)?
            }
            MatchingMethod::Stratification => {
                self.stratification_match(propensity_scores, treatment)?
            }
        };

        // Calculate balance statistics
        let smd = if let Some(feats) = features {
            calculate_smd(feats, &matched_treated, &matched_control)
        } else {
            HashMap::new()
        };

        let balance_quality = if smd.is_empty() {
            1.0
        } else {
            let max_smd = smd.values().map(|&v| v.abs()).fold(0.0f64, f64::max);
            (0.25 - max_smd).max(0.0) / 0.25 // Quality score 0-1
        };

        Ok(MatchResult {
            treated_indices: matched_treated,
            control_indices: matched_control,
            propensity_scores: propensity_scores.to_vec(),
            standardized_mean_differences: smd,
            balance_quality,
        })
    }

    fn nearest_neighbor_match(
        &self,
        ps: &[f64],
        treated_idx: &[usize],
        control_idx: &[usize],
    ) -> Result<(Vec<usize>, Vec<usize>), PropensityError> {
        let mut matched_treated = Vec::new();
        let mut matched_control = Vec::new();
        let mut used_controls = std::collections::HashSet::new();

        let ps_std = {
            let mean = ps.iter().sum::<f64>() / ps.len() as f64;
            let variance = ps.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / ps.len() as f64;
            variance.sqrt()
        };

        for &t_idx in treated_idx {
            let t_ps = ps[t_idx];

            // Find nearest control
            let mut best_distance = f64::MAX;
            let mut best_control = None;

            for &c_idx in control_idx {
                if !self.with_replacement && used_controls.contains(&c_idx) {
                    continue;
                }

                let distance = (ps[c_idx] - t_ps).abs();

                // Caliper check
                if self.method == MatchingMethod::Caliper {
                    if distance > self.caliper * ps_std {
                        continue;
                    }
                }

                if distance < best_distance {
                    best_distance = distance;
                    best_control = Some(c_idx);
                }
            }

            if let Some(c_idx) = best_control {
                matched_treated.push(t_idx);
                matched_control.push(c_idx);
                used_controls.insert(c_idx);
            }
        }

        if matched_treated.is_empty() {
            return Err(PropensityError::InsufficientOverlap);
        }

        Ok((matched_treated, matched_control))
    }

    fn stratification_match(
        &self,
        ps: &[f64],
        treatment: &[u8],
    ) -> Result<(Vec<usize>, Vec<usize>), PropensityError> {
        // Create strata based on propensity score quantiles
        let mut sorted_ps: Vec<(usize, f64)> = ps.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        sorted_ps.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let n = sorted_ps.len();
        let stratum_size = n / self.n_strata;

        let mut matched_treated = Vec::new();
        let mut matched_control = Vec::new();
        let mut rng = rand::thread_rng();

        for s in 0..self.n_strata {
            let start = s * stratum_size;
            let end = if s == self.n_strata - 1 {
                n
            } else {
                (s + 1) * stratum_size
            };

            let stratum_indices: Vec<usize> =
                sorted_ps[start..end].iter().map(|(i, _)| *i).collect();

            let treated_in_stratum: Vec<usize> = stratum_indices
                .iter()
                .filter(|&&i| treatment[i] == 1)
                .cloned()
                .collect();

            let control_in_stratum: Vec<usize> = stratum_indices
                .iter()
                .filter(|&&i| treatment[i] == 0)
                .cloned()
                .collect();

            // Match within stratum
            for t_idx in treated_in_stratum {
                if !control_in_stratum.is_empty() {
                    let c_idx = control_in_stratum[rng.gen_range(0..control_in_stratum.len())];
                    matched_treated.push(t_idx);
                    matched_control.push(c_idx);
                }
            }
        }

        if matched_treated.is_empty() {
            return Err(PropensityError::InsufficientOverlap);
        }

        Ok((matched_treated, matched_control))
    }
}

/// Inverse Probability Weighting (IPW) Estimator
pub struct IPWEstimator {
    normalize_weights: bool,
}

impl IPWEstimator {
    pub fn new(normalize_weights: bool) -> Self {
        Self { normalize_weights }
    }

    /// Estimate Average Treatment Effect using IPW
    pub fn estimate_ate(
        &self,
        outcomes: &[f64],
        treatment: &[u8],
        propensity_scores: &[f64],
    ) -> Result<CausalEstimate, PropensityError> {
        let n = outcomes.len();

        if n != treatment.len() || n != propensity_scores.len() {
            return Err(PropensityError::InvalidDimensions(
                "All inputs must have same length".to_string(),
            ));
        }

        let n_treated: f64 = treatment.iter().map(|&t| t as f64).sum();
        let n_control = n as f64 - n_treated;

        if n_treated == 0.0 || n_control == 0.0 {
            return Err(PropensityError::InsufficientOverlap);
        }

        // Calculate weights
        let mut w1_sum = 0.0;
        let mut w0_sum = 0.0;
        let mut y1_weighted = 0.0;
        let mut y0_weighted = 0.0;

        for i in 0..n {
            let t = treatment[i] as f64;
            let ps = propensity_scores[i];
            let y = outcomes[i];

            if t == 1.0 {
                let w = 1.0 / ps;
                w1_sum += w;
                y1_weighted += w * y;
            } else {
                let w = 1.0 / (1.0 - ps);
                w0_sum += w;
                y0_weighted += w * y;
            }
        }

        // Normalize if requested
        let y1_hat = if self.normalize_weights {
            y1_weighted / w1_sum
        } else {
            y1_weighted / n_treated
        };

        let y0_hat = if self.normalize_weights {
            y0_weighted / w0_sum
        } else {
            y0_weighted / n_control
        };

        let ate = y1_hat - y0_hat;

        // Bootstrap standard error
        let n_bootstrap = 500;
        let mut ate_bootstrap = Vec::with_capacity(n_bootstrap);
        let mut rng = rand::thread_rng();

        for _ in 0..n_bootstrap {
            let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();

            let mut w1_sum_b = 0.0;
            let mut w0_sum_b = 0.0;
            let mut y1_b = 0.0;
            let mut y0_b = 0.0;
            let mut n_t_b = 0.0;
            let mut n_c_b = 0.0;

            for &idx in &indices {
                let t = treatment[idx] as f64;
                let ps = propensity_scores[idx];
                let y = outcomes[idx];

                if t == 1.0 {
                    let w = 1.0 / ps;
                    w1_sum_b += w;
                    y1_b += w * y;
                    n_t_b += 1.0;
                } else {
                    let w = 1.0 / (1.0 - ps);
                    w0_sum_b += w;
                    y0_b += w * y;
                    n_c_b += 1.0;
                }
            }

            if n_t_b > 0.0 && n_c_b > 0.0 && w1_sum_b > 0.0 && w0_sum_b > 0.0 {
                let y1_hat_b = if self.normalize_weights {
                    y1_b / w1_sum_b
                } else {
                    y1_b / n_t_b
                };

                let y0_hat_b = if self.normalize_weights {
                    y0_b / w0_sum_b
                } else {
                    y0_b / n_c_b
                };

                ate_bootstrap.push(y1_hat_b - y0_hat_b);
            }
        }

        let std_error = if ate_bootstrap.len() > 1 {
            let mean = ate_bootstrap.iter().sum::<f64>() / ate_bootstrap.len() as f64;
            let variance = ate_bootstrap
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / (ate_bootstrap.len() - 1) as f64;
            variance.sqrt()
        } else {
            f64::NAN
        };

        // Confidence interval
        let ci_lower = ate - 1.96 * std_error;
        let ci_upper = ate + 1.96 * std_error;

        // P-value
        let p_value = if std_error > 0.0 {
            let z = ate.abs() / std_error;
            let normal = Normal::new(0.0, 1.0).unwrap();
            2.0 * (1.0 - normal.cdf(z))
        } else {
            f64::NAN
        };

        Ok(CausalEstimate {
            estimate: ate,
            std_error,
            confidence_interval: (ci_lower, ci_upper),
            p_value,
            method: "IPW".to_string(),
            n_treated: n_treated as usize,
            n_control: n_control as usize,
        })
    }
}

/// Doubly Robust Estimator (AIPW)
pub struct DoublyRobustEstimator {
    learning_rate: f64,
    max_iter: usize,
}

impl DoublyRobustEstimator {
    pub fn new() -> Self {
        Self {
            learning_rate: 0.01,
            max_iter: 1000,
        }
    }

    /// Estimate ATE using doubly robust method
    pub fn estimate_ate(
        &self,
        outcomes: &[f64],
        treatment: &[u8],
        propensity_scores: &[f64],
        features: &[Vec<f64>],
    ) -> Result<CausalEstimate, PropensityError> {
        let n = outcomes.len();

        // Fit outcome models (simple linear regression)
        let (mu1, mu0) = self.fit_outcome_models(outcomes, treatment, features)?;

        // Compute doubly robust estimates
        let mut y1_dr = Vec::with_capacity(n);
        let mut y0_dr = Vec::with_capacity(n);

        for i in 0..n {
            let t = treatment[i] as f64;
            let ps = propensity_scores[i];
            let y = outcomes[i];

            // AIPW for Y(1)
            let y1_i = mu1[i] + t * (y - mu1[i]) / ps;
            y1_dr.push(y1_i);

            // AIPW for Y(0)
            let y0_i = mu0[i] + (1.0 - t) * (y - mu0[i]) / (1.0 - ps);
            y0_dr.push(y0_i);
        }

        let ate = y1_dr.iter().sum::<f64>() / n as f64 - y0_dr.iter().sum::<f64>() / n as f64;

        // Bootstrap standard error
        let n_bootstrap = 500;
        let mut ate_bootstrap = Vec::with_capacity(n_bootstrap);
        let mut rng = rand::thread_rng();

        for _ in 0..n_bootstrap {
            let mut sum_diff = 0.0;
            for _ in 0..n {
                let idx = rng.gen_range(0..n);
                sum_diff += y1_dr[idx] - y0_dr[idx];
            }
            ate_bootstrap.push(sum_diff / n as f64);
        }

        let std_error = {
            let mean = ate_bootstrap.iter().sum::<f64>() / ate_bootstrap.len() as f64;
            let variance = ate_bootstrap
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / (ate_bootstrap.len() - 1) as f64;
            variance.sqrt()
        };

        let ci_lower = ate - 1.96 * std_error;
        let ci_upper = ate + 1.96 * std_error;

        let p_value = if std_error > 0.0 {
            let z = ate.abs() / std_error;
            let normal = Normal::new(0.0, 1.0).unwrap();
            2.0 * (1.0 - normal.cdf(z))
        } else {
            f64::NAN
        };

        let n_treated = treatment.iter().filter(|&&t| t == 1).count();
        let n_control = treatment.iter().filter(|&&t| t == 0).count();

        Ok(CausalEstimate {
            estimate: ate,
            std_error,
            confidence_interval: (ci_lower, ci_upper),
            p_value,
            method: "Doubly Robust (AIPW)".to_string(),
            n_treated,
            n_control,
        })
    }

    fn fit_outcome_models(
        &self,
        outcomes: &[f64],
        treatment: &[u8],
        features: &[Vec<f64>],
    ) -> Result<(Vec<f64>, Vec<f64>), PropensityError> {
        let n = features.len();
        let n_features = features[0].len();

        // Standardize features
        let (scaled_features, means, stds) = standardize_features(features);

        // Fit model for treated
        let (coef_1, intercept_1) = self.fit_linear_model(
            &scaled_features,
            outcomes,
            treatment,
            true,
            n_features,
        );

        // Fit model for control
        let (coef_0, intercept_0) = self.fit_linear_model(
            &scaled_features,
            outcomes,
            treatment,
            false,
            n_features,
        );

        // Predict for all observations
        let mut mu1 = Vec::with_capacity(n);
        let mut mu0 = Vec::with_capacity(n);

        for i in 0..n {
            let pred_1: f64 = intercept_1
                + scaled_features[i]
                    .iter()
                    .zip(coef_1.iter())
                    .map(|(x, c)| x * c)
                    .sum::<f64>();

            let pred_0: f64 = intercept_0
                + scaled_features[i]
                    .iter()
                    .zip(coef_0.iter())
                    .map(|(x, c)| x * c)
                    .sum::<f64>();

            mu1.push(pred_1);
            mu0.push(pred_0);
        }

        Ok((mu1, mu0))
    }

    fn fit_linear_model(
        &self,
        features: &[Vec<f64>],
        outcomes: &[f64],
        treatment: &[u8],
        for_treated: bool,
        n_features: usize,
    ) -> (Vec<f64>, f64) {
        let mut coef = vec![0.0; n_features];
        let mut intercept = 0.0;

        // Filter to relevant group
        let indices: Vec<usize> = treatment
            .iter()
            .enumerate()
            .filter(|(_, &t)| (t == 1) == for_treated)
            .map(|(i, _)| i)
            .collect();

        if indices.is_empty() {
            return (coef, intercept);
        }

        // Gradient descent
        for _ in 0..self.max_iter {
            let mut grad_coef = vec![0.0; n_features];
            let mut grad_intercept = 0.0;

            for &idx in &indices {
                let pred: f64 = intercept
                    + features[idx]
                        .iter()
                        .zip(coef.iter())
                        .map(|(x, c)| x * c)
                        .sum::<f64>();

                let error = pred - outcomes[idx];

                for (j, &x) in features[idx].iter().enumerate() {
                    grad_coef[j] += error * x;
                }
                grad_intercept += error;
            }

            let n_group = indices.len() as f64;
            for j in 0..n_features {
                coef[j] -= self.learning_rate * grad_coef[j] / n_group;
            }
            intercept -= self.learning_rate * grad_intercept / n_group;
        }

        (coef, intercept)
    }
}

impl Default for DoublyRobustEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

fn standardize_features(features: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let n = features.len();
    let n_features = features[0].len();

    // Calculate means
    let mut means = vec![0.0; n_features];
    for feature_vec in features {
        for (j, &x) in feature_vec.iter().enumerate() {
            means[j] += x;
        }
    }
    for mean in means.iter_mut() {
        *mean /= n as f64;
    }

    // Calculate stds
    let mut stds = vec![0.0; n_features];
    for feature_vec in features {
        for (j, &x) in feature_vec.iter().enumerate() {
            stds[j] += (x - means[j]).powi(2);
        }
    }
    for std in stds.iter_mut() {
        *std = (*std / n as f64).sqrt().max(1e-10);
    }

    // Standardize
    let scaled: Vec<Vec<f64>> = features
        .iter()
        .map(|fv| {
            fv.iter()
                .zip(means.iter().zip(stds.iter()))
                .map(|(&x, (&m, &s))| (x - m) / s)
                .collect()
        })
        .collect();

    (scaled, means, stds)
}

fn calculate_smd(
    features: &[Vec<f64>],
    treated_idx: &[usize],
    control_idx: &[usize],
) -> HashMap<String, f64> {
    let n_features = features[0].len();
    let mut smd = HashMap::new();

    for j in 0..n_features {
        let treated_vals: Vec<f64> = treated_idx.iter().map(|&i| features[i][j]).collect();
        let control_vals: Vec<f64> = control_idx.iter().map(|&i| features[i][j]).collect();

        let mean_t = treated_vals.iter().sum::<f64>() / treated_vals.len() as f64;
        let mean_c = control_vals.iter().sum::<f64>() / control_vals.len() as f64;

        let var_t = treated_vals
            .iter()
            .map(|&x| (x - mean_t).powi(2))
            .sum::<f64>()
            / treated_vals.len() as f64;
        let var_c = control_vals
            .iter()
            .map(|&x| (x - mean_c).powi(2))
            .sum::<f64>()
            / control_vals.len() as f64;

        let pooled_std = ((var_t + var_c) / 2.0).sqrt();

        let smd_value = if pooled_std > 1e-10 {
            (mean_t - mean_c) / pooled_std
        } else {
            0.0
        };

        smd.insert(format!("feature_{}", j), smd_value);
    }

    smd
}
