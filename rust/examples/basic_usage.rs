//! Basic usage example for propensity score trading
//!
//! This example demonstrates how to:
//! 1. Generate synthetic trading data
//! 2. Fit a propensity score model
//! 3. Estimate causal effects using different methods
//!
//! Run with: cargo run --example basic_usage

use propensity_score_trading::{
    generate_synthetic_data, DoublyRobustEstimator, IPWEstimator, PropensityMatcher,
    PropensityMethod, PropensityScoreModel, MatchingMethod,
};

fn main() {
    println!("=== Propensity Score Trading: Basic Usage ===\n");

    // Generate synthetic data with known treatment effect
    let true_effect = 0.005; // 0.5% return boost from signal
    let confounding_strength = 0.3;

    println!("Generating synthetic data...");
    println!("  True treatment effect: {:.2}%", true_effect * 100.0);
    println!("  Confounding strength: {:.2}", confounding_strength);

    let data = generate_synthetic_data(1000, true_effect, confounding_strength, 42);

    println!("  Samples: {}", data.metadata.n_samples);
    println!("  Treatment rate: {:.2}%", data.metadata.treatment_rate * 100.0);
    println!();

    // Calculate naive estimate (biased due to confounding)
    let treated_outcomes: Vec<f64> = data
        .outcomes
        .iter()
        .zip(data.treatment.iter())
        .filter(|(_, &t)| t == 1)
        .map(|(&o, _)| o)
        .collect();

    let control_outcomes: Vec<f64> = data
        .outcomes
        .iter()
        .zip(data.treatment.iter())
        .filter(|(_, &t)| t == 0)
        .map(|(&o, _)| o)
        .collect();

    let naive_effect = (treated_outcomes.iter().sum::<f64>() / treated_outcomes.len() as f64)
        - (control_outcomes.iter().sum::<f64>() / control_outcomes.len() as f64);

    println!("=== Naive Analysis (Biased) ===");
    println!("  Naive ATE: {:.4}%", naive_effect * 100.0);
    println!(
        "  Bias: {:.4}%",
        (naive_effect - true_effect) * 100.0
    );
    println!();

    // Fit propensity score model
    println!("=== Propensity Score Analysis ===");
    println!("Fitting propensity score model...");

    let mut ps_model = PropensityScoreModel::new(PropensityMethod::LogisticRegression);
    ps_model.fit(&data.features, &data.treatment).expect("Failed to fit model");

    let propensity_scores = ps_model
        .predict_proba(&data.features)
        .expect("Failed to predict");

    // Check propensity score distribution
    let ps_mean: f64 = propensity_scores.iter().sum::<f64>() / propensity_scores.len() as f64;
    let ps_min = propensity_scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let ps_max = propensity_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("Propensity score distribution:");
    println!("  Mean: {:.3}", ps_mean);
    println!("  Min:  {:.3}", ps_min);
    println!("  Max:  {:.3}", ps_max);
    println!();

    // Estimate ATE using IPW
    println!("=== IPW Estimator ===");
    let ipw = IPWEstimator::new(true);
    match ipw.estimate_ate(&data.outcomes, &data.treatment, &propensity_scores) {
        Ok(estimate) => {
            println!("  ATE: {:.4}%", estimate.estimate * 100.0);
            println!(
                "  95% CI: [{:.4}%, {:.4}%]",
                estimate.confidence_interval.0 * 100.0,
                estimate.confidence_interval.1 * 100.0
            );
            println!("  P-value: {:.4}", estimate.p_value);
            println!("  Bias: {:.4}%", (estimate.estimate - true_effect) * 100.0);
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // Estimate ATE using Doubly Robust method
    println!("=== Doubly Robust Estimator ===");
    let dr = DoublyRobustEstimator::new();
    match dr.estimate_ate(&data.outcomes, &data.treatment, &propensity_scores, &data.features) {
        Ok(estimate) => {
            println!("  ATE: {:.4}%", estimate.estimate * 100.0);
            println!(
                "  95% CI: [{:.4}%, {:.4}%]",
                estimate.confidence_interval.0 * 100.0,
                estimate.confidence_interval.1 * 100.0
            );
            println!("  P-value: {:.4}", estimate.p_value);
            println!("  Bias: {:.4}%", (estimate.estimate - true_effect) * 100.0);
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // Propensity score matching
    println!("=== Propensity Score Matching ===");
    let matcher = PropensityMatcher::new(MatchingMethod::NearestNeighbor)
        .with_n_neighbors(1)
        .with_replacement(false);

    match matcher.match_samples(&propensity_scores, &data.treatment, Some(&data.features)) {
        Ok(match_result) => {
            println!("  Matched pairs: {}", match_result.treated_indices.len());
            println!("  Balance quality: {:.3}", match_result.balance_quality);

            // Calculate matched ATE
            let treated_outcomes_matched: f64 = match_result
                .treated_indices
                .iter()
                .map(|&i| data.outcomes[i])
                .sum::<f64>();
            let control_outcomes_matched: f64 = match_result
                .control_indices
                .iter()
                .map(|&i| data.outcomes[i])
                .sum::<f64>();

            let n_matches = match_result.treated_indices.len() as f64;
            let matching_ate = (treated_outcomes_matched - control_outcomes_matched) / n_matches;

            println!("  Matched ATE: {:.4}%", matching_ate * 100.0);
            println!("  Bias: {:.4}%", (matching_ate - true_effect) * 100.0);
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // Summary
    println!("=== Summary ===");
    println!("True treatment effect:    {:.4}%", true_effect * 100.0);
    println!("Naive estimate (biased):  {:.4}%", naive_effect * 100.0);
    println!();
    println!("The propensity score methods help reduce the bias caused by");
    println!("confounding factors, providing more accurate causal estimates.");
}
