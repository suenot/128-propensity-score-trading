//! Backtesting example for propensity score trading strategies
//!
//! This example demonstrates how to:
//! 1. Backtest a trading strategy with propensity score weighting
//! 2. Compare causal vs naive estimates of strategy performance
//!
//! Run with: cargo run --example backtest

use propensity_score_trading::{
    generate_synthetic_data, PropensityBacktester, compare_naive_vs_causal,
};

fn main() {
    println!("=== Propensity Score Trading: Backtesting ===\n");

    // Generate synthetic data with known treatment effect
    let true_effect = 0.008; // 0.8% return boost from signal
    let confounding_strength = 0.4;

    println!("Generating synthetic data...");
    println!("  True treatment effect: {:.2}%", true_effect * 100.0);
    println!("  Confounding strength: {:.2}", confounding_strength);
    println!();

    let data = generate_synthetic_data(500, true_effect, confounding_strength, 42);

    println!("Data summary:");
    println!("  Total samples: {}", data.metadata.n_samples);
    println!("  Treated samples: {}", data.metadata.n_treated);
    println!("  Treatment rate: {:.2}%", data.metadata.treatment_rate * 100.0);
    println!();

    // Run backtest with propensity weighting
    println!("=== Backtest with Propensity Weighting ===");

    let backtester = PropensityBacktester::new()
        .with_initial_capital(100_000.0)
        .with_transaction_cost(0.001)
        .with_propensity_weighting(true)
        .with_warmup_period(50)
        .with_refit_frequency(20);

    match backtester.run(&data) {
        Ok(result) => {
            println!("{}", result.metrics);

            // Additional analysis
            println!("\n=== Additional Analysis ===");

            // Compare active returns
            let active_returns = result.active_returns();
            if !active_returns.is_empty() {
                let mean_active: f64 = active_returns.iter().sum::<f64>() / active_returns.len() as f64;
                println!("Active returns (when position is held):");
                println!("  Count: {}", active_returns.len());
                println!("  Mean: {:.4}%", mean_active * 100.0);
            }

            // Propensity score distribution
            let ps = &result.propensity_scores[50..]; // After warmup
            let ps_mean: f64 = ps.iter().sum::<f64>() / ps.len() as f64;
            println!("\nPropensity score summary (after warmup):");
            println!("  Mean: {:.3}", ps_mean);
        }
        Err(e) => println!("Backtest failed: {}", e),
    }
    println!();

    // Run backtest WITHOUT propensity weighting for comparison
    println!("=== Backtest WITHOUT Propensity Weighting ===");

    let backtester_naive = PropensityBacktester::new()
        .with_initial_capital(100_000.0)
        .with_transaction_cost(0.001)
        .with_propensity_weighting(false)
        .with_warmup_period(50);

    match backtester_naive.run(&data) {
        Ok(result) => {
            println!("Total Return:      {:>10.2}%", result.metrics.total_return * 100.0);
            println!("Sharpe Ratio:      {:>10.2}", result.metrics.sharpe_ratio);
            println!("Max Drawdown:      {:>10.2}%", result.metrics.max_drawdown * 100.0);
        }
        Err(e) => println!("Backtest failed: {}", e),
    }
    println!();

    // Compare naive vs causal estimates
    println!("=== Naive vs Causal Comparison ===");
    let comparison = compare_naive_vs_causal(&data);
    println!("{}", comparison);

    // Interpretation
    println!("\n=== Interpretation ===");
    println!("In this example, the market conditions (confounders) cause both:");
    println!("1. The trading signal to fire more often in favorable conditions");
    println!("2. Better returns during those same favorable conditions");
    println!();
    println!("Without propensity score adjustment, we overestimate the signal's");
    println!("true causal effect because we attribute the good performance to");
    println!("the signal when it was actually due to market conditions.");
    println!();
    println!("The propensity score methods help us isolate the TRUE causal");
    println!("effect of the trading signal on returns.");
}
