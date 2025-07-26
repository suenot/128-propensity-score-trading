//! Cryptocurrency data example for propensity score trading
//!
//! This example demonstrates how to use propensity score methods
//! with cryptocurrency trading data (simulated Bybit-style data).
//!
//! Run with: cargo run --example crypto_data

use propensity_score_trading::{
    generate_synthetic_data, PropensityBacktester, PropensityScoreModel,
    PropensityMethod, IPWEstimator, compare_naive_vs_causal,
};
use propensity_score_trading::data_loader::{
    PriceBar, compute_features, create_momentum_signal, create_mean_reversion_signal,
    MarketData, DatasetMetadata,
};

fn main() {
    println!("=== Propensity Score Trading: Crypto Data Example ===\n");
    println!("This example simulates cryptocurrency trading data similar to");
    println!("what you would get from Bybit's API.\n");

    // Generate synthetic crypto data
    // Crypto markets have higher volatility and different characteristics
    let data = generate_crypto_data(365, 42); // 1 year of daily data

    println!("=== Data Summary ===");
    println!("  Symbol: BTC/USDT (simulated)");
    println!("  Period: {} days", data.metadata.n_samples);
    println!("  Treatment rate: {:.2}%", data.metadata.treatment_rate * 100.0);
    println!();

    // Show some price data
    println!("=== Sample Price Data ===");
    for (i, bar) in data.prices.iter().take(5).enumerate() {
        println!(
            "  Day {}: Open=${:.2}, High=${:.2}, Low=${:.2}, Close=${:.2}, Vol={:.0}",
            i + 1, bar.open, bar.high, bar.low, bar.close, bar.volume
        );
    }
    println!("  ...");
    println!();

    // Fit propensity model
    println!("=== Propensity Score Analysis ===");

    let mut ps_model = PropensityScoreModel::new(PropensityMethod::LogisticRegression);
    match ps_model.fit(&data.features, &data.treatment) {
        Ok(_) => println!("  Propensity model fitted successfully"),
        Err(e) => {
            println!("  Failed to fit model: {}", e);
            return;
        }
    }

    let propensity_scores = match ps_model.predict_proba(&data.features) {
        Ok(ps) => ps,
        Err(e) => {
            println!("  Failed to predict: {}", e);
            return;
        }
    };

    // Propensity score statistics
    let ps_mean: f64 = propensity_scores.iter().sum::<f64>() / propensity_scores.len() as f64;
    let ps_min = propensity_scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let ps_max = propensity_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("  Propensity score stats:");
    println!("    Mean: {:.3}", ps_mean);
    println!("    Range: [{:.3}, {:.3}]", ps_min, ps_max);
    println!();

    // IPW estimate
    println!("=== Causal Effect Estimation ===");
    let ipw = IPWEstimator::new(true);
    match ipw.estimate_ate(&data.outcomes, &data.treatment, &propensity_scores) {
        Ok(estimate) => {
            println!("IPW Estimator:");
            println!("  ATE: {:.4}% per day", estimate.estimate * 100.0);
            println!(
                "  95% CI: [{:.4}%, {:.4}%]",
                estimate.confidence_interval.0 * 100.0,
                estimate.confidence_interval.1 * 100.0
            );
            println!(
                "  Significant: {}",
                if estimate.is_significant(0.05) { "Yes" } else { "No" }
            );
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // Run backtest
    println!("=== Backtest Results ===");

    let backtester = PropensityBacktester::new()
        .with_initial_capital(10_000.0) // $10k starting capital
        .with_transaction_cost(0.001) // 0.1% (typical crypto)
        .with_propensity_weighting(true)
        .with_warmup_period(30);

    match backtester.run(&data) {
        Ok(result) => {
            println!("With Propensity Weighting:");
            println!("  Total Return:    {:>10.2}%", result.metrics.total_return * 100.0);
            println!("  Annual Return:   {:>10.2}%", result.metrics.annualized_return * 100.0);
            println!("  Sharpe Ratio:    {:>10.2}", result.metrics.sharpe_ratio);
            println!("  Max Drawdown:    {:>10.2}%", result.metrics.max_drawdown * 100.0);
            println!("  Win Rate:        {:>10.2}%", result.metrics.win_rate * 100.0);
            println!("  Number of Trades: {:>9}", result.metrics.n_trades);

            if let Some(ate) = result.metrics.causal_ate {
                println!("\n  Causal Analysis:");
                println!("    Estimated Signal Effect: {:.4}% per trade", ate * 100.0);
            }
        }
        Err(e) => println!("  Backtest failed: {}", e),
    }
    println!();

    // Compare different signals
    println!("=== Signal Comparison ===");
    compare_signals(&data.prices);
}

/// Generate simulated cryptocurrency data
fn generate_crypto_data(n_days: usize, seed: u64) -> MarketData {
    use rand::prelude::*;
    use rand_distr::{LogNormal, Normal};

    let mut rng = StdRng::seed_from_u64(seed);

    // Crypto has higher volatility than stocks
    let daily_vol = 0.04; // 4% daily volatility
    let normal = Normal::new(0.0, daily_vol).unwrap();
    let volume_dist = LogNormal::new(20.0, 1.0).unwrap();

    let mut prices = Vec::with_capacity(n_days);
    let mut current_price = 40000.0; // Starting BTC price

    for i in 0..n_days {
        let daily_return = rng.sample(normal);
        let new_price = current_price * (1.0 + daily_return);

        let high = new_price * (1.0 + rng.gen::<f64>() * 0.02);
        let low = new_price * (1.0 - rng.gen::<f64>() * 0.02);
        let open = current_price * (1.0 + (rng.gen::<f64>() - 0.5) * 0.01);
        let volume = rng.sample(volume_dist);

        prices.push(PriceBar {
            timestamp: i as i64 * 86400,
            open,
            high,
            low,
            close: new_price,
            volume,
        });

        current_price = new_price;
    }

    // Compute features
    let features = compute_features(&prices);

    // Generate momentum signal with crypto-appropriate threshold
    let treatment = create_momentum_signal(&prices, 5, 0.03); // 3% threshold

    // Calculate forward returns
    let outcomes: Vec<f64> = (0..n_days - 1)
        .map(|i| (prices[i + 1].close - prices[i].close) / prices[i].close)
        .chain(std::iter::once(0.0))
        .collect();

    let n_treated = treatment.iter().filter(|&&t| t == 1).count();

    MarketData {
        prices,
        features,
        outcomes,
        treatment,
        metadata: DatasetMetadata {
            n_samples: n_days,
            n_treated,
            treatment_rate: n_treated as f64 / n_days as f64,
            true_treatment_effect: None, // Unknown for real-style data
            confounding_strength: None,
            data_source: "simulated_bybit".to_string(),
        },
    }
}

/// Compare different trading signals
fn compare_signals(prices: &[PriceBar]) {
    // Momentum signal
    let momentum_signal = create_momentum_signal(prices, 5, 0.03);
    let n_momentum = momentum_signal.iter().filter(|&&t| t == 1).count();

    // Mean reversion signal
    let mean_rev_signal = create_mean_reversion_signal(prices, 20, -2.0);
    let n_mean_rev = mean_rev_signal.iter().filter(|&&t| t == 1).count();

    println!("Momentum Signal (5-day, >3%):");
    println!("  Triggers: {} ({:.1}%)", n_momentum, n_momentum as f64 / prices.len() as f64 * 100.0);

    println!("\nMean Reversion Signal (20-day, z<-2):");
    println!("  Triggers: {} ({:.1}%)", n_mean_rev, n_mean_rev as f64 / prices.len() as f64 * 100.0);

    println!("\nTo properly compare signal effectiveness, you would use");
    println!("propensity score methods to estimate the causal effect of");
    println!("each signal while controlling for market conditions.");
}
