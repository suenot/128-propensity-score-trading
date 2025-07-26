//! Utility functions
//!
//! Common utilities used across the crate.

use std::f64;

/// Check if two floats are approximately equal
pub fn approx_equal(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
}

/// Calculate mean of a slice
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Calculate variance of a slice
pub fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}

/// Calculate standard deviation of a slice
pub fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Calculate the correlation between two slices
pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x = mean(x);
    let mean_y = mean(y);

    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f64>()
        / (n - 1.0);

    let std_x = std_dev(x);
    let std_y = std_dev(y);

    if std_x > 0.0 && std_y > 0.0 {
        cov / (std_x * std_y)
    } else {
        0.0
    }
}

/// Calculate percentile of a slice
pub fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    if p <= 0.0 {
        return *data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    }
    if p >= 100.0 {
        return *data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let rank = p / 100.0 * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let weight = rank - lower as f64;

    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

/// Calculate quantiles (splits data into n groups)
pub fn quantiles(data: &[f64], n: usize) -> Vec<f64> {
    (1..n)
        .map(|i| percentile(data, (i as f64 / n as f64) * 100.0))
        .collect()
}

/// Clip values to a range
pub fn clip(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

/// Sigmoid function
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// Inverse sigmoid (logit) function
pub fn logit(p: f64) -> f64 {
    if p <= 0.0 {
        f64::NEG_INFINITY
    } else if p >= 1.0 {
        f64::INFINITY
    } else {
        (p / (1.0 - p)).ln()
    }
}

/// Calculate rolling window statistics
pub struct RollingWindow {
    window_size: usize,
    data: Vec<f64>,
}

impl RollingWindow {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            data: Vec::with_capacity(window_size),
        }
    }

    pub fn push(&mut self, value: f64) {
        self.data.push(value);
        if self.data.len() > self.window_size {
            self.data.remove(0);
        }
    }

    pub fn mean(&self) -> f64 {
        mean(&self.data)
    }

    pub fn std_dev(&self) -> f64 {
        std_dev(&self.data)
    }

    pub fn is_full(&self) -> bool {
        self.data.len() >= self.window_size
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&data) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_std_dev() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = std_dev(&data);
        assert!((sd - 2.138).abs() < 0.001);
    }

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&data, 50.0) - 3.0).abs() < 1e-10);
        assert!((percentile(&data, 0.0) - 1.0).abs() < 1e-10);
        assert!((percentile(&data, 100.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_rolling_window() {
        let mut window = RollingWindow::new(3);
        window.push(1.0);
        window.push(2.0);
        window.push(3.0);
        assert!((window.mean() - 2.0).abs() < 1e-10);

        window.push(4.0);
        assert!((window.mean() - 3.0).abs() < 1e-10);
    }
}
