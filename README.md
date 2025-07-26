# Chapter 107: Propensity Score Methods for Trading

## Overview

Propensity Score Methods (PSM) are a powerful set of causal inference techniques originally developed in biostatistics to estimate causal effects from observational data. In trading, these methods allow us to answer critical questions: "What is the true causal effect of a trading signal on returns?" rather than simply identifying correlations. By estimating the probability (propensity) that a certain treatment (e.g., a buy signal, a regime change, or a specific market condition) occurs given observed covariates, we can construct matched comparison groups and estimate unbiased treatment effects.

The seminal work by Rosenbaum & Rubin (1983) introduced propensity score matching as a method to reduce selection bias in observational studies. In financial markets, this approach helps traders distinguish genuine alpha signals from spurious correlations caused by confounding factors like market regime, volatility conditions, or liquidity.

This chapter covers the theory of propensity scores, their application to causal trading strategies, implementations in Python and Rust, and practical examples using both stock market and cryptocurrency data from Bybit.

## Table of Contents

1. [Introduction to Causal Inference in Trading](#introduction-to-causal-inference-in-trading)
2. [Propensity Score Fundamentals](#propensity-score-fundamentals)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Propensity Score Estimation Methods](#propensity-score-estimation-methods)
5. [Matching and Weighting Techniques](#matching-and-weighting-techniques)
6. [Applications to Trading](#applications-to-trading)
7. [Implementation in Python](#implementation-in-python)
8. [Implementation in Rust](#implementation-in-rust)
9. [Practical Examples with Stock and Crypto Data](#practical-examples-with-stock-and-crypto-data)
10. [Backtesting Framework](#backtesting-framework)
11. [Performance Evaluation](#performance-evaluation)
12. [Future Directions](#future-directions)
13. [References](#references)

---

## Introduction to Causal Inference in Trading

### The Problem of Confounding

In traditional machine learning for trading, we often find correlations between features and future returns. However, correlation does not imply causation. Consider a common scenario:

- A momentum signal shows strong correlation with future returns
- But momentum tends to work better in low-volatility regimes
- In high-volatility regimes, the same signal fails

The true relationship might be:

```
Volatility Regime → Momentum Signal Success
Volatility Regime → Future Returns
```

Here, volatility regime is a **confounder** — it affects both the treatment (signal activation) and the outcome (returns). Without accounting for this, our estimated signal effectiveness is biased.

### The Potential Outcomes Framework

The Rubin Causal Model (RCM) formalizes causal inference:

- For each unit i, there are two potential outcomes: Y_i(1) if treated, Y_i(0) if untreated
- The individual treatment effect: τ_i = Y_i(1) - Y_i(0)
- The fundamental problem: we only observe one outcome per unit

The **Average Treatment Effect (ATE)**:

```
ATE = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]
```

In trading terms:
- **Treatment**: Applying a trading signal (e.g., going long when signal > threshold)
- **Outcome**: The realized return
- **ATE**: The true causal effect of following the signal

### Why Propensity Scores?

Propensity scores solve the confounding problem by:
1. Estimating the probability of treatment given covariates: e(X) = P(T=1|X)
2. Using this probability to create balanced comparison groups
3. Estimating treatment effects that are unbiased by confounding

---

## Propensity Score Fundamentals

### Definition

The propensity score is the conditional probability of receiving treatment given observed covariates:

```
e(X) = P(T = 1 | X)
```

Where:
- T ∈ {0, 1} is the treatment indicator
- X is the vector of observed covariates (confounders)
- e(X) is the propensity score

### The Balancing Property

The key theoretical result (Rosenbaum & Rubin, 1983):

**If e(X) = e(X'), then P(X | T=1, e(X)) = P(X | T=0, e(X))**

This means that within strata defined by the propensity score, the distribution of covariates is the same for treated and untreated units. This "balances" the covariates, removing confounding bias.

### Assumptions for Causal Inference

1. **Unconfoundedness (Ignorability)**: Y(0), Y(1) ⊥ T | X
   - Given observed covariates, treatment assignment is as good as random

2. **Positivity (Overlap)**: 0 < P(T=1|X) < 1 for all X
   - Every unit has a non-zero probability of receiving either treatment

3. **Stable Unit Treatment Value Assumption (SUTVA)**:
   - No interference between units
   - No hidden variations in treatment

In trading, these assumptions translate to:
- All factors affecting both signal activation and returns are observed
- For any market condition, the signal could plausibly fire or not fire
- One trade doesn't affect the outcome of another (reasonable for liquid markets)

---

## Mathematical Foundation

### Average Treatment Effect Estimators

**Inverse Probability Weighting (IPW)**:

```
ATE_IPW = (1/n) Σ [T_i Y_i / e(X_i) - (1-T_i) Y_i / (1-e(X_i))]
```

Each observation is weighted by the inverse of its propensity of receiving its actual treatment.

**Augmented IPW (AIPW / Doubly Robust)**:

```
ATE_AIPW = (1/n) Σ [μ_1(X_i) - μ_0(X_i) + T_i(Y_i - μ_1(X_i))/e(X_i) - (1-T_i)(Y_i - μ_0(X_i))/(1-e(X_i))]
```

Where μ_t(X) = E[Y | T=t, X] is the outcome model. AIPW is doubly robust: consistent if either the propensity or outcome model is correctly specified.

**Matching Estimator**:

```
ATE_match = (1/n) Σ [T_i(Y_i - Y_j(i)) + (1-T_i)(Y_j(i) - Y_i)]
```

Where j(i) is the matched unit for i with the closest propensity score from the opposite treatment group.

### Variance Estimation

For IPW:

```
Var(ATE_IPW) ≈ (1/n) Var[T Y / e(X) - (1-T) Y / (1-e(X))]
```

Bootstrap is commonly used for variance estimation, especially for matching estimators.

### Subclassification

Dividing the propensity score into K strata and estimating within-stratum effects:

```
ATE_strat = Σ_k (n_k/n) [Ȳ_1k - Ȳ_0k]
```

Cochran (1968) showed that 5 strata remove 90% of bias from a single covariate.

---

## Propensity Score Estimation Methods

### Logistic Regression

The classic approach:

```
e(X) = 1 / (1 + exp(-X'β))
```

Fit via maximum likelihood. Simple, interpretable, but assumes linear relationship in log-odds.

### Gradient Boosting (GBM)

```python
from sklearn.ensemble import GradientBoostingClassifier
ps_model = GradientBoostingClassifier(n_estimators=100, max_depth=3)
ps_model.fit(X, T)
propensity_scores = ps_model.predict_proba(X)[:, 1]
```

Advantages:
- Captures non-linear relationships
- Handles high-dimensional covariates
- Often better calibrated for extreme propensities

### Neural Networks

For complex, high-dimensional data:

```python
class PropensityNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
```

### Generalized Propensity Scores (GPS)

For continuous or multi-valued treatments:

```
GPS(t, X) = f(T=t | X)
```

This extends PSM to questions like: "What is the effect of position size on returns?"

---

## Matching and Weighting Techniques

### Nearest Neighbor Matching

Match each treated unit to the closest untreated unit in propensity score:

```python
def nearest_neighbor_match(ps_treated, ps_control, caliper=None):
    matches = []
    for i, ps_t in enumerate(ps_treated):
        distances = np.abs(ps_control - ps_t)
        if caliper is not None:
            valid = distances <= caliper
            if not valid.any():
                continue
            distances = np.where(valid, distances, np.inf)
        j = np.argmin(distances)
        matches.append((i, j))
    return matches
```

Options:
- **With replacement**: Control units can be matched multiple times
- **Without replacement**: Each control used once (reduces bias, increases variance)
- **Caliper**: Maximum allowed distance for a match

### Kernel Matching

Weight all control units by their distance to each treated unit:

```
Ŷ_0(i) = Σ_j K((e_i - e_j) / h) Y_j / Σ_j K((e_i - e_j) / h)
```

Where K is a kernel function (e.g., Gaussian) and h is the bandwidth.

### Coarsened Exact Matching (CEM)

1. Coarsen covariates into bins
2. Match exactly within bins
3. Use weights to correct for different group sizes

```python
def coarsened_exact_match(X, T, n_bins=5):
    # Coarsen each covariate
    X_coarse = np.floor(X * n_bins) / n_bins
    # Find unique strata
    strata = [tuple(row) for row in X_coarse]
    # Match within strata
    weights = compute_cem_weights(strata, T)
    return weights
```

### Inverse Probability Weighting

Weights for each unit:

- Treated: w_i = 1 / e(X_i)
- Control: w_i = 1 / (1 - e(X_i))

For ATT (Average Treatment Effect on the Treated):
- Treated: w_i = 1
- Control: w_i = e(X_i) / (1 - e(X_i))

**Stabilized weights** to reduce variance:

```
sw_i = T_i * P(T=1) / e(X_i) + (1-T_i) * P(T=0) / (1-e(X_i))
```

### Overlap Weights

Addresses positivity violations by downweighting extreme propensities:

```
w_i = T_i * (1 - e(X_i)) + (1 - T_i) * e(X_i)
```

This targets the Average Treatment Effect among the Overlap population (ATO).

---

## Applications to Trading

### Use Case 1: Evaluating Signal Effectiveness

**Question**: Does our momentum signal actually cause positive returns, or is it confounded by market conditions?

**Setup**:
- Treatment: Momentum signal fires (T=1) or not (T=0)
- Outcome: Next-period return
- Covariates: Volatility, volume, spread, time of day, etc.

**Process**:
1. Estimate propensity scores: P(signal fires | market conditions)
2. Match signal-firing periods with similar non-firing periods
3. Compare average returns in matched pairs
4. Interpret the difference as the causal effect of the signal

### Use Case 2: Regime-Adjusted Strategy Evaluation

**Question**: What is the true performance of our strategy controlling for market regime?

**Setup**:
- Treatment: Strategy is "active" (generating trades)
- Outcome: Daily P&L
- Covariates: VIX level, trend strength, sector rotation, etc.

This separates strategy skill from lucky regime exposure.

### Use Case 3: Trade Execution Analysis

**Question**: Does trading at a specific time improve execution quality?

**Setup**:
- Treatment: Trade executed at market open vs. other times
- Outcome: Slippage relative to arrival price
- Covariates: Order size, spread, volatility, stock characteristics

### Use Case 4: Cryptocurrency Market Analysis

**Question**: Do on-chain metrics causally predict returns, or are they confounded by market sentiment?

**Setup**:
- Treatment: On-chain activity exceeds threshold
- Outcome: 24h forward return
- Covariates: BTC correlation, volume profile, funding rate, social sentiment

---

## Implementation in Python

### Propensity Score Model

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from typing import Optional, Tuple, List, Dict

class PropensityScoreEstimator:
    """
    Propensity Score Estimation using various methods.

    Methods:
    - logistic: Logistic regression
    - gbm: Gradient Boosting Machine
    - neural: Neural network
    """

    def __init__(self, method: str = 'gbm'):
        self.method = method
        self.model = None

    def fit(self, X: np.ndarray, T: np.ndarray) -> 'PropensityScoreEstimator':
        """Fit propensity score model."""
        if self.method == 'logistic':
            self.model = LogisticRegression(max_iter=1000, C=1.0)
            self.model.fit(X, T)
        elif self.method == 'gbm':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8
            )
            self.model.fit(X, T)
        elif self.method == 'neural':
            self.model = self._fit_neural(X, T)
        return self

    def _fit_neural(self, X: np.ndarray, T: np.ndarray, epochs: int = 100):
        """Fit neural network propensity model."""
        X_tensor = torch.tensor(X, dtype=torch.float32)
        T_tensor = torch.tensor(T, dtype=torch.float32).unsqueeze(1)

        model = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        for _ in range(epochs):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, T_tensor)
            loss.backward()
            optimizer.step()

        return model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict propensity scores."""
        if self.method in ['logistic', 'gbm']:
            return self.model.predict_proba(X)[:, 1]
        elif self.method == 'neural':
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                return self.model(X_tensor).numpy().flatten()


class PropensityScoreMatcher:
    """
    Propensity Score Matching and Treatment Effect Estimation.
    """

    def __init__(
        self,
        method: str = 'nearest',
        caliper: Optional[float] = None,
        replacement: bool = True
    ):
        self.method = method
        self.caliper = caliper
        self.replacement = replacement
        self.matches_ = None

    def match(
        self,
        propensity_scores: np.ndarray,
        treatment: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform propensity score matching.

        Returns:
            matched_treated: Indices of matched treated units
            matched_control: Indices of matched control units
        """
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]

        ps_treated = propensity_scores[treated_idx]
        ps_control = propensity_scores[control_idx]

        matched_treated = []
        matched_control = []
        available_controls = set(range(len(control_idx)))

        for i, ps_t in enumerate(ps_treated):
            if not available_controls:
                break

            # Find distances to available controls
            ctrl_list = list(available_controls)
            distances = np.abs(ps_control[ctrl_list] - ps_t)

            # Apply caliper
            if self.caliper is not None:
                valid = distances <= self.caliper
                if not valid.any():
                    continue
                distances = np.where(valid, distances, np.inf)

            best_idx = ctrl_list[np.argmin(distances)]

            matched_treated.append(treated_idx[i])
            matched_control.append(control_idx[best_idx])

            if not self.replacement:
                available_controls.remove(best_idx)

        self.matches_ = (np.array(matched_treated), np.array(matched_control))
        return self.matches_

    def estimate_ate(
        self,
        outcomes: np.ndarray,
        propensity_scores: np.ndarray,
        treatment: np.ndarray,
        method: str = 'matching'
    ) -> Dict[str, float]:
        """
        Estimate Average Treatment Effect.

        Args:
            outcomes: Outcome variable (e.g., returns)
            propensity_scores: Estimated propensity scores
            treatment: Treatment indicator
            method: 'matching', 'ipw', or 'aipw'

        Returns:
            Dictionary with ATE estimate and standard error
        """
        if method == 'matching':
            return self._ate_matching(outcomes, treatment)
        elif method == 'ipw':
            return self._ate_ipw(outcomes, propensity_scores, treatment)
        elif method == 'aipw':
            return self._ate_aipw(outcomes, propensity_scores, treatment)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _ate_matching(
        self,
        outcomes: np.ndarray,
        treatment: np.ndarray
    ) -> Dict[str, float]:
        """ATE via matching."""
        if self.matches_ is None:
            raise ValueError("Must call match() first")

        matched_t, matched_c = self.matches_
        y_treated = outcomes[matched_t]
        y_control = outcomes[matched_c]

        ate = np.mean(y_treated - y_control)
        se = np.std(y_treated - y_control) / np.sqrt(len(matched_t))

        return {
            'ate': ate,
            'se': se,
            'ci_lower': ate - 1.96 * se,
            'ci_upper': ate + 1.96 * se,
            'n_matched': len(matched_t)
        }

    def _ate_ipw(
        self,
        outcomes: np.ndarray,
        ps: np.ndarray,
        treatment: np.ndarray
    ) -> Dict[str, float]:
        """ATE via Inverse Probability Weighting."""
        # Clip propensity scores for stability
        ps = np.clip(ps, 0.01, 0.99)

        # IPW estimator
        weighted_treated = treatment * outcomes / ps
        weighted_control = (1 - treatment) * outcomes / (1 - ps)

        ate = np.mean(weighted_treated) - np.mean(weighted_control)

        # Variance via influence function
        influence = (
            treatment * (outcomes - np.mean(weighted_treated)) / ps -
            (1 - treatment) * (outcomes - np.mean(weighted_control)) / (1 - ps)
        )
        se = np.std(influence) / np.sqrt(len(outcomes))

        return {
            'ate': ate,
            'se': se,
            'ci_lower': ate - 1.96 * se,
            'ci_upper': ate + 1.96 * se,
            'n': len(outcomes)
        }

    def _ate_aipw(
        self,
        outcomes: np.ndarray,
        ps: np.ndarray,
        treatment: np.ndarray
    ) -> Dict[str, float]:
        """ATE via Augmented IPW (Doubly Robust)."""
        ps = np.clip(ps, 0.01, 0.99)

        # Fit outcome models
        from sklearn.linear_model import Ridge
        X_with_t = np.column_stack([ps, treatment])

        # E[Y|T=1]
        mask_t = treatment == 1
        model_1 = Ridge(alpha=1.0)
        model_1.fit(ps[mask_t].reshape(-1, 1), outcomes[mask_t])
        mu_1 = model_1.predict(ps.reshape(-1, 1))

        # E[Y|T=0]
        mask_c = treatment == 0
        model_0 = Ridge(alpha=1.0)
        model_0.fit(ps[mask_c].reshape(-1, 1), outcomes[mask_c])
        mu_0 = model_0.predict(ps.reshape(-1, 1))

        # AIPW estimator
        aipw_treated = mu_1 + treatment * (outcomes - mu_1) / ps
        aipw_control = mu_0 + (1 - treatment) * (outcomes - mu_0) / (1 - ps)

        ate = np.mean(aipw_treated - aipw_control)

        # Variance
        influence = aipw_treated - aipw_control - ate
        se = np.std(influence) / np.sqrt(len(outcomes))

        return {
            'ate': ate,
            'se': se,
            'ci_lower': ate - 1.96 * se,
            'ci_upper': ate + 1.96 * se,
            'n': len(outcomes)
        }


class CausalTradingStrategy:
    """
    Trading strategy that uses propensity score methods
    to evaluate and adjust signal effectiveness.
    """

    def __init__(
        self,
        ps_method: str = 'gbm',
        ate_method: str = 'aipw',
        min_ate: float = 0.0,
        confidence_threshold: float = 0.95
    ):
        self.ps_estimator = PropensityScoreEstimator(method=ps_method)
        self.matcher = PropensityScoreMatcher()
        self.ate_method = ate_method
        self.min_ate = min_ate
        self.confidence_threshold = confidence_threshold
        self.ate_history_ = []

    def fit(
        self,
        features: np.ndarray,
        signals: np.ndarray,
        returns: np.ndarray
    ) -> 'CausalTradingStrategy':
        """
        Fit the causal trading strategy.

        Args:
            features: Confounding covariates (market conditions)
            signals: Trading signals (treatment)
            returns: Realized returns (outcome)
        """
        # Estimate propensity scores
        self.ps_estimator.fit(features, signals)
        ps = self.ps_estimator.predict(features)

        # Match and estimate ATE
        self.matcher.match(ps, signals)
        self.ate_result_ = self.matcher.estimate_ate(
            returns, ps, signals, method=self.ate_method
        )

        self.ate_history_.append(self.ate_result_)
        return self

    def should_trade(self) -> bool:
        """Determine if signal has statistically significant positive effect."""
        if not hasattr(self, 'ate_result_'):
            return False

        ate = self.ate_result_['ate']
        ci_lower = self.ate_result_['ci_lower']

        # Trade if ATE is positive and confidence interval doesn't include 0
        return ate > self.min_ate and ci_lower > 0

    def get_signal_strength(self) -> float:
        """Get causal signal strength (ATE normalized by SE)."""
        if not hasattr(self, 'ate_result_'):
            return 0.0

        ate = self.ate_result_['ate']
        se = self.ate_result_['se']

        return ate / (se + 1e-10)  # t-statistic
```

### Data Loader

```python
import pandas as pd
import numpy as np
import requests
from typing import Optional, Tuple

class TradingDataLoader:
    """Data loader for stock and cryptocurrency data with feature engineering."""

    def __init__(self, window: int = 60):
        self.window = window

    def load_stock_data(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Load stock data using yfinance."""
        import yfinance as yf
        data = yf.download(symbol, start=start, end=end)
        data.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in data.columns]
        return data

    def fetch_bybit_data(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "D",
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch kline data from Bybit API."""
        url = "https://api.bybit.com/v5/market/kline"
        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        resp = requests.get(url, params=params).json()

        if resp['retCode'] != 0:
            raise ValueError(f"Bybit API error: {resp['retMsg']}")

        records = resp['result']['list']
        df = pd.DataFrame(records, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df['open_time'] = pd.to_datetime(df['open_time'].astype(int), unit='ms')
        df = df.sort_values('open_time').reset_index(drop=True)

        return df

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute trading features for propensity score estimation."""
        features = pd.DataFrame(index=df.index)

        # Price-based features
        features['returns'] = np.log(df['close'] / df['close'].shift(1))
        features['volatility_20'] = features['returns'].rolling(20).std()
        features['volatility_60'] = features['returns'].rolling(60).std()

        # Momentum
        features['momentum_5'] = df['close'].pct_change(5)
        features['momentum_20'] = df['close'].pct_change(20)

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        features['rsi'] = 100 - 100 / (1 + gain / (loss + 1e-10))

        # Volume features
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_std'] = df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()

        # Range features
        features['range'] = (df['high'] - df['low']) / df['close']
        features['range_ratio'] = features['range'] / features['range'].rolling(20).mean()

        # Moving average features
        features['ma_ratio_20'] = df['close'] / df['close'].rolling(20).mean()
        features['ma_ratio_60'] = df['close'] / df['close'].rolling(60).mean()

        # Trend strength
        features['adx'] = self._compute_adx(df)

        return features

    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute ADX indicator."""
        high = df['high']
        low = df['low']
        close = df['close']

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()

        return adx

    def generate_signal(
        self,
        df: pd.DataFrame,
        signal_type: str = 'momentum'
    ) -> pd.Series:
        """Generate trading signal (treatment indicator)."""
        if signal_type == 'momentum':
            momentum = df['close'].pct_change(20)
            signal = (momentum > momentum.rolling(60).mean()).astype(int)
        elif signal_type == 'mean_reversion':
            zscore = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
            signal = (zscore < -1).astype(int)
        elif signal_type == 'volatility_breakout':
            volatility = np.log(df['close'] / df['close'].shift(1)).rolling(20).std()
            signal = (volatility > volatility.rolling(60).quantile(0.8)).astype(int)
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")

        return signal

    def prepare_causal_dataset(
        self,
        df: pd.DataFrame,
        signal_type: str = 'momentum',
        horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare dataset for causal inference.

        Returns:
            features: Confounding covariates (X)
            treatment: Signal indicator (T)
            outcome: Future returns (Y)
        """
        features = self.compute_features(df)
        treatment = self.generate_signal(df, signal_type)

        # Forward returns as outcome
        outcome = np.log(df['close'].shift(-horizon) / df['close'])

        # Align and drop NaN
        combined = pd.concat([features, treatment.rename('treatment'),
                            outcome.rename('outcome')], axis=1)
        combined = combined.dropna()

        X = combined.drop(['treatment', 'outcome'], axis=1).values
        T = combined['treatment'].values
        Y = combined['outcome'].values

        return X, T, Y
```

### Backtesting Engine

```python
import numpy as np
import pandas as pd
from typing import Dict, Optional

class PropensityScoreBacktester:
    """
    Backtesting framework for propensity score trading strategies.

    This backtester evaluates strategies that use causal inference
    to adjust position sizing based on estimated treatment effects.
    """

    def __init__(
        self,
        strategy: 'CausalTradingStrategy',
        lookback: int = 252,
        refit_freq: int = 20,
        transaction_cost: float = 0.001
    ):
        self.strategy = strategy
        self.lookback = lookback
        self.refit_freq = refit_freq
        self.transaction_cost = transaction_cost

    def run(
        self,
        features: np.ndarray,
        signals: np.ndarray,
        returns: np.ndarray,
        prices: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Run backtest with periodic strategy refitting.

        Args:
            features: Feature matrix for propensity estimation
            signals: Raw trading signals
            returns: Realized returns
            prices: Price series (optional, for P&L calculation)

        Returns:
            Dictionary of performance metrics
        """
        n = len(returns)
        positions = np.zeros(n)
        strategy_returns = np.zeros(n)
        ate_estimates = []

        for t in range(self.lookback, n):
            # Refit strategy periodically
            if (t - self.lookback) % self.refit_freq == 0:
                train_slice = slice(t - self.lookback, t)
                self.strategy.fit(
                    features[train_slice],
                    signals[train_slice],
                    returns[train_slice]
                )
                ate_estimates.append({
                    'time': t,
                    **self.strategy.ate_result_
                })

            # Generate position
            if self.strategy.should_trade():
                # Scale position by signal strength (t-statistic)
                strength = self.strategy.get_signal_strength()
                position = np.sign(strength) * min(abs(strength) / 2, 1.0)
                positions[t] = position * signals[t]
            else:
                positions[t] = 0

        # Calculate returns with transaction costs
        position_changes = np.diff(positions, prepend=0)
        costs = np.abs(position_changes) * self.transaction_cost
        strategy_returns = positions * returns - costs

        # Compute metrics
        metrics = self._compute_metrics(strategy_returns, positions)
        metrics['ate_estimates'] = ate_estimates

        return metrics

    def _compute_metrics(
        self,
        returns: np.ndarray,
        positions: np.ndarray
    ) -> Dict[str, float]:
        """Compute performance metrics."""
        # Remove initial zeros
        start_idx = np.argmax(positions != 0) if (positions != 0).any() else len(returns)
        returns = returns[start_idx:]
        positions = positions[start_idx:]

        if len(returns) == 0:
            return {'error': 'No trades executed'}

        # Total and annual returns
        total_return = np.exp(np.sum(np.log1p(returns))) - 1
        n_periods = len(returns)
        annual_return = (1 + total_return) ** (252 / max(n_periods, 1)) - 1

        # Volatility
        annual_vol = np.std(returns) * np.sqrt(252)

        # Sharpe ratio
        sharpe = annual_return / (annual_vol + 1e-10)

        # Max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Sortino ratio
        downside = returns[returns < 0]
        downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 1e-10
        sortino = annual_return / downside_vol

        # Calmar ratio
        calmar = annual_return / (abs(max_drawdown) + 1e-10)

        # Win rate
        active_returns = returns[positions != 0]
        win_rate = np.mean(active_returns > 0) if len(active_returns) > 0 else 0

        # Number of trades
        n_trades = np.sum(np.abs(np.diff(positions)) > 0.01)

        # Average position size
        avg_position = np.mean(np.abs(positions[positions != 0])) if (positions != 0).any() else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_trades': int(n_trades),
            'avg_position': avg_position,
            'n_periods': n_periods
        }

    def compare_with_naive(
        self,
        features: np.ndarray,
        signals: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare causal strategy with naive signal-following.

        Returns metrics for both strategies for comparison.
        """
        # Causal strategy
        causal_metrics = self.run(features, signals, returns)

        # Naive strategy: just follow signals
        naive_returns = signals[self.lookback:] * returns[self.lookback:]
        naive_positions = signals[self.lookback:]
        naive_metrics = self._compute_metrics(naive_returns, naive_positions)

        return {
            'causal': causal_metrics,
            'naive': naive_metrics
        }
```

---

## Implementation in Rust

### Project Structure

```
107_propensity_score_trading/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── model/
│   │   ├── mod.rs
│   │   ├── propensity.rs
│   │   └── matching.rs
│   ├── data/
│   │   ├── mod.rs
│   │   ├── features.rs
│   │   └── bybit.rs
│   ├── backtest/
│   │   ├── mod.rs
│   │   └── engine.rs
│   └── trading/
│       ├── mod.rs
│       └── strategy.rs
└── examples/
    ├── basic_propensity.rs
    ├── causal_trading.rs
    └── backtest_strategy.rs
```

### Core Propensity Score Implementation (Rust)

```rust
use std::collections::HashMap;

/// Logistic regression for propensity score estimation.
pub struct LogisticRegression {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    max_iter: usize,
}

impl LogisticRegression {
    pub fn new(n_features: usize) -> Self {
        LogisticRegression {
            weights: vec![0.0; n_features],
            bias: 0.0,
            learning_rate: 0.01,
            max_iter: 1000,
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        let n = x.len();
        let n_features = self.weights.len();

        for _ in 0..self.max_iter {
            let mut grad_w = vec![0.0; n_features];
            let mut grad_b = 0.0;

            for i in 0..n {
                let pred = self.predict_proba_single(&x[i]);
                let error = pred - y[i];

                for j in 0..n_features {
                    grad_w[j] += error * x[i][j] / n as f64;
                }
                grad_b += error / n as f64;
            }

            // Update weights
            for j in 0..n_features {
                self.weights[j] -= self.learning_rate * grad_w[j];
            }
            self.bias -= self.learning_rate * grad_b;
        }
    }

    pub fn predict_proba_single(&self, x: &[f64]) -> f64 {
        let z: f64 = self.weights.iter()
            .zip(x.iter())
            .map(|(w, xi)| w * xi)
            .sum::<f64>() + self.bias;
        sigmoid(z)
    }

    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|xi| self.predict_proba_single(xi)).collect()
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Propensity Score Matcher for causal inference.
pub struct PropensityMatcher {
    caliper: Option<f64>,
    replacement: bool,
}

impl PropensityMatcher {
    pub fn new(caliper: Option<f64>, replacement: bool) -> Self {
        PropensityMatcher { caliper, replacement }
    }

    /// Match treated units to control units based on propensity scores.
    pub fn match_units(
        &self,
        propensity_scores: &[f64],
        treatment: &[u8],
    ) -> Vec<(usize, usize)> {
        let treated: Vec<usize> = treatment.iter()
            .enumerate()
            .filter(|(_, &t)| t == 1)
            .map(|(i, _)| i)
            .collect();

        let control: Vec<usize> = treatment.iter()
            .enumerate()
            .filter(|(_, &t)| t == 0)
            .map(|(i, _)| i)
            .collect();

        let mut matches = Vec::new();
        let mut available: std::collections::HashSet<usize> =
            control.iter().cloned().collect();

        for &t_idx in &treated {
            if available.is_empty() {
                break;
            }

            let ps_t = propensity_scores[t_idx];
            let mut best_match: Option<(usize, f64)> = None;

            for &c_idx in &available {
                let distance = (propensity_scores[c_idx] - ps_t).abs();

                // Apply caliper
                if let Some(cal) = self.caliper {
                    if distance > cal {
                        continue;
                    }
                }

                match best_match {
                    None => best_match = Some((c_idx, distance)),
                    Some((_, best_dist)) if distance < best_dist => {
                        best_match = Some((c_idx, distance));
                    }
                    _ => {}
                }
            }

            if let Some((c_idx, _)) = best_match {
                matches.push((t_idx, c_idx));
                if !self.replacement {
                    available.remove(&c_idx);
                }
            }
        }

        matches
    }

    /// Estimate Average Treatment Effect from matched pairs.
    pub fn estimate_ate_matched(
        &self,
        outcomes: &[f64],
        matches: &[(usize, usize)],
    ) -> ATEResult {
        let n = matches.len();
        if n == 0 {
            return ATEResult {
                ate: 0.0,
                se: f64::INFINITY,
                ci_lower: f64::NEG_INFINITY,
                ci_upper: f64::INFINITY,
                n_matched: 0,
            };
        }

        let diffs: Vec<f64> = matches.iter()
            .map(|(t_idx, c_idx)| outcomes[*t_idx] - outcomes[*c_idx])
            .collect();

        let ate = diffs.iter().sum::<f64>() / n as f64;
        let variance = diffs.iter()
            .map(|d| (d - ate).powi(2))
            .sum::<f64>() / (n - 1).max(1) as f64;
        let se = (variance / n as f64).sqrt();

        ATEResult {
            ate,
            se,
            ci_lower: ate - 1.96 * se,
            ci_upper: ate + 1.96 * se,
            n_matched: n,
        }
    }

    /// Estimate ATE using Inverse Probability Weighting.
    pub fn estimate_ate_ipw(
        &self,
        outcomes: &[f64],
        propensity_scores: &[f64],
        treatment: &[u8],
    ) -> ATEResult {
        let n = outcomes.len();
        let mut weighted_treated = 0.0;
        let mut weighted_control = 0.0;
        let mut n_treated = 0;
        let mut n_control = 0;

        for i in 0..n {
            let ps = propensity_scores[i].clamp(0.01, 0.99);
            let y = outcomes[i];

            if treatment[i] == 1 {
                weighted_treated += y / ps;
                n_treated += 1;
            } else {
                weighted_control += y / (1.0 - ps);
                n_control += 1;
            }
        }

        let mean_treated = if n_treated > 0 { weighted_treated / n_treated as f64 } else { 0.0 };
        let mean_control = if n_control > 0 { weighted_control / n_control as f64 } else { 0.0 };
        let ate = mean_treated - mean_control;

        // Simplified variance estimation
        let se = (1.0 / n_treated.max(1) as f64 + 1.0 / n_control.max(1) as f64).sqrt() * 0.1;

        ATEResult {
            ate,
            se,
            ci_lower: ate - 1.96 * se,
            ci_upper: ate + 1.96 * se,
            n_matched: n,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ATEResult {
    pub ate: f64,
    pub se: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub n_matched: usize,
}

impl ATEResult {
    pub fn is_significant(&self, alpha: f64) -> bool {
        let z = 1.96; // For alpha = 0.05
        self.ate.abs() > z * self.se
    }
}
```

### Trading Strategy (Rust)

```rust
use crate::model::{LogisticRegression, PropensityMatcher, ATEResult};

/// Causal trading strategy using propensity scores.
pub struct CausalTradingStrategy {
    ps_model: LogisticRegression,
    matcher: PropensityMatcher,
    min_ate: f64,
    current_ate: Option<ATEResult>,
}

impl CausalTradingStrategy {
    pub fn new(n_features: usize, min_ate: f64) -> Self {
        CausalTradingStrategy {
            ps_model: LogisticRegression::new(n_features),
            matcher: PropensityMatcher::new(Some(0.1), true),
            min_ate,
            current_ate: None,
        }
    }

    /// Fit the strategy on historical data.
    pub fn fit(
        &mut self,
        features: &[Vec<f64>],
        signals: &[u8],
        returns: &[f64],
    ) {
        // Convert signals to f64 for training
        let signals_f64: Vec<f64> = signals.iter().map(|&s| s as f64).collect();

        // Fit propensity model
        self.ps_model.fit(features, &signals_f64);

        // Estimate propensity scores
        let ps = self.ps_model.predict_proba(features);

        // Match and estimate ATE
        let matches = self.matcher.match_units(&ps, signals);
        self.current_ate = Some(self.matcher.estimate_ate_matched(returns, &matches));
    }

    /// Determine if we should trade based on causal effect.
    pub fn should_trade(&self) -> bool {
        match &self.current_ate {
            Some(ate) => ate.ate > self.min_ate && ate.ci_lower > 0.0,
            None => false,
        }
    }

    /// Get the signal strength (t-statistic).
    pub fn signal_strength(&self) -> f64 {
        match &self.current_ate {
            Some(ate) => ate.ate / (ate.se + 1e-10),
            None => 0.0,
        }
    }

    /// Generate trading position.
    pub fn generate_position(&self, signal: u8) -> f64 {
        if !self.should_trade() {
            return 0.0;
        }

        let strength = self.signal_strength();
        let scale = (strength.abs() / 2.0).min(1.0);

        if signal == 1 {
            scale * strength.signum()
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone)]
pub struct TradingSignal {
    pub timestamp: i64,
    pub direction: SignalDirection,
    pub strength: f64,
    pub causal_effect: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SignalDirection {
    Long,
    Short,
    Neutral,
}
```

### Bybit Data Fetcher (Rust)

```rust
use serde::Deserialize;
use std::error::Error;

#[derive(Debug, Deserialize)]
pub struct BybitKline {
    pub open_time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Fetch kline data from Bybit public API.
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<BybitKline>, Box<dyn Error>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let resp: BybitResponse = reqwest::get(&url).await?.json().await?;

    if resp.ret_code != 0 {
        return Err(format!("Bybit API error: {}", resp.ret_msg).into());
    }

    let klines: Vec<BybitKline> = resp.result.list
        .iter()
        .filter_map(|arr| {
            if arr.len() < 6 {
                return None;
            }
            Some(BybitKline {
                open_time: arr[0].parse().ok()?,
                open: arr[1].parse().ok()?,
                high: arr[2].parse().ok()?,
                low: arr[3].parse().ok()?,
                close: arr[4].parse().ok()?,
                volume: arr[5].parse().ok()?,
            })
        })
        .collect();

    Ok(klines)
}

/// Compute trading features from kline data.
pub fn compute_features(klines: &[BybitKline]) -> Vec<Vec<f64>> {
    let n = klines.len();
    let mut features = Vec::new();

    for i in 60..n {
        let mut f = Vec::new();

        // Returns
        let ret = (klines[i].close / klines[i - 1].close).ln();
        f.push(ret);

        // 20-period volatility
        let rets: Vec<f64> = (i - 20..i)
            .map(|j| (klines[j].close / klines[j - 1].close).ln())
            .collect();
        let vol_20 = std_dev(&rets);
        f.push(vol_20);

        // 20-period momentum
        let momentum_20 = (klines[i].close / klines[i - 20].close) - 1.0;
        f.push(momentum_20);

        // Volume ratio
        let avg_vol: f64 = (i - 20..i).map(|j| klines[j].volume).sum::<f64>() / 20.0;
        let vol_ratio = klines[i].volume / (avg_vol + 1e-10);
        f.push(vol_ratio);

        // Price range
        let range = (klines[i].high - klines[i].low) / klines[i].close;
        f.push(range);

        // RSI (simplified)
        let gains: f64 = (i - 14..i)
            .filter_map(|j| {
                let d = klines[j].close - klines[j - 1].close;
                if d > 0.0 { Some(d) } else { None }
            })
            .sum();
        let losses: f64 = (i - 14..i)
            .filter_map(|j| {
                let d = klines[j - 1].close - klines[j].close;
                if d > 0.0 { Some(d) } else { None }
            })
            .sum();
        let rsi = gains / (gains + losses + 1e-10);
        f.push(rsi);

        features.push(f);
    }

    features
}

fn std_dev(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    variance.sqrt()
}

/// Generate momentum signal.
pub fn generate_momentum_signal(klines: &[BybitKline], lookback: usize) -> Vec<u8> {
    let n = klines.len();
    let mut signals = vec![0u8; n];

    for i in lookback..n {
        let momentum = klines[i].close / klines[i - lookback].close - 1.0;
        let avg_momentum: f64 = (i - 60..i)
            .map(|j| klines[j].close / klines[j - lookback].close - 1.0)
            .sum::<f64>() / 60.0;

        if momentum > avg_momentum {
            signals[i] = 1;
        }
    }

    signals
}
```

---

## Practical Examples with Stock and Crypto Data

### Example 1: Evaluating Momentum Signal on Stocks (Python)

```python
import numpy as np
from propensity_score import (
    TradingDataLoader,
    PropensityScoreEstimator,
    PropensityScoreMatcher,
    CausalTradingStrategy,
    PropensityScoreBacktester
)

# Load stock data
loader = TradingDataLoader()
data = loader.load_stock_data('AAPL', '2018-01-01', '2024-01-01')

# Prepare causal dataset
X, T, Y = loader.prepare_causal_dataset(data, signal_type='momentum', horizon=1)

print(f"Dataset: {len(Y)} observations")
print(f"Treatment rate: {np.mean(T):.2%}")

# Estimate propensity scores
ps_estimator = PropensityScoreEstimator(method='gbm')
ps_estimator.fit(X, T)
ps = ps_estimator.predict(X)

# Match and estimate ATE
matcher = PropensityScoreMatcher(caliper=0.1)
matcher.match(ps, T)

# Compare estimation methods
methods = ['matching', 'ipw', 'aipw']
for method in methods:
    result = matcher.estimate_ate(Y, ps, T, method=method)
    print(f"\n{method.upper()} Results:")
    print(f"  ATE: {result['ate']:.4f} (SE: {result['se']:.4f})")
    print(f"  95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    sig = "Significant" if result['ci_lower'] > 0 or result['ci_upper'] < 0 else "Not Significant"
    print(f"  {sig}")
```

### Example 2: Crypto Trading with Bybit Data (Python)

```python
# Fetch BTC and ETH data
btc_data = loader.fetch_bybit_data("BTCUSDT", "D", 1000)
eth_data = loader.fetch_bybit_data("ETHUSDT", "D", 1000)

# Prepare datasets
X_btc, T_btc, Y_btc = loader.prepare_causal_dataset(btc_data, signal_type='momentum')
X_eth, T_eth, Y_eth = loader.prepare_causal_dataset(eth_data, signal_type='momentum')

# Initialize causal strategy
strategy = CausalTradingStrategy(ps_method='gbm', ate_method='aipw')

# Train on BTC
strategy.fit(X_btc, T_btc, Y_btc)
print(f"\nBTC Momentum Signal Causal Effect:")
print(f"  ATE: {strategy.ate_result_['ate']:.4f}")
print(f"  Should trade: {strategy.should_trade()}")
print(f"  Signal strength: {strategy.get_signal_strength():.2f}")

# Cross-validate on ETH
strategy_eth = CausalTradingStrategy(ps_method='gbm', ate_method='aipw')
strategy_eth.fit(X_eth, T_eth, Y_eth)
print(f"\nETH Momentum Signal Causal Effect:")
print(f"  ATE: {strategy_eth.ate_result_['ate']:.4f}")
print(f"  Should trade: {strategy_eth.should_trade()}")

# Backtest causal strategy
backtester = PropensityScoreBacktester(strategy, lookback=252, refit_freq=20)
comparison = backtester.compare_with_naive(X_btc, T_btc, Y_btc)

print("\n=== Backtest Comparison ===")
print(f"\nCausal Strategy:")
print(f"  Sharpe: {comparison['causal']['sharpe_ratio']:.2f}")
print(f"  Max DD: {comparison['causal']['max_drawdown']:.2%}")
print(f"  Win Rate: {comparison['causal']['win_rate']:.2%}")

print(f"\nNaive Strategy:")
print(f"  Sharpe: {comparison['naive']['sharpe_ratio']:.2f}")
print(f"  Max DD: {comparison['naive']['max_drawdown']:.2%}")
print(f"  Win Rate: {comparison['naive']['win_rate']:.2%}")
```

### Example 3: Rust Trading Example

```rust
use propensity_score::{
    data::bybit::{fetch_bybit_klines, compute_features, generate_momentum_signal},
    model::{LogisticRegression, PropensityMatcher},
    trading::CausalTradingStrategy,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Fetch BTC data from Bybit
    let klines = fetch_bybit_klines("BTCUSDT", "D", 500).await?;
    println!("Fetched {} klines from Bybit", klines.len());

    // Compute features and signals
    let features = compute_features(&klines);
    let signals = generate_momentum_signal(&klines, 20);

    // Compute returns (outcome)
    let returns: Vec<f64> = klines.windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect();

    // Align data (features start at index 60)
    let start_idx = 60;
    let features = &features[..];
    let signals = &signals[start_idx..start_idx + features.len()];
    let returns = &returns[start_idx..start_idx + features.len()];

    // Initialize and fit causal strategy
    let mut strategy = CausalTradingStrategy::new(6, 0.0);
    strategy.fit(features, signals, returns);

    println!("\n=== Causal Analysis Results ===");
    println!("Should trade: {}", strategy.should_trade());
    println!("Signal strength: {:.2}", strategy.signal_strength());

    // Generate signals for recent data
    println!("\n=== Recent Trading Signals ===");
    for i in (features.len() - 5)..features.len() {
        let position = strategy.generate_position(signals[i]);
        let direction = if position > 0.0 { "LONG" }
                       else if position < 0.0 { "SHORT" }
                       else { "NEUTRAL" };
        println!(
            "Day {}: Signal={}, Position={:.2} ({})",
            i, signals[i], position, direction
        );
    }

    Ok(())
}
```

---

## Backtesting Framework

### Walk-Forward Validation with Causal Estimation

```python
def walk_forward_causal_backtest(
    data: pd.DataFrame,
    signal_type: str = 'momentum',
    train_period: int = 500,
    test_period: int = 60,
    refit_every: int = 60,
):
    """Walk-forward backtesting with causal estimation."""
    loader = TradingDataLoader()
    results = []

    for end in range(train_period + 60, len(data) - test_period, refit_every):
        # Training window
        train_data = data.iloc[:end]
        X_train, T_train, Y_train = loader.prepare_causal_dataset(
            train_data, signal_type=signal_type
        )

        # Fit causal strategy
        strategy = CausalTradingStrategy(ps_method='gbm', ate_method='aipw')
        strategy.fit(X_train[-train_period:], T_train[-train_period:], Y_train[-train_period:])

        # Test window
        test_data = data.iloc[end:end + test_period]
        X_test, T_test, Y_test = loader.prepare_causal_dataset(
            test_data, signal_type=signal_type
        )

        if len(Y_test) == 0:
            continue

        # Generate positions
        positions = np.zeros(len(Y_test))
        if strategy.should_trade():
            strength = strategy.get_signal_strength()
            positions = T_test * np.sign(strength) * min(abs(strength) / 2, 1.0)

        # Calculate returns
        test_returns = positions * Y_test

        results.append({
            'period_end': data.index[end] if hasattr(data, 'index') else end,
            'ate': strategy.ate_result_['ate'],
            'ate_se': strategy.ate_result_['se'],
            'ate_significant': strategy.ate_result_['ci_lower'] > 0,
            'traded': strategy.should_trade(),
            'return': np.sum(test_returns),
            'sharpe': np.mean(test_returns) / (np.std(test_returns) + 1e-10) * np.sqrt(252),
        })

    return pd.DataFrame(results)
```

### Sensitivity Analysis

```python
def sensitivity_analysis(X, T, Y, n_simulations: int = 100):
    """
    Assess sensitivity of ATE to unobserved confounding.

    Uses the approach of Rosenbaum bounds.
    """
    results = []

    # Baseline ATE
    ps_estimator = PropensityScoreEstimator(method='gbm')
    ps_estimator.fit(X, T)
    ps = ps_estimator.predict(X)
    matcher = PropensityScoreMatcher()
    matcher.match(ps, T)
    baseline = matcher.estimate_ate(Y, ps, T, method='aipw')

    # Simulate hidden confounders with varying strength
    for gamma in [1.0, 1.25, 1.5, 1.75, 2.0]:
        ate_samples = []

        for _ in range(n_simulations):
            # Simulate confounder effect
            noise = np.random.randn(len(Y)) * 0.01 * (gamma - 1)
            Y_perturbed = Y + noise * (2 * T - 1)

            result = matcher.estimate_ate(Y_perturbed, ps, T, method='aipw')
            ate_samples.append(result['ate'])

        results.append({
            'gamma': gamma,
            'ate_mean': np.mean(ate_samples),
            'ate_std': np.std(ate_samples),
            'ate_lower': np.percentile(ate_samples, 2.5),
            'ate_upper': np.percentile(ate_samples, 97.5),
        })

    return pd.DataFrame(results)
```

---

## Performance Evaluation

### Metrics Summary

| Metric | Description | Target |
|--------|-------------|--------|
| ATE | Average Treatment Effect (causal signal strength) | > 0 with CI excluding 0 |
| Sharpe Ratio | Risk-adjusted return (annualized) | > 1.0 |
| Sortino Ratio | Downside risk-adjusted return | > 1.5 |
| Max Drawdown | Largest peak-to-trough decline | > -20% |
| Win Rate | Fraction of profitable trades | > 52% |
| Calmar Ratio | Annual return / Max drawdown | > 0.5 |

### Covariate Balance Diagnostics

Before trusting causal estimates, verify that matching achieves balance:

```python
def assess_balance(X, T, ps, feature_names=None):
    """Assess covariate balance before and after matching."""
    if feature_names is None:
        feature_names = [f'X{i}' for i in range(X.shape[1])]

    # Before matching
    treated_mean = X[T == 1].mean(axis=0)
    control_mean = X[T == 0].mean(axis=0)
    pooled_std = np.sqrt((X[T == 1].var(axis=0) + X[T == 0].var(axis=0)) / 2)
    smd_before = (treated_mean - control_mean) / (pooled_std + 1e-10)

    # After matching (weighting by PS)
    weights_t = 1.0 / ps[T == 1]
    weights_c = 1.0 / (1 - ps[T == 0])

    weighted_mean_t = np.average(X[T == 1], weights=weights_t, axis=0)
    weighted_mean_c = np.average(X[T == 0], weights=weights_c, axis=0)
    smd_after = (weighted_mean_t - weighted_mean_c) / (pooled_std + 1e-10)

    balance = pd.DataFrame({
        'Feature': feature_names,
        'SMD_Before': smd_before,
        'SMD_After': smd_after,
        'Improvement': np.abs(smd_before) - np.abs(smd_after)
    })

    return balance
```

Good balance: |SMD| < 0.1 for all covariates after matching.

---

## Future Directions

1. **Continuous Treatments**: Extend to generalized propensity scores for position sizing optimization
2. **Double Machine Learning**: Use ML for both propensity and outcome models with cross-fitting for unbiased estimation
3. **Instrumental Variables**: Combine PSM with IV methods for stronger identification
4. **Heterogeneous Treatment Effects**: Estimate how signal effectiveness varies across market conditions
5. **Synthetic Control**: Use propensity scores to construct synthetic benchmarks for individual assets
6. **Meta-Learners**: Apply T-learner, S-learner, X-learner approaches for treatment effect heterogeneity
7. **Causal Discovery**: Automatically identify confounders from data using causal discovery algorithms

---

## References

1. Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41-55.
2. Rosenbaum, P. R., & Rubin, D. B. (1984). Reducing bias in observational studies using subclassification on the propensity score. *Journal of the American Statistical Association*, 79(387), 516-524.
3. Hirano, K., & Imbens, G. W. (2004). The propensity score with continuous treatments. *Applied Bayesian Modeling and Causal Inference from Incomplete-Data Perspectives*, 226164, 73-84.
4. Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression coefficients when some regressors are not always observed. *Journal of the American Statistical Association*, 89(427), 846-866.
5. Bang, H., & Robins, J. M. (2005). Doubly robust estimation in missing data and causal inference models. *Biometrics*, 61(4), 962-973.
6. Imbens, G. W. (2004). Nonparametric estimation of average treatment effects under exogeneity: A review. *Review of Economics and Statistics*, 86(1), 4-29.
7. Austin, P. C. (2011). An introduction to propensity score methods for reducing the effects of confounding in observational studies. *Multivariate Behavioral Research*, 46(3), 399-424.

---

## Running the Examples

### Python

```bash
cd 107_propensity_score_trading/python
pip install -r requirements.txt
python model.py          # Test propensity score models
python data_loader.py    # Test data loading and features
python backtest.py       # Run backtesting examples
```

### Rust

```bash
cd 107_propensity_score_trading
cargo build
cargo run --example basic_propensity
cargo run --example causal_trading
cargo run --example backtest_strategy
```
