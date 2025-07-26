"""
Propensity Score Models for Trading

This module implements various propensity score estimation and
causal effect estimation methods.
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal, Tuple, List, Union
from dataclasses import dataclass
from enum import Enum
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PropensityMethod(Enum):
    """Propensity score estimation methods."""
    LOGISTIC = "logistic"
    GRADIENT_BOOSTING = "gradient_boosting"
    RANDOM_FOREST = "random_forest"


class MatchingMethod(Enum):
    """Matching methods for propensity score matching."""
    NEAREST_NEIGHBOR = "nearest_neighbor"
    CALIPER = "caliper"
    STRATIFICATION = "stratification"


@dataclass
class MatchResult:
    """Result of propensity score matching."""
    treated_indices: np.ndarray
    control_indices: np.ndarray
    propensity_scores: np.ndarray
    standardized_mean_differences: dict
    balance_stats: dict


@dataclass
class CausalEstimate:
    """Result of causal effect estimation."""
    estimate: float
    std_error: float
    confidence_interval: Tuple[float, float]
    p_value: float
    method: str
    n_treated: int
    n_control: int
    balance_diagnostics: Optional[dict] = None


class PropensityScoreModel:
    """
    Propensity Score Model for Trading Applications

    Estimates the probability of treatment (e.g., signal firing) given
    observed covariates (market conditions).

    Example usage:
    ```python
    model = PropensityScoreModel(method='gradient_boosting')
    model.fit(X_covariates, treatment_indicator)
    propensity_scores = model.predict_proba(X_covariates)
    ```
    """

    def __init__(
        self,
        method: Union[str, PropensityMethod] = PropensityMethod.LOGISTIC,
        regularization: float = 1.0,
        max_depth: int = 3,
        n_estimators: int = 100,
        clip_bounds: Tuple[float, float] = (0.01, 0.99),
        random_state: int = 42,
    ):
        """
        Initialize the propensity score model.

        Args:
            method: Estimation method ('logistic', 'gradient_boosting', 'random_forest')
            regularization: Regularization strength for logistic regression
            max_depth: Max depth for tree-based methods
            n_estimators: Number of estimators for ensemble methods
            clip_bounds: Bounds to clip propensity scores (prevent extreme weights)
            random_state: Random seed for reproducibility
        """
        if isinstance(method, str):
            method = PropensityMethod(method)

        self.method = method
        self.regularization = regularization
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.clip_bounds = clip_bounds
        self.random_state = random_state

        self._model = None
        self._scaler = StandardScaler()
        self._is_fitted = False
        self._feature_names: List[str] = []

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        treatment: Union[pd.Series, np.ndarray]
    ) -> 'PropensityScoreModel':
        """
        Fit the propensity score model.

        Args:
            X: Covariate matrix (n_samples, n_features)
            treatment: Binary treatment indicator (n_samples,)

        Returns:
            self: Fitted model
        """
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X = X.values

        if isinstance(treatment, pd.Series):
            treatment = treatment.values

        # Validate inputs
        if X.shape[0] != len(treatment):
            raise ValueError("X and treatment must have same number of samples")

        if not np.all(np.isin(treatment, [0, 1])):
            raise ValueError("Treatment must be binary (0 or 1)")

        # Scale features
        X_scaled = self._scaler.fit_transform(X)

        # Initialize model based on method
        if self.method == PropensityMethod.LOGISTIC:
            self._model = LogisticRegression(
                C=self.regularization,
                max_iter=1000,
                random_state=self.random_state,
                solver='lbfgs'
            )
        elif self.method == PropensityMethod.GRADIENT_BOOSTING:
            self._model = GradientBoostingClassifier(
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                random_state=self.random_state
            )
        elif self.method == PropensityMethod.RANDOM_FOREST:
            self._model = RandomForestClassifier(
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Fit model
        self._model.fit(X_scaled, treatment)
        self._is_fitted = True

        logger.info(f"Propensity model fitted using {self.method.value}")

        return self

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict propensity scores (probability of treatment).

        Args:
            X: Covariate matrix

        Returns:
            Propensity scores clipped to bounds
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values

        X_scaled = self._scaler.transform(X)
        proba = self._model.predict_proba(X_scaled)[:, 1]

        # Clip to prevent extreme weights
        proba = np.clip(proba, self.clip_bounds[0], self.clip_bounds[1])

        return proba

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the propensity model."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        if self.method == PropensityMethod.LOGISTIC:
            importance = np.abs(self._model.coef_[0])
        elif self.method in (PropensityMethod.GRADIENT_BOOSTING, PropensityMethod.RANDOM_FOREST):
            importance = self._model.feature_importances_
        else:
            raise ValueError(f"Feature importance not supported for {self.method}")

        feature_names = self._feature_names if self._feature_names else [f"feature_{i}" for i in range(len(importance))]

        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


class PropensityMatcher:
    """
    Propensity Score Matching for Creating Balanced Comparison Groups

    Matches treated observations with control observations based on
    propensity scores to create balanced comparison groups.
    """

    def __init__(
        self,
        method: Union[str, MatchingMethod] = MatchingMethod.NEAREST_NEIGHBOR,
        n_neighbors: int = 1,
        caliper: Optional[float] = None,
        replacement: bool = False,
        n_strata: int = 5,
    ):
        """
        Initialize the matcher.

        Args:
            method: Matching method
            n_neighbors: Number of nearest neighbors for NN matching
            caliper: Maximum distance for caliper matching (in std dev of propensity score)
            replacement: Whether to match with replacement
            n_strata: Number of strata for stratification
        """
        if isinstance(method, str):
            method = MatchingMethod(method)

        self.method = method
        self.n_neighbors = n_neighbors
        self.caliper = caliper
        self.replacement = replacement
        self.n_strata = n_strata

    def match(
        self,
        propensity_scores: np.ndarray,
        treatment: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> MatchResult:
        """
        Perform propensity score matching.

        Args:
            propensity_scores: Estimated propensity scores
            treatment: Binary treatment indicator
            X: Original covariates (for balance diagnostics)

        Returns:
            MatchResult with matched indices and diagnostics
        """
        treatment = np.asarray(treatment).flatten()
        propensity_scores = np.asarray(propensity_scores).flatten()

        treated_mask = treatment == 1
        control_mask = treatment == 0

        treated_ps = propensity_scores[treated_mask]
        control_ps = propensity_scores[control_mask]

        treated_indices_original = np.where(treated_mask)[0]
        control_indices_original = np.where(control_mask)[0]

        if self.method == MatchingMethod.NEAREST_NEIGHBOR:
            matched_treated, matched_control = self._nn_matching(
                treated_ps, control_ps,
                treated_indices_original, control_indices_original
            )
        elif self.method == MatchingMethod.CALIPER:
            matched_treated, matched_control = self._caliper_matching(
                treated_ps, control_ps,
                treated_indices_original, control_indices_original
            )
        elif self.method == MatchingMethod.STRATIFICATION:
            matched_treated, matched_control = self._stratification(
                propensity_scores, treatment
            )
        else:
            raise ValueError(f"Unknown matching method: {self.method}")

        # Calculate balance diagnostics
        balance_stats = {}
        smd = {}

        if X is not None:
            X = np.asarray(X)
            matched_X_treated = X[matched_treated]
            matched_X_control = X[matched_control]

            for i in range(X.shape[1]):
                mean_treated = matched_X_treated[:, i].mean()
                mean_control = matched_X_control[:, i].mean()
                pooled_std = np.sqrt(
                    (matched_X_treated[:, i].var() + matched_X_control[:, i].var()) / 2
                )

                if pooled_std > 0:
                    smd[f'feature_{i}'] = (mean_treated - mean_control) / pooled_std
                else:
                    smd[f'feature_{i}'] = 0.0

            balance_stats['max_smd'] = max(abs(v) for v in smd.values())
            balance_stats['mean_smd'] = np.mean([abs(v) for v in smd.values()])

        return MatchResult(
            treated_indices=matched_treated,
            control_indices=matched_control,
            propensity_scores=propensity_scores,
            standardized_mean_differences=smd,
            balance_stats=balance_stats
        )

    def _nn_matching(
        self,
        treated_ps: np.ndarray,
        control_ps: np.ndarray,
        treated_indices: np.ndarray,
        control_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform nearest neighbor matching."""
        nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')
        nn.fit(control_ps.reshape(-1, 1))

        distances, indices = nn.kneighbors(treated_ps.reshape(-1, 1))

        matched_treated = []
        matched_control = []
        used_controls = set() if not self.replacement else None

        for i, (dist, idx) in enumerate(zip(distances, indices)):
            for j, (d, ctrl_idx) in enumerate(zip(dist, idx)):
                if self.caliper is not None:
                    ps_std = np.std(np.concatenate([treated_ps, control_ps]))
                    if d > self.caliper * ps_std:
                        continue

                if not self.replacement and ctrl_idx in used_controls:
                    continue

                matched_treated.append(treated_indices[i])
                matched_control.append(control_indices[ctrl_idx])

                if not self.replacement:
                    used_controls.add(ctrl_idx)
                break

        return np.array(matched_treated), np.array(matched_control)

    def _caliper_matching(
        self,
        treated_ps: np.ndarray,
        control_ps: np.ndarray,
        treated_indices: np.ndarray,
        control_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform caliper matching."""
        if self.caliper is None:
            # Default caliper: 0.2 * std of propensity score
            self.caliper = 0.2

        return self._nn_matching(
            treated_ps, control_ps,
            treated_indices, control_indices
        )

    def _stratification(
        self,
        propensity_scores: np.ndarray,
        treatment: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform stratification matching."""
        # Create strata based on propensity score quantiles
        strata_bounds = np.percentile(
            propensity_scores,
            np.linspace(0, 100, self.n_strata + 1)
        )

        matched_treated = []
        matched_control = []

        for i in range(self.n_strata):
            lower = strata_bounds[i]
            upper = strata_bounds[i + 1]

            in_stratum = (propensity_scores >= lower) & (propensity_scores <= upper)
            treated_in_stratum = in_stratum & (treatment == 1)
            control_in_stratum = in_stratum & (treatment == 0)

            treated_idx = np.where(treated_in_stratum)[0]
            control_idx = np.where(control_in_stratum)[0]

            # Add all pairs within stratum
            for t_idx in treated_idx:
                if len(control_idx) > 0:
                    matched_treated.append(t_idx)
                    # Random control from same stratum
                    matched_control.append(np.random.choice(control_idx))

        return np.array(matched_treated), np.array(matched_control)


class CausalEffectEstimator:
    """
    Base class for causal effect estimation.

    Estimates Average Treatment Effect (ATE), Average Treatment Effect
    on the Treated (ATT), and Average Treatment Effect on the Control (ATC).
    """

    def estimate_ate(
        self,
        outcome: np.ndarray,
        treatment: np.ndarray,
        propensity_scores: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
    ) -> CausalEstimate:
        """
        Estimate Average Treatment Effect.

        Args:
            outcome: Observed outcomes (e.g., returns)
            treatment: Binary treatment indicator
            propensity_scores: Estimated propensity scores
            X: Covariates

        Returns:
            CausalEstimate with point estimate and uncertainty
        """
        raise NotImplementedError("Subclasses must implement estimate_ate")


class IPWEstimator(CausalEffectEstimator):
    """
    Inverse Probability Weighting (IPW) Estimator

    Uses propensity scores to weight observations and estimate
    the average treatment effect.

    ATE_IPW = E[Y*T/e(X)] - E[Y*(1-T)/(1-e(X))]
    """

    def __init__(self, normalize_weights: bool = True):
        """
        Args:
            normalize_weights: Whether to normalize weights (stabilized IPW)
        """
        self.normalize_weights = normalize_weights

    def estimate_ate(
        self,
        outcome: np.ndarray,
        treatment: np.ndarray,
        propensity_scores: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> CausalEstimate:
        """Estimate ATE using Inverse Probability Weighting."""
        outcome = np.asarray(outcome).flatten()
        treatment = np.asarray(treatment).flatten()
        propensity_scores = np.asarray(propensity_scores).flatten()

        n = len(outcome)

        # Calculate weights
        w1 = treatment / propensity_scores  # Treated weights
        w0 = (1 - treatment) / (1 - propensity_scores)  # Control weights

        if self.normalize_weights:
            w1 = w1 / w1.sum() * treatment.sum()
            w0 = w0 / w0.sum() * (1 - treatment).sum()

        # Estimate potential outcomes
        y1_hat = np.sum(w1 * outcome) / treatment.sum()
        y0_hat = np.sum(w0 * outcome) / (1 - treatment).sum()

        ate = y1_hat - y0_hat

        # Bootstrap standard error
        n_bootstrap = 1000
        ate_bootstrap = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            y_b, t_b, ps_b = outcome[idx], treatment[idx], propensity_scores[idx]

            w1_b = t_b / ps_b
            w0_b = (1 - t_b) / (1 - ps_b)

            if self.normalize_weights:
                w1_sum = w1_b.sum()
                w0_sum = w0_b.sum()
                if w1_sum > 0:
                    w1_b = w1_b / w1_sum * t_b.sum()
                if w0_sum > 0:
                    w0_b = w0_b / w0_sum * (1 - t_b).sum()

            t_sum = t_b.sum()
            c_sum = (1 - t_b).sum()

            if t_sum > 0 and c_sum > 0:
                y1_b = np.sum(w1_b * y_b) / t_sum
                y0_b = np.sum(w0_b * y_b) / c_sum
                ate_bootstrap.append(y1_b - y0_b)

        std_error = np.std(ate_bootstrap) if ate_bootstrap else np.nan

        # Confidence interval
        ci_lower = ate - 1.96 * std_error
        ci_upper = ate + 1.96 * std_error

        # P-value (two-sided test against zero)
        if std_error > 0:
            z_score = ate / std_error
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            p_value = np.nan

        return CausalEstimate(
            estimate=ate,
            std_error=std_error,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            method="IPW",
            n_treated=int(treatment.sum()),
            n_control=int((1 - treatment).sum())
        )


class DoublyRobustEstimator(CausalEffectEstimator):
    """
    Doubly Robust (DR) Estimator

    Combines outcome regression with inverse probability weighting.
    Consistent if either the propensity model OR the outcome model
    is correctly specified (but not necessarily both).
    """

    def __init__(self, outcome_model=None):
        """
        Args:
            outcome_model: Model to predict outcomes (default: LinearRegression)
        """
        from sklearn.linear_model import Ridge
        self.outcome_model = outcome_model or Ridge(alpha=1.0)

    def estimate_ate(
        self,
        outcome: np.ndarray,
        treatment: np.ndarray,
        propensity_scores: np.ndarray,
        X: np.ndarray,
    ) -> CausalEstimate:
        """Estimate ATE using Doubly Robust method."""
        outcome = np.asarray(outcome).flatten()
        treatment = np.asarray(treatment).flatten()
        propensity_scores = np.asarray(propensity_scores).flatten()
        X = np.asarray(X)

        n = len(outcome)

        # Fit outcome models separately for treated and control
        from sklearn.linear_model import Ridge

        treated_mask = treatment == 1
        control_mask = treatment == 0

        model_treated = Ridge(alpha=1.0)
        model_control = Ridge(alpha=1.0)

        if treated_mask.sum() > 0:
            model_treated.fit(X[treated_mask], outcome[treated_mask])
        if control_mask.sum() > 0:
            model_control.fit(X[control_mask], outcome[control_mask])

        # Predict potential outcomes for all observations
        mu1 = model_treated.predict(X)  # E[Y(1)|X]
        mu0 = model_control.predict(X)  # E[Y(0)|X]

        # Doubly robust estimator
        ps = propensity_scores

        # AIPW for Y(1)
        y1_dr = mu1 + (treatment * (outcome - mu1)) / ps

        # AIPW for Y(0)
        y0_dr = mu0 + ((1 - treatment) * (outcome - mu0)) / (1 - ps)

        ate = np.mean(y1_dr - y0_dr)

        # Bootstrap standard error
        n_bootstrap = 1000
        ate_bootstrap = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            ate_bootstrap.append(np.mean(y1_dr[idx] - y0_dr[idx]))

        std_error = np.std(ate_bootstrap)

        ci_lower = ate - 1.96 * std_error
        ci_upper = ate + 1.96 * std_error

        z_score = ate / std_error if std_error > 0 else np.nan
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score))) if not np.isnan(z_score) else np.nan

        return CausalEstimate(
            estimate=ate,
            std_error=std_error,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            method="Doubly Robust (AIPW)",
            n_treated=int(treatment.sum()),
            n_control=int((1 - treatment).sum())
        )


class MatchingEstimator(CausalEffectEstimator):
    """
    Matching-based estimator for ATE.

    Uses propensity score matching to create balanced groups
    and estimates ATE as the mean difference in outcomes.
    """

    def __init__(self, matcher: Optional[PropensityMatcher] = None):
        """
        Args:
            matcher: PropensityMatcher instance (default: nearest neighbor)
        """
        self.matcher = matcher or PropensityMatcher()

    def estimate_ate(
        self,
        outcome: np.ndarray,
        treatment: np.ndarray,
        propensity_scores: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> CausalEstimate:
        """Estimate ATE using propensity score matching."""
        outcome = np.asarray(outcome).flatten()
        treatment = np.asarray(treatment).flatten()
        propensity_scores = np.asarray(propensity_scores).flatten()

        # Perform matching
        match_result = self.matcher.match(propensity_scores, treatment, X)

        # Get matched outcomes
        y_treated = outcome[match_result.treated_indices]
        y_control = outcome[match_result.control_indices]

        # ATE = mean difference in matched pairs
        ate = np.mean(y_treated - y_control)

        # Standard error using paired t-test
        diff = y_treated - y_control
        std_error = np.std(diff) / np.sqrt(len(diff))

        ci_lower = ate - 1.96 * std_error
        ci_upper = ate + 1.96 * std_error

        t_stat = ate / std_error if std_error > 0 else np.nan
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(diff)-1)) if not np.isnan(t_stat) else np.nan

        return CausalEstimate(
            estimate=ate,
            std_error=std_error,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            method="Propensity Score Matching",
            n_treated=len(match_result.treated_indices),
            n_control=len(match_result.control_indices),
            balance_diagnostics=match_result.balance_stats
        )


def check_overlap(
    propensity_scores: np.ndarray,
    treatment: np.ndarray,
    plot: bool = True,
) -> dict:
    """
    Check propensity score overlap between treated and control groups.

    Good overlap is essential for valid causal inference.

    Args:
        propensity_scores: Estimated propensity scores
        treatment: Binary treatment indicator
        plot: Whether to create visualization

    Returns:
        Dictionary with overlap statistics
    """
    ps_treated = propensity_scores[treatment == 1]
    ps_control = propensity_scores[treatment == 0]

    # Calculate overlap region
    common_min = max(ps_treated.min(), ps_control.min())
    common_max = min(ps_treated.max(), ps_control.max())

    # Proportion in common support
    treated_in_common = np.mean((ps_treated >= common_min) & (ps_treated <= common_max))
    control_in_common = np.mean((ps_control >= common_min) & (ps_control <= common_max))

    stats_dict = {
        'treated_mean': ps_treated.mean(),
        'treated_std': ps_treated.std(),
        'control_mean': ps_control.mean(),
        'control_std': ps_control.std(),
        'common_support_min': common_min,
        'common_support_max': common_max,
        'treated_in_common_support': treated_in_common,
        'control_in_common_support': control_in_common,
        'overlap_quality': 'good' if (treated_in_common > 0.9 and control_in_common > 0.9) else 'poor'
    }

    if plot:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(ps_treated, bins=30, alpha=0.5, label='Treated', density=True)
            ax.hist(ps_control, bins=30, alpha=0.5, label='Control', density=True)
            ax.axvline(common_min, color='red', linestyle='--', label='Common support')
            ax.axvline(common_max, color='red', linestyle='--')
            ax.set_xlabel('Propensity Score')
            ax.set_ylabel('Density')
            ax.set_title('Propensity Score Distribution: Overlap Check')
            ax.legend()
            plt.tight_layout()

            stats_dict['figure'] = fig
        except ImportError:
            logger.warning("matplotlib not available for plotting")

    return stats_dict


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate synthetic data
    n = 1000

    # Covariates (market conditions)
    volatility = np.random.exponential(0.2, n)
    volume = np.random.normal(100, 20, n)
    trend = np.random.normal(0, 0.1, n)

    X = np.column_stack([volatility, volume, trend])

    # Treatment probability depends on covariates (confounding)
    propensity_true = 1 / (1 + np.exp(-(-1 + 2*volatility - 0.01*volume + 5*trend)))
    treatment = np.random.binomial(1, propensity_true)

    # Outcome depends on treatment and covariates
    # True treatment effect = 0.02 (2% return boost)
    noise = np.random.normal(0, 0.05, n)
    outcome = 0.01 - 0.1*volatility + 0.001*volume + 0.5*trend + 0.02*treatment + noise

    # Estimate propensity scores
    ps_model = PropensityScoreModel(method='gradient_boosting')
    ps_model.fit(X, treatment)
    ps_estimated = ps_model.predict_proba(X)

    print("Feature Importance:")
    print(ps_model.get_feature_importance())

    # Check overlap
    overlap_stats = check_overlap(ps_estimated, treatment, plot=False)
    print(f"\nOverlap Quality: {overlap_stats['overlap_quality']}")

    # Estimate ATE using different methods
    print("\n--- Causal Effect Estimates ---")

    # IPW Estimator
    ipw = IPWEstimator()
    ipw_result = ipw.estimate_ate(outcome, treatment, ps_estimated)
    print(f"\nIPW Estimator:")
    print(f"  ATE: {ipw_result.estimate:.4f}")
    print(f"  95% CI: ({ipw_result.confidence_interval[0]:.4f}, {ipw_result.confidence_interval[1]:.4f})")
    print(f"  p-value: {ipw_result.p_value:.4f}")

    # Doubly Robust Estimator
    dr = DoublyRobustEstimator()
    dr_result = dr.estimate_ate(outcome, treatment, ps_estimated, X)
    print(f"\nDoubly Robust Estimator:")
    print(f"  ATE: {dr_result.estimate:.4f}")
    print(f"  95% CI: ({dr_result.confidence_interval[0]:.4f}, {dr_result.confidence_interval[1]:.4f})")
    print(f"  p-value: {dr_result.p_value:.4f}")

    # Matching Estimator
    matcher = PropensityMatcher(method='nearest_neighbor', n_neighbors=1)
    match_est = MatchingEstimator(matcher)
    match_result = match_est.estimate_ate(outcome, treatment, ps_estimated, X)
    print(f"\nMatching Estimator:")
    print(f"  ATE: {match_result.estimate:.4f}")
    print(f"  95% CI: ({match_result.confidence_interval[0]:.4f}, {match_result.confidence_interval[1]:.4f})")
    print(f"  p-value: {match_result.p_value:.4f}")

    print(f"\n[True ATE: 0.0200]")
