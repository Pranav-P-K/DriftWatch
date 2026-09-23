"""
drift_metrics_categorical.py
----------------------------
Statistical tests for categorical drift detection:
1. Categorical Population Stability Index (PSI) with Laplace smoothing
2. Chi-Square Goodness-of-Fit Test (p-value and chi2 statistic)
3. Cramer's V effect size association metric
4. Multi-feature categorical drift aggregator
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Tuple, List


def calculate_categorical_psi(
    baseline_distribution: Dict[str, float],
    current_counts: Dict[str, int],
    epsilon: float = 1e-4
) -> float:
    """
    Computes categorical PSI using pre-calculated baseline reference proportions
    and current micro-batch category counts.
    
    PSI = SUM( (Actual_i - Expected_i) * ln(Actual_i / Expected_i) )
    Applies Laplace smoothing (epsilon) to avoid division by zero or log(0).
    """
    all_categories = sorted(list(set(baseline_distribution.keys()).union(set(current_counts.keys()))))
    
    total_current = sum(current_counts.values())
    if total_current == 0:
        return 0.0
    
    psi_total = 0.0
    for cat in all_categories:
        # Expected proportion from baseline
        expected_prop = baseline_distribution.get(cat, 0.0)
        # Observed proportion from current batch
        observed_prop = current_counts.get(cat, 0) / total_current
        
        # Smooth
        p_exp = expected_prop + epsilon
        p_act = observed_prop + epsilon
        
        psi_total += (p_act - p_exp) * np.log(p_act / p_exp)
        
    return float(max(0.0, psi_total))


def calculate_chi_square(
    baseline_distribution: Dict[str, float],
    current_counts: Dict[str, int],
    epsilon: float = 1e-3
) -> Tuple[float, float]:
    """
    Chi-Square Goodness-of-Fit test comparing observed counts in current batch
    against expected counts based on baseline proportion:
    
    Expected_i = N_current * Expected_prop_i
    
    Returns (chi2_statistic, p_value).
    """
    all_categories = sorted(list(set(baseline_distribution.keys()).union(set(current_counts.keys()))))
    total_current = sum(current_counts.values())
    
    if total_current == 0 or len(all_categories) <= 1:
        return 0.0, 1.0
        
    observed = []
    expected = []
    
    for cat in all_categories:
        obs = float(current_counts.get(cat, 0))
        exp = float(baseline_distribution.get(cat, 0.0) * total_current)
        
        # Avoid zero expected counts for numerical stability
        obs = max(epsilon, obs)
        exp = max(epsilon, exp)
        
        observed.append(obs)
        expected.append(exp)
        
    # Normalize expected to match observed sum exactly
    scale = sum(observed) / sum(expected)
    expected = [e * scale for e in expected]
    
    try:
        chi2_stat, p_val = stats.chisquare(f_obs=observed, f_exp=expected)
        return float(chi2_stat), float(p_val)
    except Exception:
        return 0.0, 1.0


def calculate_cramers_v(
    chi2_stat: float,
    n_samples: int,
    k_categories: int
) -> float:
    """
    Calculates Cramer's V measure of association:
    V = sqrt( chi2 / (n * min(r-1, c-1)) )
    For goodness-of-fit with k categories: V = sqrt( chi2 / (n * (k - 1)) )
    """
    if n_samples <= 0 or k_categories <= 1:
        return 0.0
    val = np.sqrt(chi2_stat / (n_samples * (k_categories - 1)))
    return float(np.clip(val, 0.0, 1.0))


def evaluate_batch_categorical_drift(
    baseline_profiles: Dict[str, Dict[str, float]],
    batch_df: pd.DataFrame,
    categorical_cols: List[str],
    psi_threshold: float = 0.25,
    p_val_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Evaluates all categorical columns in a micro-batch against baseline profiles.
    Returns per-feature metrics and sustained drift indicators.
    """
    results = {}
    any_drift = False
    
    for col in categorical_cols:
        if col not in batch_df.columns or col not in baseline_profiles:
            continue
            
        current_counts = batch_df[col].astype(str).value_counts().to_dict()
        base_dist = baseline_profiles[col]
        
        psi = calculate_categorical_psi(base_dist, current_counts)
        chi2_stat, p_val = calculate_chi_square(base_dist, current_counts)
        cramers_v = calculate_cramers_v(chi2_stat, len(batch_df), len(base_dist))
        
        # Dual-confirmation drift condition for categorical
        is_drift = (psi >= psi_threshold) and (p_val < p_val_threshold)
        if is_drift:
            any_drift = True
            
        results[col] = {
            "psi": round(psi, 4),
            "chi2": round(chi2_stat, 2),
            "p_val": round(p_val, 6),
            "cramers_v": round(cramers_v, 4),
            "drift_detected": is_drift
        }
        
    return {
        "features": results,
        "any_drift_detected": any_drift,
        "sample_size": len(batch_df)
    }
